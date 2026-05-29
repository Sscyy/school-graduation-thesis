"""SAM2 Hiera image encoder wrapper for SAM-Mamba-HoverNet.

Loads SAM2.1 Hiera Base+ image encoder and exposes its FPN feature maps
for downstream Mamba fusion.

SAM2 Base+ architecture (image encoder part only):
    Hiera trunk:
        - Patch embed: (B, 3, 1024, 1024) → (B, 112, 256, 256)  [stride=4]
        - Stage 0 (2 blocks, no pool):   (B, 112, 256, 256)
        - Stage 1 (3 blocks, Q-pool):    (B, 224, 128, 128)
        - Stage 2 (16 blocks, Q-pool):   (B, 448,  64,  64)
        - Stage 3 (3 blocks, Q-pool):    (B, 896,  64,  64)
    FPN neck:
        - Projects each stage to 256ch via Conv1x1
        - Top-down fusion on levels [2, 3]
        - scalp=1 → drops deepest (64×64 duplicate)
    Output (backbone_fpn, 3 levels):
        - level 0: (B, 256, 256, 256)   high-res shallow features
        - level 1: (B, 256, 128, 128)
        - level 2: (B, 256,  64,  64)   low-res deep features

Compared to SAM1 (ViT-H) where all 4 tapped features were 64×64,
SAM2 provides genuinely multi-scale features — FPN fusion is meaningful.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine


def _build_sam2_image_encoder() -> ImageEncoder:
    """Instantiate SAM2.1 Hiera Base+ image encoder from scratch.

    Parameters match sam2/configs/sam2.1/sam2.1_hiera_b+.yaml exactly.
    No hydra required.
    """
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
    )

    neck = FpnNeck(
        position_encoding=position_encoding,
        d_model=256,
        backbone_channel_list=[896, 448, 224, 112],
        kernel_size=1,
        stride=1,
        padding=0,
        fpn_interp_model="nearest",
        fuse_type="sum",
        fpn_top_down_levels=[2, 3],
    )

    trunk = Hiera(
        embed_dim=112,
        num_heads=2,
    )

    encoder = ImageEncoder(
        trunk=trunk,
        neck=neck,
        scalp=1,
    )

    return encoder


class SAM2Encoder(nn.Module):
    """Wraps SAM2.1 Hiera Base+ image encoder for SAM-Mamba-HoverNet.

    Loads pretrained weights from a SAM2.1 checkpoint (full model .pt file),
    extracts only the image_encoder weights, and exposes backbone_fpn features
    for downstream Mamba fusion.

    Args:
        checkpoint    : path to sam2.1_hiera_base_plus.pt
        freeze_stages : number of Hiera stages to freeze from the bottom.
                        Stage indices: 0 (shallowest) … 3 (deepest).
                        Default 2: freeze stages 0-1, fine-tune stages 2-3.
        img_size      : input image size for SAM2 (default 1024)
    """

    # SAM2 ImageNet normalisation (same as SAM1)
    PIXEL_MEAN = [123.675, 116.28,  103.53]
    PIXEL_STD  = [ 58.395,  57.12,   57.375]

    def __init__(
        self,
        checkpoint: str,
        freeze_stages: int = 2,
        img_size: int = 1024,
    ):
        super().__init__()
        self.img_size = img_size

        self.encoder = _build_sam2_image_encoder()
        self._load_checkpoint(checkpoint)
        self._freeze_stages(freeze_stages)

        # register normalisation constants as buffers (moves with .to(device))
        self.register_buffer(
            "pixel_mean",
            torch.tensor(self.PIXEL_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(self.PIXEL_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def _load_checkpoint(self, checkpoint: str):
        """Load image_encoder weights from a full SAM2 checkpoint."""
        sd_full = torch.load(checkpoint, map_location="cpu", weights_only=True)

        # SAM2 checkpoints use key "model"
        if "model" in sd_full:
            sd_full = sd_full["model"]

        # extract only image_encoder.* keys
        prefix    = "image_encoder."
        sd_encoder = {
            k[len(prefix):]: v
            for k, v in sd_full.items()
            if k.startswith(prefix)
        }

        missing, unexpected = self.encoder.load_state_dict(sd_encoder, strict=True)
        if missing:
            raise RuntimeError(f"Missing keys when loading SAM2 encoder: {missing}")
        print(f"SAM2 encoder loaded from {checkpoint}  "
              f"(unexpected keys ignored: {len(unexpected)})")

    def _freeze_stages(self, freeze_stages: int):
        """Freeze the first `freeze_stages` Hiera stages.

        Stage boundaries in Hiera are tracked by self.encoder.trunk.stage_ends.
        Stage 0 is the shallowest (highest resolution).
        """
        trunk = self.encoder.trunk

        # always freeze patch_embed and pos_embed
        for p in trunk.patch_embed.parameters():
            p.requires_grad = False
        if hasattr(trunk, "pos_embed"):
            trunk.pos_embed.requires_grad = False

        if freeze_stages <= 0:
            return

        # collect block indices belonging to each stage
        stage_ends = trunk.stage_ends  # list of last block idx per stage
        stage_starts = [0] + [e + 1 for e in stage_ends[:-1]]

        for stage_idx in range(min(freeze_stages, len(stage_ends))):
            start = stage_starts[stage_idx]
            end   = stage_ends[stage_idx] + 1
            for i in range(start, end):
                for p in trunk.blocks[i].parameters():
                    p.requires_grad = False

        # neck is always trainable (projects to 256ch)
        for p in self.encoder.neck.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> list:
        """Extract multi-scale features from SAM2 image encoder.

        Args:
            x: (B, 3, H, W) float32, pixel values in [0, 255]

        Returns:
            backbone_fpn: list of 3 tensors (B, 256, H_i, W_i)
                level 0: (B, 256, 256, 256)  — high-res
                level 1: (B, 256, 128, 128)
                level 2: (B, 256,  64,  64)  — low-res deep
        """
        B, C, H, W = x.shape

        # resize to SAM2 expected input size
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(
                x, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            )

        # normalise: pixel values [0,255] → (x - mean) / std
        x = (x - self.pixel_mean) / self.pixel_std

        out = self.encoder(x)  # dict with backbone_fpn, vision_features, vision_pos_enc
        return out["backbone_fpn"]  # list of 3 × (B, 256, H_i, W_i)

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
