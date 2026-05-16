"""SAM ViT-H encoder wrapper for SAM-Mamba-HoverNet.

Loads SAM ViT-H image encoder and extracts multi-scale features
by tapping intermediate transformer blocks. The original SAM encoder
outputs a single-scale feature map; we extract 4 intermediate feature
maps at different depths to feed the Mamba fusion module.

SAM ViT-H architecture:
    - Patch embed:  (B, 3, 1024, 1024) -> (B, 64, 64, 1280)
    - 32 transformer blocks
    - Neck: (B, 64, 64, 1280) -> (B, 256, 64, 64)

We extract features after blocks 7, 15, 23, 31 (quarter, half,
three-quarter, full depth) and project each to 256 channels.
These 4 feature maps feed the Mamba fusion module.

Input size note:
    SAM was trained on 1024x1024. For PanNuke's 256x256 patches we
    resize on the fly inside forward(). This is standard practice
    (e.g., CellViT does the same).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Indices of SAM ViT-H blocks to tap for multi-scale features.
# ViT-H has 32 blocks (0-indexed). We tap at 4 evenly-spaced depths.
TAP_INDICES = [7, 15, 23, 31]

# SAM ViT-H window attention global blocks (from SAM source)
GLOBAL_ATTN_INDEXES = [7, 15, 23, 31]


class SAMEncoder(nn.Module):
    """Wraps SAM image encoder to expose multi-scale intermediate features.

    Args:
        checkpoint    : path to sam_vit_h_*.pth weights file
        out_channels  : output channels for each scale (default 256)
        freeze_layers : number of transformer blocks to freeze from the
                        bottom (0 = freeze none, 32 = freeze all).
                        Default: freeze first 20 blocks, fine-tune last 12.
        img_size      : size to resize input images to before feeding SAM
                        encoder (SAM expects 1024; smaller speeds things up)
    """

    def __init__(
        self,
        checkpoint: str,
        out_channels: int = 256,
        freeze_layers: int = 20,
        img_size: int = 1024,
    ):
        super().__init__()
        self.img_size = img_size
        self.out_channels = out_channels

        # --- load SAM encoder ---
        try:
            from segment_anything import sam_model_registry
        except ImportError as e:
            raise ImportError(
                "segment-anything not installed. "
                "Run: pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from e

        sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        self.encoder = sam.image_encoder
        del sam  # free memory for decoder/prompt encoder we don't need

        # --- selective freezing ---
        # freeze patch embed and positional embedding always
        for p in self.encoder.patch_embed.parameters():
            p.requires_grad = False
        if hasattr(self.encoder, 'pos_embed') and self.encoder.pos_embed is not None:
            self.encoder.pos_embed.requires_grad = False

        # freeze first `freeze_layers` transformer blocks
        for i, block in enumerate(self.encoder.blocks):
            if i < freeze_layers:
                for p in block.parameters():
                    p.requires_grad = False
            else:
                for p in block.parameters():
                    p.requires_grad = True

        # neck is always trainable
        for p in self.encoder.neck.parameters():
            p.requires_grad = True

        # --- projection heads: map each tapped feature to out_channels ---
        # SAM ViT-H intermediate block output dim = 1280
        sam_hidden_dim = 1280
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(sam_hidden_dim),
                nn.Linear(sam_hidden_dim, out_channels),
            )
            for _ in TAP_INDICES
        ])

    def forward(self, x: torch.Tensor):
        """Extract multi-scale features from SAM encoder.

        Args:
            x: (B, 3, H, W) float32 images, pixel values in [0, 255]

        Returns:
            features: list of 4 tensors, each (B, out_channels, h, w)
                      spatial sizes depend on img_size:
                        img_size=1024 -> 64x64 for all scales
                        (SAM ViT-H uses fixed 16x downsampling)
                      After projection they are all the same spatial size,
                      so the Mamba fusion module receives uniform-resolution
                      multi-scale features distinguished by depth (semantics).
        """
        # normalize to SAM's expected range [0,1] then to [-mean/std, ...]
        # SAM pixel_mean/std are applied inside the model; we just resize
        B, C, H, W = x.shape

        # resize to SAM's expected input size
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(
                x, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False
            )

        # normalize: SAM expects pixel_mean subtracted, divided by pixel_std
        # SAM pixel_mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375]
        pixel_mean = torch.tensor([123.675, 116.28, 103.53],
                                   device=x.device).view(1, 3, 1, 1)
        pixel_std  = torch.tensor([58.395, 57.12, 57.375],
                                   device=x.device).view(1, 3, 1, 1)
        x = (x - pixel_mean) / pixel_std

        # patch embed: (B, 3, 1024, 1024) -> (B, 64, 64, 1280)
        x = self.encoder.patch_embed(x)
        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed

        # run transformer blocks, tapping at TAP_INDICES
        tapped = []
        tap_set = set(TAP_INDICES)
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in tap_set:
                tapped.append(x)  # each: (B, 64, 64, 1280)

        # project and reshape: (B, 64, 64, 1280) -> (B, out_channels, 64, 64)
        features = []
        for feat, proj in zip(tapped, self.proj_layers):
            feat = proj(feat)                          # (B, 64, 64, out_channels)
            feat = feat.permute(0, 3, 1, 2).contiguous()  # (B, out_channels, 64, 64)
            features.append(feat)

        return features  # list of 4 x (B, 256, 64, 64)

    def get_trainable_params(self):
        """Return only trainable parameters (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]
