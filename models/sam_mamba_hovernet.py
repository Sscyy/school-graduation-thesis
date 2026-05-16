"""SAM-Mamba-HoverNet: Full model definition.

Assembles:
    SAMEncoder      — feature extraction (SAM ViT-H backbone)
    MambaFusion     — long-range feature fusion (VSS Blocks + FPN)
    HoverNet decoder — three-branch output (NP / HoVer / NC)

The HoverNet decoder is taken verbatim from hover_net/models/hovernet/net_desc.py
and wired to receive features from MambaFusion instead of ResNet50.

Input:  (B, 3, H, W)  uint8 or float images, pixel values in [0, 255]
Output: dict with keys
    'np' : (B, 2,       h, w)  nuclear pixel logits
    'hv' : (B, 2,       h, w)  hover map regression
    'tp' : (B, nr_types,h, w)  type logits  (only if nr_types > 0)
"""

import sys
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# HoverNet decoder utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hover_net'))
from models.hovernet.net_utils import DenseBlock, TFSamepaddingLayer, UpSample2x
from models.hovernet.utils import crop_op

from .sam_encoder import SAMEncoder
from .mamba_fusion import MambaFusion


def _create_hovernet_decoder_branch(in_ch: int, out_ch: int, ksize: int = 5):
    """Recreate a single HoverNet decoder branch.

    Identical to hover_net/models/hovernet/net_desc.py::create_decoder_branch
    but parameterised on input channels (in_ch) so we can wire it to
    MambaFusion output (1024 ch) instead of the original ResNet bottleneck.
    """
    module_list = [
        ("conva", nn.Conv2d(in_ch, 256, ksize, stride=1, padding=0, bias=False)),
        ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
        ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False)),
    ]
    u3 = nn.Sequential(OrderedDict(module_list))

    module_list = [
        ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
        ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
        ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False)),
    ]
    u2 = nn.Sequential(OrderedDict(module_list))

    module_list = [
        ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
        ("conva",     nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False)),
    ]
    u1 = nn.Sequential(OrderedDict(module_list))

    module_list = [
        ("bn",   nn.BatchNorm2d(64, eps=1e-5)),
        ("relu", nn.ReLU(inplace=True)),
        ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True)),
    ]
    u0 = nn.Sequential(OrderedDict(module_list))

    return nn.Sequential(OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)]))


class SAMMambaHoverNet(nn.Module):
    """SAM-Mamba-HoverNet full model.

    Args:
        sam_checkpoint : path to sam_vit_h_*.pth
        nr_types       : number of nuclear types including background (e.g. 6
                         for PanNuke). Set to None to disable classification branch.
        freeze_layers  : how many SAM transformer blocks to freeze (default 20)
        vss_depth      : VSS blocks per FPN scale level (default 1)
        d_state        : Mamba SSM state dimension (default 16)
        mode           : 'original' or 'fast' (controls decoder kernel size,
                         same meaning as in HoverNet)
    """

    def __init__(
        self,
        sam_checkpoint: str,
        nr_types: int = None,
        freeze_layers: int = 20,
        vss_depth: int = 1,
        d_state: int = 16,
        mode: str = 'original',
    ):
        super().__init__()
        self.nr_types = nr_types
        self.mode = mode

        # --- encoder ---
        self.sam_encoder = SAMEncoder(
            checkpoint=sam_checkpoint,
            out_channels=256,
            freeze_layers=freeze_layers,
        )

        # --- fusion ---
        self.mamba_fusion = MambaFusion(
            in_channels=256,
            out_channels=1024,
            vss_depth=vss_depth,
            d_state=d_state,
        )

        # adapter: bring skip dims to what HoverNet decoder expects
        # HoverNet original: d0=256, d1=512, d2=1024, d3=1024
        # MambaFusion skips: already [256, 512, 1024] — no adapters needed

        # --- decoder (HoverNet-style three branches) ---
        ksize = 5 if mode == 'original' else 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(OrderedDict([
                ("np", _create_hovernet_decoder_branch(1024, 2,        ksize)),
                ("hv", _create_hovernet_decoder_branch(1024, 2,        ksize)),
            ]))
        else:
            self.decoder = nn.ModuleDict(OrderedDict([
                ("tp", _create_hovernet_decoder_branch(1024, nr_types, ksize)),
                ("np", _create_hovernet_decoder_branch(1024, 2,        ksize)),
                ("hv", _create_hovernet_decoder_branch(1024, 2,        ksize)),
            ]))

        self.upsample2x = UpSample2x()

    def forward(self, imgs: torch.Tensor):
        """
        Args:
            imgs: (B, 3, H, W) float32, pixel values in [0, 255]

        Returns:
            out_dict: OrderedDict with keys 'np', 'hv', and optionally 'tp'
                      each value: (B, out_ch, h, w)
        """
        # 1. SAM encoder: 4 multi-scale features
        features = self.sam_encoder(imgs)   # list of 4 x (B, 256, 64, 64)

        # 2. Mamba fusion: bottleneck + skip features
        d3, skips = self.mamba_fusion(features)
        # d3:      (B, 1024, 64, 64)
        # skips:   [(B,256,64,64), (B,512,64,64), (B,1024,64,64)]

        # Mirror HoverNet's d = [d0, d1, d2, d3] list used in decoder
        d0 = skips[0]   # (B, 256,  64, 64)  shallow features
        d1 = skips[1]   # (B, 512,  64, 64)
        d2 = skips[2]   # (B, 1024, 64, 64)
        # d3 already (B, 1024, 64, 64)

        # 3. Crop skip connections as HoverNet does
        if self.mode == 'original':
            d0 = crop_op(d0, [184, 184])
            d1 = crop_op(d1, [72, 72])
        else:
            d0 = crop_op(d0, [92, 92])
            d1 = crop_op(d1, [36, 36])

        d = [d0, d1, d2, d3]

        # 4. Decoder branches (identical logic to HoverNet forward)
        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict

    def get_param_groups(self, lr_encoder=1e-5, lr_fusion=1e-4, lr_decoder=1e-4):
        """Return parameter groups with different learning rates.

        Encoder (SAM) should be fine-tuned with a smaller lr to preserve
        pretrained features; fusion and decoder are trained from scratch.
        """
        return [
            {'params': self.sam_encoder.get_trainable_params(), 'lr': lr_encoder},
            {'params': self.mamba_fusion.parameters(),           'lr': lr_fusion},
            {'params': self.decoder.parameters(),                'lr': lr_decoder},
        ]
