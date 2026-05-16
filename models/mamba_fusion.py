"""Mamba feature fusion module for SAM-Mamba-HoverNet.

Takes the 4 multi-scale feature maps from SAMEncoder and fuses them
into a single feature map that is fed to the HoverNet decoder.

Architecture:
    Input:  4 x (B, C, 64, 64)  — from SAM encoder tapped at depths 7,15,23,31
    Per-scale: VSS Block (2D Mamba) models long-range dependencies
    Fusion:    FPN-style top-down upsampling + lateral connections
    Output: (B, 1024, H_out, W_out)  — matches HoverNet's conv_bot output dim

Why Mamba here:
    SAM ViT-H uses windowed attention (window size 14) in most blocks,
    meaning each token only attends to its local 14x14 window.
    The Mamba SSM processes the full 64x64 = 4096 token sequence
    without quadratic cost, capturing cross-window (cross-patch) context.
    This is the core mechanism for resolving patch-boundary artifacts.

VSS Block reference: VMamba (arXiv 2401.13088), Visual State Space Model.
We use a simplified 2D-selective-scan version compatible with mamba-ssm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SSMScan2D(nn.Module):
    """Simplified 2D state space scan using mamba-ssm.

    Scans the 2D feature map in 4 directions (H-forward, H-backward,
    W-forward, W-backward) and merges the outputs. This is the core
    of the VSS Block from VMamba.

    Falls back to a depthwise conv approximation if mamba-ssm is not
    installed, so the code remains importable without GPU dependencies.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        try:
            from mamba_ssm import Mamba
            self.use_mamba = True
            # 4 directional scans, each a 1D Mamba on the flattened sequence
            self.mamba_h_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=expand)
            self.mamba_h_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=expand)
            self.mamba_w_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=expand)
            self.mamba_w_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=expand)
        except ImportError:
            # fallback: depthwise separable conv approximates local SSM
            self.use_mamba = False
            self.fallback_conv = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model),
                nn.Conv2d(d_model, d_model, kernel_size=1),
                nn.GELU(),
            )

        self.merge_proj = nn.Linear(d_model * 4 if self.use_mamba else d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape

        if not self.use_mamba:
            return self.fallback_conv(x)

        # reshape to sequence for each scan direction
        # H-direction scan: (B, W, H, C) -> (B*W, H, C)
        x_hw = x.permute(0, 3, 2, 1).contiguous()        # (B, W, H, C)
        seq_h = x_hw.view(B * W, H, C)
        h_fwd = self.mamba_h_fwd(seq_h)                  # (B*W, H, C)
        h_bwd = self.mamba_h_bwd(seq_h.flip(1)).flip(1)  # (B*W, H, C)

        # W-direction scan: (B, H, W, C) -> (B*H, W, C)
        x_wh = x.permute(0, 2, 3, 1).contiguous()        # (B, H, W, C)
        seq_w = x_wh.view(B * H, W, C)
        w_fwd = self.mamba_w_fwd(seq_w)                  # (B*H, W, C)
        w_bwd = self.mamba_w_bwd(seq_w.flip(1)).flip(1)  # (B*H, W, C)

        # reshape back to (B, H, W, C)
        h_fwd = h_fwd.view(B, W, H, C).permute(0, 2, 1, 3)  # (B, H, W, C)
        h_bwd = h_bwd.view(B, W, H, C).permute(0, 2, 1, 3)
        w_fwd = w_fwd.view(B, H, W, C)
        w_bwd = w_bwd.view(B, H, W, C)

        # merge 4 directions
        merged = torch.cat([h_fwd, h_bwd, w_fwd, w_bwd], dim=-1)  # (B, H, W, 4C)
        out = self.merge_proj(merged)                               # (B, H, W, C)
        return out.permute(0, 3, 1, 2).contiguous()                 # (B, C, H, W)


class VSSBlock(nn.Module):
    """Visual State Space Block (simplified VMamba-style).

    Structure:
        LayerNorm -> SSMScan2D -> residual
        LayerNorm -> FFN        -> residual

    Args:
        dim      : feature channels
        d_state  : SSM state dimension
        expand   : SSM inner expand ratio
        mlp_ratio: FFN hidden dim ratio
    """

    def __init__(self, dim: int, d_state: int = 16, expand: int = 2, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ssm   = SSMScan2D(dim, d_state=d_state, expand=expand)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # SSM branch
        x_norm = x.permute(0, 2, 3, 1)           # (B, H, W, C)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)       # (B, C, H, W)
        x = x + self.ssm(x_norm)

        # FFN branch
        x_norm = x.permute(0, 2, 3, 1)            # (B, H, W, C)
        x_norm = self.ffn(self.norm2(x_norm))
        x = x + x_norm.permute(0, 3, 1, 2)

        return x


class MambaFusion(nn.Module):
    """Fuses 4 SAM encoder feature maps using Mamba (VSS Blocks) + FPN.

    Input:  list of 4 tensors, each (B, in_channels, 64, 64)
            ordered from shallow (block 7) to deep (block 31)
    Output: (B, out_channels, 64, 64)
            This is fed to the HoverNet decoder as `d3` (the bottleneck).

    The HoverNet decoder also needs d0, d1, d2 for skip connections.
    We upsample the fused features and provide them as pseudo skip maps.

    Pipeline per scale:
        feature_i -> VSSBlock -> lateral_proj_i
    Top-down FPN fusion:
        P4 = lateral_4
        P3 = lateral_3 + upsample(P4)
        P2 = lateral_2 + upsample(P3)
        P1 = lateral_1 + upsample(P2)
    Final output: P1 (finest scale) projected to out_channels.

    Args:
        in_channels  : channels of each input feature map (256 from SAMEncoder)
        out_channels : output channels for the fused feature map (1024 to match
                       HoverNet's conv_bot output)
        vss_depth    : number of VSS blocks per scale level
        d_state      : SSM state dimension
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 1024,
        vss_depth: int = 1,
        d_state: int = 16,
    ):
        super().__init__()
        self.n_scales = 4

        # per-scale VSS blocks
        self.vss_blocks = nn.ModuleList([
            nn.Sequential(*[VSSBlock(in_channels, d_state=d_state) for _ in range(vss_depth)])
            for _ in range(self.n_scales)
        ])

        # lateral projections (all scales -> in_channels, already same)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
            for _ in range(self.n_scales)
        ])

        # top-down fusion projections (after adding adjacent scales)
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            for _ in range(self.n_scales - 1)
        ])

        # final projection to HoverNet bottleneck dimension
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # skip connection projections for HoverNet decoder
        # HoverNet decoder expects: d0(256ch), d1(512ch), d2(1024ch), d3(1024ch)
        # We provide pseudo skips from the fused feature at different scales
        self.skip_proj = nn.ModuleList([
            nn.Conv2d(in_channels, ch, kernel_size=1)
            for ch in [256, 512, 1024]
        ])

    def forward(self, features):
        """
        Args:
            features: list of 4 tensors (B, in_channels, 64, 64)
                      ordered shallow -> deep

        Returns:
            d3:   (B, out_channels, 64, 64)  — bottleneck for HoverNet decoder
            skips: list of 3 tensors for HoverNet skip connections
                   [d0: (B,256,H,W), d1: (B,512,H,W), d2: (B,1024,H,W)]
                   spatial sizes match what HoverNet decoder expects after crop
        """
        assert len(features) == self.n_scales

        # step 1: per-scale Mamba processing
        processed = []
        for i, (feat, vss) in enumerate(zip(features, self.vss_blocks)):
            processed.append(vss(feat))  # each (B, in_channels, 64, 64)

        # step 2: lateral projections
        laterals = [conv(p) for conv, p in zip(self.lateral_convs, processed)]

        # step 3: top-down FPN fusion (deep -> shallow)
        # laterals[3] is deepest, laterals[0] is shallowest
        fpn = [None] * self.n_scales
        fpn[-1] = laterals[-1]
        for i in range(self.n_scales - 2, -1, -1):
            upsampled = F.interpolate(
                fpn[i + 1], size=laterals[i].shape[-2:],
                mode='nearest'
            )
            fpn[i] = self.fusion_convs[i](laterals[i] + upsampled)

        # step 4: output — use finest (shallowest) fused feature as d3
        d3 = self.out_proj(fpn[0])  # (B, out_channels, 64, 64)

        # step 5: skip connections for HoverNet decoder
        # HoverNet crops d0 and d1 during forward, so spatial size just needs
        # to be large enough; we keep all at 64x64 and let the decoder crop
        skips = [proj(fpn[i]) for i, proj in enumerate(self.skip_proj)]
        # skips[0]: (B,256,64,64)  -> used as d0 equivalent
        # skips[1]: (B,512,64,64)  -> used as d1 equivalent
        # skips[2]: (B,1024,64,64) -> used as d2 equivalent

        return d3, skips
