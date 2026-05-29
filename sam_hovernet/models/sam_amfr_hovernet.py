"""SAM2-BBE-HoverNet: 面向病理图像细胞核实例分割的改进框架

改进模块：
  1. SAM2.1 Hiera Base+ Encoder 替换 ResNet50（解决预训练域偏移）
  2. BBE 双向分支增强模块（NP/HoVer 分支互相调制，强化多任务协同）
  3. Edge Branch + EA-Skip 边缘感知跳跃连接（第四章，强化边界感知）
"""

import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hover_net_new.models.hovernet.net_utils import DenseBlock, TFSamepaddingLayer, UpSample2x
from hover_net_new.models.hovernet.utils import crop_op
from .sam2_encoder import SAM2Encoder


# ─────────────────────────────────────────────────────────────────────────────
# Decoder branch factory
# ─────────────────────────────────────────────────────────────────────────────

def _create_decoder_branch(in_ch: int, out_ch: int, ksize: int = 3):
    u3 = nn.Sequential(OrderedDict([
        ("conva", nn.Conv2d(in_ch, 256, ksize, stride=1, padding=0, bias=False)),
        ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
        ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False)),
    ]))
    u2 = nn.Sequential(OrderedDict([
        ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
        ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
        ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False)),
    ]))
    u1 = nn.Sequential(OrderedDict([
        ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
        ("conva",     nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False)),
    ]))
    u0 = nn.Sequential(OrderedDict([
        ("bn",   nn.BatchNorm2d(64, eps=1e-5)),
        ("relu", nn.ReLU(inplace=True)),
        ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True)),
    ]))
    return nn.Sequential(OrderedDict([
        ("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)
    ]))


# ─────────────────────────────────────────────────────────────────────────────
# EdgeAwareModule（EA-Skip，第四章使用）
# ─────────────────────────────────────────────────────────────────────────────

class EdgeAwareModule(nn.Module):
    """边缘感知增强模块（EA-Skip）。

    对 SAM2.1 FPN Level 0 细粒度特征做边缘增强，
    使 skip connection 携带更丰富的边界几何信息。
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.edge_conv     = nn.Conv2d(channels, channels, 3,
                                       padding=1, groups=channels, bias=False)
        self.semantic_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn            = nn.BatchNorm2d(channels)
        self.relu          = nn.ReLU(inplace=True)
        self._init_laplacian()

    def _init_laplacian(self):
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        self.edge_conv.weight.data = (
            lap.view(1, 1, 3, 3)
               .expand(self.edge_conv.weight.shape[0], 1, 3, 3)
               .clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.edge_conv(x) + self.semantic_conv(x)))


# ─────────────────────────────────────────────────────────────────────────────
# FPNAdapter：FPN 特征 → Decoder skip connections
# ─────────────────────────────────────────────────────────────────────────────

class FPNAdapter(nn.Module):
    """FPN 多尺度特征适配器。

    将 SAM2.1 FPN 输出的三层特征（256ch 均匀）投影到 HoverNet decoder
    所需的通道维度，并生成 bottleneck。每个分支使用独立的投影层，
    允许两个分支从相同的 FPN 特征中学习各自所需的通道表示。

    输入：f0(256×256,256ch), f1(128×128,256ch), f2(64×64,256ch)
    输出：
        np_skips : [(B,256,256,256), (B,512,128,128), (B,1024,64,64)]
        hv_skips : 同结构
        d3       : (B, 1024, 32, 32)
    """

    def __init__(self, fpn_ch: int = 256):
        super().__init__()
        skip_chs = [256, 512, 1024]
        # 两个分支各自独立的通道投影
        self.np_proj = nn.ModuleList([nn.Conv2d(fpn_ch, c, 1) for c in skip_chs])
        self.hv_proj = nn.ModuleList([nn.Conv2d(fpn_ch, c, 1) for c in skip_chs])
        # bottleneck：f2(64×64) → d3(32×32, 1024ch)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(fpn_ch, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

    def forward(self, f0: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor):
        fpn_levels = [f0, f1, f2]
        np_skips = [self.np_proj[k](fpn_levels[k]) for k in range(3)]
        hv_skips = [self.hv_proj[k](fpn_levels[k]) for k in range(3)]
        d3 = self.bottleneck(f2)
        return np_skips, hv_skips, d3


# ─────────────────────────────────────────────────────────────────────────────
# BBE：双向分支增强模块（第三章核心创新）
# ─────────────────────────────────────────────────────────────────────────────

class BidirectionalBranchEnhancement(nn.Module):
    """双向分支增强模块（Bidirectional Branch Enhancement, BBE）。

    NP 和 HoVer 两个解码分支在完成第一层解码（u3）后，
    互相以对方的全局语义统计为条件，生成各自的通道级注意力权重：

        HV → NP：HV 分支的距离/位置特征指导 NP 分支关注核边界相关通道
        NP → HV：NP 分支的分割置信度指导 HV 分支在核内区域精确回归距离

    实现采用 SE（Squeeze-and-Excitation）风格的跨分支通道注意力，
    通过全局平均池化提取分支级语义统计，计算复杂度 O(C²)，
    避免空间注意力的二次复杂度，适合高分辨率特征图。

    Args:
        channels  : 输入特征通道数（u3 输出 = 512）
        reduction : 通道压缩比
    """

    def __init__(self, channels: int = 512, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 32)

        # HV → NP：用 HV 全局特征生成 NP 通道权重
        self.hv_to_np = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )
        # NP → HV：用 NP 全局特征生成 HV 通道权重
        self.np_to_hv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )
        # 零初始化最后的线性层，确保训练初期为恒等映射
        nn.init.zeros_(self.hv_to_np[-2].weight)
        nn.init.zeros_(self.np_to_hv[-2].weight)

    def forward(self, F_np: torch.Tensor, F_hv: torch.Tensor):
        B, C, H, W = F_np.shape

        # HV 全局信息 → NP 通道权重
        gate_np = self.hv_to_np(F_hv).view(B, C, 1, 1)
        # NP 全局信息 → HV 通道权重
        gate_hv = self.np_to_hv(F_np).view(B, C, 1, 1)

        # 残差调制：F_out = F + F * gate（初始 gate≈0 → 恒等映射）
        return F_np + F_np * gate_np, F_hv + F_hv * gate_hv


# ─────────────────────────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────────────────────────

class SAM2AMFRHoverNet(nn.Module):
    """SAM2-BBE-HoverNet 完整模型。

    Args:
        sam2_checkpoint : SAM2.1 Hiera Base+ 权重文件路径
        freeze_stages   : 冻结前 N 个 Hiera stage（默认 2）
        mode            : 'fast'（3×3）或 'original'（5×5）
        use_bbe         : True=A3（启用 BBE 双向增强）False=A2（消融基线）
    """

    def __init__(
        self,
        sam2_checkpoint: str,
        freeze_stages: int = 2,
        mode: str = 'fast',
        use_bbe: bool = True,
        use_ea_skip: bool = True,
        use_edge_branch: bool = True,
    ):
        super().__init__()
        self.mode            = mode
        self.use_bbe         = use_bbe
        self.use_ea_skip     = use_ea_skip
        self.use_edge_branch = use_edge_branch

        # SAM2.1 Hiera encoder
        self.sam2_encoder = SAM2Encoder(
            checkpoint=sam2_checkpoint,
            freeze_stages=freeze_stages,
        )

        # EA-Skip：Level 0 边缘增强（第四章，可选）
        if use_ea_skip:
            self.ea_skip = EdgeAwareModule(channels=256)

        # FPN Adapter：多尺度特征 → decoder skip connections
        self.fpn_adapter = FPNAdapter(fpn_ch=256)

        # BBE：双向分支增强（第三章，可选）
        if use_bbe:
            self.bbe = BidirectionalBranchEnhancement(channels=512, reduction=8)

        # HoverNet 解码分支
        ksize = 5 if mode == 'original' else 3
        branches = [
            ("np",   _create_decoder_branch(1024, 2, ksize)),
            ("hv",   _create_decoder_branch(1024, 2, ksize)),
        ]
        if use_edge_branch:
            branches.append(("edge", _create_decoder_branch(1024, 1, ksize)))
        self.decoder = nn.ModuleDict(OrderedDict(branches))
        self.upsample = UpSample2x()

    def forward(self, imgs: torch.Tensor) -> dict:
        # ── 1. SAM2 encoder ────────────────────────────────────────────────
        features = self.sam2_encoder(imgs)
        f0 = features[0]   # (B, 256, 256, 256)
        f1 = features[1]   # (B, 256, 128, 128)
        f2 = features[2]   # (B, 256,  64,  64)

        # ── 2. EA-Skip（第四章，可选）──────────────────────────────────────
        f0_enhanced = self.ea_skip(f0) if self.use_ea_skip else f0

        # ── 3. FPN Adapter → skip connections + bottleneck ─────────────────
        np_skips, hv_skips, d3 = self.fpn_adapter(f0_enhanced, f1, f2)

        # ── 4. 裁剪 skip 至解码器目标尺寸 ──────────────────────────────────
        if self.mode == 'fast':
            crop_np = [crop_op(np_skips[0], [92, 92]),
                       crop_op(np_skips[1], [36, 36]),
                       np_skips[2]]
            crop_hv = [crop_op(hv_skips[0], [92, 92]),
                       crop_op(hv_skips[1], [36, 36]),
                       hv_skips[2]]
        else:
            crop_np = [crop_op(np_skips[0], [184, 184]),
                       crop_op(np_skips[1], [72,  72]),
                       np_skips[2]]
            crop_hv = [crop_op(hv_skips[0], [184, 184]),
                       crop_op(hv_skips[1], [72,  72]),
                       hv_skips[2]]

        # ── 5. 解码：NP 和 HV 先跑 u3，再用 BBE 互相增强，再继续 u2/u1/u0 ─
        d0_np, d1_np, d2_np = crop_np
        d0_hv, d1_hv, d2_hv = crop_hv

        # u3
        np_u3 = self.decoder["np"][0](self._align_add(self.upsample(d3), d2_np))
        hv_u3 = self.decoder["hv"][0](self._align_add(self.upsample(d3), d2_hv))

        # BBE（双向分支增强，仅 A3 启用）
        if self.use_bbe:
            np_u3, hv_u3 = self.bbe(np_u3, hv_u3)

        # NP: u2 → u1 → u0
        np_u2  = self.decoder["np"][1](self._align_add(self.upsample(np_u3), d1_np))
        np_u1  = self.decoder["np"][2](self._align_add(self.upsample(np_u2), d0_np))
        np_out = self.decoder["np"][3](np_u1)

        # HV: u2 → u1 → u0
        hv_u2  = self.decoder["hv"][1](self._align_add(self.upsample(hv_u3), d1_hv))
        hv_u1  = self.decoder["hv"][2](self._align_add(self.upsample(hv_u2), d0_hv))
        hv_out = self.decoder["hv"][3](hv_u1)

        out = {"np": np_out, "hv": hv_out}

        # Edge branch（第四章，可选）
        if self.use_edge_branch:
            out["edge"] = self._decode(self.decoder["edge"], crop_np, d3)

        return out

    def _decode(self, branch: nn.Module, skips: list, d3: torch.Tensor) -> torch.Tensor:
        """通用单分支解码（不经过 BBE）。"""
        d0, d1, d2 = skips
        u3 = branch[0](self._align_add(self.upsample(d3), d2))
        u2 = branch[1](self._align_add(self.upsample(u3), d1))
        u1 = branch[2](self._align_add(self.upsample(u2), d0))
        return branch[3](u1)

    @staticmethod
    def _align_add(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            dh = x.shape[2] - skip.shape[2]
            dw = x.shape[3] - skip.shape[3]
            x = x[:, :,
                  dh // 2: x.shape[2] - (dh - dh // 2),
                  dw // 2: x.shape[3] - (dw - dw // 2)]
        return x + skip

    def get_param_groups(
        self,
        lr_encoder: float = 1e-5,
        lr_amfr:    float = 1e-4,
        lr_decoder: float = 1e-4,
    ) -> list:
        extra = list(self.fpn_adapter.parameters())
        if self.use_ea_skip:
            extra += list(self.ea_skip.parameters())
        if self.use_bbe:
            extra += list(self.bbe.parameters())
        return [
            {'params': self.sam2_encoder.get_trainable_params(),
             'lr': lr_encoder, 'name': 'encoder'},
            {'params': extra,
             'lr': lr_amfr, 'name': 'fpn_bbe'},
            {'params': self.decoder.parameters(),
             'lr': lr_decoder, 'name': 'decoder'},
        ]
