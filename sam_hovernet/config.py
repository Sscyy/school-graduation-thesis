"""SAM2-AMFR-HoverNet 训练配置。"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SAMMambaConfig:
    # ── 数据 ──────────────────────────────────────────────────────────────────
    pannuke_root: str = (
        "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/PanNuke"
    )
    train_folds: List[str] = field(default_factory=lambda: ["fold1", "fold2"])
    val_folds:   List[str] = field(default_factory=lambda: ["fold3"])

    input_shape: List[int] = field(default_factory=lambda: [256, 256])
    mask_shape:  List[int] = field(default_factory=lambda: [164, 164])

    # ── SAM2 Encoder ──────────────────────────────────────────────────────────
    sam2_checkpoint: str = (
        "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/SAM_ckp/sam2.1_hiera_base_plus.pt"
    )
    freeze_stages: int = 2   # 冻结前 N 个 Hiera stage（0=全部微调，4=全部冻结）

    # ── 模型 ──────────────────────────────────────────────────────────────────
    mode: str = "fast"            # "fast"（3×3 卷积核）or "original"（5×5 卷积核）
    use_bbe:          bool = True  # 第三章：BBE 双向分支增强
    use_ea_skip:      bool = True  # 第四章：EA-Skip 边缘感知跳跃连接
    use_edge_branch:  bool = True  # 第四章：Edge Branch 边缘引导分支

    # ── 训练 ──────────────────────────────────────────────────────────────────
    epochs:      int   = 50
    batch_size:  int   = 4     # 每 GPU（SAM2 显存约束）
    num_workers: int   = 8
    seed:        int   = 42

    # 差异化学习率
    lr_encoder:  float = 1e-5  # SAM2 encoder（小 lr，保护预训练特征）
    lr_amfr:     float = 1e-4  # EA-Skip + AMFR（从零训练）
    lr_decoder:  float = 1e-4  # HoverNet decoder（从零训练）
    weight_decay: float = 1e-4

    # ── 损失权重 ──────────────────────────────────────────────────────────────
    loss_np_bce:   float = 1.0
    loss_np_dice:  float = 1.0
    loss_hv_mse:   float = 1.0
    loss_hv_msge:  float = 1.0
    loss_edge_dice: float = 1.0  # Edge Branch Dice 损失权重

    # ── 输出 ──────────────────────────────────────────────────────────────────
    output_dir:   str = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/results/sam_amfr_hovernet"
    save_every:   int = 10
    log_interval: int = 50

    @property
    def train_parquet(self) -> List[str]:
        return [f"{self.pannuke_root}/{f}-00000-of-00001.parquet" for f in self.train_folds]

    @property
    def val_parquet(self) -> List[str]:
        return [f"{self.pannuke_root}/{f}-00000-of-00001.parquet" for f in self.val_folds]
