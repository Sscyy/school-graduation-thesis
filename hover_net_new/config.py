"""Training configuration for HoverNet on PanNuke.

All fields can be overridden from the command line via train.py argparse.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    # ── Data ──────────────────────────────────────────────────────────────
    pannuke_root: str = (
        "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/PanNuke"
    )
    # fold1+fold2 for training, fold3 for validation
    train_folds: List[str] = field(default_factory=lambda: ["fold1", "fold2"])
    val_folds:   List[str] = field(default_factory=lambda: ["fold3"])

    input_shape: List[int] = field(default_factory=lambda: [256, 256])
    mask_shape:  List[int] = field(default_factory=lambda: [164, 164])
    nr_types:    int = None  # None = 只做分割，不做核类型分类

    # ── Model ─────────────────────────────────────────────────────────────
    mode: str = "fast"     # "original" (270x270→80x80) or "fast" (256x256→164x164)

    # ── Training ──────────────────────────────────────────────────────────
    epochs:      int   = 50
    batch_size:  int   = 16    # per GPU
    num_workers: int   = 8
    seed:        int   = 42

    lr:          float = 2e-4
    weight_decay:float = 1e-4

    # ── Loss weights ──────────────────────────────────────────────────────
    loss_np_bce:  float = 1.0
    loss_np_dice: float = 1.0
    loss_hv_mse:  float = 1.0
    loss_hv_msge: float = 1.0

    # ── Checkpointing ─────────────────────────────────────────────────────
    output_dir:  str = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/results/hovernet_baseline"
    save_every:  int = 10   # save checkpoint every N epochs
    log_interval:int = 50   # log every N steps

    # ── DDP ───────────────────────────────────────────────────────────────
    # Set automatically by torchrun via environment variables — do not set manually.
    # Exposed here only for documentation.
    # RANK, LOCAL_RANK, WORLD_SIZE are read from os.environ in train.py.

    def parquet_paths(self, folds: List[str]) -> List[str]:
        return [
            f"{self.pannuke_root}/{fold}-00000-of-00001.parquet"
            for fold in folds
        ]

    @property
    def train_parquet(self) -> List[str]:
        return self.parquet_paths(self.train_folds)

    @property
    def val_parquet(self) -> List[str]:
        return self.parquet_paths(self.val_folds)
