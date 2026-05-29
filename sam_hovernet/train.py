import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sam_hovernet.config import SAMMambaConfig
from sam_hovernet.models.sam_amfr_hovernet import SAM2AMFRHoverNet
from datasets.pannuke import get_loaders


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def _sobel(x: torch.Tensor):
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device
    ).view(1, 1, 3, 3)
    ky  = kx.transpose(2, 3)
    x   = x.unsqueeze(1)
    gx  = nn.functional.conv2d(x, kx, padding=1).squeeze(1)
    gy  = nn.functional.conv2d(x, ky, padding=1).squeeze(1)
    return gx, gy


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-3):
    """NP/HV 分支的 Dice 损失（softmax 激活，双通道）。"""
    pred   = torch.softmax(pred, dim=1)[:, 1]
    target = target.float()
    inter  = (pred * target).sum(dim=(1, 2))
    union  = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def dice_loss_edge(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-3):
    """Edge 分支的 Dice 损失（sigmoid 激活，单通道）。"""
    pred   = torch.sigmoid(pred[:, 0])   # (B, H, W)
    target = target.float()
    inter  = (pred * target).sum(dim=(1, 2))
    union  = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def msge_loss(pred: torch.Tensor, target: torch.Tensor, focus: torch.Tensor):
    """HoVer 分支的梯度均方误差损失。"""
    loss = 0.0
    for c in range(pred.shape[1]):
        pg_x, pg_y = _sobel(pred[:, c])
        tg_x, tg_y = _sobel(target[:, c])
        loss += ((focus * (pg_x - tg_x) ** 2).sum()
               + (focus * (pg_y - tg_y) ** 2).sum()) / (focus.sum() + 1e-6)
    return loss / pred.shape[1]


def compute_loss(out: dict, batch: dict, cfg: SAMMambaConfig) -> torch.Tensor:
    dev = out["np"].device

    np_pred = out["np"]
    hv_pred = out["hv"]
    np_true = batch["np_map"].to(dev)
    hv_true = batch["hv_map"].to(dev)

    focus = (np_true > 0).float()

    loss = (
        cfg.loss_np_bce  * nn.functional.cross_entropy(np_pred, np_true)
        + cfg.loss_np_dice * dice_loss(np_pred, np_true)
        + cfg.loss_hv_mse  * nn.functional.mse_loss(hv_pred, hv_true)
        + cfg.loss_hv_msge * msge_loss(hv_pred, hv_true, focus)
    )

    # Edge Branch 损失（仅在有 edge_map GT 时计算）
    if "edge" in out and "edge_map" in batch:
        edge_pred = out["edge"]
        edge_true = batch["edge_map"].to(dev)
        loss += cfg.loss_edge_dice * dice_loss_edge(edge_pred, edge_true)

    return loss


# ─────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_ddp():
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp(world_size: int):
    if world_size > 1:
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Train / validate
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, cfg, device, epoch, writer, rank):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        imgs = batch["img"].to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = compute_loss(out, batch, cfg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        if rank == 0 and (step + 1) % cfg.log_interval == 0:
            avg         = total_loss / (step + 1)
            global_step = epoch * len(loader) + step
            writer.add_scalar("train/loss", avg, global_step)
            print(f"  Epoch {epoch}  Step {step+1}/{len(loader)}  loss={avg:.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, cfg, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        imgs = batch["img"].to(device)
        out  = model(imgs)
        total_loss += compute_loss(out, batch, cfg).item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SAM2-AMFR-HoverNet training")
    # data
    p.add_argument("--pannuke_root",   type=str)
    p.add_argument("--output_dir",     type=str)
    # model
    p.add_argument("--sam2_checkpoint", type=str)
    p.add_argument("--freeze_stages",   type=int)
    p.add_argument("--mode",            type=str, choices=["original", "fast"])
    p.add_argument("--use_bbe", type=lambda x: x.lower() != "false",
                   default=None, metavar="BOOL",
                   help="True=启用BBE（第三章）; False=关闭")
    p.add_argument("--use_ea_skip", type=lambda x: x.lower() != "false",
                   default=None, metavar="BOOL",
                   help="True=启用EA-Skip（第四章）; False=关闭")
    p.add_argument("--use_edge_branch", type=lambda x: x.lower() != "false",
                   default=None, metavar="BOOL",
                   help="True=启用Edge Branch（第四章）; False=关闭")
    # training
    p.add_argument("--epochs",          type=int)
    p.add_argument("--batch_size",      type=int)
    p.add_argument("--lr_encoder",      type=float)
    p.add_argument("--lr_amfr",         type=float)
    p.add_argument("--lr_decoder",      type=float)
    p.add_argument("--weight_decay",    type=float)
    p.add_argument("--num_workers",     type=int)
    p.add_argument("--seed",            type=int)
    p.add_argument("--save_every",      type=int)
    p.add_argument("--log_interval",    type=int)
    # 损失权重
    p.add_argument("--loss_np_bce",    type=float)
    p.add_argument("--loss_np_dice",   type=float)
    p.add_argument("--loss_hv_mse",    type=float)
    p.add_argument("--loss_hv_msge",   type=float)
    p.add_argument("--loss_edge_dice", type=float)
    return p.parse_args()


def apply_overrides(cfg: SAMMambaConfig, args) -> SAMMambaConfig:
    for field in vars(args):
        val = getattr(args, field)
        if val is not None and hasattr(cfg, field):
            setattr(cfg, field, val)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    cfg = apply_overrides(SAMMambaConfig(), parse_args())

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    out_dir = Path(cfg.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir:    {out_dir}")
        print(f"World size:    {world_size}")
        print(f"SAM2 ckpt:     {cfg.sam2_checkpoint}")
        print(f"freeze_stages: {cfg.freeze_stages}  mode: {cfg.mode}")

    # 数据
    if rank == 0:
        print("Loading datasets...")
    train_loader, val_loader, train_sampler = get_loaders(cfg, rank, world_size)
    if rank == 0:
        print(f"  train batches: {len(train_loader)},  val batches: {len(val_loader)}")

    # 模型
    model = SAM2AMFRHoverNet(
        sam2_checkpoint=cfg.sam2_checkpoint,
        freeze_stages=cfg.freeze_stages,
        mode=cfg.mode,
        use_bbe=cfg.use_bbe,
        use_ea_skip=cfg.use_ea_skip,
        use_edge_branch=cfg.use_edge_branch,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    inner = model.module if world_size > 1 else model

    # 差异化学习率 optimizer
    param_groups = inner.get_param_groups(
        lr_encoder=cfg.lr_encoder,
        lr_amfr=cfg.lr_amfr,
        lr_decoder=cfg.lr_decoder,
    )
    optimizer = optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs")) if rank == 0 else None

    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0         = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, cfg, device, epoch, writer, rank
        )
        val_loss = validate(model, val_loader, cfg, device)
        scheduler.step()

        if world_size > 1:
            val_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_tensor.item()

        if rank == 0:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch}/{cfg.epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"time={elapsed:.1f}s"
            )
            writer.add_scalar("val/loss", val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(inner.state_dict(), out_dir / "best.pth")

            if epoch % cfg.save_every == 0:
                torch.save(inner.state_dict(), out_dir / f"epoch_{epoch:03d}.pth")

    if rank == 0:
        writer.close()
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")

    cleanup_ddp(world_size)


if __name__ == "__main__":
    main()
