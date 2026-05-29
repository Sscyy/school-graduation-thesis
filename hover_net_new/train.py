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

from hover_net_new.config import TrainConfig
from datasets.pannuke import get_loaders
from hover_net_new.models.hovernet.net_desc import HoVerNet


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

def _sobel(x: torch.Tensor) -> tuple:
    """Apply Sobel filter to (B, H, W) tensor. Returns (gx, gy)."""
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    x  = x.unsqueeze(1)
    gx = nn.functional.conv2d(x, kx, padding=1).squeeze(1)
    gy = nn.functional.conv2d(x, ky, padding=1).squeeze(1)
    return gx, gy


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-3) -> torch.Tensor:
    """Soft Dice loss. pred: (B,2,H,W) logits, target: (B,H,W) long."""
    pred   = torch.softmax(pred, dim=1)[:, 1]
    target = target.float()
    inter  = (pred * target).sum(dim=(1, 2))
    union  = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def msge_loss(
    pred: torch.Tensor, target: torch.Tensor, focus: torch.Tensor
) -> torch.Tensor:
    """MSE on spatial gradient of hover maps (gradient term in HoverNet loss)."""
    loss = 0.0
    for c in range(pred.shape[1]):
        pg_x, pg_y = _sobel(pred[:, c])
        tg_x, tg_y = _sobel(target[:, c])
        loss += ((focus * (pg_x - tg_x) ** 2).sum()
               + (focus * (pg_y - tg_y) ** 2).sum()) / (focus.sum() + 1e-6)
    return loss / pred.shape[1]


def compute_loss(out: dict, batch: dict, cfg: TrainConfig) -> torch.Tensor:
    dev = out["np"].device

    np_pred = out["np"]                            # (B, 2, H, W)
    hv_pred = out["hv"]                            # (B, 2, H, W)
    np_true = batch["np_map"].to(dev)              # (B, H, W) long
    hv_true = batch["hv_map"].to(dev)              # (B, 2, H, W)

    # NP branch
    loss_np = (
        cfg.loss_np_bce  * nn.functional.cross_entropy(np_pred, np_true)
        + cfg.loss_np_dice * dice_loss(np_pred, np_true)
    )

    # HoVer branch
    focus = (np_true > 0).float()
    loss_hv = (
        cfg.loss_hv_mse  * nn.functional.mse_loss(hv_pred, hv_true)
        + cfg.loss_hv_msge * msge_loss(hv_pred, hv_true, focus)
    )

    loss = loss_np + loss_hv

    # TP branch
    if "tp" in out:
        tp_pred = out["tp"]                        # (B, nr_types, H, W)
        tp_true = batch["tp_map"].to(dev)          # (B, H, W) long
        binary_tp = torch.cat(
            [tp_pred[:, :1], tp_pred[:, 1:].sum(1, keepdim=True)], dim=1
        )
        loss += (
            cfg.loss_tp_bce  * nn.functional.cross_entropy(tp_pred, tp_true)
            + cfg.loss_tp_dice * dice_loss(binary_tp, (tp_true > 0).long())
        )

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ──────────────────────────────────────────────────────────────────────────────

def setup_ddp():
    """Initialize DDP from torchrun environment variables."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Train / validate
# ──────────────────────────────────────────────────────────────────────────────

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
            avg = total_loss / (step + 1)
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


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing — overrides TrainConfig fields
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HoverNet PanNuke training")
    p.add_argument("--pannuke_root",  type=str)
    p.add_argument("--output_dir",    type=str)
    p.add_argument("--epochs",        type=int)
    p.add_argument("--batch_size",    type=int)
    p.add_argument("--lr",            type=float)
    p.add_argument("--num_workers",   type=int)
    p.add_argument("--seed",          type=int)
    p.add_argument("--save_every",    type=int)
    p.add_argument("--log_interval",  type=int)
    p.add_argument("--mode",          type=str, choices=["original", "fast"])
    return p.parse_args()


def apply_overrides(cfg: TrainConfig, args) -> TrainConfig:
    for field in vars(args):
        val = getattr(args, field)
        if val is not None and hasattr(cfg, field):
            setattr(cfg, field, val)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    cfg = apply_overrides(TrainConfig(), parse_args())

    # reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # output dir (rank 0 only)
    out_dir = Path(cfg.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {out_dir}")
        print(f"World size: {world_size}")

    # data
    if rank == 0:
        print("Loading datasets...")
    train_loader, val_loader, train_sampler = get_loaders(cfg, rank, world_size)
    if rank == 0:
        print(f"  train batches: {len(train_loader)},  val batches: {len(val_loader)}")

    # model
    model = HoVerNet(
        input_ch=3,
        nr_types=cfg.nr_types,
        freeze=False,
        mode=cfg.mode,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    inner = model.module if world_size > 1 else model

    # optimizer + scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # tensorboard (rank 0 only)
    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs")) if rank == 0 else None

    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, cfg, device, epoch, writer, rank
        )
        val_loss = validate(model, val_loader, cfg, device)
        scheduler.step()

        # aggregate val_loss across ranks
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
