"""Training script for SAM-Mamba-HoverNet on PanNuke.

Usage:
    python train.py --config configs/sam_mamba_pannuke.yaml

Or with overrides:
    python train.py --config configs/sam_mamba_pannuke.yaml \
        --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
        --fold_dirs data/pannuke/Fold1 data/pannuke/Fold2 \
        --output_dir results/run1 \
        --seed 42
"""

import argparse
import os
import sys
import json
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hover_net'))
from models.hovernet.run_desc import proc_valid_step_output, ProcTrainStep
from models.hovernet.targets import gen_targets

sys.path.insert(0, os.path.dirname(__file__))
from models.sam_mamba_hovernet import SAMMambaHoverNet
from datasets.pannuke import get_pannuke_loaders


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions (same as HoverNet)
# ──────────────────────────────────────────────────────────────────────────────

def dice_loss(pred, target, smooth=1e-3):
    """Soft Dice loss for binary segmentation."""
    pred   = torch.softmax(pred, dim=1)[:, 1]   # foreground prob
    target = target.float()
    inter  = (pred * target).sum(dim=(1, 2))
    union  = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def mse_loss(pred, target):
    return nn.functional.mse_loss(pred, target)


def msge_loss(pred, target, focus_mask):
    """MSE on spatial gradient of hover maps (Lb in HoverNet paper)."""
    def _sobel(x):
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        ky = kx.transpose(2, 3)
        gx = nn.functional.conv2d(x.unsqueeze(1), kx, padding=1).squeeze(1)
        gy = nn.functional.conv2d(x.unsqueeze(1), ky, padding=1).squeeze(1)
        return gx, gy

    focus = focus_mask.float()
    loss = 0.0
    for c in range(pred.shape[1]):
        pg_x, pg_y = _sobel(pred[:, c])
        tg_x, tg_y = _sobel(target[:, c])
        loss += ((focus * (pg_x - tg_x) ** 2).sum()
               + (focus * (pg_y - tg_y) ** 2).sum()) / (focus.sum() + 1e-6)
    return loss / pred.shape[1]


def compute_loss(out_dict, batch, nr_types):
    """Compute total HoverNet-style loss."""
    np_pred = out_dict['np']                         # (B,2,H,W)
    hv_pred = out_dict['hv']                         # (B,2,H,W)
    np_true = batch['np_map'].to(np_pred.device)     # (B,H,W) long
    hv_true = batch['hv_map'].to(hv_pred.device)     # (B,2,H,W) float

    # NP branch: cross-entropy + dice
    loss_np_ce   = nn.functional.cross_entropy(np_pred, np_true)
    loss_np_dice = dice_loss(np_pred, np_true)

    # HoVer branch: MSE + gradient MSE
    focus = (np_true > 0).float()                    # only compute on nuclei
    loss_hv_mse  = mse_loss(hv_pred, hv_true)
    loss_hv_msge = msge_loss(hv_pred, hv_true, focus)

    loss = loss_np_ce + loss_np_dice + 2.0 * loss_hv_mse + 2.0 * loss_hv_msge

    # Type branch (optional)
    if nr_types is not None and 'tp' in out_dict:
        tp_pred = out_dict['tp']
        tp_true = batch['tp_map'].to(tp_pred.device)
        loss_tp_ce   = nn.functional.cross_entropy(tp_pred, tp_true)
        loss_tp_dice = dice_loss(
            torch.cat([tp_pred[:, :1], tp_pred[:, 1:].sum(1, keepdim=True)], dim=1),
            (tp_true > 0).long()
        )
        loss = loss + loss_tp_ce + loss_tp_dice

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device, epoch, writer, log_interval=50):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        imgs = batch['img'].to(device)              # (B,3,H,W)
        optimizer.zero_grad()
        out_dict = model(imgs)
        loss = compute_loss(out_dict, batch, model.nr_types)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        if (step + 1) % log_interval == 0:
            avg = total_loss / (step + 1)
            global_step = epoch * len(loader) + step
            writer.add_scalar('train/loss', avg, global_step)
            print(f'  Epoch {epoch}  Step {step+1}/{len(loader)}  loss={avg:.4f}')

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        imgs = batch['img'].to(device)
        out_dict = model(imgs)
        loss = compute_loss(out_dict, batch, model.nr_types)
        total_loss += loss.item()
    return total_loss / len(loader)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sam_checkpoint', type=str,
                   default='checkpoints/sam_vit_h_4b8939.pth')
    p.add_argument('--fold_dirs', nargs='+',
                   default=['../Auto-claude-code-research-in-sleep/data/pannuke/Fold1'])
    p.add_argument('--output_dir',   type=str, default='results/run_debug')
    p.add_argument('--nr_types',     type=int, default=6,
                   help='number of nuclear types incl background (PanNuke=6)')
    p.add_argument('--epochs',       type=int, default=50)
    p.add_argument('--batch_size',   type=int, default=8)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--lr_encoder',   type=float, default=1e-5,
                   help='SAM encoder learning rate (smaller to preserve pretrained features)')
    p.add_argument('--freeze_layers',type=int, default=20,
                   help='number of SAM transformer blocks to freeze')
    p.add_argument('--vss_depth',    type=int, default=1)
    p.add_argument('--d_state',      type=int, default=16)
    p.add_argument('--mode',         type=str, default='original',
                   choices=['original', 'fast'])
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--val_split',    type=float, default=0.1)
    p.add_argument('--seed',         type=int, default=42)
    p.add_argument('--save_every',   type=int, default=10,
                   help='save checkpoint every N epochs')
    p.add_argument('--log_interval', type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # data
    print('Building dataloaders...')
    train_loader, val_loader = get_pannuke_loaders(
        fold_dirs_train=args.fold_dirs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )
    print(f'  train batches: {len(train_loader)},  val batches: {len(val_loader)}')

    # model
    print('Building model...')
    model = SAMMambaHoverNet(
        sam_checkpoint=args.sam_checkpoint,
        nr_types=args.nr_types,
        freeze_layers=args.freeze_layers,
        vss_depth=args.vss_depth,
        d_state=args.d_state,
        mode=args.mode,
    ).to(device)

    # multi-GPU
    if torch.cuda.device_count() > 1:
        print(f'  Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)

    # optimizer with per-group lr
    inner = model.module if isinstance(model, nn.DataParallel) else model
    param_groups = inner.get_param_groups(
        lr_encoder=args.lr_encoder,
        lr_fusion=args.lr,
        lr_decoder=args.lr,
    )
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter(log_dir=str(out_dir / 'tb_logs'))

    # training loop
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, writer, args.log_interval
        )
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f'Epoch {epoch}/{args.epochs}  '
              f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
              f'time={elapsed:.1f}s')

        writer.add_scalar('val/loss', val_loss, epoch)

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                (model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
                out_dir / 'best.pth'
            )

        # periodic save
        if epoch % args.save_every == 0:
            torch.save(
                (model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
                out_dir / f'epoch_{epoch:03d}.pth'
            )

    writer.close()
    print(f'Training complete. Best val loss: {best_val_loss:.4f}')
    print(f'Checkpoints saved to: {out_dir}')


if __name__ == '__main__':
    main()
