"""SAM2-AMFR-HoverNet 推理与评估（PanNuke fold3）。

用法：
    python infer.py --checkpoint /path/to/best.pth
"""

import argparse
import io
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sam_hovernet.config import SAMMambaConfig
from sam_hovernet.models.sam_amfr_hovernet import SAM2AMFRHoverNet
from hover_net_new.misc.utils import cropping_center
from hover_net_new.metrics.stats_utils import get_fast_pq, get_fast_aji, remap_label
from hover_net_new.models.hovernet.post_proc import process
from viz_html import save_viz_html


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_parquet(path: str):
    import pandas as pd
    df = pd.read_parquet(path)
    samples = []
    for _, row in df.iterrows():
        img = row["image"]
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        img = np.array(img, dtype=np.uint8)

        masks = np.zeros((256, 256, 6), dtype=np.float64)
        inst_id = 1
        for inst_img, cls_idx in zip(row["instances"], row["categories"]):
            if isinstance(inst_img, dict) and "bytes" in inst_img:
                inst_img = Image.open(io.BytesIO(inst_img["bytes"]))
            inst_arr = np.array(inst_img, dtype=bool)
            masks[inst_arr, cls_idx] = inst_id
            inst_id += 1
        samples.append((img, masks))
    return samples


def masks_to_inst(masks: np.ndarray):
    """从 (H,W,6) PanNuke 掩码提取实例 ID 图（不需要类型信息）。"""
    H, W     = masks.shape[:2]
    inst_map = np.zeros((H, W), dtype=np.int32)
    cur_id   = 0
    for cls_idx in range(5):
        cls_mask = masks[..., cls_idx].astype(np.int32)
        for inst_id in np.unique(cls_mask):
            if inst_id == 0:
                continue
            cur_id += 1
            inst_map[cls_mask == inst_id] = cur_id
    return inst_map


# ─────────────────────────────────────────────────────────────────────────────
# 模型输出 → post_proc 输入
# ─────────────────────────────────────────────────────────────────────────────

def model_out_to_pred_map(out: dict) -> np.ndarray:
    """将模型输出转换为 post_proc.process() 所需的 pred_map。

    不做分类（nr_types=None），pred_map shape = (H, W, 3) = [np_prob, hv_x, hv_y]。
    """
    np_prob = F.softmax(out["np"], dim=1)[0, 1].cpu().numpy()
    hv      = out["hv"][0].cpu().numpy().transpose(1, 2, 0)   # (H, W, 2)
    return np.dstack([np_prob, hv])   # (H, W, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def make_viz(img: np.ndarray, true_inst: np.ndarray, pred_inst: np.ndarray):
    """返回 (orig_rgb, gt_rgb, pred_rgb) 三张独立 RGB 图，用于 HTML 可视化。

    - 绿色轮廓：真实标注（GT）
    - 橙色轮廓：模型预测
    """
    mask_h, mask_w = true_inst.shape[:2]
    img_crop = cropping_center(img, (mask_h, mask_w))   # RGB
    img_bgr  = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)

    def draw_contours(base_bgr, inst_map, color_bgr):
        out = base_bgr.copy()
        for inst_id in np.unique(inst_map):
            if inst_id == 0:
                continue
            mask = (inst_map == inst_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color_bgr, 1)
        return out

    gt_bgr   = draw_contours(img_bgr, true_inst, color_bgr=(0, 255, 0))    # 绿
    pred_bgr = draw_contours(img_bgr, pred_inst, color_bgr=(0, 128, 255))  # 橙

    # 返回 RGB 给 viz_html 使用
    return (
        img_crop,
        cv2.cvtColor(gt_bgr,   cv2.COLOR_BGR2RGB),
        cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    cfg = SAMMambaConfig()
    p = argparse.ArgumentParser(description="SAM2-AMFR-HoverNet inference + evaluation")
    p.add_argument("--checkpoint",      type=str, required=True)
    p.add_argument("--sam2_checkpoint", type=str, default=cfg.sam2_checkpoint)
    p.add_argument("--parquet",         type=str,
                   default=f"{cfg.pannuke_root}/fold3-00000-of-00001.parquet")
    p.add_argument("--output_dir",      type=str,
                   default=str(Path(cfg.output_dir).parent / "sam_amfr_infer"))
    p.add_argument("--viz_samples",  type=int, default=100)
    p.add_argument("--mode",         type=str, default=cfg.mode)
    p.add_argument("--freeze_stages",type=int, default=cfg.freeze_stages)
    p.add_argument("--use_bbe",         type=lambda x: x.lower() != "false",
                   default=cfg.use_bbe, metavar="BOOL")
    p.add_argument("--use_ea_skip",     type=lambda x: x.lower() != "false",
                   default=cfg.use_ea_skip, metavar="BOOL")
    p.add_argument("--use_edge_branch", type=lambda x: x.lower() != "false",
                   default=cfg.use_edge_branch, metavar="BOOL")
    p.add_argument("--device",          type=str, default="cuda")
    return p.parse_args()


def main():
    args    = parse_args()
    device  = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 模型
    print(f"Loading model checkpoint: {args.checkpoint}")
    model = SAM2AMFRHoverNet(
        sam2_checkpoint=args.sam2_checkpoint,
        freeze_stages=args.freeze_stages,
        mode=args.mode,
        use_bbe=args.use_bbe,
        use_ea_skip=args.use_ea_skip,
        use_edge_branch=args.use_edge_branch,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    # 数据
    print(f"Loading parquet: {args.parquet}")
    samples = load_parquet(args.parquet)
    print(f"  {len(samples)} images")

    input_shape = (256, 256)
    mask_shape  = (164, 164)

    # 固定种子，保证跨实验选取相同的图片（与训练随机状态完全隔离）
    _viz_rng   = random.Random(42)
    viz_indices = set(
        _viz_rng.sample(range(len(samples)), min(args.viz_samples, len(samples)))
    ) if args.viz_samples > 0 else set()
    viz_data = []   # 收集 (orig_rgb, gt_rgb, pred_rgb) 元组

    all_pq   = []
    all_aji  = []
    all_dice = []

    for idx, (img, masks) in enumerate(tqdm(samples, desc="Inferring")):
        true_inst      = masks_to_inst(masks)
        true_inst_crop = cropping_center(true_inst, mask_shape)

        img_crop   = cropping_center(img, input_shape)
        img_tensor = (
            torch.from_numpy(img_crop).permute(2, 0, 1).float().unsqueeze(0).to(device)
        )

        with torch.no_grad():
            out = model(img_tensor)

        pred_map  = model_out_to_pred_map(out)
        pred_inst, _ = process(pred_map, nr_types=None, return_centroids=True)

        true_inst_r = remap_label(true_inst_crop)
        pred_inst_r = remap_label(pred_inst)

        if true_inst_r.max() == 0 and pred_inst_r.max() == 0:
            continue

        # DICE：像素级二值对比
        pred_bin = (pred_inst_r > 0).astype(np.float32)
        true_bin = (true_inst_r > 0).astype(np.float32)
        inter    = (pred_bin * true_bin).sum()
        union    = pred_bin.sum() + true_bin.sum()
        dice     = float(2 * inter / (union + 1e-6))
        all_dice.append(dice)

        if true_inst_r.max() == 0 or pred_inst_r.max() == 0:
            all_pq.append([0.0, 0.0, 0.0])
            all_aji.append(0.0)
        else:
            [dq, sq, pq], _ = get_fast_pq(true_inst_r, pred_inst_r)
            aji             = get_fast_aji(true_inst_r, pred_inst_r)
            all_pq.append([dq, sq, pq])
            all_aji.append(aji)

        if idx in viz_indices:
            orig, gt, pred = make_viz(img_crop, true_inst_crop, pred_inst)
            viz_data.append((orig, gt, pred))

    all_pq   = np.array(all_pq)
    all_aji  = np.array(all_aji)
    all_dice = np.array(all_dice)

    results = {
        "n_images": len(samples),
        "overall": {
            "DICE": float(all_dice.mean()),
            "AJI":  float(all_aji.mean()),
            "DQ":   float(all_pq[:, 0].mean()),
            "SQ":   float(all_pq[:, 1].mean()),
            "PQ":   float(all_pq[:, 2].mean()),
        },
    }

    print("\n── Results ──────────────────────────────────")
    print(f"  DICE: {results['overall']['DICE']:.4f}")
    print(f"  AJI : {results['overall']['AJI']:.4f}")
    print(f"  DQ  : {results['overall']['DQ']:.4f}")
    print(f"  SQ  : {results['overall']['SQ']:.4f}")
    print(f"  PQ  : {results['overall']['PQ']:.4f}")

    json_path = out_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to: {json_path}")
    if viz_data:
        exp_name = Path(args.output_dir).parent.name  # 取实验目录名作为标题
        html_path = out_dir / "viz.html"
        save_viz_html(viz_data, str(html_path), title=exp_name)


if __name__ == "__main__":
    main()
