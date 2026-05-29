"""PanNuke dataset loader.

读取 HuggingFace parquet 格式，生成 HoverNet 训练所需的 target（np_map,
hv_map, edge_map），支持 albumentations 数据增强和 DDP DistributedSampler。
"""

import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data
import albumentations as A
from scipy.ndimage import measurements
from skimage import morphology as morph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hover_net_new.misc.utils import cropping_center, get_bounding_box, fix_mirror_padding


PANNUKE_CLASSES = {
    0: "background",
    1: "neoplastic",
    2: "inflammatory",
    3: "connective",
    4: "dead",
    5: "epithelial",
}


# ─────────────────────────────────────────────────────────────────────────────
# Target generation
# ─────────────────────────────────────────────────────────────────────────────

def _gen_hv_map(inst_map: np.ndarray) -> np.ndarray:
    """从实例 ID 图生成 (H, W, 2) HoVer 距离图。"""
    fixed_ann      = fix_mirror_padding(inst_map.copy())
    inst_map_clean = morph.remove_small_objects(fixed_ann, min_size=30)

    x_map = np.zeros(inst_map.shape[:2], dtype=np.float32)
    y_map = np.zeros(inst_map.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(inst_map_clean))
    if 0 in inst_list:
        inst_list.remove(0)

    for inst_id in inst_list:
        inst_mask = np.array(fixed_ann == inst_id, dtype=np.uint8)
        inst_box  = get_bounding_box(inst_mask)

        inst_box[0] -= 2
        inst_box[2] -= 2
        inst_box[1] += 2
        inst_box[3] += 2

        inst_crop = inst_mask[inst_box[0]:inst_box[1], inst_box[2]:inst_box[3]]
        if inst_crop.shape[0] < 2 or inst_crop.shape[1] < 2:
            continue

        inst_com    = list(measurements.center_of_mass(inst_crop))
        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_crop.shape[1] + 1) - inst_com[1]
        inst_y_range = np.arange(1, inst_crop.shape[0] + 1) - inst_com[0]
        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        inst_x[inst_crop == 0] = 0
        inst_y[inst_crop == 0] = 0
        inst_x = inst_x.astype(np.float32)
        inst_y = inst_y.astype(np.float32)

        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        x_map[inst_box[0]:inst_box[1], inst_box[2]:inst_box[3]][inst_crop > 0] = inst_x[inst_crop > 0]
        y_map[inst_box[0]:inst_box[1], inst_box[2]:inst_box[3]][inst_crop > 0] = inst_y[inst_crop > 0]

    return np.dstack([x_map, y_map])


def _gen_edge_map(np_map: np.ndarray) -> np.ndarray:
    """从二值核掩码生成边缘图（拉普拉斯算子）。

    边缘信息反映核的几何形态而非染色外观，有助于提升跨域泛化能力。

    Args:
        np_map: (H, W) int32 二值图，核像素=1，背景=0
    Returns:
        edge_map: (H, W) float32 二值图，边缘像素=1，其余=0
    """
    np_uint8  = (np_map * 255).astype(np.uint8)
    laplacian = cv2.Laplacian(np_uint8, cv2.CV_64F)
    return (np.abs(laplacian) > 0).astype(np.float32)


def _masks_to_targets(masks: np.ndarray, mask_shape) -> dict:
    """将 PanNuke (H, W, 6) 掩码转换为 HoverNet 训练 targets。

    Returns:
        dict with keys: np_map, hv_map, tp_map, edge_map
        （tp_map 保留用于兼容 hover_net_new baseline 训练；edge_map 为新增）
    """
    H, W        = masks.shape[:2]
    inst_map    = np.zeros((H, W), dtype=np.int32)
    tp_map      = np.zeros((H, W), dtype=np.int32)
    current_max = 0

    for cls_idx in range(5):
        cls_mask = masks[..., cls_idx].astype(np.int32)
        for inst_id in np.unique(cls_mask):
            if inst_id == 0:
                continue
            region           = cls_mask == inst_id
            inst_map[region] = current_max + 1
            tp_map[region]   = cls_idx + 1
            current_max     += 1

    hv_map   = _gen_hv_map(inst_map)
    np_map   = (inst_map > 0).astype(np.int32)
    edge_map = _gen_edge_map(np_map)

    return {
        "np_map":   cropping_center(np_map,   mask_shape),
        "hv_map":   cropping_center(hv_map,   mask_shape),
        "tp_map":   cropping_center(tp_map,   mask_shape),   # 保留，兼容 baseline
        "edge_map": cropping_center(edge_map, mask_shape),   # 新增
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parquet loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_parquet(path: str):
    """加载一个 fold 的 parquet 文件 → (N,256,256,3) uint8, (N,256,256,6)。"""
    import pandas as pd
    from PIL import Image

    df     = pd.read_parquet(path)
    N      = len(df)
    images = np.zeros((N, 256, 256, 3), dtype=np.uint8)
    masks  = np.zeros((N, 256, 256, 6), dtype=np.float64)

    for i, row in df.iterrows():
        img = row["image"]
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        images[i] = np.array(img, dtype=np.uint8)

        inst_id = 1
        for inst_img, cls_idx in zip(row["instances"], row["categories"]):
            if isinstance(inst_img, dict) and "bytes" in inst_img:
                inst_img = Image.open(io.BytesIO(inst_img["bytes"]))
            inst_arr              = np.array(inst_img, dtype=bool)
            masks[i, inst_arr, cls_idx] = inst_id
            inst_id              += 1

    return images, masks


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def _build_augmentation(mode: str, seed: int):
    additional_targets = {f"mask{c}": "mask" for c in range(6)}

    if mode == "train":
        return A.Compose([
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.01, 0.01),
                shear=(-5, 5),
                rotate=(-179, 179),
                interpolation=cv2.INTER_NEAREST,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussNoise(p=1.0),
            ], p=1.0),
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=8,
                    sat_shift_limit=int(0.2 * 255),
                    val_shift_limit=26,
                    p=1.0,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=26 / 255,
                    contrast_limit=0.25,
                    p=1.0,
                ),
            ], p=1.0),
        ], additional_targets=additional_targets, seed=seed)
    else:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PanNukeDataset(torch.utils.data.Dataset):
    """PanNuke dataset，输出 np_map / hv_map / tp_map / edge_map 四个 target。"""

    def __init__(
        self,
        parquet_paths: list,
        mode: str = "train",
        input_shape=(256, 256),
        mask_shape=(164, 164),
        seed: int = 42,
    ):
        self.mode        = mode
        self.input_shape = input_shape
        self.mask_shape  = mask_shape

        images_list, masks_list = [], []
        for p in parquet_paths:
            imgs, msks = _load_parquet(p)
            images_list.append(imgs)
            masks_list.append(msks)

        self.images = np.concatenate(images_list, axis=0)
        self.masks  = np.concatenate(masks_list,  axis=0)
        self.aug    = _build_augmentation(mode, seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = self.images[idx].copy()
        masks = self.masks[idx].copy()

        if self.aug is not None:
            mask_dict = {f"mask{c}": masks[..., c].astype(np.uint8) for c in range(6)}
            result    = self.aug(image=img, **mask_dict)
            img       = result["image"]
            for c in range(6):
                masks[..., c] = result[f"mask{c}"].astype(masks.dtype)

        img     = cropping_center(img, self.input_shape)
        targets = _masks_to_targets(masks, self.mask_shape)

        img_tensor  = torch.from_numpy(img).permute(2, 0, 1).float()
        np_tensor   = torch.from_numpy(targets["np_map"]).long()
        hv_tensor   = torch.from_numpy(targets["hv_map"]).permute(2, 0, 1).float()
        tp_tensor   = torch.from_numpy(targets["tp_map"]).long()
        edge_tensor = torch.from_numpy(targets["edge_map"]).float()

        return {
            "img":      img_tensor,
            "np_map":   np_tensor,
            "hv_map":   hv_tensor,
            "tp_map":   tp_tensor,       # 保留，兼容 hover_net_new baseline
            "edge_map": edge_tensor,     # 新增
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(cfg, rank: int = 0, world_size: int = 1):
    """构建 train/val DataLoader，支持 DDP DistributedSampler。"""
    train_ds = PanNukeDataset(
        parquet_paths=cfg.train_parquet,
        mode="train",
        input_shape=tuple(cfg.input_shape),
        mask_shape=tuple(cfg.mask_shape),
        seed=cfg.seed,
    )
    val_ds = PanNukeDataset(
        parquet_paths=cfg.val_parquet,
        mode="valid",
        input_shape=tuple(cfg.input_shape),
        mask_shape=tuple(cfg.mask_shape),
        seed=cfg.seed,
    )

    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        if world_size > 1 else None
    )
    val_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        if world_size > 1 else None
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler
