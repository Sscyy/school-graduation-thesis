import csv
import glob
import io
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch.utils.data
from scipy.ndimage import measurements
from skimage import morphology as morph

from misc.utils import cropping_center, get_bounding_box

from .augs import (
    fix_mirror_padding,
)

# albumentations — used by PanNukeLoader (avoids imgaug/numpy compat issues)
import albumentations as A


####
class FileLoader(torch.utils.data.Dataset):
    """Data Loader. Loads images from a file list and 
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are 
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'
        
    """

    # TODO: doc string

    def __init__(
        self,
        file_list,
        with_type=False,
        input_shape=None,
        mask_shape=None,
        mode="train",
        setup_augmentor=True,
        target_gen=None,
    ):
        assert input_shape is not None and mask_shape is not None
        self.mode = mode
        self.info_list = file_list
        self.with_type = with_type
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        self.id = 0
        self.target_gen_func = target_gen[0]
        self.target_gen_kwargs = target_gen[1]
        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):
        import imgaug as ia
        from imgaug import augmenters as iaa
        from .augs import add_to_brightness, add_to_contrast, add_to_hue, add_to_saturation, gaussian_blur, median_blur
        self.augmentor = self.__get_augmentation(self.mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        path = self.info_list[idx]
        data = np.load(path)

        # split stacked channel into image and label
        img = (data[..., :3]).astype("uint8")  # RGB images
        ann = (data[..., 3:]).astype("int32")  # instance ID map and type map

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            ann = shape_augs.augment_image(ann)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        img = cropping_center(img, self.input_shape)
        feed_dict = {"img": img}

        inst_map = ann[..., 0]  # HW1 -> HW
        if self.with_type:
            type_map = (ann[..., 1]).copy()
            type_map = cropping_center(type_map, self.mask_shape)
            #type_map[type_map == 5] = 1  # merge neoplastic and non-neoplastic
            feed_dict["tp_map"] = type_map

        # TODO: document hard coded assumption about #input
        target_dict = self.target_gen_func(
            inst_map, self.mask_shape, **self.target_gen_kwargs
        )
        feed_dict.update(target_dict)

        return feed_dict

    def __get_augmentation(self, mode, rng):
        import imgaug as ia
        from imgaug import augmenters as iaa
        from .augs import add_to_brightness, add_to_contrast, add_to_hue, add_to_saturation, gaussian_blur, median_blur
        if mode == "train":
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -A to +A percent (per axis)
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    order=0,  # use nearest neighbour
                    backend="cv2",  # opencv for fast processing
                    seed=rng,
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        elif mode == "valid":
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs


####
def _gen_hv_map(inst_map):
    """Generate HoVer maps from an instance ID map (H, W) int32.

    Replicates gen_instance_hv_map from targets.py but operates on a
    pre-cropped 256x256 PanNuke patch (no additional crop needed).
    Returns hv_map (H, W, 2) float32.
    """
    fixed_ann = fix_mirror_padding(inst_map.copy())
    inst_map_clean = morph.remove_small_objects(fixed_ann, min_size=30)

    x_map = np.zeros(inst_map.shape[:2], dtype=np.float32)
    y_map = np.zeros(inst_map.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(inst_map_clean))
    if 0 in inst_list:
        inst_list.remove(0)

    for inst_id in inst_list:
        inst_mask = np.array(fixed_ann == inst_id, dtype=np.uint8)
        inst_box = get_bounding_box(inst_mask)

        inst_box[0] -= 2
        inst_box[2] -= 2
        inst_box[1] += 2
        inst_box[3] += 2

        inst_crop = inst_mask[inst_box[0]:inst_box[1], inst_box[2]:inst_box[3]]
        if inst_crop.shape[0] < 2 or inst_crop.shape[1] < 2:
            continue

        inst_com = list(measurements.center_of_mass(inst_crop))
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


def _pannuke_masks_to_targets(masks, mask_shape):
    """Convert PanNuke (H, W, 6) float64 mask array to HoverNet targets.

    channels 0-4: instance IDs for each of the 5 nuclear types
    channel 5:    background (ignored)

    Returns:
        feed dict keys: 'np_map' (H,W) int32, 'hv_map' (H,W,2) float32,
                        'tp_map' (H,W) int32
    """
    H, W = masks.shape[:2]
    inst_map = np.zeros((H, W), dtype=np.int32)
    tp_map   = np.zeros((H, W), dtype=np.int32)

    current_max_id = 0
    for cls_idx in range(5):  # channels 0-4 are the 5 nuclear types
        cls_mask = masks[..., cls_idx].astype(np.int32)
        for inst_id in np.unique(cls_mask):
            if inst_id == 0:
                continue
            region = cls_mask == inst_id
            new_id = current_max_id + 1
            inst_map[region] = new_id
            tp_map[region]   = cls_idx + 1  # 1-indexed type
            current_max_id  += 1

    hv_map = _gen_hv_map(inst_map)
    np_map = (inst_map > 0).astype(np.int32)

    hv_map = cropping_center(hv_map, mask_shape)
    np_map = cropping_center(np_map, mask_shape)
    tp_map = cropping_center(tp_map, mask_shape)

    return {"np_map": np_map, "hv_map": hv_map, "tp_map": tp_map}


####
class PanNukeLoader(torch.utils.data.Dataset):
    """Dataset loader for PanNuke in HuggingFace parquet format.

    Replaces FileLoader for PanNuke. Reads one or more parquet fold files,
    builds a unified image/mask array in memory, and exposes the same
    __getitem__ interface as FileLoader (returns the same feed_dict keys).

    Args:
        parquet_paths : list of .parquet file paths
        input_shape   : (H, W) network input size, e.g. [256, 256]
        mask_shape    : (H, W) network output size, e.g. [164, 164]
        mode          : 'train' or 'valid'
        with_type     : whether to include tp_map in feed_dict
        setup_augmentor: whether to build augmentor immediately (False for
                         multi-worker DataLoader, workers call setup_augmentor)
        val_split     : fraction of data held out for validation
        seed          : random seed for train/val split
    """

    def __init__(
        self,
        parquet_paths,
        input_shape=None,
        mask_shape=None,
        mode="train",
        with_type=True,
        setup_augmentor=True,
        val_split=0.1,
        seed=42,
        target_gen=None,  # ignored, kept for API compatibility with FileLoader
    ):
        assert input_shape is not None and mask_shape is not None
        self.mode = mode
        self.with_type = with_type
        self.input_shape = input_shape
        self.mask_shape = mask_shape
        self.id = 0

        images_list, masks_list = [], []
        for p in parquet_paths:
            imgs, msks = self._load_parquet_fold(p)
            images_list.append(imgs)
            masks_list.append(msks)

        all_images = np.concatenate(images_list, axis=0)
        all_masks  = np.concatenate(masks_list,  axis=0)

        N = len(all_images)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(N)
        val_n = max(1, int(N * val_split))

        if mode == "train":
            sel = indices[val_n:]
        else:
            sel = indices[:val_n]

        self.images = all_images[sel]
        self.masks  = all_masks[sel]

        self.shape_augs = None
        self.input_augs = None
        if setup_augmentor:
            self.setup_augmentor(0, 0)

    @staticmethod
    def _load_parquet_fold(parquet_path):
        """Load one parquet fold into (N,256,256,3) uint8 and (N,256,256,6) float64."""
        import pandas as pd
        from PIL import Image

        df = pd.read_parquet(parquet_path)
        N = len(df)
        images = np.zeros((N, 256, 256, 3), dtype=np.uint8)
        masks  = np.zeros((N, 256, 256, 6), dtype=np.float64)

        for i, row in df.iterrows():
            img = row['image']
            if isinstance(img, dict) and 'bytes' in img:
                img = Image.open(io.BytesIO(img['bytes'])).convert('RGB')
            images[i] = np.array(img, dtype=np.uint8)

            instances  = row['instances']
            categories = row['categories']
            inst_id = 1
            for inst_img, cls_idx in zip(instances, categories):
                if isinstance(inst_img, dict) and 'bytes' in inst_img:
                    inst_img = Image.open(io.BytesIO(inst_img['bytes']))
                inst_arr = np.array(inst_img, dtype=bool)
                masks[i, inst_arr, cls_idx] = inst_id
                inst_id += 1

        return images, masks

    def setup_augmentor(self, worker_id, seed):
        self.aug_pipeline = self._get_augmentation(self.mode, seed)
        self.id = self.id + worker_id

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = self.images[idx].copy()   # (256, 256, 3) uint8
        masks = self.masks[idx].copy()    # (256, 256, 6) float64

        if self.aug_pipeline is not None:
            # albumentations expects uint8 image; masks passed as additional_targets
            mask_dict = {f"mask{c}": masks[..., c].astype(np.uint8) for c in range(6)}
            result = self.aug_pipeline(image=img, **mask_dict)
            img = result["image"]
            for c in range(6):
                masks[..., c] = result[f"mask{c}"].astype(masks.dtype)

        img = cropping_center(img, self.input_shape)
        feed_dict = {"img": img}

        targets = _pannuke_masks_to_targets(masks, self.mask_shape)
        feed_dict["np_map"] = targets["np_map"]
        feed_dict["hv_map"] = targets["hv_map"]
        if self.with_type:
            feed_dict["tp_map"] = targets["tp_map"]

        return feed_dict

    def _get_augmentation(self, mode, seed):
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
                    A.GaussNoise(var_limit=(0.0, 0.05 * 255) , p=1.0),
                ], p=1.0),
                A.OneOf([
                    A.HueSaturationValue(
                        hue_shift_limit=8,
                        sat_shift_limit=int(0.2 * 255),
                        val_shift_limit=26,
                        p=1.0,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=26/255,
                        contrast_limit=0.25,
                        p=1.0,
                    ),
                ], p=1.0),
            ], additional_targets=additional_targets, seed=seed)
        else:
            return None
