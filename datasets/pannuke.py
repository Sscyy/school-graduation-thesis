"""PanNuke dataset loader for HoverNet-style training.

PanNuke format:
    images/fold{n}/images.npy  : (N, 256, 256, 3)  float64, range [0, 255]
    masks/fold{n}/masks.npy    : (N, 256, 256, 6)  float64
        channel 0: Neoplastic
        channel 1: Inflammatory
        channel 2: Connective
        channel 3: Dead
        channel 4: Epithelial
        channel 5: Background
    images/fold{n}/types.npy   : (N,)  str, tissue type

Each mask channel contains instance IDs (0 = background, 1,2,... = instances).
We convert to HoverNet format:
    np_map  : (H, W)     binary foreground mask
    hv_map  : (H, W, 2)  horizontal/vertical distance maps
    tp_map  : (H, W)     nuclear type map (1-indexed, 0=background)
"""

import sys
import os
import numpy as np
import torch
import torch.utils.data
import imgaug as ia
from imgaug import augmenters as iaa
from scipy.ndimage import measurements
from skimage import morphology as morph

# Allow importing from hover_net directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hover_net'))
from dataloader.augs import fix_mirror_padding
from misc.utils import cropping_center, get_bounding_box


# PanNuke class index → type name (1-indexed to match HoverNet convention)
PANNUKE_CLASSES = {
    0: 'background',
    1: 'neoplastic',
    2: 'inflammatory',
    3: 'connective',
    4: 'dead',
    5: 'epithelial',
}
NR_TYPES = 6  # including background


def gen_hv_map(inst_map):
    """Generate horizontal and vertical distance maps from instance map.

    Replicates hover_net/models/hovernet/targets.py::gen_instance_hv_map
    but operates directly on a pre-loaded instance map (no crop needed
    since PanNuke patches are already 256x256).

    Args:
        inst_map: (H, W) int array, 0=background, 1..N=instance IDs

    Returns:
        hv_map: (H, W, 2) float32, horizontal and vertical distance maps
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


def pannuke_masks_to_hovernet(masks):
    """Convert PanNuke 6-channel mask to HoverNet targets.

    Args:
        masks: (H, W, 6) float64 array, each channel = instance IDs for that class

    Returns:
        inst_map : (H, W) int32  — merged instance ID map (unique across classes)
        tp_map   : (H, W) int32  — type map (0=bg, 1=neoplastic, ..., 5=epithelial)
        np_map   : (H, W) int32  — binary foreground (0=bg, 1=nucleus)
        hv_map   : (H, W, 2) float32 — hover maps
    """
    H, W = masks.shape[:2]
    inst_map = np.zeros((H, W), dtype=np.int32)
    tp_map = np.zeros((H, W), dtype=np.int32)

    current_max_id = 0
    # channels 0-4 are the 5 nuclear types; channel 5 is background (skip)
    for cls_idx in range(5):
        cls_mask = masks[..., cls_idx].astype(np.int32)
        inst_ids = np.unique(cls_mask)
        inst_ids = inst_ids[inst_ids > 0]
        for inst_id in inst_ids:
            region = cls_mask == inst_id
            new_id = current_max_id + 1
            inst_map[region] = new_id
            tp_map[region] = cls_idx + 1  # 1-indexed type
            current_max_id += 1

    np_map = (inst_map > 0).astype(np.int32)
    hv_map = gen_hv_map(inst_map)

    return inst_map, tp_map, np_map, hv_map


class PanNukeDataset(torch.utils.data.Dataset):
    """PanNuke dataset for HoverNet-style training.

    Args:
        fold_dirs : list of fold directories, e.g.
                    ['/path/to/pannuke/Fold1', '/path/to/pannuke/Fold2']
        mode      : 'train' or 'valid'
        input_shape : (H, W) input patch size fed to network
        mask_shape  : (H, W) output mask size (center crop of input)
        val_split   : fraction of data to use for validation (only used
                      when mode='valid')
        seed        : random seed for train/val split
    """

    def __init__(
        self,
        fold_dirs,
        mode='train',
        input_shape=(256, 256),
        mask_shape=(164, 164),
        val_split=0.1,
        seed=42,
    ):
        self.mode = mode
        self.input_shape = input_shape
        self.mask_shape = mask_shape

        images_list, masks_list = [], []
        for fold_dir in fold_dirs:
            # find fold number from directory name
            fold_name = os.path.basename(fold_dir.rstrip('/'))  # e.g. 'Fold1'
            fold_num = ''.join(filter(str.isdigit, fold_name))  # '1'
            fold_key = f'fold{fold_num}'

            img_path  = os.path.join(fold_dir, 'images', fold_key, 'images.npy')
            mask_path = os.path.join(fold_dir, 'masks',  fold_key, 'masks.npy')

            imgs  = np.load(img_path)   # (N, 256, 256, 3)
            masks = np.load(mask_path)  # (N, 256, 256, 6)
            images_list.append(imgs)
            masks_list.append(masks)

        all_images = np.concatenate(images_list, axis=0)  # (N_total, 256, 256, 3)
        all_masks  = np.concatenate(masks_list,  axis=0)  # (N_total, 256, 256, 6)

        # train / val split
        N = len(all_images)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(N)
        val_n = max(1, int(N * val_split))

        if mode == 'train':
            sel = indices[val_n:]
        else:
            sel = indices[:val_n]

        self.images = all_images[sel]
        self.masks  = all_masks[sel]

        self.augmentor = self._build_augmentor(mode, seed)

    def _build_augmentor(self, mode, seed):
        if mode == 'train':
            return iaa.Sequential([
                iaa.Fliplr(0.5, seed=seed),
                iaa.Flipud(0.5, seed=seed),
                iaa.Affine(
                    rotate=(-179, 179),
                    order=0,
                    backend='cv2',
                    seed=seed,
                ),
                iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
                iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))),
                iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.25))),
            ])
        else:
            return None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = self.images[idx].astype(np.float32)   # (256, 256, 3)
        masks = self.masks[idx]                        # (256, 256, 6)

        # augmentation (shape-preserving, applied jointly to img and masks)
        if self.augmentor is not None:
            aug_det = self.augmentor.to_deterministic()
            img_uint8 = img.astype(np.uint8)
            img_uint8 = aug_det.augment_image(img_uint8)
            img = img_uint8.astype(np.float32)
            # augment each mask channel independently (same transform)
            for c in range(masks.shape[-1]):
                masks[..., c] = aug_det.augment_image(
                    masks[..., c].astype(np.uint8)
                ).astype(masks.dtype)

        # convert masks → HoverNet targets
        _, tp_map, np_map, hv_map = pannuke_masks_to_hovernet(masks)

        # center-crop to mask_shape
        tp_map = cropping_center(tp_map, self.mask_shape)
        np_map = cropping_center(np_map, self.mask_shape)
        hv_map = cropping_center(hv_map, self.mask_shape)
        img_cropped = cropping_center(img, self.input_shape)

        # to tensors
        img_tensor = torch.from_numpy(img_cropped).permute(2, 0, 1).float()  # (3, H, W)
        np_tensor  = torch.from_numpy(np_map).long()
        hv_tensor  = torch.from_numpy(hv_map).permute(2, 0, 1).float()      # (2, H, W)
        tp_tensor  = torch.from_numpy(tp_map).long()

        return {
            'img':    img_tensor,
            'np_map': np_tensor,
            'hv_map': hv_tensor,
            'tp_map': tp_tensor,
        }


def get_pannuke_loaders(
    fold_dirs_train,
    fold_dirs_val=None,
    input_shape=(256, 256),
    mask_shape=(164, 164),
    batch_size=8,
    num_workers=4,
    val_split=0.1,
    seed=42,
):
    """Convenience function to build train and val DataLoaders.

    Args:
        fold_dirs_train : list of fold dirs used for training
        fold_dirs_val   : list of fold dirs used for validation (if None,
                          val_split fraction of fold_dirs_train is used)
        input_shape     : (H, W) network input size
        mask_shape      : (H, W) network output size
        batch_size      : training batch size
        num_workers     : DataLoader workers
        val_split       : validation fraction (ignored if fold_dirs_val given)
        seed            : reproducibility seed

    Returns:
        train_loader, val_loader
    """
    if fold_dirs_val is not None:
        train_ds = PanNukeDataset(
            fold_dirs_train, mode='train',
            input_shape=input_shape, mask_shape=mask_shape,
            val_split=0.0, seed=seed,
        )
        val_ds = PanNukeDataset(
            fold_dirs_val, mode='valid',
            input_shape=input_shape, mask_shape=mask_shape,
            val_split=1.0, seed=seed,
        )
    else:
        train_ds = PanNukeDataset(
            fold_dirs_train, mode='train',
            input_shape=input_shape, mask_shape=mask_shape,
            val_split=val_split, seed=seed,
        )
        val_ds = PanNukeDataset(
            fold_dirs_train, mode='valid',
            input_shape=input_shape, mask_shape=mask_shape,
            val_split=val_split, seed=seed,
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
