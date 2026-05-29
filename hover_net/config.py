import importlib
import random

import cv2
import numpy as np


class Config(object):
    """Configuration file."""

    def __init__(self):
        self.seed = 10

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False

        model_name = "hovernet"
        model_mode = "fast" # PanNuke uses 256x256 patches → fast mode

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        nr_type = 6 # PanNuke: 5 nuclear types + background = 6

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True

        # fast mode: input 256x256, output 164x164
        act_shape = [256, 256]
        out_shape = [164, 164]

        if model_mode == "original":
            if act_shape != [270,270] or out_shape != [80,80]:
                raise Exception("If using `original` mode, input shape must be [270,270] and output shape must be [80,80]")
        if model_mode == "fast":
            if act_shape != [256,256] or out_shape != [164,164]:
                raise Exception("If using `fast` mode, input shape must be [256,256] and output shape must be [164,164]")

        self.dataset_name = "pannuke"
        self.log_dir = "logs/pannuke_baseline/"

        # Paths to PanNuke parquet files.
        # Use fold1+fold2 for training and fold3 for validation.
        pannuke_root = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/PanNuke"
        self.train_parquet_list = [
            f"{pannuke_root}/fold1-00000-of-00001.parquet",
            f"{pannuke_root}/fold2-00000-of-00001.parquet",
        ]
        self.valid_parquet_list = [
            f"{pannuke_root}/fold3-00000-of-00001.parquet",
        ]

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        module = importlib.import_module(
            "models.%s.opt" % model_name
        )
        self.model_config = module.get_config(nr_type, model_mode)
