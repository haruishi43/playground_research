#!/usr/bin/env python3

import os
from pathlib import Path
import yaml

import matplotlib.pyplot as plt

from PIL import Image


if __name__ == "__main__":
    # Imports

    # Variables
    dataset_root = "./dataset/kitti360"
    seq = "2013_05_28_drive_0000_sync"
    frame = 0

    # Directories:
    img2_calib_path = Path(
        os.path.join(
            dataset_root,
            "calibration",
            "image_02.yaml",
        )
    )
    img3_calib_path = Path(
        os.path.join(
            dataset_root,
            "calibration",
            "image_03.yaml",
        )
    )
    assert (
        img2_calib_path.exists() and img3_calib_path.exists()
    ), f"{img2_calib_path} and {img3_calib_path} doesn't exist!"
    img2_path = Path(
        os.path.join(
            dataset_root,
            seq,
            "image_02",
            "data_rgb",
            str(frame).zfill(10) + ".png",
        )
    )
    img3_path = Path(
        os.path.join(
            dataset_root,
            seq,
            "image_03",
            "data_rgb",
            str(frame).zfill(10) + ".png",
        )
    )
    assert (
        img2_path.exists() and img3_path.exists()
    ), f"{img2_path} and {img3_path} doesn't exist!"

    # Load 2 fisheye images
    print(img2_path)
    print(img3_path)

    img2 = Image.open(img2_path)
    img3 = Image.open(img3_path)
    # plt.imshow(img2)
    # plt.show()

    # Load camera intrinsics
    with open(img2_calib_path) as f:
        img2_params = yaml.safe_load(f)
    with open(img3_calib_path) as f:
        img3_params = yaml.safe_load(f)

    print(type(img2_params))
