import logging

import albumentations
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from src.utils.path_utils import build_path_rewrites, normalize_path


class ColonDataset(Dataset):
    def __init__(self, mode, cfg, is_labeled=False):
        super().__init__()
        self.mode = mode
        self.is_labeled = is_labeled
        self.path_rewrites = build_path_rewrites(cfg)

        csv_file = normalize_path(cfg["dataset"][mode], self.path_rewrites)
        self.img_paths, self.mask_paths = self._load_rows(csv_file)

        height = int(cfg["dataset"]["resize"]["height"])
        width = int(cfg["dataset"]["resize"]["width"])
        self.transforms = {
            "weak": albumentations.Compose(
                [
                    albumentations.RandomScale(scale_limit=(0.0, 0.5), p=0.5),
                    albumentations.RandomCrop(height=height, width=width, p=0.5),
                    albumentations.ShiftScaleRotate(
                        shift_limit=0,
                        scale_limit=0,
                        rotate_limit=(-20, 20),
                        interpolation=cv2.INTER_LANCZOS4,
                        border_mode=0,
                        p=0.5,
                    ),
                    albumentations.HorizontalFlip(p=0.5),
                    albumentations.ElasticTransform(),
                    albumentations.Resize(height, width),
                ]
            ),
            "strong": albumentations.Compose(
                [
                    albumentations.RandomBrightnessContrast(brightness_limit=(0.5, 1.5), contrast_limit=(0.5, 1.5), p=1),
                    albumentations.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=1.0),
                ]
            ),
            "normal": albumentations.Compose(
                [
                    albumentations.Normalize(cfg["dataset"]["mean"], cfg["dataset"]["std"]),
                    ToTensorV2(),
                ]
            ),
        }

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        image = Image.open(image_path).convert("RGB")
        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path)

        if self.mode == "train":
            if self.is_labeled:
                transformed = self.transforms["weak"](image=np.array(image), mask=np.array(mask))
                tensor_transformed = self.transforms["normal"](image=transformed["image"], mask=transformed["mask"])
                return image_path, tensor_transformed["image"], tensor_transformed["mask"]

            weak = self.transforms["weak"](image=np.array(image), mask=np.array(mask))
            weak_image = weak["image"]
            strong = self.transforms["strong"](image=np.array(weak_image))
            tensor_weak_image = self.transforms["normal"](image=np.array(weak_image))["image"]
            tensor_strong_image = self.transforms["normal"](image=np.array(strong["image"]))["image"]
            return image_path, tensor_weak_image, tensor_strong_image

        transformed = self.transforms["normal"](image=np.array(image), mask=np.array(mask))
        return image_path, transformed["image"], transformed["mask"]

    def __len__(self):
        return len(self.img_paths)

    def _load_rows(self, csv_path):
        data = pd.read_csv(csv_path)

        imgs = [normalize_path(path, self.path_rewrites) for path in data["image_path"].values]
        masks = [normalize_path(path, self.path_rewrites) for path in data["segmentation_mask_path"].values]

        logging.info("Total images: %d, labels: %d", len(imgs), len(masks))
        return imgs, masks
