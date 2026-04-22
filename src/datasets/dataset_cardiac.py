import logging
import os

import albumentations
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from src.utils.path_utils import build_path_rewrites, normalize_path


class CardiacDataset(Dataset):
    def __init__(self, mode, cfg, is_labeled=False):
        super().__init__()
        self.mode = mode
        self.is_labeled = is_labeled
        self.num_classes = int(cfg["model"]["num_classes"])
        self.path_rewrites = build_path_rewrites(cfg)

        csv_file = normalize_path(cfg["dataset"][mode], self.path_rewrites)
        self.img_paths, self.train_np_paths, self.mask_paths = self._load_rows(csv_file)

        height = int(cfg["dataset"]["resize"]["height"])
        width = int(cfg["dataset"]["resize"]["width"])
        self.transforms = {
            "resize": albumentations.Compose([albumentations.Resize(height, width)]),
            "weak": albumentations.Compose(
                [
                    albumentations.Resize(height, width),
                    albumentations.HorizontalFlip(p=0.5),
                    albumentations.ShiftScaleRotate(
                        shift_limit=0,
                        scale_limit=0.1,
                        rotate_limit=(-20, 20),
                        interpolation=cv2.INTER_LANCZOS4,
                        p=0.5,
                    ),
                    albumentations.ElasticTransform(p=0.2),
                    albumentations.Resize(height, width),
                ]
            ),
            "strong": albumentations.Compose(
                [
                    albumentations.RandomBrightnessContrast(brightness_limit=(0.1, 2), contrast_limit=(0.1, 2), p=1),
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
        image = np.load(self.train_np_paths[index])
        mask_path = self.mask_paths[index]

        if self.mode == "train":
            if self.is_labeled:
                mask = self.load_mask(mask_path, image)
                transformed = self.transforms["weak"](image=np.array(image), mask=np.array(mask))
                tensor = self.transforms["normal"](image=transformed["image"], mask=transformed["mask"])
                return image_path, tensor["image"], tensor["mask"]

            resized = self.transforms["resize"](image=np.array(image))["image"]
            strong = self.transforms["strong"](image=np.array(resized))["image"]
            weak_tensor = self.transforms["normal"](image=np.array(resized))["image"]
            strong_tensor = self.transforms["normal"](image=np.array(strong))["image"]
            return image_path, weak_tensor, strong_tensor

        if self.mode in {"eval", "test"}:
            mask = self.load_mask(mask_path, image)
            transformed = self.transforms["normal"](image=np.array(image), mask=np.array(mask))
            return image_path, transformed["image"], transformed["mask"]

        transformed = self.transforms["normal"](image=np.array(image))
        return image_path, transformed["image"]

    def load_mask(self, mask_path, image):
        normalized = normalize_path(mask_path, self.path_rewrites)
        if isinstance(normalized, str) and normalized and os.path.exists(normalized):
            try:
                return Image.open(normalized)
            except (IOError, OSError):
                pass

        h, w = image.shape[:2]
        return Image.fromarray(np.zeros((h, w, self.num_classes), dtype=np.uint8))

    def __len__(self):
        return len(self.img_paths)

    def _load_rows(self, csv_path):
        data = pd.read_csv(csv_path)
        imgs = [normalize_path(path, self.path_rewrites) for path in data["image_vis_path"].values]
        train_imgs = [normalize_path(path, self.path_rewrites) for path in data["image_train_path"].values]
        masks = [normalize_path(path, self.path_rewrites) for path in data["segmentation_mask_path"].values]
        logging.info("Total images: %d, labels: %d", len(imgs), len(masks))
        return imgs, train_imgs, masks
