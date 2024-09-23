# encoding: utf-8

"""
CXR14
Read images and corresponding labels.
"""

import cv2
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import logging

import albumentations
from albumentations.pytorch import ToTensorV2

class_names =  ['base']
domain_names = {1:'BIDMC', 2:'BMC', 3:'HK', 4:'I2CVB', 5:'RUNMC', 6:'UCL'}

class_name_to_id = {}
for i, each_label in enumerate(class_names):
    class_id = i  
    class_name = each_label
    class_name_to_id[class_name] = class_id

class ProstateDataset(Dataset):
    def __init__(self, mode, cfg, is_labeled=False):
        """
        Args:
            mode: train, eval
            is_labeled: whether the dataset is labeled
            cfg: configuration file
        """
        super(ProstateDataset, self).__init__()
        csv_file = cfg['dataset'][mode]

        self.img_paths, self.train_np_paths, self.mask_paths = self.__load_imgs__(csv_file)
        self.mode = mode
        self.is_labeled = is_labeled

        height, width= cfg['dataset']['resize']['height'], cfg['dataset']['resize']['width']
        self.transforms = {
            'weak': albumentations.Compose([
                albumentations.RandomScale(scale_limit=(0.0, 0.5), p=0.5),
                albumentations.RandomCrop(height=height, width=width, p=0.5),
                albumentations.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=(-20, 20), interpolation=cv2.INTER_LANCZOS4,p=0.5),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.ElasticTransform(),
                albumentations.Resize(height, width),
                ]),
            'strong': albumentations.Compose([
                albumentations.RandomBrightnessContrast(brightness_limit=(0.1, 2), contrast_limit=(0.1, 2), p=1),
                albumentations.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=1.0),
            ]),
            'normal': albumentations.Compose([
                albumentations.Normalize(cfg['dataset']['mean'], cfg['dataset']['std']),
                ToTensorV2()
            ])
        }
        
    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_path = self.img_paths[index]
        train_img_path = self.train_np_paths[index]
        image = np.load(train_img_path)
        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path)
        
        if self.mode == 'train':
            if self.is_labeled:
                transformed = self.transforms['weak'](image=np.array(image), mask=np.array(mask))
                tensor_transformed = self.transforms['normal'](image = transformed['image'], mask = transformed['mask'])
                return image_path, tensor_transformed['image'], tensor_transformed['mask']
            else:
                image = Image.fromarray(image)
                # weak = self.transforms['weak'](image=np.array(image), mask=np.array(mask))
                # weak_image = weak['image']
                strong = self.transforms['strong'](image=np.array(image))
                tensor_weak_image = self.transforms['normal'](image=np.array(image))['image']
                tensor_strong_image = self.transforms['normal'](image=np.array(strong['image']))['image']
                return image_path, tensor_weak_image, tensor_strong_image
        else: # test or val
            transformed = self.transforms['normal'](image=np.array(image), mask=np.array(mask))
            image = transformed['image']
            mask = transformed['mask']
            return image_path, image, mask
        
    def __len__(self):
        return len(self.img_paths)

    def __load_imgs__(self, csv_path):
        data = pd.read_csv(csv_path)
        imgs = data['image_vis_path'].values
        train_imgs_np = data['image_train_path'].values
        masks = data['segmentation_mask_path'].values
        logging.info(f'Total # images:{len(imgs)}, labels:{len(masks)}')
        return imgs, train_imgs_np, masks