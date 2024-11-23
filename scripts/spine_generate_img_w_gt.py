import os
import re
import sys
import cv2

import pandas as pd
import random
import numpy as np
from glob import glob
import nibabel as nib

import yaml
from pathlib import Path
from PIL import Image
import torch

def draw_mask_and_save(img:np.ndarray, pred:torch.Tensor, save_path:str='./img/1/example.png')->None:
    color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(0)
    if len(img.shape) == 2: # If img is HxW
        img = np.expand_dims(img, axis=-1)  # Convert to HxWx1
    if len(img) == 1: # If img is 1xHxW
        img = np.tile(img, (1, 1, 3))  # Convert to HxWx3 by repeating the channel

    h, w, _ = img.shape
    rgb = np.zeros((h, w, 3))
    mask = np.ones((h, w, 3))
    classes = len(pred)
    pred = pred.cpu().numpy()
    for i in reversed(range(classes)):
        mask[pred[i] == 1] = (0.5, 0.5, 0.5)
        rgb[pred[i] == 1] = color_list[i]
    res = (img+rgb)*mask
    res = res.astype(np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, res)

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

filepath = Path(__file__).resolve().parent
config = yaml.safe_load(open(filepath.joinpath("../configs/spine/scripts_conf.yaml")))

target_folder = config.get("target_folder", "~/spine/fed_semi")
raw_data_folder = config.get("raw_data_folder", f"spine/data")
random.seed(config.get("seed", 0))
resize = (config.get("resize")['width'], config.get("resize")['height'])

domain_names = {1:'A', 2:'B', 3:'C', 4:'D'}

# image on site 1 don't need to be cropped
centor_crop_ratio = {
    2: 0.35,
    3: 0.25,
    4: 0.3,
}

def crop_from_center(image,  ratio):
    """
    use the center_crop_ratio to crop the image
    """
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    crop_height = int(h * ratio)
    crop_width = int(w * ratio)

    top = max(0, center_y - crop_height // 2)
    left = max(0, center_x - crop_width // 2)
    bottom = min(h, top + crop_height)
    right = min(w, left + crop_width)
    
    return image[top:bottom, left:right, ...]

Path(target_folder).mkdir(parents=True, exist_ok=True)

total_slices_with_label = 0 
processed_count = 0

for domain in domain_names.keys():
    client_folder = f'{target_folder}/client_{domain}'
    Path(client_folder).mkdir(parents=True, exist_ok=True)
    client_data_folder = f'{client_folder}/data'
    Path(client_data_folder).mkdir(parents=True, exist_ok=True)

img_files = glob(f'{raw_data_folder}/*-image.nii.gz')


pattern = r'^(site(\d+)-(sc\d+))-'

for img_file in img_files:
    img_name = os.path.basename(img_file)
    match = re.match(pattern, img_name)
    case_idx = match.group(1)
    site = match.group(2)

    mask_files = glob(f'{raw_data_folder}/{case_idx}-mask-r*.nii.gz')
    masks = []
    for mask_file in mask_files:
        mask = nib.load(mask_file).get_fdata()
        masks.append(mask)

    if not masks:
        print(f"No masks found for case {case_idx}")
        is_test = True
    else:
        is_test = False
        # Stack masks along a new axis to create gt_list
        gt_list = np.stack(masks, axis=0)

    img = nib.load(img_file).get_fdata()

    client_data_folder = f'{target_folder}/client_{site}/data'
    case_img_gt_vis_folder = f'{client_data_folder}/{case_idx}/img_vis_gt'
   
    Path(case_img_gt_vis_folder).mkdir(parents=True, exist_ok=True)

    for slice in range(img.shape[2]):
        img_slice = img[:, :, slice]
        img_slice = np.rot90(img_slice, k=3)
        if not np.any(img_slice):
            print(f'Warning: {case_idx} slice {slice} has no data')
            continue

        # handle test data
        if is_test:
            continue

        gt_list_slice = gt_list[:, :, :, slice].astype(np.uint8)
        # Generate spinal cord mask using NumPy
        spinal_cord_mask = ((np.mean((gt_list_slice == 2).astype(float), axis=0)) > 0.5).astype(float)
        # Generate gray matter mask using NumPy
        gm_mask = ((np.mean((gt_list_slice == 1).astype(float), axis=0)) > 0.5).astype(float)
        h, w = spinal_cord_mask.shape
        mask_slice = np.concatenate((spinal_cord_mask.reshape(h,w,1), gm_mask.reshape(h,w,1)), axis=2).astype(np.uint8)
        if not np.any(gm_mask):
            print(f'Warning: {case_idx} slice {slice} has no grey matter')
            continue
        mask_slice = np.rot90(mask_slice, k=3, axes=(0, 1))

        total_slices_with_label += np.count_nonzero(np.any(mask_slice, axis=(0, 1)))
        img_slice_original = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)
        
        # center crop the image 
        if site != '1':
            img_slice = crop_from_center(img_slice, centor_crop_ratio[int(site)])
            mask_slice = crop_from_center(mask_slice, centor_crop_ratio[int(site)])
        # for visualization, normalize the image slice to [0, 255]
        # z-score normalization
        lower_percentile = np.percentile(img_slice, 1)
        upper_percentile = np.percentile(img_slice, 99)
        img_slice_clipped = np.clip(img_slice, lower_percentile, upper_percentile)
        img_slice_norm = (img_slice_clipped - img_slice_clipped.min()) / (img_slice_clipped.max() - img_slice_clipped.min())
        img_slice_vis = (img_slice_norm * 255).astype(np.uint8)

        img_slice_vis_resized = cv2.resize(img_slice_vis, resize, interpolation=cv2.INTER_LINEAR)
        mask_slice_resized = cv2.resize(mask_slice, resize, interpolation=cv2.INTER_NEAREST)
        mask_slice_resized = torch.from_numpy(mask_slice_resized).permute(2, 0, 1)
        draw_mask_and_save(img_slice_vis_resized, mask_slice_resized, f'{case_img_gt_vis_folder}/{slice}.png')
        print(f'domain: {site}, case: {case_idx}, slice: {slice}')

print(f'total slices with label: {total_slices_with_label}')
print(f'processed count: {processed_count}')