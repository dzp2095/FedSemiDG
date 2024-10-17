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

# create a dataframe to store the image path and mask path    
train_df = pd.DataFrame(columns=[
    'image_id', 
    'image_train_path', 
    'image_vis_path',
    'segmentation_mask_path', 
])
test_df = pd.DataFrame(columns=[
    'image_id', 
    'image_train_path', 
    'image_vis_path',
    'segmentation_mask_path', 
])

train_dfs = [train_df] * 4
test_dfs = [test_df] * 4


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
    case_mask_folder = f'{client_data_folder}/{case_idx}/mask'
    case_mask_vis_folder = f'{client_data_folder}/{case_idx}/mask_vis'
    case_img_vis_folder = f'{client_data_folder}/{case_idx}/img_vis'
    case_img_train_folder = f'{client_data_folder}/{case_idx}/img_train'
    case_img_original_folder = f'{client_data_folder}/{case_idx}/img_original'

    Path(case_mask_folder).mkdir(parents=True, exist_ok=True)
    Path(case_mask_vis_folder).mkdir(parents=True, exist_ok=True)
    Path(case_img_vis_folder).mkdir(parents=True, exist_ok=True)
    Path(case_img_train_folder).mkdir(parents=True, exist_ok=True)
    Path(case_img_original_folder).mkdir(parents=True, exist_ok=True)

    for slice in range(img.shape[2]):
        img_slice = img[:, :, slice]
        img_slice = np.rot90(img_slice, k=3)
        if not np.any(img_slice):
            print(f'Warning: {case_idx} slice {slice} has no data')
            continue

        # handle test data
        if is_test:
            img_slice_original = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)
            Image.fromarray(img_slice_original).save(f'{case_img_original_folder}/{slice}.png')
            if site != '1':
                img_slice = crop_from_center(img_slice, centor_crop_ratio[int(site)])
            # for visualization, normalize the image slice to [0, 255]
            lower_percentile = np.percentile(img_slice, 1)
            upper_percentile = np.percentile(img_slice, 99)
            img_slice_clipped = np.clip(img_slice, lower_percentile, upper_percentile)
            img_slice_norm = (img_slice_clipped - img_slice_clipped.min()) / (img_slice_clipped.max() - img_slice_clipped.min())
            img_slice_vis = (img_slice_norm * 255).astype(np.uint8)
            
            # img_slice_train is for future use, it is used as the training data, which preserves the original information as possible.
            # for training
            # img_slice_train = img_slice.astype(np.float32)
            img_slice_train = img_slice_vis.astype(np.float32)

            # Save img_slice_vis as PNG
            Image.fromarray(img_slice_vis).resize(resize, Image.Resampling.LANCZOS).save(f'{case_img_vis_folder}/{slice}.png')

            # Resize img_slice_train and save as .npy
            img_slice_train_resized = np.array(Image.fromarray(img_slice_train).resize(resize, Image.Resampling.LANCZOS))
            np.save(f'{case_img_train_folder}/{slice}.npy', img_slice_train_resized)

            row = pd.DataFrame([{
            'image_id': f'{case_idx}_{slice}',
            'image_train_path': f'{case_img_train_folder}/{slice}.npy',
            'image_vis_path': f'{case_img_vis_folder}/{slice}.png',
            'segmentation_mask_path': '',
        }])
            test_dfs[int(site)-1] = pd.concat([test_dfs[int(site)-1], row], ignore_index=True)
            processed_count += 1
            print(f'domain: {site}, case: {case_idx}, slice: {slice}')
            continue

        gt_list_slice = gt_list[:, :, :, slice].astype(np.uint8)
        # Generate spinal cord mask using NumPy
        spinal_cord_mask = ((np.mean((gt_list_slice > 0).astype(float), axis=0)) > 0.5).astype(float)
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
        Image.fromarray(img_slice_original).save(f'{case_img_original_folder}/{slice}.png')
        
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

        # img_slice_train is for future use, it is used as the training data, which preserves the original information as possible.
        # for training
        # img_slice_train = img_slice.astype(np.float32)
        img_slice_train = img_slice_vis.astype(np.float32)

        # Save img_slice_vis as PNG
        Image.fromarray(img_slice_vis).resize(resize, Image.Resampling.LANCZOS).save(f'{case_img_vis_folder}/{slice}.png')

        # Resize img_slice_train and save as .npy
        img_slice_train_resized = np.array(Image.fromarray(img_slice_train).resize(resize, Image.Resampling.LANCZOS))
        np.save(f'{case_img_train_folder}/{slice}.npy', img_slice_train_resized)

        # Save mask_slice as PNG
        Image.fromarray(mask_slice).resize(resize, Image.Resampling.NEAREST).save(f'{case_mask_folder}/{slice}.png')
        
        # mask visualization
        mask_rgb = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 3), dtype=np.uint8)
        mask_rgb[:, :, 0] = (mask_slice[:, :, 0] != 0) * 255
        mask_rgb[:, :, 1] = (mask_slice[:, :, 1] != 0) * 255
        
        # Resize and save the black and white mask_slice
        Image.fromarray(np.array(Image.fromarray(mask_rgb).resize(resize, Image.Resampling.NEAREST))).save(f'{case_mask_vis_folder}/{slice}.png')

        row = pd.DataFrame([{
            'image_id': f'{case_idx}_{slice}',
            'image_train_path': f'{case_img_train_folder}/{slice}.npy',
            'image_vis_path': f'{case_img_vis_folder}/{slice}.png',
            'segmentation_mask_path': f'{case_mask_folder}/{slice}.png',
        }])
        train_dfs[int(site)-1] = pd.concat([train_dfs[int(site)-1], row], ignore_index=True)
        processed_count += 1
        print(f'domain: {site}, case: {case_idx}, slice: {slice}')

for i in range(4):
    train_dfs[i].to_csv(f'{target_folder}/client_{i+1}/train.csv', index=False)
    test_dfs[i].to_csv(f'{target_folder}/client_{i+1}/test.csv', index=False)

print(f'total slices with label: {total_slices_with_label}')
print(f'processed count: {processed_count}')