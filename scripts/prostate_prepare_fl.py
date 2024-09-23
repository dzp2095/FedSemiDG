import os
import sys
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
config = yaml.safe_load(open(filepath.joinpath("../configs/prostate/scripts_conf.yaml")))

target_folder = config.get("target_folder", "~/prostate_mri/fed_semi")
raw_data_folder = config.get("raw_data_folder", f"prostate_mri/data")
random.seed(config.get("seed", 0))
resize = (config.get("resize")['width'], config.get("resize")['height'])

domain_names = {1:'BIDMC', 2:'BMC', 3:'HK', 4:'I2CVB', 5:'RUNMC', 6:'UCL'}

Path(target_folder).mkdir(parents=True, exist_ok=True)

test_ratio = config.get("test_ratio", 0.2)

total_slices_with_label = 0 
processed_count = 0

for domain in domain_names.keys():
    domain_name = domain_names[domain]

    client_folder = f'{target_folder}/client_{domain}'
    Path(client_folder).mkdir(parents=True, exist_ok=True)
    source_folder = f'{raw_data_folder}/{domain_name}'

    client_data_folder = f'{client_folder}/data'
    Path(client_data_folder).mkdir(parents=True, exist_ok=True)
    
    mask_files = glob(f'{source_folder}/*_segmentation.nii.gz')

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

    # train-test split on case level
    train_mask_files = random.sample(mask_files, int(len(mask_files)*(1-test_ratio)))
    test_mask_files = list(set(mask_files) - set(train_mask_files))
    
    for mask_file in mask_files:
        mask_name = os.path.basename(mask_file)
        case_idx = mask_name.split('_')[0]
        img_file = f'{source_folder}/{case_idx}.nii.gz'
        img = nib.load(img_file).get_fdata()
        mask = nib.load(mask_file).get_fdata()

        case_mask_folder = f'{client_data_folder}/{case_idx}/mask'
        case_mask_vis_folder = f'{client_data_folder}/{case_idx}/mask_vis'
        case_img_vis_folder = f'{client_data_folder}/{case_idx}/img_vis'
        case_img_train_folder = f'{client_data_folder}/{case_idx}/img_train'

        Path(case_mask_folder).mkdir(parents=True, exist_ok=True)
        Path(case_mask_vis_folder).mkdir(parents=True, exist_ok=True)
        Path(case_img_vis_folder).mkdir(parents=True, exist_ok=True)
        Path(case_img_train_folder).mkdir(parents=True, exist_ok=True)

        total_slices_with_label += np.count_nonzero(np.any(mask, axis=(0, 1)))

        for slice in range(img.shape[2]):
            mask_slice = mask[:, :, slice].astype(np.uint8)
            if not np.any(mask_slice):
                continue
            img_slice = img[:, :, slice]
            min_value = img_slice.min()
            max_value = img_slice.max() 

            # for visualization, normalize the image slice to [0, 255]
            img_slice_vis = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)

            # img_slice_train is for future use, it is used as the training data, which preserves the original information as possible.
            # for training
            # img_slice_train = img_slice.astype(np.float32)
            img_slice_train = img_slice_vis.astype(np.float32)


            # Save img_slice_vis as PNG
            Image.fromarray(img_slice_vis).save(f'{case_img_vis_folder}/{slice}.png')

            # Resize img_slice_train and save as .npy
            img_slice_train_resized = np.array(Image.fromarray(img_slice_train).resize(resize, Image.Resampling.LANCZOS))
            np.save(f'{case_img_train_folder}/{slice}.npy', img_slice_train_resized)

            # Save mask_slice as PNG
            Image.fromarray(mask_slice).resize(resize, Image.Resampling.NEAREST).save(f'{case_mask_folder}/{slice}.png')
            
            # Convert the mask_slice to a binary (black and white) image
            mask_slice_vis = np.where(mask_slice != 0, 255, 0).astype(np.uint8)
            
            # Resize and save the black and white mask_slice
            Image.fromarray(np.array(Image.fromarray(mask_slice_vis).resize(resize, Image.Resampling.NEAREST))).save(f'{case_mask_vis_folder}/{slice}.png')

            row = pd.DataFrame([{
                'image_id': f'{case_idx}_{slice}',
                'image_train_path': f'{case_img_train_folder}/{slice}.npy',
                'image_vis_path': f'{case_img_vis_folder}/{slice}.png',
                'segmentation_mask_path': f'{case_mask_folder}/{slice}.png',
            }])

            if mask_file in train_mask_files:
                train_df = pd.concat([train_df, row], ignore_index=True)
            else:
                test_df = pd.concat([test_df, row], ignore_index=True)
            processed_count += 1
            print(f'domain: {domain_name}, case: {case_idx}, slice: {slice}')
            
    train_df.to_csv(f'{client_folder}/train.csv', index=False)
    test_df.to_csv(f'{client_folder}/test.csv', index=False)

print(f'total slices with label: {total_slices_with_label}')
print(f'processed count: {processed_count}')