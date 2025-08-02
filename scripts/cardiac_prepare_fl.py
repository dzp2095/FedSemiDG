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

repo_root = Path(__file__).resolve().parent.parent
cfg_path = repo_root / 'configs' / 'cardiac' / 'scripts.yaml' 

with cfg_path.open() as f:
    config = yaml.safe_load(f)

target_folder = Path(config.get("target_folder", "~/cardiac/fed_semi")).expanduser()
raw_data_folder = Path(config.get("raw_data_folder", "cardiac/OpenDataset")).expanduser()

csv_file = pd.read_csv(f'{raw_data_folder}/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
resize = (config.get("resize")['width'], config.get("resize")['height'])

random.seed(config.get("seed", 0))

vendor2client = {'A':1, 'B':2, 'C':3, 'D':4}
Path(target_folder).mkdir(parents=True, exist_ok=True)
test_ratio = config.get("test_ratio", 0.2)

df = pd.DataFrame(columns=[
    'image_id', 
    'image_train_path', 
    'image_vis_path',
    'segmentation_mask_path', 
])

client_dfs = {
    'A': {
        'train': df.copy(),
        'test': df.copy()
    },  
    'B': {
        'train': df.copy(),
        'test': df.copy()
    },
    'C': {
        'train': df.copy(),
        'test': df.copy()
    },
    'D': {
        'train': df.copy(),
        'test': df.copy()
    }
}
# Train/Test split based on the vendor
case2vendor = {'train': {}, 'test': {}}

for vendor in vendor2client.keys():
    vendor_csv = csv_file.loc[csv_file['Vendor'] == vendor]
    cases = vendor_csv['External code'].values
    train_cases = random.sample(list(cases), int(len(cases)*(1-test_ratio)))
    test_cases = list(set(cases) - set(train_cases))
    case2vendor['train'].update({case: vendor for case in train_cases})
    case2vendor['test'].update({case: vendor for case in test_cases})

# create client folders
for domain in vendor2client.values():
    client_folder = f'{target_folder}/client_{domain}'
    Path(client_folder).mkdir(parents=True, exist_ok=True)
    client_data_folder = f'{client_folder}/data'
    Path(client_data_folder).mkdir(parents=True, exist_ok=True)

total_slices_with_labels_per_time = 0 # 2D slices
total_time_points_with_labels = 0 # 3D volumes
processed_count = 0

# The directory structure is as in the original dataset.
# Process the data in the Testing folder:
test_mask_files = glob(f'{raw_data_folder}/Testing/*/*_gt.nii.gz')
train_mask_files = glob(f'{raw_data_folder}/Training/*/*/*_gt.nii.gz')

for mask_file in test_mask_files + train_mask_files:
    case_id = os.path.basename(mask_file).split('_')[0]
    vendor = csv_file.loc[csv_file['External code'] == case_id]['Vendor'].values[0]
    img_file = mask_file.replace('_gt.nii.gz', '.nii.gz')
    img = nib.load(img_file)
    mask = nib.load(mask_file)
    img_data = img.get_fdata()
    mask_data = mask.get_fdata()

    target_img_train_folder = f'{target_folder}/client_{vendor2client[vendor]}/data/{case_id}/img_train'
    target_img_vis_folder = f'{target_folder}/client_{vendor2client[vendor]}/data/{case_id}/img_vis'
    target_mask_folder = f'{target_folder}/client_{vendor2client[vendor]}/data/{case_id}/mask'
    target_mask_vis_folder = f'{target_folder}/client_{vendor2client[vendor]}/data/{case_id}/mask_vis'
    Path(target_img_train_folder).mkdir(parents=True, exist_ok=True)
    Path(target_img_vis_folder).mkdir(parents=True, exist_ok=True)
    Path(target_mask_folder).mkdir(parents=True, exist_ok=True)
    Path(target_mask_vis_folder).mkdir(parents=True, exist_ok=True)
    
    # count the number of slices with labels
    slices_with_labels_per_time = np.count_nonzero(np.any(mask_data, axis=(0, 1)))

    # count the number of time points with labels
    time_points_with_labels = np.count_nonzero(np.any(mask_data, axis=(0, 1, 2)))

    total_slices_with_labels_per_time += slices_with_labels_per_time
    total_time_points_with_labels += time_points_with_labels

    for slice in range(img_data.shape[2]):
        for frame in range(img_data.shape[3]):
            mask_slice = mask_data[:, :, slice, frame].astype(np.uint8)
            if not np.any(mask_slice):
                continue
            img_slice = img_data[:, :, slice, frame]

            # for visualization, normalize the image slice to [0, 255]
            img_slice_vis = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)
            # for training

            # img_slice_train is for future use, it is used as the training data, which preserves the original information as possible.
            # img_slice_train = img_slice.astype(np.float32)
            img_slice_train = img_slice_vis.astype(np.float32)


            # Resize img_slice_vis and convert to RGB
            img_slice_vis_resized = Image.fromarray(img_slice_vis).resize(resize, Image.Resampling.LANCZOS).convert('RGB')
            img_slice_vis_resized.save(f'{target_img_vis_folder}/{slice}_{frame}.png')

            # Resize img_slice_train and save as .npy
            img_slice_train_resized = np.array(Image.fromarray(img_slice_train).resize(resize, Image.Resampling.LANCZOS))
            np.save(f'{target_img_train_folder}/{slice}_{frame}.npy', img_slice_train_resized)

            # Save mask_slice as PNG
            Image.fromarray(mask_slice).resize(resize, Image.Resampling.NEAREST).save(f'{target_mask_folder}/{slice}_{frame}.png')

            # Convert image to grayscale if not already
            value_mapping = {1: 85, 2: 170, 3: 255}  # Map 1 -> 85, 2 -> 170, 3 -> 255
            mask_slice_vis = mask_slice.copy()

            for original_value, new_value in value_mapping.items():
                mask_slice_vis[mask_slice == original_value] = new_value

            # Resize and save mask_slice
            Image.fromarray(np.array(Image.fromarray(mask_slice_vis).resize(resize, Image.Resampling.NEAREST))).save(f'{target_mask_vis_folder}/{slice}_{frame}.png')

            row = pd.DataFrame([{
                'image_id': f'{case_id}_{slice}_{frame}',
                'image_train_path': f'{target_img_train_folder}/{slice}_{frame}.npy',
                'image_vis_path': f'{target_img_vis_folder}/{slice}_{frame}.png',
                'segmentation_mask_path': f'{target_mask_folder}/{slice}_{frame}.png',
            }])

            if case_id in case2vendor['train']:
                client_dfs[case2vendor['train'][case_id]]['train'] = pd.concat([client_dfs[case2vendor['train'][case_id]]['train'], row])
            else:
                client_dfs[case2vendor['test'][case_id]]['test'] = pd.concat([client_dfs[case2vendor['test'][case_id]]['test'], row])
            processed_count += 1
            print(f'Processed {case_id}_{slice}_{frame}')

for vendor in vendor2client.keys():
    client_idx = vendor2client[vendor]
    client_dfs[vendor]['train'].to_csv(f'{target_folder}/client_{client_idx}/train.csv', index=False)
    client_dfs[vendor]['test'].to_csv(f'{target_folder}/client_{client_idx}/test.csv', index=False)

print(f'Total slices with labels per time: {total_slices_with_labels_per_time}')
print(f'Total time points with labels: {total_time_points_with_labels}')
print(f'Total processed: {processed_count}')
