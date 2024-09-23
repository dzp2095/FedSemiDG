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
import torch

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

filepath = Path(__file__).resolve().parent
config = yaml.safe_load(open(filepath.joinpath("../configs/cardiac/scripts_conf.yaml")))

target_folder = config.get("target_folder", "~/cardiac/fed_semi")
raw_data_folder = config.get("raw_data_folder", f"cardiac/OpenDataset")
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

    # Convert image to grayscale if not already
    value_mapping = {1: 85, 2: 170, 3: 255}  # Map 1 -> 85, 2 -> 170, 3 -> 255
    mask_data = mask_data.astype(np.uint8)
    mask_data = torch.tensor(mask_data).long().to('cuda:0')
    n_abnormal = torch.count_nonzero(mask_data > 3)
    if n_abnormal > 0:
        print(f'abnormal value in {case_id}')

