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
import torch.nn.functional as F
import cv2
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
config = yaml.safe_load(open(filepath.joinpath("../configs/cardiac/scripts_conf.yaml")))

target_folder = config.get("target_folder", "~/cardiac/fed_semi")
raw_data_folder = config.get("raw_data_folder", f"cardiac/OpenDataset")
csv_file = pd.read_csv(f'{raw_data_folder}/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
resize = (config.get("resize")['width'], config.get("resize")['height'])

random.seed(config.get("seed", 0))

vendor2client = {'A':1, 'B':2, 'C':3, 'D':4}
Path(target_folder).mkdir(parents=True, exist_ok=True)
test_ratio = config.get("test_ratio", 0.2)

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

    target_img_vis_folder = f'{target_folder}/client_{vendor2client[vendor]}/data/{case_id}/img_gt_vis'

    Path(target_img_vis_folder).mkdir(parents=True, exist_ok=True)

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

            # Convert image to grayscale if not already
            value_mapping = {1: 85, 2: 170, 3: 255}  # Map 1 -> 85, 2 -> 170, 3 -> 255
            mask_slice_one_hot = F.one_hot(torch.tensor(mask_slice).long(), num_classes=4).permute(2, 0, 1).float()
            draw_mask_and_save(img_slice_vis, mask_slice_one_hot[1:], f'{target_img_vis_folder}/{slice}_{frame}.png')

            processed_count += 1
            print(f'Processed {case_id}_{slice}_{frame}')
print(f'Total slices with labels per time: {total_slices_with_labels_per_time}')
print(f'Total time points with labels: {total_time_points_with_labels}')
print(f'Total processed: {processed_count}')
