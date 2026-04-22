# Partially from https://github.com/MedcAILab/FedBCa

import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

global nii_name
import pandas as pd
import cv2

import sys
import pandas as pd
import random
import numpy as np
from glob import glob

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

def getListIndex(arr, value):
    dim1_list = dim2_list = dim3_list = []
    if (arr.ndim == 3):
        index = np.argwhere(arr == value)
        dim1_list = index[:, 0].tolist()
        dim2_list = index[:, 1].tolist()
        dim3_list = index[:, 2].tolist()

    else:
        raise ValueError('The ndim of array must be 3!!')

    return dim1_list, dim2_list, dim3_list

def ROI_cutting(o_data, o_roi, expend_voxel=1):
    print('#ROI cutting...')
    data = o_data
    roi = o_roi

    [I1, I2, I3] = getListIndex(roi, 1)
    d1_min = min(I1)
    d1_max = max(I1)
    d2_min = min(I2)
    d2_max = max(I2)
    d3_min = min(I3)
    d3_max = max(I3)
    print(d3_min, d3_max)

    if expend_voxel > 0:
        d1_min -= expend_voxel
        d1_max += expend_voxel
        d2_min -= expend_voxel
        d2_max += expend_voxel
        d3_min -= 1
        d3_max += 1

        d1_min = d1_min if d1_min > 0 else 0
        d1_max = d1_max if d1_max < data.shape[0] - 1 else data.shape[0] - 1
        d2_min = d2_min if d2_min > 0 else 0
        d2_max = d2_max if d2_max < data.shape[1] - 1 else data.shape[1] - 1
        d3_min = d3_min if d3_min > 0 else 0
        d3_max = d3_max if d3_max < data.shape[2] - 1 else data.shape[2] - 1

    data = data[d1_min:d1_max, d2_min:d2_max, d3_min:d3_max]
    print(data.shape)
    roi = roi[d1_min:d1_max, d2_min:d2_max, d3_min:d3_max]

    print("--Cutting size:", data.shape)
    return data, roi

def centre_window_cropping(o_data, reshapesize=None):
    print('#Centre window cropping...')
    data = o_data
    or_size = data.shape
    target_size = (reshapesize[0], reshapesize[1], or_size[2])

    # pad if or_size is smaller than target_size
    if (target_size[0] > or_size[0]) | (target_size[1] > or_size[1]):
        if target_size[0] > or_size[0]:
            pad_size = int((target_size[0] - or_size[0]) / 2)
            data = np.pad(data, ((pad_size, pad_size), (0, 0), (0, 0)))
        if target_size[1] > or_size[1]:
            pad_size = int((target_size[1] - or_size[1]) / 2)
            data = np.pad(data, ((0, 0), (pad_size, pad_size), (0, 0)))

    #  centre_window_cropping
    cur_size = data.shape
    centre_x = float(cur_size[0] / 2)
    centre_y = float(cur_size[1] / 2)
    dx = float(target_size[0] / 2)
    dy = float(target_size[1] / 2)
    data = data[int(centre_x - dx + 1):int(centre_x + dx), int(centre_y - dy + 1): int(centre_y + dy), :]

    data_resize = np.zeros((reshapesize[0], reshapesize[1], cur_size[2]))
    for kk in range(cur_size[2]):
        data_resize[:, :, kk] = cv2.resize(data[:, :, kk], (reshapesize[0], reshapesize[1]),
                                           interpolation=cv2.INTER_NEAREST)

    return data_resize

def linear_normalizing(o_data):
    print('#Linear_normalizing...')
    data = o_data
    data_min = np.min(data)
    data_max = np.max(data)
    data = (data - data_min) / (data_max - data_min)

    return data


current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

filepath = Path(__file__).resolve().parent
config = yaml.safe_load(open(filepath.joinpath("../configs/bladder/scripts.yaml")))

target_folder = os.path.expandvars(config.get("target_folder", "~/bladder/fed_semi"))
raw_data_folder = os.path.expandvars(config.get("raw_data_folder", f"bladder/data"))
resize = (config.get("resize")['width'], config.get("resize")['height'])

random.seed(config.get("seed", 0))

center2client = {'Center1':1, 'Center2':2, 'Center3':3, 'Center4':4}
Path(target_folder).mkdir(parents=True, exist_ok=True)
test_ratio = config.get("test_ratio", 0.2)


df = pd.DataFrame(columns=[
    'image_id', 
    'image_train_path', 
    'image_vis_path',
    'segmentation_mask_path', 
])
dfs = [df] * 4

if __name__ == "__main__":

    Path(target_folder).mkdir(parents=True, exist_ok=True)

    total_slices_with_label = 0 
    processed_count = 0

    expend_voxel = 15
    deep = 1
    step = 1

    for domain in center2client.values():
        client_folder = f'{target_folder}/client_{domain}'
        Path(client_folder).mkdir(parents=True, exist_ok=True)
        client_data_folder = f'{client_folder}/data'
        Path(client_data_folder).mkdir(parents=True, exist_ok=True)

    for center in center2client.keys():
        client_raw_data_folder = f'{raw_data_folder}/{center}'
        roi_files = glob(f'{client_raw_data_folder}/Annotation/*.nii.gz')

        label_list = pd.read_excel(f'{raw_data_folder}/{center}/{center}_label.xlsx')
        label_list = np.array(label_list)

        for roi_file in roi_files:
            print(f'Processing {roi_file}...')
            roi_id = os.path.basename(roi_file).split(".nii.gz")[0]
            case_id = roi_id.split("_")[0]

            client_data_folder = f'{target_folder}/client_{center2client[center]}/data'

            target_img_train_folder = f'{client_data_folder}/{roi_id}/img_train'
            target_img_vis_folder = f'{client_data_folder}/{roi_id}/img_vis'
            target_mask_folder = f'{client_data_folder}/{roi_id}/mask'
            target_mask_vis_folder = f'{client_data_folder}/{roi_id}/mask_vis'
            target_img_gt_vis_folder = f'{client_data_folder}/{roi_id}/img_vis_gt'

            Path(target_img_train_folder).mkdir(parents=True, exist_ok=True)
            Path(target_img_vis_folder).mkdir(parents=True, exist_ok=True)
            Path(target_mask_folder).mkdir(parents=True, exist_ok=True)
            Path(target_mask_vis_folder).mkdir(parents=True, exist_ok=True)
            Path(target_img_gt_vis_folder).mkdir(parents=True, exist_ok=True)

            img_file = os.path.join(f'{client_raw_data_folder}/T2WI', case_id + ".nii.gz")
            img_data = nib.load(img_file)
            img_arr = np.array(img_data.dataobj, dtype='float32')
            
            roi_data = nib.load(roi_file)
            roi_arr = np.array(roi_data.dataobj, dtype='float32')
            roi_arr[roi_arr < 0.5] = 0
            roi_arr[roi_arr >= 0.5] = 1
            img_arr, roi_arr = ROI_cutting(img_arr, roi_arr, expend_voxel=expend_voxel)

            # img_arr = centre_window_cropping(img_arr, reshapesize=resize)
            # roi_arr = centre_window_cropping(roi_arr, reshapesize=resize)
            img_arr = linear_normalizing(img_arr)

            for slice in range(img_arr.shape[-1]):
                mask_slice = roi_arr[:,:,slice]
                if np.count_nonzero(mask_slice) == 0:
                    print(f'No label found for slice {slice}')
                    continue
                img_slice = img_arr[:,:,slice]
                img_slice_vis = (img_slice * 255).astype(np.uint8)
                # for training
                img_slice_train_resized = np.array(Image.fromarray(img_slice).resize(resize, Image.Resampling.LANCZOS))
                np.save(f'{target_img_train_folder}/{slice}.npy', img_slice_train_resized)

                # for visualization
                Image.fromarray(img_slice_vis).resize(resize, Image.Resampling.LANCZOS).save(f'{target_img_vis_folder}/{slice}.png')

                Image.fromarray(mask_slice.astype(np.uint8)).resize(resize, Image.Resampling.NEAREST).save(f'{target_mask_folder}/{slice}.png')
                # mask visualization
                mask_rgb = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 3), dtype=np.uint8)
                mask_rgb[:, :, 0] = (mask_slice!= 0) * 255
                Image.fromarray(mask_rgb).resize(resize, Image.Resampling.LANCZOS).save(f'{target_mask_vis_folder}/{slice}.png')

                # img visualization with gt
                mask_slice = np.array(Image.fromarray(mask_slice).resize(resize, Image.Resampling.NEAREST))
                img_slice_vis = np.array(Image.fromarray(img_slice_vis).resize(resize, Image.Resampling.LANCZOS))

                mask_slice = torch.from_numpy(mask_slice)
                draw_mask_and_save(img_slice_vis, mask_slice, f'{target_img_gt_vis_folder}/{slice}.png')

                row = pd.DataFrame([{
                    'image_id': f'{roi_id}_{slice}',
                    'image_train_path': f'{target_img_train_folder}/{slice}.npy',
                    'image_vis_path': f'{target_img_vis_folder}/{slice}.png',
                    'segmentation_mask_path': f'{target_mask_folder}/{slice}.png',
                }])
                processed_count += 1
                dfs[center2client[center]-1] = pd.concat([dfs[center2client[center]-1], row], ignore_index=True)
                print(f'center: {center}, case: {roi_id}, slice: {slice}')

for i in range(1,5):
    # train test split
    test_size = int(len(dfs[i-1]) * test_ratio)
    test_df = dfs[i-1].sample(test_size)
    train_df = dfs[i-1].drop(test_df.index)
    test_df.to_csv(f'{target_folder}/client_{i}/test.csv', index=False)
    train_df.to_csv(f'{target_folder}/client_{i}/train.csv', index=False)

print(f'total slices with label: {total_slices_with_label}')
print(f'processed count: {processed_count}')


