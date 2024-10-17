from PIL import Image
import numpy as np 
from glob import glob
from pathlib import Path
import os

pred_img_path = "/storage/zhipengdeng/data/segmentation/cardiac/fed_semi/client_1/data/A0S9V9/img_vis/result/fl_cardiac_fixmatch_labeled_[2, 3, 4]_unseen_1_round_95_client_1/3_0_dice_0.9328.png"
original_img_path = "/storage/zhipengdeng/data/segmentation/cardiac/fed_semi/client_1/data/A0S9V9/img_vis/3_0.png"

pred_img_path = "/storage/zhipengdeng/data/segmentation/cardiac/fed_semi/*/data/*/img_vis/result/*/*.png"
pred_img_files = glob(pred_img_path)

for pred_img_file in pred_img_files:
    file_name = os.path.basename(pred_img_file)
    original_file_name = file_name.split('_dice')[0] + '.png'
    original_img_path = Path(pred_img_file).parents[2].joinpath(original_file_name)

    pred_vis = Image.open(pred_img_file)
    original_img_path = Image.open(original_img_path)
    pred_vis_np = np.array(pred_vis).astype(np.int32)
    original_img_np = np.array(original_img_path).astype(np.int32)
    pred_mask = pred_vis_np * 2 - original_img_np

    pred_mask = (pred_vis_np * 2) - original_img_np

    pred_mask[pred_mask==-1]=0
    pred_mask[pred_mask==254]=255
    pred_mask[pred_mask==1]=0
    pred_mask[pred_mask==256]=255



    new_pred_img = np.zeros_like(pred_mask)

    colors_to_replace = {
        (0, 255, 0): (255, 0, 0),
        (0, 0, 255): (0, 255, 0),
        (255, 255, 0): (0, 0, 255)
    }

    for old_color, new_color in colors_to_replace.items():
        mask = np.all(pred_mask == old_color, axis=-1)
        new_pred_img[mask] = new_color

    new_pred_img = ((new_pred_img + original_img_np)* (0.5,0.5,0.5)).astype(np.uint8)

    save_file_name = Path(pred_img_file).parent.joinpath(f'{file_name.split("_dice")[0]}_new.png')
    Image.fromarray(new_pred_img).save(save_file_name)
    print(f"Saved {save_file_name}")

