import os
import sys
import pandas as pd
import random

from glob import glob

import yaml
from pathlib import Path
from PIL import Image

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

filepath = Path(__file__).resolve().parent
config = yaml.safe_load(open(filepath.joinpath("../configs/fundus/scripts_conf.yaml")))

target_folder = config.get("target_folder", "~/fundus_dofe/fed_semi")
raw_data_folder = config.get("raw_data_folder", f"fundus_dofe/Fundus")

resize = (config.get("resize")['width'], config.get("resize")['height'])
labeled_slice_num = config.get("labeled_slice_num", 5)

domain_names = {1:'DGS', 2:'RIM', 3:'REF', 4:'REF_val'}

Path(target_folder).mkdir(parents=True, exist_ok=True)

for domain in domain_names.keys():
    client_folder = f'{target_folder}/client_{domain}'
    Path(client_folder).mkdir(parents=True, exist_ok=True)
    source_folder = f'{raw_data_folder}/Domain{domain}'

    client_img_folder = f'{client_folder}/data/img'
    client_mask_folder = f'{client_folder}/data/mask'
    Path(client_img_folder).mkdir(parents=True, exist_ok=True)
    Path(client_mask_folder).mkdir(parents=True, exist_ok=True)

    for split in ['train', 'test']:
        img_folder = f'{source_folder}/{split}/ROIs/image'
        mask_folder = f'{source_folder}/{split}/ROIs/mask'
        img_files = glob(f'{img_folder}/*.png')

        # create a dataframe to store the image path and mask path
        df = pd.DataFrame(columns=[
            'image_id', 
            'image_path', 
            'segmentation_mask_path', 
        ])
        
        for img_file in img_files:
            img_name = os.path.basename(img_file)
            mask_file = f'{mask_folder}/{img_name}'
            img = Image.open(img_file).convert('RGB').resize(resize, Image.Resampling.LANCZOS)
            mask = Image.open(mask_file)
            if mask.mode == 'RGB':
                mask = mask.convert('L')
            mask = mask.resize(resize, Image.Resampling.NEAREST)
            updated_img_file = f'{client_img_folder}/{img_name}'
            updated_mask_file = f'{client_mask_folder}/{img_name}'
            img.save(updated_img_file)
            mask.save(updated_mask_file)

            row = pd.DataFrame([{
                'image_id': img_name,
                'image_path': updated_img_file,
                'segmentation_mask_path': updated_mask_file,
            }])
            df = pd.concat([df, row], ignore_index=True)
            print(f'Processed {img_name}')

        df.to_csv(f'{client_folder}/{split}.csv', index=False)

    # generate the labeled/unlabeled split
    fully_train_df = pd.read_csv(f'{client_folder}/train.csv')
    labeled_df = fully_train_df.sample(n=labeled_slice_num, random_state=1)
    labeled_df.to_csv(f'{client_folder}/labeled.csv', index=False)
    unlabeled_df = fully_train_df.drop(labeled_df.index)
    unlabeled_df.to_csv(f'{client_folder}/unlabeled.csv', index=False)
    
