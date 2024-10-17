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

# Description: This script generates the labeled/unlabeled split for the clients in the federated semi-supervised learning setting.
# 1. Fundus Task
config = yaml.safe_load(open(filepath.joinpath("../configs/fundus/scripts_conf.yaml")))

target_folder = config.get("target_folder", "~/fundus_dofe/fed_semi")

labeled_slice_num = config.get("labeled_slice_num", 10)

domain_names = {1:'DGS', 2:'RIM', 3:'REF', 4:'REF_val'}

for domain in domain_names.keys():
    client_folder = f'{target_folder}/client_{domain}'
    # generate the labeled/unlabeled split
    train_df = pd.read_csv(f'{client_folder}/train.csv')
    test_df = pd.read_csv(f'{client_folder}/test.csv')
    data_df = pd.concat([train_df, test_df])

    labeled_df = data_df.sample(n=labeled_slice_num, random_state=1)
    labeled_df.to_csv(f'{client_folder}/labeled.csv', index=False)
    unlabeled_df = data_df.drop(labeled_df.index)
    unlabeled_df.to_csv(f'{client_folder}/unlabeled.csv', index=False)
    data_df.to_csv(f'{client_folder}/all.csv', index=False)

# 2. Prostate Task
config = yaml.safe_load(open(filepath.joinpath("../configs/prostate/scripts_conf.yaml")))
target_folder = config.get("target_folder", "~/prostate_mri/fed_semi")
labeled_slice_num = config.get("labeled_slice_num", 20)
domain_names = {1:'BIDMC', 2:'BMC', 3:'HK', 4:'I2CVB', 5:'RUNMC', 6:'UCL'}

for domain in domain_names.keys():
    client_folder = f'{target_folder}/client_{domain}'
    # generate the labeled/unlabeled split
    train_df = pd.read_csv(f'{client_folder}/train.csv')
    test_df = pd.read_csv(f'{client_folder}/test.csv')
    data_df = pd.concat([train_df, test_df])

    # notice: labeled data should be selected from consecutive slices within the same case
    all_cases = list(set(data_df['image_id'].str.extract(r'(Case\d+)_')[0].to_list()))
    # Randomly shuffle cases
    random.shuffle(all_cases)

    labeled_slices = []
    total_slices = 0
    
    for case in all_cases:
        # Select all slices from the current case
        case_slices = data_df[data_df['image_id'].str.startswith(case)]
        
        # Check if adding these slices exceeds the required labeled_slice_num
        if total_slices + len(case_slices) > labeled_slice_num:
            remaining_slices = labeled_slice_num - total_slices
            labeled_slices.append(case_slices.sample(n=remaining_slices, random_state=1))
            break
        else:
            labeled_slices.append(case_slices)
            total_slices += len(case_slices)

    labeled_df = pd.concat(labeled_slices)
    labeled_df.to_csv(f'{client_folder}/labeled.csv', index=False)

    unlabeled_df = data_df[~data_df['image_id'].isin(labeled_df['image_id'])]
    unlabeled_df.to_csv(f'{client_folder}/unlabeled.csv', index=False)
    data_df.to_csv(f'{client_folder}/all.csv', index=False)

# 3. Cardiac Task
config = yaml.safe_load(open(filepath.joinpath("../configs/cardiac/scripts_conf.yaml")))
target_folder = config.get("target_folder", "~/cardiac/fed_semi")
labeled_slice_num = config.get("labeled_slice_num", 20)
domain_names = {1:'A', 2:'B', 3:'C', 4:'D'}

for domain in domain_names.keys():
    client_folder = f'{target_folder}/client_{domain}'
    # generate the labeled/unlabeled split
    train_df = pd.read_csv(f'{client_folder}/train.csv')
    test_df = pd.read_csv(f'{client_folder}/test.csv')
    data_df = pd.concat([train_df, test_df])
    # notice: labeled data should be selected from consecutive slices within the same case
    all_cases = list(set(data_df['image_id'].str.split(r'_').apply(lambda x: x[0]).to_list()))
    # Randomly shuffle cases
    random.shuffle(all_cases)

    labeled_slices = []
    total_slices = 0
    
    for case in all_cases:
        # Select all slices from the current case
        case_slices = data_df[data_df['image_id'].str.startswith(case)]
        
        # Check if adding these slices exceeds the required labeled_slice_num
        if total_slices + len(case_slices) > labeled_slice_num:
            remaining_slices = labeled_slice_num - total_slices
            labeled_slices.append(case_slices.sample(n=remaining_slices, random_state=1))
            break
        else:
            labeled_slices.append(case_slices)
            total_slices += len(case_slices)

    labeled_df = pd.concat(labeled_slices)
    labeled_df.to_csv(f'{client_folder}/labeled.csv', index=False)

    unlabeled_df = data_df[~data_df['image_id'].isin(labeled_df['image_id'])]
    unlabeled_df.to_csv(f'{client_folder}/unlabeled.csv', index=False)
    data_df.to_csv(f'{client_folder}/all.csv', index=False)

# 4. Spine Task
config = yaml.safe_load(open(filepath.joinpath("../configs/spine/scripts_conf.yaml")))
target_folder = config.get("target_folder", "~/spine/fed_semi")
labeled_slice_num = config.get("labeled_slice_num", 20)
domain_names = {1:'A', 2:'B', 3:'C', 4:'D'}

for domain in domain_names.keys():
    client_folder = f'{target_folder}/client_{domain}'
    # generate the labeled/unlabeled split
    train_df = pd.read_csv(f'{client_folder}/train.csv')
    # data in test.csv is unlabeled
    test_df = pd.read_csv(f'{client_folder}/test.csv')
    
    # notice: labeled data should be selected from consecutive slices within the same case
    all_cases = list(set(train_df['image_id'].str.split(r'_').apply(lambda x: x[0]).to_list()))
    # Randomly shuffle cases
    random.shuffle(all_cases)

    labeled_slices = []
    total_slices = 0
    
    for case in all_cases:
        # Select all slices from the current case
        case_slices = train_df[train_df['image_id'].str.startswith(case)]
        
        # Check if adding these slices exceeds the required labeled_slice_num
        if total_slices + len(case_slices) > labeled_slice_num:
            remaining_slices = labeled_slice_num - total_slices
            labeled_slices.append(case_slices.sample(n=remaining_slices, random_state=1))
            break
        else:
            labeled_slices.append(case_slices)
            total_slices += len(case_slices)

    labeled_df = pd.concat(labeled_slices)
    labeled_df.to_csv(f'{client_folder}/labeled.csv', index=False)

    unlabeled_df = train_df[~train_df['image_id'].isin(labeled_df['image_id'])]
    unlabeled_df = pd.concat([unlabeled_df, test_df])
    unlabeled_df.to_csv(f'{client_folder}/unlabeled.csv', index=False)
    train_df.to_csv(f'{client_folder}/all.csv', index=False)

