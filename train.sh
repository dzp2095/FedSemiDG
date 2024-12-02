# # Description: Train the model on the localized data: supervised learning
# for i in {1..4}
# do
#     for j in {1..3}
#     do
#         python3 local_train.py \
#             --config "/storage/zhipengdeng/project/fed_semi/configs/fundus/run_conf.yaml" \
#             --run_name "localized_client_${i}_run_${j}" \
#             --train_path "/storage/zhipengdeng/data/segmentation/fundus_dofe/fed_semi/client_$i" \
#             --test_path "/storage/zhipengdeng/data/segmentation/fundus_dofe/fed_semi/" \
#             --deterministic 0 \
#             --trainer "supervised" 
#     done
# done


# # Description: Train the model on the localized data: semi-supervised learning
# for i in {1..4}
# do
#     for j in {1..3}
#     do
#         python3 local_train.py \
#             --config "/storage/zhipengdeng/project/fed_semi/configs/fundus/run_conf.yaml" \
#             --run_name "localized_client_${i}_run_${j}" \
#             --train_path "/storage/zhipengdeng/data/segmentation/fundus_dofe/fed_semi/client_$i" \
#             --test_path "/storage/zhipengdeng/data/segmentation/fundus_dofe/fed_semi/" \
#             --deterministic 0 \
#             --trainer "semi" 
#     done
# done

# fundus task
python3 local_train.py \
    --config "/storage/zhipengdeng/project/fed_semi/configs/fundus/run_conf.yaml" \
    --run_name "localized_supervised_client_4" \
    --train_path "/storage/zhipengdeng/data/segmentation/fundus_dofe/fed_semi/client_4" \
    --test_path "/storage/zhipengdeng/data/segmentation/fundus_dofe/fed_semi/" \
    --trainer "supervised" 

python3 fl_train.py \
    --config "/zhipengdeng/project/fed_semi/configs/fundus/fl_run_conf.yaml" \
    --run_name "fl_semi_ga_labeled_1_2_3_unseen_4" \
    --train_path "/storage/zhipengdeng/data/segmentation/fundus_dofe/fed_semi" \
    --test_path "/storage/zhipengdeng/data/segmentation/fundus_dofe/fed_semi" \
    --labeled_clients client_1 client_2 client_3 \
    --trainer semi \
    --unseen_client client_4 \
    --use_ga

python3 fl_train.py \
    --config "/zhipengdeng/project/fed_semi/configs/fundus/fl_run_conf.yaml" \
    --run_name "fl_fully_labeled_fundus" \
    --train_path "/storage/zhipengdeng/data/segmentation/fundus_dofe/fed_semi/" \
    --test_path "/storage/zhipengdeng/data/segmentation/fundus_dofe/fed_semi/" \
    --labeled_clients client_1 client_2 client_3 \
    --unseen_client client_4 \
    --trainer "supervised" 

# prostate task
python3 local_train.py \
    --config "/storage/zhipengdeng/project/fed_semi/configs/prostate/run_conf.yaml" \
    --run_name "localized_supervised_prostate_client_2" \
    --train_path "/storage/zhipengdeng/data/segmentation/prostate_mri/fed_semi/client_2" \
    --test_path "/storage/zhipengdeng/data/segmentation/prostate_mri/fed_semi/" \
    --deterministic 0 \
    --trainer "supervised" 

python3 local_train.py \
    --config "/storage/zhipengdeng/project/fed_semi/configs/prostate/run_conf.yaml" \
    --run_name "localized_semi_prostate_client_2" \
    --train_path "/storage/zhipengdeng/data/segmentation/prostate_mri/fed_semi/client_2" \
    --test_path "/storage/zhipengdeng/data/segmentation/prostate_mri/fed_semi/" \
    --deterministic 0 \
    --trainer "semi" 

python3 fl_train.py \
    --config "/zhipengdeng/project/fed_semi/configs/prostate/fl_run_conf.yaml" \
    --run_name "fl_semi_prostate_labeled_1_2_3_4_5_unseen_6" \
    --train_path "/storage/zhipengdeng/data/segmentation/prostate_mri/fed_semi/" \
    --test_path "/storage/zhipengdeng/data/segmentation/prostate_mri/fed_semi/" \
    --labeled_clients client_1 client_2 client_3 client_4 client_5 \
    --trainer semi \
    --unseen_client client_6 \
    --use_ga

# cardiac task
python3 local_train.py \
    --config "/zhipengdeng/project/fed_semi/configs/cardiac/run_conf.yaml" \
    --run_name "localized_supervised_cardiac_client_4" \
    --train_path "/storage/zhipengdeng/data/segmentation/cardiac/fed_semi/client_4" \
    --test_path "/storage/zhipengdeng/data/segmentation/cardiac/fed_semi/" \
    --deterministic 0 \
    --trainer "supervised" 

python3 local_train.py \
    --config "/storage/zhipengdeng/project/fed_semi/configs/cardiac/run_conf_debug.yaml" \
    --run_name "localized_semi_cardiac_client_4" \
    --train_path "/storage/zhipengdeng/data/segmentation/cardiac/fed_semi/client_4" \
    --test_path "/storage/zhipengdeng/data/segmentation/cardiac/fed_semi/" \
    --deterministic 0 \
    --trainer "semi" 

python3 fl_train.py \
    --config "/zhipengdeng/project/fed_semi_dev/configs/cardiac/fl_run_conf.yaml" \
    --run_name "fl_semi_labeled_1_2_3_unseen_4" \
    --train_path "/storage/zhipengdeng/data/segmentation/cardiac/fed_semi/" \
    --test_path "/storage/zhipengdeng/data/segmentation/cardiac/fed_semi/" \
    --deterministic 0 \
    --labeled_clients client_1 client_2 client_3\
    --trainer semi \
    --unseen_client client_4 \
    --use_ga

# spine task
python3 fl_train.py \
    --config "/zhipengdeng/project/fed_semi/configs/spine/fl_run_conf.yaml" \
    --run_name "fl_fully_spine" \
    --train_path "/storage/zhipengdeng/data/segmentation/spine/fed_semi/" \
    --test_path "/storage/zhipengdeng/data/segmentation/spine/fed_semi/" \
    --labeled_clients client_1 client_2 client_3 \
    --unseen_client client_4 \
    --trainer "supervised" 

python3 fl_train.py \
    --config "/zhipengdeng/project/fed_semi/configs/spine/fl_run_conf.yaml" \
    --run_name "fl_semi_spine" \
    --train_path "/storage/zhipengdeng/data/segmentation/spine/fed_semi/" \
    --test_path "/storage/zhipengdeng/data/segmentation/spine/fed_semi/" \
    --labeled_clients client_1 client_2 client_3 \
    --unseen_client client_4 \
    --trainer "semi" 

python3 fl_train.py \
    --config "/zhipengdeng/project/fed_semi_dev/configs/spine/fl_run_conf.yaml" \
    --run_name "fl_ga_spine" \
    --train_path "/storage/zhipengdeng/data/segmentation/spine/fed_semi/" \
    --test_path "/storage/zhipengdeng/data/segmentation/spine/fed_semi/" \
    --labeled_clients client_1 client_2 client_3 \
    --unseen_client client_4 \
    --trainer "semi" \
    --use_ga

python3 local_train.py \
    --config "/zhipengdeng/project/fed_semi/configs/spine/run_conf.yaml" \
    --run_name "localized_semi_spine_client_4" \
    --train_path "/storage/zhipengdeng/data/segmentation/spine/fed_semi/client_4" \
    --test_path "/storage/zhipengdeng/data/segmentation/spine/fed_semi/" \
    --trainer "semi" 


# bladder task
python3 fl_train.py \
    --config "/zhipengdeng/project/fed_semi/configs/bladder/fl_run_conf.yaml" \
    --run_name "fl_fully_bladder" \
    --train_path "/storage/zhipengdeng/data/segmentation/bladder/fed_semi/" \
    --test_path "/storage/zhipengdeng/data/segmentation/bladder/fed_semi/" \
    --labeled_clients client_1 client_2 client_3 \
    --unseen_client client_4 \
    --trainer "supervised" 

python3 fl_train.py \
    --config "/zhipengdeng/project/fed_semi/configs/bladder/fl_run_conf.yaml" \
    --run_name "fl_semi_bladder" \
    --train_path "/storage/zhipengdeng/data/segmentation/bladder/fed_semi/" \
    --test_path "/storage/zhipengdeng/data/segmentation/bladder/fed_semi/" \
    --labeled_clients client_1 client_2 client_3 \
    --unseen_client client_4 \
    --trainer "semi" 

python3 fl_train.py \
    --config "/zhipengdeng/project/fed_semi_dev/configs/bladder/fl_run_conf.yaml" \
    --run_name "fl_ga_bladder" \
    --train_path "/storage/zhipengdeng/data/segmentation/bladder/fed_semi/" \
    --test_path "/storage/zhipengdeng/data/segmentation/bladder/fed_semi/" \
    --labeled_clients client_1 client_2 client_3 \
    --unseen_client client_4 \
    --trainer "semi" \
    --use_ga

python3 local_train.py \
    --config "/zhipengdeng/project/fed_semi/configs/bladder/run_conf.yaml" \
    --run_name "localized_semi_bladder_client_4" \
    --train_path "/storage/zhipengdeng/data/segmentation/bladder/fed_semi/client_4" \
    --test_path "/storage/zhipengdeng/data/segmentation/bladder/fed_semi/" \
    --trainer "supervised" 