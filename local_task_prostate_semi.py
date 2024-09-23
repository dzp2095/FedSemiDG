import subprocess

# fundus task
# for i in range(1, 5):
#     for j in range(1, 4):
#         run_name = f"localized_semi_client_{i}_run_{j}"
#         train_path = f"/storage/zhipengdeng/data/segmentation/fundus_dofe/fed_semi/client_{i}"
        
#         command = [
#             "python3", "local_train.py",
#             "--config", "/storage/zhipengdeng/project/fed_semi/configs/fundus/run_conf.yaml",
#             "--run_name", run_name,
#             "--train_path", train_path,
#             "--test_path", "/storage/zhipengdeng/data/segmentation/fundus_dofe/fed_semi/",
#             "--deterministic", "0",
#             "--trainer", "semi"
#         ]
        
#         subprocess.run(command)

# prostate task
for i in range(1, 7):
    for j in range(1, 4):
        run_name = f"localized_semi_prostate_client_{i}_run_{j}"
        train_path = f"/storage/zhipengdeng/data/segmentation/prostate_mri/fed_semi/client_{i}"
        
        command = [
            "python3", "local_train.py",
            "--config", "/storage/zhipengdeng/project/fed_semi/configs/prostate/run_conf.yaml",
            "--run_name", run_name,
            "--train_path", train_path,
            "--test_path", "/storage/zhipengdeng/data/segmentation/prostate_mri/fed_semi/",
            "--deterministic", "0",
            "--trainer", "semi"
        ]
        subprocess.run(command)