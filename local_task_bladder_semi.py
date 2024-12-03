import subprocess

# spine task
for i in range(1, 5):
    for j in range(1, 4):
        run_name = f"localized_fixmatch_bladder_client_{i}_run_{j}"
        train_path = f"/storage/zhipengdeng/data/segmentation/bladder/fed_semi/client_{i}"
        
        command = [
            "python3", "local_train.py",
            "--config", "/zhipengdeng/project/fed_semi/configs/bladder/run_conf.yaml",
            "--run_name", run_name,
            "--train_path", train_path,
            "--test_path", "/storage/zhipengdeng/data/segmentation/bladder/fed_semi/",
            "--trainer", "semi"
        ]
        subprocess.run(command)