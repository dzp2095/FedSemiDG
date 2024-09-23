import subprocess

# cardiac task
for i in range(1, 5):
    for j in range(1, 4):
        run_name = f"localized_supervised_cardiac_client_{i}_run_{j}"
        train_path = f"/storage/zhipengdeng/data/segmentation/cardiac/fed_semi/client_{i}"
        
        command = [
            "python3", "local_train.py",
            "--config", "/storage/zhipengdeng/project/fed_semi/configs/cardiac/run_conf.yaml",
            "--run_name", run_name,
            "--train_path", train_path,
            "--test_path", "/storage/zhipengdeng/data/segmentation/cardiac/fed_semi/",
            "--deterministic", "0",
            "--trainer", "supervised"
        ]
        subprocess.run(command)