import subprocess

# colon task
gpu_pool = ["2"]
for i in range(1, 5):
    for j in range(1, 4):
        run_name = f"localized_labelonly_colon_client_{i}_run_{j}"
        train_path = f"/home/dengzhipeng/data/segmentation/colon/fed_semi/client_{i}"
        command = [
            "python3", "local_train.py",
            "--config", "/home/dengzhipeng/project/fed_semi/configs/colon/run_conf.yaml",
            "--run_name", run_name,
            "--train_path", train_path,
            "--test_path", "/home/dengzhipeng/data/segmentation/colon/fed_semi/",
            "--trainer", "supervised",
            "--gpu_pool", ",".join(map(str, gpu_pool)),
        ]
        subprocess.run(command, check=True)
