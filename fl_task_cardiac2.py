import subprocess
import itertools

all_labeled_sites = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
all_unseen_sites = [4, 3, 2, 1]

task = "cardiac"
for i in range(len(all_labeled_sites)):
    labeled_sites = all_labeled_sites[i]
    unseen_site = all_unseen_sites[i]
    flw = 0.0
    fpr = 0.0
    run_name = f"fl_{task}_ours_only_dual_labeled_{labeled_sites}_unseen_{unseen_site}"
    labeled_clients = ["client_" + str(client_id) for client_id in labeled_sites]
    unseen_client = f'client_{unseen_site}'
    command = [
        "python3", "fl_train.py",
        "--config", f"/zhipengdeng/project/fed_semi/configs/{task}/fl_run_conf.yaml",
        "--run_name", run_name,
        "--train_path", f"/storage/zhipengdeng/data/segmentation/{task}/fed_semi",
        "--test_path", f"/storage/zhipengdeng/data/segmentation/{task}/fed_semi",
        "--trainer", "semi",
        "--use_ga",
        "--unseen_client", unseen_client,
        "--labeled_clients"
    ] + labeled_clients + ["--feature_loss_weight", str(flw), "--fp_rate", str(fpr)]

    subprocess.run(command)