import subprocess
import itertools

all_labeled_sites = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
all_unseen_sites = [[4], [3], [2], [1]]


for i in range(len(all_labeled_sites)):
    labeled_sites = all_labeled_sites[i]
    unseen_sites = all_unseen_sites[i]
    run_name = f"fl_cardiac_labelonly_labeled_{labeled_sites}_unseen_{unseen_sites}"
    labeled_clients = ["client_" + str(client_id) for client_id in labeled_sites]
    unseen_clients = ["client_" + str(client_id) for client_id in unseen_sites]
    command = [
        "python3", "fl_train.py",
        "--config", "/zhipengdeng/project/fed_semi/configs/cardiac/fl_run_conf.yaml",
        "--run_name", run_name,
        "--train_path", "/storage/zhipengdeng/data/segmentation/cardiac/fed_semi",
        "--test_path", "/storage/zhipengdeng/data/segmentation/cardiac/fed_semi",
        "--deterministic", "0",
        "--trainer", "supervised",
        "--labeled_clients"
    ] + labeled_clients + [
        "--unseen_clients"
    ] + unseen_clients

    subprocess.run(command)