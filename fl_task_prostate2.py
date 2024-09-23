import subprocess
import itertools


all_labeled_sites = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 4], [2, 4], [2, 4], [2, 4], [2, 5], [2, 5], [2, 5], [2, 5], [2, 6], [2, 6], [2, 6], [2, 6], [3, 4], [3, 4], [3, 4], [3, 4]]

all_unlabed_sites = [[4, 5, 6], [1, 5, 6], [1, 4, 6], [1, 4, 5], [3, 5, 6], [1, 5, 6], [1, 3, 6], [1, 3, 5], [3, 4, 6], [1, 4, 6], [1, 3, 6], [1, 3, 4], [3, 4, 5], [1, 4, 5], [1, 3, 5], [1, 3, 4], [2, 5, 6], [1, 5, 6], [1, 2, 6], [1, 2, 5]]

all_unseen_sites = [[1], [4], [5], [6], [1], [3], [5], [6], [1], [3], [4], [6], [1], [3], [4], [5], [1], [2], [5], [6]]


for i in range(len(all_labeled_sites)):
    labeled_sites = all_labeled_sites[i]
    unlabeled_sites = all_unlabed_sites[i]
    unseen_sites = all_unseen_sites[i]
    run_name = f"fl_semi_prostate_labeled_{labeled_sites}_unlabeled_{unlabeled_sites}_unseen_{unseen_sites}"
    labeled_clients = ["client_" + str(client_id) for client_id in labeled_sites]
    unlabeled_clients = ["client_" + str(client_id) for client_id in unlabeled_sites]
    unseen_clients = ["client_" + str(client_id) for client_id in unseen_sites]
    command = [
        "python3", "fl_train.py",
        "--config", "/storage/zhipengdeng/project/fed_semi/configs/prostate/fl_run_conf.yaml",
        "--run_name", run_name,
        "--train_path", "/storage/zhipengdeng/data/segmentation/prostate_mri/fed_semi/",
        "--test_path", "/storage/zhipengdeng/data/segmentation/prostate_mri/fed_semi/",
        "--deterministic", "0",
        "--labeled_clients"
    ] + labeled_clients + [
        "--unlabeled_clients"
    ] + unlabeled_clients + [
        "--unseen_clients"
    ] + unseen_clients

    subprocess.run(command)