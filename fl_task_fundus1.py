import subprocess
import itertools

all_labeled_sites = [[2, 3], [1, 3], [1, 2], [2, 4], [1, 4], [1, 2]]
all_unlabed_sites = [[1], [2], [3], [1], [2], [4]]
all_unseen_sites = [[4], [4], [4], [3], [3], [3]]

for i in range(len(all_labeled_sites)):
    labeled_sites = all_labeled_sites[i]
    unlabeled_sites = all_unlabed_sites[i]
    unseen_sites = all_unseen_sites[i]
    run_name = f"fl_semi_labeled_{labeled_sites}_unlabeled_{unlabeled_sites}_unseen_{unseen_sites}"
    labeled_clients = ["client_" + str(client_id) for client_id in labeled_sites]
    unlabeled_clients = ["client_" + str(client_id) for client_id in unlabeled_sites]
    unseen_clients = ["client_" + str(client_id) for client_id in unseen_sites]
    command = [
        "python3", "fl_train.py",
        "--config", "/storage/zhipengdeng/project/fed_semi/configs/fundus/fl_run_conf.yaml",
        "--run_name", run_name,
        "--train_path", "/storage/zhipengdeng/data/segmentation/fundus_dofe/fed_semi/",
        "--test_path", "/storage/zhipengdeng/data/segmentation/fundus_dofe/fed_semi/",
        "--deterministic", "0",
        "--labeled_clients"
    ] + labeled_clients + [
        "--unlabeled_clients"
    ] + unlabeled_clients + [
        "--unseen_clients"
    ] + unseen_clients

    subprocess.run(command)