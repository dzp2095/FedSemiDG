import subprocess
import itertools


all_labeled_sites = [
   [1,2,3,4,5], [1,2,3,4,6], [1,2,3,5,6], [1,2,4,5,6], [1,3,4,5,6], [2,3,4,5,6]
]

all_unseen_sites = [ 6, 5, 4, 3, 2, 1]

task = "prostate"
data_dir = "prostate_mri"

for i in range(len(all_labeled_sites)):
    labeled_sites = all_labeled_sites[i]
    unseen_site = all_unseen_sites[i]
    run_name = f"fl_{task}_fixmatch_labeled_{labeled_sites}_unseen_{unseen_site}"
    labeled_clients = ["client_" + str(client_id) for client_id in labeled_sites]
    unseen_client = f'client_{unseen_site}'
    command = [
        "python3", "fl_train.py",
        "--config", f"/zhipengdeng/project/fed_semi/configs/{task}/fl_run_conf.yaml",
        "--run_name", run_name,
        "--train_path", f"/storage/zhipengdeng/data/segmentation/{data_dir}/fed_semi",
        "--test_path", f"/storage/zhipengdeng/data/segmentation/{data_dir}/fed_semi",
        "--trainer", "semi",
        "--unseen_client", unseen_client,
        "--labeled_clients"
    ] + labeled_clients

    subprocess.run(command)