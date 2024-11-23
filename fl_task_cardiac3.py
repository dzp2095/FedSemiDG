import subprocess
import itertools

all_labeled_sites = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
all_unseen_sites = [4, 3, 2, 1]

task = "cardiac"

feature_loss_weight = [0.1, 0.3, 0.5]
fp_rate = [0.3, 0.5]
entropy_start_ratio = [0.1, 0.2, 0.3]
entropy_end_ratio = [0.7]

for flw, fpr, esr, eer in itertools.product(feature_loss_weight, fp_rate, entropy_start_ratio, entropy_end_ratio):
    for i in range(len(all_labeled_sites)):
        labeled_sites = all_labeled_sites[i]
        unseen_site = all_unseen_sites[i]
        run_name = f"fl_{task}_ours_dynamic_flw_{flw}_fpr_{fpr}_esr_{esr}_eer_{eer}_labeled_{labeled_sites}_unseen_{unseen_site}"
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
        ] + labeled_clients + ["--feature_loss_weight", str(flw), "--fp_rate", str(fpr), "--entropy_start_ratio", str(esr), "--entropy_end_ratio", str(eer)]
        subprocess.run(command)