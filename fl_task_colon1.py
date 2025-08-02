import os
import signal
import subprocess
import itertools
import multiprocessing as mp
from typing import Any
import time
from multiprocessing.managers import ListProxy

all_labeled_sites = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
all_unseen_sites = [4, 3, 2, 1]

task = "colon"
data_dir = "colon"
gpu_ids = [0, 1, 2, 3]

feature_loss_weight = [0.3, 0.5]
fp_rate = [0.3, 0.5]
entropy_pairs = [(0.15, 0.3), (0.1, 0.5)]

param_grid = [
    (flw, fpr, esr, eer)
    for flw, fpr in itertools.product(feature_loss_weight, fp_rate)
    for (esr, eer) in entropy_pairs
]

def build_commands(group_idx: int):
    labeled_sites = all_labeled_sites[group_idx]
    unseen_site = all_unseen_sites[group_idx]

    commands = []
    for flw, fpr, esr, eer in param_grid:
        run_name = (f"fl_{task}_fgasl_flw_{flw}_fpr_{fpr}"
                    f"_esr_{esr}_eer_{eer}_labeled_{labeled_sites}_unseen_{unseen_site}")
        labeled_clients = ["client_" + str(cid) for cid in labeled_sites]
        unseen_client = f"client_{unseen_site}"

        cmd = [
            "python3", "fl_train.py",
            "--config", f"/home/dengzhipeng/project/fed_semi/configs/{task}/fl_run_conf.yaml",
            "--run_name", run_name,
            "--train_path", f"/home/dengzhipeng/data/segmentation/{data_dir}/fed_semi",
            "--test_path", f"/home/dengzhipeng/data/segmentation/{data_dir}/fed_semi",
            "--trainer", "semi",
            "--use_ga",
            "--unseen_client", unseen_client,
            "--labeled_clients", *labeled_clients,
            # 如果需要，还可加上 "--gpu_pool", "0"
        ]
        cmd += ["--feature_loss_weight", str(flw),
                "--fp_rate", str(fpr),
                "--entropy_start_ratio", str(esr),
                "--entropy_end_ratio", str(eer)]
        commands.append((run_name, cmd))
    return commands

def kill_process_group(pid: int):
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except Exception:
        # 兜底
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

def worker(group_idx: int, gpu_id: int, stop_event: Any, running_pids: ListProxy):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    for run_name, cmd in build_commands(group_idx):
        if stop_event.is_set():
            break

        log_path = os.path.join(log_dir, f"{run_name}.log")
        with open(log_path, "w") as f:
            p = subprocess.Popen(
                cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                start_new_session=True  # 便于整组终止
            )
            running_pids.append(p.pid)
            try:
                # 轮询：一旦收到停止信号，立刻终止当前训练
                while True:
                    ret = p.poll()
                    if ret is not None:
                        break
                    if stop_event.is_set():
                        kill_process_group(p.pid)
                        p.wait()
                        break
                    time.sleep(1)
            finally:
                try:
                    running_pids.remove(p.pid)
                except ValueError:
                    pass

if __name__ == "__main__":
    manager = mp.Manager()
    stop_event = manager.Event()     # EventProxy（可在进程间共享）
    running_pids = manager.list()    # ListProxy

    assignments = list(zip(range(len(all_labeled_sites)), gpu_ids))
    pool = mp.Pool(processes=len(assignments))
    try:
        pool.starmap(worker, [(gi, gpu, stop_event, running_pids) for gi, gpu in assignments])
    except KeyboardInterrupt:
        stop_event.set()
        for pid in list(running_pids):
            kill_process_group(pid)
        pool.terminate()
    finally:
        pool.join()