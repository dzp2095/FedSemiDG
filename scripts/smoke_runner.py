#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


@dataclass
class DatasetSpec:
    name: str
    config: Path
    fl_config: Path
    data_root: Path
    clients: Tuple[str, ...] = ("client_1", "client_2", "client_3", "client_4")


CSV_CANDIDATES = ("train.csv", "test.csv", "labeled.csv", "unlabeled.csv", "all.csv")


def infer_storage_root(data_root: Path) -> Path:
    if data_root.parent.name == "data" and data_root.parent.parent != Path("/"):
        return data_root.parent.parent
    return data_root.parent


def build_prefix_rewrites(env: Mapping[str, str]) -> Dict[str, str]:
    data_root = env.get("FEDSEMI_DATA_ROOT", "/data/segmentation")
    storage_root = env.get("FEDSEMI_STORAGE_ROOT", "/data")
    project_root = env.get("FEDSEMI_PROJECT_ROOT", "/workspace/fed_semi")
    return {
        "/storage/zhipengdeng/data/segmentation": data_root,
        "/raid/zhipeng/data/segmentation": data_root,
        "/storage/zhipengdeng/project/fed_semi": project_root,
        "/zhipengdeng/project/fed_semi": project_root,
        "/storage/zhipengdeng": storage_root,
        "/raid/zhipeng": storage_root,
        "/zhipengdeng/project": project_root,
    }


def run_command(cmd: List[str], cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def maybe_generate_colon_fedsemi(repo_root: Path, colon_root: Path, env: Dict[str, str]) -> None:
    fedsemi_root = colon_root / "fed_semi"
    if fedsemi_root.exists() and (fedsemi_root / "client_1").exists():
        return
    if fedsemi_root.exists() and not (fedsemi_root / "client_1").exists():
        shutil.rmtree(fedsemi_root, ignore_errors=True)

    script = repo_root / "scripts" / "colon_prepare_fl.py"
    if not script.exists():
        raise FileNotFoundError(f"Colon prepare script not found: {script}")

    prepare_env = env.copy()
    prepare_env.setdefault("FEDSEMI_DATA_ROOT", str(colon_root.parent))
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(repo_root),
        env=prepare_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to generate colon fed_semi:\n{proc.stdout}")


def rewrite_csv_paths(
    src_csv: Path,
    dst_csv: Path,
    do_rewrite: bool,
    prefix_rewrites: Mapping[str, str],
    max_rows_per_csv: int,
) -> None:
    with src_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if max_rows_per_csv > 0:
        rows = rows[:max_rows_per_csv]

    if do_rewrite:
        for row in rows:
            for key, value in row.items():
                if not isinstance(value, str):
                    continue
                new_value = value
                for src_prefix, dst_prefix in prefix_rewrites.items():
                    if new_value.startswith(src_prefix):
                        new_value = dst_prefix + new_value[len(src_prefix) :]
                row[key] = new_value

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with dst_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_semi_csvs(client_root: Path) -> None:
    labeled_csv = client_root / "labeled.csv"
    unlabeled_csv = client_root / "unlabeled.csv"
    if labeled_csv.exists() and unlabeled_csv.exists():
        return

    train_csv = client_root / "train.csv"
    if not train_csv.exists():
        return

    with train_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows or not fieldnames:
        return

    split = max(1, len(rows) // 2)
    labeled_rows = rows[:split]
    unlabeled_rows = rows[split:] or rows[:split]

    for target, subset in ((labeled_csv, labeled_rows), (unlabeled_csv, unlabeled_rows)):
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subset)


def prepare_runtime_dataset(
    spec: DatasetSpec,
    runtime_root: Path,
    prefix_rewrites: Mapping[str, str],
    max_rows_per_csv: int,
) -> Path:
    if not spec.data_root.exists():
        raise FileNotFoundError(f"Dataset root missing: {spec.data_root}")

    target_root = runtime_root / spec.name
    for client in spec.clients:
        src_client = spec.data_root / client
        if not src_client.exists():
            raise FileNotFoundError(f"Missing client folder: {src_client}")

        dst_client = target_root / client
        dst_client.mkdir(parents=True, exist_ok=True)

        for csv_name in CSV_CANDIDATES:
            src_csv = src_client / csv_name
            if src_csv.exists():
                rewrite_csv_paths(
                    src_csv,
                    dst_client / csv_name,
                    do_rewrite=spec.name in {"spine", "bladder"},
                    prefix_rewrites=prefix_rewrites,
                    max_rows_per_csv=max_rows_per_csv,
                )

        ensure_semi_csvs(dst_client)

    return target_root


def build_specs(repo_root: Path, data_root: Path) -> List[DatasetSpec]:
    return [
        DatasetSpec(
            "cardiac",
            repo_root / "configs" / "cardiac" / "run_conf.yaml",
            repo_root / "configs" / "cardiac" / "fl_run_conf.yaml",
            data_root / "cardiac" / "fed_semi",
        ),
        DatasetSpec(
            "colon",
            repo_root / "configs" / "colon" / "run_conf.yaml",
            repo_root / "configs" / "colon" / "fl_run_conf.yaml",
            data_root / "colon" / "fed_semi",
        ),
        DatasetSpec(
            "spine",
            repo_root / "configs" / "spine" / "run_conf.yaml",
            repo_root / "configs" / "spine" / "fl_run_conf.yaml",
            data_root / "spine" / "fed_semi",
        ),
        DatasetSpec(
            "bladder",
            repo_root / "configs" / "bladder" / "run_conf.yaml",
            repo_root / "configs" / "bladder" / "fl_run_conf.yaml",
            data_root / "bladder" / "fed_semi",
        ),
    ]


def run_dataset_smoke(
    repo_root: Path,
    spec: DatasetSpec,
    runtime_dataset_root: Path,
    env: Dict[str, str],
    gpu_pool: str,
    batch_size: int,
    num_workers: int,
    max_iter: int,
) -> Dict[str, Tuple[bool, str]]:
    results: Dict[str, Tuple[bool, str]] = {}

    shared = [
        "--gpu_pool",
        gpu_pool,
        "--batch_size_override",
        str(batch_size),
        "--num_workers_override",
        str(num_workers),
    ]

    local_supervised_cmd = [
        sys.executable,
        "local_train.py",
        "--config",
        str(spec.config),
        "--run_name",
        f"smoke_local_sup_{spec.name}",
        "--train_path",
        str(runtime_dataset_root / "client_1"),
        "--trainer",
        "supervised",
        "--max_iter_override",
        str(max_iter),
    ] + shared

    code, out = run_command(local_supervised_cmd, repo_root, env)
    results["local_supervised"] = (code == 0, out)

    local_semi_cmd = [
        sys.executable,
        "local_train.py",
        "--config",
        str(spec.config),
        "--run_name",
        f"smoke_local_semi_{spec.name}",
        "--train_path",
        str(runtime_dataset_root / "client_1"),
        "--trainer",
        "semi",
        "--max_iter_override",
        str(max_iter),
    ] + shared

    code, out = run_command(local_semi_cmd, repo_root, env)
    results["local_semi"] = (code == 0, out)

    fl_semi_cmd = [
        sys.executable,
        "fl_train.py",
        "--config",
        str(spec.fl_config),
        "--run_name",
        f"smoke_fl_semi_{spec.name}",
        "--train_path",
        str(runtime_dataset_root),
        "--trainer",
        "semi",
        "--labeled_clients",
        "client_1",
        "client_2",
        "client_3",
        "--unseen_client",
        "client_4",
        "--rounds_override",
        "1",
        "--local_iter_override",
        "1",
        "--iter_per_round_override",
        "1",
    ] + shared

    code, out = run_command(fl_semi_cmd, repo_root, env)
    results["fl_semi"] = (code == 0, out)

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(os.environ.get("FEDSEMI_DATA_ROOT", "/data/segmentation")),
    )
    parser.add_argument("--gpu_pool", type=str, default="3,4", help="Visible physical GPU ids")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_iter", type=int, default=2)
    parser.add_argument("--max_rows_per_csv", type=int, default=64, help="Limit rows per CSV for fast smoke")
    parser.add_argument("--keep_runtime", action="store_true", help="Keep generated runtime csv workspace")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_root = args.data_root.resolve()

    env = os.environ.copy()
    env.setdefault("FEDSEMI_DATA_ROOT", str(data_root))
    env.setdefault("FEDSEMI_RAW_DATA_ROOT", str(data_root))
    env.setdefault("FEDSEMI_STORAGE_ROOT", str(infer_storage_root(data_root)))
    env.setdefault("FEDSEMI_PROJECT_ROOT", str(repo_root))
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_pool

    prefix_rewrites = build_prefix_rewrites(env)

    maybe_generate_colon_fedsemi(repo_root, data_root / "colon", env)

    runtime_root = Path(tempfile.mkdtemp(prefix="fedsemi_smoke_"))
    matrix: Dict[str, Dict[str, Tuple[bool, str]]] = {}

    try:
        for spec in build_specs(repo_root, data_root):
            if not spec.config.exists() or not spec.fl_config.exists():
                matrix[spec.name] = {"config": (False, f"Missing config: {spec.config} or {spec.fl_config}")}
                continue

            try:
                dataset_runtime_root = prepare_runtime_dataset(
                    spec,
                    runtime_root,
                    prefix_rewrites,
                    args.max_rows_per_csv,
                )
                matrix[spec.name] = run_dataset_smoke(
                    repo_root,
                    spec,
                    dataset_runtime_root,
                    env,
                    args.gpu_pool,
                    args.batch_size,
                    args.num_workers,
                    args.max_iter,
                )
            except Exception as exc:
                matrix[spec.name] = {"data": (False, str(exc))}

        print("=== Smoke Matrix ===")
        all_ok = True
        for dataset_name, results in matrix.items():
            print(f"[{dataset_name}]")
            for stage, (ok, output) in results.items():
                status = "PASS" if ok else "FAIL"
                print(f"  - {stage}: {status}")
                if not ok:
                    all_ok = False
                    tail = "\n".join(output.strip().splitlines()[-20:])
                    print(f"    tail:\n{tail}")

        return 0 if all_ok else 1
    finally:
        if args.keep_runtime:
            print(f"Runtime workspace kept at: {runtime_root}")
        else:
            shutil.rmtree(runtime_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
