import argparse
from typing import Any, Dict, List

from src.utils.path_utils import expand_cfg_paths, expand_path


def parse_gpu_list(value: str) -> List[int]:
    if not value or value.strip() == "":
        return []
    return [int(x) for x in value.split(",") if x.strip().isdigit()]


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--run_name", type=str, required=True, help="Experiment run name")

    parser.add_argument("--resume_path", type=str, default="", help="Checkpoint path to resume from")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--gpu", type=str, default="1", help="Use GPU when available")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training")

    parser.add_argument("--train_path", type=str, default=None, help="Override training data root")
    parser.add_argument("--trainer", type=str, default="supervised", help="Trainer type: supervised | semi")
    parser.add_argument("--labeled_clients", type=str, nargs="+", default=None, help="Labeled FL clients")
    parser.add_argument("--unseen_client", type=str, default=None, help="Held-out FL client")

    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch_size_override", type=int, default=None, help="Override train.batch_size")
    parser.add_argument("--num_workers_override", type=int, default=None, help="Override train.num_workers")

    parser.add_argument(
        "--gpu_exclude_list",
        type=parse_gpu_list,
        default=[],
        help="Comma-separated physical GPU ids to skip, for example: 0,3",
    )
    parser.add_argument(
        "--gpu_pool",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids exported through CUDA_VISIBLE_DEVICES",
    )

    parser.add_argument("--rounds_override", type=int, default=None, help="Override fl.rounds")
    parser.add_argument("--local_iter_override", type=int, default=None, help="Override fl.local_iter")
    parser.add_argument("--max_iter_override", type=int, default=None, help="Override train.max_iter")
    parser.add_argument(
        "--iter_per_round_override",
        type=int,
        default=None,
        help="Override local.iter_per_round.iter for FL client training",
    )

    args, _unknown = parser.parse_known_args()
    return args


args = args_parser()


def _apply_runtime_overrides(cfg: Dict[str, Any], run_args: argparse.Namespace) -> None:
    cfg.setdefault("train", {})

    if run_args.batch_size_override is not None:
        cfg["train"]["batch_size"] = run_args.batch_size_override

    if run_args.num_workers_override is not None:
        cfg["train"]["num_workers"] = run_args.num_workers_override

    if run_args.rounds_override is not None:
        cfg.setdefault("fl", {})
        cfg["fl"]["rounds"] = run_args.rounds_override

    if run_args.local_iter_override is not None:
        cfg.setdefault("fl", {})
        cfg["fl"]["local_iter"] = run_args.local_iter_override

    if run_args.max_iter_override is not None:
        cfg["train"]["max_iter"] = run_args.max_iter_override

    if run_args.iter_per_round_override is not None:
        cfg.setdefault("local", {})
        cfg["local"].setdefault("iter_per_round", {})
        cfg["local"]["iter_per_round"]["iter"] = run_args.iter_per_round_override
        cfg["local"]["iter_per_round"]["epoch"] = None


def args2cfg(cfg: Dict[str, Any], run_args: argparse.Namespace) -> Dict[str, Any]:
    cfg = expand_cfg_paths(cfg)

    cfg.setdefault("wandb", {})
    cfg["wandb"]["run_name"] = run_args.run_name

    if run_args.train_path is not None:
        cfg.setdefault("dataset", {})
        cfg["dataset"]["train"] = expand_path(run_args.train_path)

    if run_args.resume_path:
        cfg.setdefault("train", {})
        cfg["train"]["resume_path"] = expand_path(run_args.resume_path)

    if run_args.lr is not None:
        cfg.setdefault("train", {})
        cfg["train"].setdefault("optimizer", {})
        cfg["train"]["optimizer"]["lr"] = run_args.lr

    _apply_runtime_overrides(cfg, run_args)
    return cfg
