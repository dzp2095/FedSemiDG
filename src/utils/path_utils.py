import os
from typing import Dict, Mapping

DEFAULT_DATA_ROOT = os.environ.get("FEDSEMI_DATA_ROOT", "/data/segmentation")
DEFAULT_STORAGE_ROOT = os.environ.get("FEDSEMI_STORAGE_ROOT", "/data")
DEFAULT_PROJECT_ROOT = os.environ.get("FEDSEMI_PROJECT_ROOT", "/workspace/fed_semi")

# Keep specific legacy prefixes before generic ones to avoid duplicated segments
# like /data/data/segmentation when normalizing old CSV paths.
DEFAULT_PATH_REWRITES = {
    "/storage/zhipengdeng/data/segmentation": DEFAULT_DATA_ROOT,
    "/raid/zhipeng/data/segmentation": DEFAULT_DATA_ROOT,
    "/storage/zhipengdeng/project/fed_semi": DEFAULT_PROJECT_ROOT,
    "/zhipengdeng/project/fed_semi": DEFAULT_PROJECT_ROOT,
    "/storage/zhipengdeng": DEFAULT_STORAGE_ROOT,
    "/raid/zhipeng": DEFAULT_STORAGE_ROOT,
    "/zhipengdeng/project": DEFAULT_PROJECT_ROOT,
}


def expand_path(value: str) -> str:
    if not isinstance(value, str):
        return value

    value = value.replace("${FEDSEMI_DATA_ROOT}", os.environ.get("FEDSEMI_DATA_ROOT", DEFAULT_DATA_ROOT))
    value = value.replace("${FEDSEMI_STORAGE_ROOT}", os.environ.get("FEDSEMI_STORAGE_ROOT", DEFAULT_STORAGE_ROOT))
    value = value.replace("${FEDSEMI_PROJECT_ROOT}", os.environ.get("FEDSEMI_PROJECT_ROOT", DEFAULT_PROJECT_ROOT))
    value = value.replace("${FEDSEMI_RAW_DATA_ROOT}", os.environ.get("FEDSEMI_RAW_DATA_ROOT", DEFAULT_DATA_ROOT))
    return os.path.expanduser(os.path.expandvars(value))


def expand_cfg_paths(node):
    if isinstance(node, dict):
        return {key: expand_cfg_paths(value) for key, value in node.items()}
    if isinstance(node, list):
        return [expand_cfg_paths(value) for value in node]
    if isinstance(node, str):
        return expand_path(node)
    return node


def build_path_rewrites(cfg) -> Dict[str, str]:
    rewrites = dict(DEFAULT_PATH_REWRITES)
    user_rewrites = cfg.get("dataset", {}).get("path_rewrites", {})
    if isinstance(user_rewrites, Mapping):
        for src_prefix, dst_prefix in user_rewrites.items():
            rewrites[expand_path(str(src_prefix))] = expand_path(str(dst_prefix))
    return rewrites


def normalize_path(path: str, rewrites: Mapping[str, str]) -> str:
    if not isinstance(path, str):
        return path

    normalized = expand_path(path)
    for src_prefix, dst_prefix in rewrites.items():
        if normalized.startswith(src_prefix):
            return dst_prefix + normalized[len(src_prefix) :]
    return normalized
