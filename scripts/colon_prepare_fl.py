import os
import os
import sys
import json
import yaml
import random
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------
# 0) Load config
cfg_file = Path(__file__).resolve().parent.parent / "configs" / "colon" / "scripts.yaml"
cfg = yaml.safe_load(cfg_file.read_text())

RAW_ROOT = Path(os.path.expandvars(cfg.get("raw_data_folder", "colon"))).expanduser()
OUT_ROOT = Path(os.path.expandvars(cfg.get("target_folder", "~/colon/experiments_fed"))).expanduser()
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEED = int(cfg.get("seed", 42))
random.seed(SEED)
np.random.seed(SEED)

w = int(cfg["resize"]["width"])
h = int(cfg["resize"]["height"])
RESIZE = (w, h)

TEST_RATIO = float(cfg.get("test_ratio", 0.1))   # split ratio used for test set
TRAINABLE = set(cfg.get("trainable_domains", ["kvasir", "cvc_clinic"]))

# ---------------------------------------------
# 1) Dataset (domain) mapping -> each will be one client
colon_sets = {
    "kvasir": {
        "img": RAW_ROOT / "Kvasir-SEG" / "images",
        "msk": RAW_ROOT / "Kvasir-SEG" / "masks",
    },
    "cvc_clinic": {
        "img": RAW_ROOT / "CVC-ClinicDB" / "PNG" / "Original",
        "msk": RAW_ROOT / "CVC-ClinicDB" / "PNG" / "Ground Truth",
    },
    "cvc_colon": {
        "img": RAW_ROOT / "CVC-ColonDB" / "images",
        "msk": RAW_ROOT / "CVC-ColonDB" / "masks",
    },
    "etis": {
        "img": RAW_ROOT / "ETIS" / "images",
        "msk": RAW_ROOT / "ETIS" / "masks",
    },
}


# ---------------------------------------------
# 2) Helpers

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(dir_path: Path):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob(str(dir_path / f"*{ext}")))
    files = sorted(files)
    return [Path(f) for f in files]


def build_stem_map(files):
    """Map lowercase stem -> Path (keep last one if conflicts)."""
    mp = {}
    for p in files:
        mp[p.stem.lower()] = p
    return mp


def ensure_mask_binary(msk_img: Image.Image) -> Image.Image:
    """
    Convert mask to binary {0,255}:
    - Convert to 'L'
    - If near-white (>=245) -> 255 else 0
    """
    m = msk_img.convert("L")
    arr = np.array(m, dtype=np.uint8)
    uniq = np.unique(arr)
    if not np.array_equal(uniq, np.array([0, 255], dtype=np.uint8)):
        bin_arr = np.where(arr >= 245, 255, 0).astype(np.uint8)
    else:
        bin_arr = arr
    return Image.fromarray(bin_arr, mode="L")


def split_train_test(n, test_ratio=0.1, seed=42):
    """Case-index level split into train/test only."""
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    n_test = int(round(n * test_ratio))
    test_idxs = set(idxs[:n_test])
    train_idxs = set(idxs[n_test:])
    return train_idxs, test_idxs


def save_sample(img_p: Path, msk_p: Path, out_img_dir: Path, out_msk_dir: Path, resize_hw):
    """Load, resize, save PNG; return saved paths."""
    # image -> RGB, LANCZOS
    img = Image.open(img_p).convert("RGB")
    img = img.resize(resize_hw, Image.Resampling.LANCZOS)

    # mask -> binary L, NEAREST
    msk_raw = Image.open(msk_p)
    msk_bin = ensure_mask_binary(msk_raw)
    msk_bin = msk_bin.resize(resize_hw, Image.Resampling.NEAREST)

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    fname = img_p.stem + ".png"
    img_out = out_img_dir / fname
    msk_out = out_msk_dir / fname

    img.save(img_out)
    msk_bin.save(msk_out)

    return img_out, msk_out


def write_split_csv(rows, out_csv: Path):
    df = pd.DataFrame(rows, columns=["image_id", "image_path", "segmentation_mask_path"])
    out_csv.write_text(df.to_csv(index=False))


# ---------------------------------------------
# 3) Main: per domain -> per client
def main():
    domains = list(colon_sets.keys())

    for cid, dname in enumerate(domains, start=1):
        print(f"\n=== Processing domain '{dname}' as client_{cid} ===")
        dinfo = colon_sets[dname]
        img_dir, msk_dir = dinfo["img"], dinfo["msk"]

        assert img_dir.exists(), f"Image dir not found: {img_dir}"
        assert msk_dir.exists(), f"Mask dir not found:  {msk_dir}"

        # per-client output dirs
        client_root = OUT_ROOT / f"client_{cid}"
        data_img_dir = client_root / "data" / "img"
        data_msk_dir = client_root / "data" / "mask"
        client_root.mkdir(parents=True, exist_ok=True)

        # pair images and masks by stem
        img_files = list_images(img_dir)
        msk_files = list_images(msk_dir)
        msk_map = build_stem_map(msk_files)

        pairs = []
        missing = 0
        for ip in img_files:
            key = ip.stem.lower()
            mp = msk_map.get(key, None)
            if mp is None:
                missing += 1
                continue
            pairs.append((ip, mp))

        if missing > 0:
            print(f"[{dname}] Warning: {missing} images had no matching mask; skipped.")

        n = len(pairs)
        print(f"[{dname}] paired samples: {n}")

        tr, te = split_train_test(n, TEST_RATIO, seed=SEED)

        rows_train, rows_test = [], []

        # save samples
        for idx in tqdm(range(n), desc=f"Saving {dname}", unit="img"):
            img_p, msk_p = pairs[idx]
            img_out, msk_out = save_sample(img_p, msk_p, data_img_dir, data_msk_dir, RESIZE)
            row = [img_p.stem, str(img_out), str(msk_out)]
            if idx in tr:
                rows_train.append(row)
            else:
                rows_test.append(row)

        # write split csvs
        if rows_train:
            write_split_csv(rows_train, client_root / "train.csv")
        if rows_test:
            write_split_csv(rows_test, client_root / "test.csv")

        print(f"[{dname}] train/test = {len(rows_train)}/{len(rows_test)}")
    print("\nAll domains processed. Output root:", OUT_ROOT)

if __name__ == "__main__":
    main()
