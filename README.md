# FedSemi

## News & Citation

Our paper **"FedSemiDG: Domain Generalized Federated Semi-Supervised Medical Image Segmentation"** has been accepted by **Medical Image Analysis**.

If you find this repository useful for your research, please consider citing our work:

```bibtex
@article{deng2025fedsemidg,
  title={Fedsemidg: Domain generalized federated semi-supervised medical image segmentation},
  author={Deng, Zhipeng and Xu, Zhe and Isshiki, Tsuyoshi and Zheng, Yefeng},
  journal={arXiv preprint arXiv:2501.07378},
  year={2025}
}
```

This repository provides a cleaned public training pipeline for four datasets:

- `cardiac`
- `spine`
- `colon`
- `bladder`

## Scope

- Supported training modes: `supervised`, `semi`
- Supported entrypoints: `local_train.py`, `fl_train.py`
- Training-time evaluation/test hooks are removed from the training loop
- `fundus` and `prostate` code paths are removed


## Data root

Use `FEDSEMI_DATA_ROOT` to point to your dataset root (default fallback is `/data/segmentation` in current scripts).

## Quick launch

See `train.sh` for minimal local and FL examples.
