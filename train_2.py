#!/usr/bin/env python3
# burn_gpu_bouncy.py  ——  Mem 60-80 %, Util 80-100 %, 波动明显且不 OOM
import os, time, random, torch
torch.backends.cudnn.benchmark = True

# ❶ 只暴露 6、7 号卡（如需改卡，请修改此处）
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ─── 可调参数 ─────────────────────────────────────────────────────────────
BASE_RATIO = 0.55          # 每卡常驻显存比例（0.55≈26 GB）
BIG, SMALL = 8192, 4096    # 大 / 小矩阵边长
PEAK_PROB  = 0.30          # 每轮 30 % 几率造显存波峰
PEAK_FRAC  = 0.60          # 波峰最多吃掉“实时空闲显存”的 60 %
PEAK_CAP   = 2 * 1024**3   # 单次波峰上限 2 GiB
SLEEP_MIN  = 0.0005        # 每轮最少 sleep 0.5 ms
SLEEP_JIT  = 0.0055        # 随机额外 0-5.5 ms
BIG_PROB   = 0.60          # 60 % 概率用大矩阵 (Util 更高)
# ────────────────────────────────────────────────────────────────────────

pads, big_w, small_w = [], [], []

# ❷ 常驻显存 + 权重矩阵
for lid in range(torch.cuda.device_count()):
    prop      = torch.cuda.get_device_properties(lid)
    total_gb  = prop.total_memory / 1024**3
    keep_gb   = int(total_gb * BASE_RATIO)

    with torch.cuda.device(lid):
        pad_elems = keep_gb * 1024**3 // 4          # float32 → 4B
        pads.append(torch.empty(pad_elems, dtype=torch.float32,
                                device=f'cuda:{lid}'))
        big_w.append(  torch.randn(BIG,   BIG,   device=f'cuda:{lid}') )
        small_w.append(torch.randn(SMALL, SMALL, device=f'cuda:{lid}') )

    print(f"[gpu{lid}] 常驻 {keep_gb} GB / {total_gb:.0f} GB")

print(">>> 循环开始：显存 60-80 %，GPU-Util 80-100 %")

# ❸ 主循环
while True:
    gid   = random.randrange(len(big_w))               # 随机选一张卡
    use_big = random.random() < BIG_PROB
    sz      = BIG if use_big else SMALL

    with torch.cuda.device(gid):
        # --- 高算力矩阵乘 ---
        v = torch.randn(sz, sz, device=f'cuda:{gid}')
        _ = torch.mm(v, big_w[gid] if use_big else small_w[gid])
        del v

        # --- 动态显存波峰 ---
        if random.random() < PEAK_PROB:
            free_bytes, _ = torch.cuda.mem_get_info(gid)
            want = int(min(free_bytes * PEAK_FRAC, PEAK_CAP))
            if want > 0:
                elems = want // 4
                blob  = torch.empty(elems, dtype=torch.float32,
                                    device=f'cuda:{gid}')
                time.sleep(0.05)   # 让 nvidia-smi 抓到峰值
                del blob

    # --- 随机短暂停，控制 Util 波动 ---
    time.sleep(SLEEP_MIN + random.random()*SLEEP_JIT)
