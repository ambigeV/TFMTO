"""
benchmark_chunk.py
------------------
Measures TabPFN predict throughput at different chunk sizes to find the
optimal _CHUNK value for this machine.

Simulates the actual workload: n_train = 60 (20 init + some BO iters),
feature_dim = 11 (10D + scalar task_id), n_estimators = 8.

Run with:
    python benchmark_chunk.py
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ddmtolab.Methods.Algo_Methods.tfm_utils import _build_model

# --- Workload parameters matching the real BO loop ---
N_TRAIN      = 60       # realistic mid-run training set size
FEATURE_DIM  = 11       # 10D + 1 scalar task_id  (worst case: 10 + 2 one-hot = 12)
N_ESTIMATORS = 8
N_TOTAL_TEST = 1000     # total candidates per iteration
REPEATS      = 3        # average over this many trials per chunk size

CHUNK_SIZES = [64, 128, 256, 500, 512, 750, 1000]

# --- Fixed training data ---
rng = np.random.default_rng(42)
X_train = rng.random((N_TRAIN, FEATURE_DIM)).astype(np.float32)
y_train = rng.random(N_TRAIN).astype(np.float32)
X_test_full = rng.random((N_TOTAL_TEST, FEATURE_DIM)).astype(np.float32)

print(f"TabPFN chunk-size benchmark")
print(f"  n_train={N_TRAIN}, feature_dim={FEATURE_DIM}, "
      f"n_estimators={N_ESTIMATORS}, n_total_test={N_TOTAL_TEST}")
print(f"  Each trial: fit once, predict {N_TOTAL_TEST} points in chunks\n")
print(f"{'Chunk':>8}  {'Avg time (s)':>14}  {'Throughput (pts/s)':>20}  {'Notes'}")
print("-" * 65)

results = {}
for chunk in CHUNK_SIZES:
    times = []
    for _ in range(REPEATS):
        model = _build_model(N_ESTIMATORS, random_state=42)
        model.fit(X_train, y_train)

        t0 = time.perf_counter()
        parts = []
        for i in range(0, N_TOTAL_TEST, chunk):
            out = model.predict(X_test_full[i:i + chunk], output_type="main")
            parts.append(out["mean"])
        _ = np.concatenate(parts)
        times.append(time.perf_counter() - t0)

    avg = np.mean(times)
    throughput = N_TOTAL_TEST / avg
    n_chunks = -(-N_TOTAL_TEST // chunk)   # ceiling division
    results[chunk] = avg
    print(f"{chunk:>8}  {avg:>14.4f}  {throughput:>20.1f}  ({n_chunks} chunks)")

best_chunk = min(results, key=results.get)
print(f"\n  Optimal chunk size: {best_chunk}  "
      f"(fastest at {results[best_chunk]:.4f}s avg)")
print(f"\n  To apply, set in tfm_utils.py:  _CHUNK = {best_chunk}")
