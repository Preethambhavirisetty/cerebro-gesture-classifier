"""
Inference latency benchmark.

Measures per-sample ONNX inference latency and estimates whether the C++
implementation would meet the sub-15 ms budget on Cortex-M4.

Rules of thumb used for C++ estimate:
  • Each float MAC (multiply-accumulate) takes ~1 cycle on Cortex-M4 with FPU.
  • Cortex-M4 at 168 MHz → ~168 M ops/s.
  • MLP: (14×32) + (32×16) + (16×8) = 448 + 512 + 128 = 1088 MACs per inference.
  • Plus bias adds and activations ≈ 1500 ops total.
  • Estimated C++ time: 1500 / 168e6 ≈ 0.009 ms (well within 15 ms budget).

Usage:
  python -m validation.benchmark
  python -m validation.benchmark --n 10000 --batch 1
"""

import argparse
import os
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trainer.data_generator   import generate_dataset
from trainer.feature_engineer import extract_batch, K_NUM_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
ONNX_PATH  = os.path.join(MODELS_DIR, "gesture_net.onnx")

# Cortex-M4 @ 168 MHz with single-precision FPU
CORTEX_M4_MHTZ   = 168.0e6
MLP_MAC_COUNT     = (14 * 32) + (32 * 16) + (16 * 8)   # 1088
OVERHEAD_FACTOR   = 1.4   # bias adds, ReLU, softmax overhead
ESTIMATED_CPP_OPS = MLP_MAC_COUNT * OVERHEAD_FACTOR * 2  # each MAC = 2 ops
BUDGET_MS         = 15.0


def benchmark_onnx(
    session:    ort.InferenceSession,
    X:          np.ndarray,
    n_reps:     int = 2000,
    batch_size: int = 1,
) -> dict:
    latencies_ms = []

    # Warmup
    for _ in range(50):
        session.run(["logits"], {"features": X[:batch_size]})

    for i in range(0, min(n_reps * batch_size, len(X)), batch_size):
        xb = X[i : i + batch_size]
        if len(xb) == 0:
            break
        t0 = time.perf_counter()
        session.run(["logits"], {"features": xb})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0 / len(xb))  # per-sample

    arr = np.array(latencies_ms)
    return {
        "n":       len(arr),
        "mean_ms": arr.mean(),
        "p50_ms":  np.percentile(arr, 50),
        "p95_ms":  np.percentile(arr, 95),
        "p99_ms":  np.percentile(arr, 99),
        "min_ms":  arr.min(),
        "max_ms":  arr.max(),
    }


def estimate_cpp_latency_ms() -> float:
    return (ESTIMATED_CPP_OPS / CORTEX_M4_MHTZ) * 1000.0


def run_benchmark(
    onnx_path:  str = ONNX_PATH,
    n_reps:     int = 2000,
    batch_size: int = 1,
    seed:       int = 7,
) -> None:
    if not os.path.exists(onnx_path):
        sys.exit(
            f"ONNX model not found: {onnx_path}\n"
            "Run: python -m trainer.train && python -m trainer.export_onnx"
        )

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    print(f"Generating benchmark inputs (seed={seed})…")
    events, labels = generate_dataset(n_per_class=max(n_reps // 8, 50), seed=seed)
    X = extract_batch(events).astype(np.float32)

    print(f"Running ONNX benchmark  n={n_reps}  batch={batch_size}…\n")
    stats = benchmark_onnx(session, X, n_reps=n_reps, batch_size=batch_size)

    print("── ONNX Inference Latency (per sample, host CPU) ──────────────────")
    print(f"  Samples measured : {stats['n']}")
    print(f"  Mean             : {stats['mean_ms']:.4f} ms")
    print(f"  p50              : {stats['p50_ms']:.4f} ms")
    print(f"  p95              : {stats['p95_ms']:.4f} ms")
    print(f"  p99              : {stats['p99_ms']:.4f} ms")
    print(f"  Min / Max        : {stats['min_ms']:.4f} / {stats['max_ms']:.4f} ms")

    cpp_est = estimate_cpp_latency_ms()
    budget_ok = cpp_est < BUDGET_MS

    print("\n── C++ Cortex-M4 Estimate (168 MHz with FPU) ──────────────────────")
    print(f"  MLP MACs         : {MLP_MAC_COUNT}")
    print(f"  Estimated ops    : {ESTIMATED_CPP_OPS:.0f}")
    print(f"  Estimated time   : {cpp_est:.4f} ms")
    print(f"  Budget           : {BUDGET_MS:.1f} ms")
    print(f"  Budget status    : {'PASS ✓' if budget_ok else 'FAIL ✗'}")

    if not budget_ok:
        print(f"\nWARNING: estimated C++ latency {cpp_est:.2f} ms exceeds {BUDGET_MS:.1f} ms budget.")
        print("Consider reducing MLP depth or quantizing to fixed-point.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx",  default=ONNX_PATH)
    parser.add_argument("--n",     type=int, default=2000, dest="n_reps",
                        help="Number of inference repetitions")
    parser.add_argument("--batch", type=int, default=1,    dest="batch_size")
    parser.add_argument("--seed",  type=int, default=7)
    args = parser.parse_args()

    run_benchmark(
        onnx_path  = args.onnx,
        n_reps     = args.n_reps,
        batch_size = args.batch_size,
        seed       = args.seed,
    )


if __name__ == "__main__":
    main()
