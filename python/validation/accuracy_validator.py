"""
Automated accuracy validation suite.

Loads the exported ONNX model (models/gesture_net.onnx) and evaluates it
against a held-out test set generated with a different seed than training.

Outputs:
  • Per-class precision, recall, F1
  • Confusion matrix
  • Overall accuracy (must be ≥ TARGET_ACCURACY to pass)
  • Exit code 0 = pass, 1 = fail

Usage:
  python -m validation.accuracy_validator
  python -m validation.accuracy_validator --target 0.90 --n-per-class 500
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trainer.data_generator   import generate_dataset, GESTURE_NAMES
from trainer.feature_engineer import extract_batch

MODELS_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "models")
ONNX_PATH      = os.path.join(MODELS_DIR, "gesture_net.onnx")
TARGET_ACCURACY = 0.88   # minimum acceptable overall accuracy


def run_onnx(session: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    logits = session.run(["logits"], {"features": X.astype(np.float32)})[0]
    return logits.argmax(axis=1)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> dict:
    n = cm.shape[0]
    metrics = {}
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        metrics[GESTURE_NAMES[i]] = {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "support":   int(cm[i, :].sum()),
        }
    return metrics


def validate(
    onnx_path:   str   = ONNX_PATH,
    target_acc:  float = TARGET_ACCURACY,
    n_per_class: int   = 400,
    seed:        int   = 99,     # different from training seed (42)
) -> bool:
    if not os.path.exists(onnx_path):
        sys.exit(
            f"ONNX model not found: {onnx_path}\n"
            "Run: python -m trainer.train && python -m trainer.export_onnx"
        )

    print(f"Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    print(f"Generating test set (seed={seed}, {n_per_class}×8={n_per_class*8} samples)…")
    events, labels = generate_dataset(n_per_class=n_per_class, seed=seed)
    X = extract_batch(events)
    y_true = np.array(labels, dtype=np.int64)

    print("Running inference…")
    y_pred = run_onnx(session, X)

    overall_acc = (y_true == y_pred).mean()
    cm          = confusion_matrix(y_true, y_pred, n=8)
    class_m     = per_class_metrics(cm)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\nOverall accuracy: {overall_acc:.1%}  (target ≥ {target_acc:.0%})")
    print()
    print(f"{'Gesture':<20}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'Support':>8}")
    print("-" * 55)
    for name, m in class_m.items():
        print(f"{name:<20}  {m['precision']:>5.1%}  {m['recall']:>6.1%}  "
              f"{m['f1']:>5.1%}  {m['support']:>8d}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = " " * 22 + "  ".join(f"{n[:4]:>4}" for n in GESTURE_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>4d}" for v in row)
        print(f"  {GESTURE_NAMES[i]:<20}  {row_str}")

    passed = overall_acc >= target_acc
    status = "PASS" if passed else "FAIL"
    print(f"\n{status}: overall accuracy {overall_acc:.1%} "
          f"{'≥' if passed else '<'} target {target_acc:.0%}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx",        default=ONNX_PATH)
    parser.add_argument("--target",      type=float, default=TARGET_ACCURACY)
    parser.add_argument("--n-per-class", type=int,   default=400)
    parser.add_argument("--seed",        type=int,   default=99)
    args = parser.parse_args()

    passed = validate(
        onnx_path   = args.onnx,
        target_acc  = args.target,
        n_per_class = args.n_per_class,
        seed        = args.seed,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
