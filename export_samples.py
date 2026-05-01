"""
export_samples.py — run inside your private training repo.
───────────────────────────────────────────────────────────
Saves 2 pre-processed gait windows (one PD, one healthy)
from your held-out test set as .npy files.

These are from the PUBLIC PhysioNet dataset, so sharing them
as demo samples is fine — but double-check your IRB/license.

Usage:
    python export_samples.py --out /path/to/pd-screening-demo/samples
"""

import argparse
import numpy as np
from pathlib import Path


def export(out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Replace the lines below with your actual test-set tensors ────────────
    # Each should be a numpy array of shape (N_windows, 16, 256)
    # Pick one clearly-PD and one clearly-healthy sample from your test split.

    # Example (replace with real data):
    # from your_training_module import test_loader
    # for gait, label, subject_id in test_loader:
    #     if label == 1 and pd_sample is None:
    #         pd_sample = gait.numpy()      # shape (N_windows, 16, 256)
    #     if label == 0 and hc_sample is None:
    #         hc_sample = gait.numpy()

    # np.save(out / "sample_pd_gait.npy",      pd_sample)
    # np.save(out / "sample_healthy_gait.npy", hc_sample)

    print(f"✅ Sample files saved to {out}")
    print("   These are safe to commit IF they are from the public PhysioNet dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    export(args.out)
