"""
export_samples.py — run inside your private training repo.
───────────────────────────────────────────────────────────
Saves 2 pre-processed gait windows (one PD, one healthy)
from your held-out test set as .npy files.

These are from the PUBLIC PhysioNet dataset, so sharing them
as demo samples is fine.

Usage (from parkinsons_research directory):
    python3 /Users/aakritijain/Desktop/Projects/pd-screening-demo/export_samples.py \
        --gait-dir /Users/aakritijain/Desktop/parkinsons_research/gait-in-parkinsons-disease-1.0.0 \
        --out /Users/aakritijain/Desktop/Projects/pd-screening-demo/samples
"""

import argparse
import numpy as np
from pathlib import Path


WINDOW_SIZE = 256
STRIDE      = 128


def load_and_window(filepath: Path) -> np.ndarray:
    """
    Load a PhysioNet .txt file and return windowed array (N, 19, W).
    Keeps ALL 19 columns including time — matches PhysioNetGaitDataset
    in training which does np.loadtxt() without dropping any columns.
    """
    data = np.loadtxt(filepath).astype(np.float32)  # (T, 19) — all cols

    windows = []
    T = len(data)
    start = 0
    while start + WINDOW_SIZE <= T:
        w = data[start : start + WINDOW_SIZE]   # (256, 19)
        mean = w.mean(axis=0, keepdims=True)
        std  = w.std(axis=0, keepdims=True) + 1e-8
        w    = (w - mean) / std
        windows.append(w.T)                      # (19, 256)
        start += STRIDE

    return np.stack(windows, axis=0).astype(np.float32)  # (N, 19, 256)


def export(gait_dir: str, out_dir: str):
    out      = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(gait_dir)

    skip = {'demographics', 'format', 'SHA256SUMS'}
    pd_sample      = None
    healthy_sample = None

    txt_files = sorted(data_dir.glob('*.txt'))
    print(f"Scanning {len(txt_files)} files...")

    for filepath in txt_files:
        name = filepath.stem
        if name in skip:
            continue

        is_pd      = 'Pt' in name
        is_healthy = 'Co' in name

        if pd_sample is not None and healthy_sample is not None:
            break

        try:
            windows = load_and_window(filepath)
            if len(windows) < 5:
                continue   # skip very short recordings

            if is_pd and pd_sample is None:
                pd_sample = windows
                print(f"  ✓ PD sample: {name} — {windows.shape[0]} windows")

            if is_healthy and healthy_sample is None:
                healthy_sample = windows
                print(f"  ✓ Healthy sample: {name} — {windows.shape[0]} windows")

        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue

    if pd_sample is None or healthy_sample is None:
        print("❌ Could not find both sample types. Check your gait_dir path.")
        return

    np.save(out / "sample_pd_gait.npy",      pd_sample)
    np.save(out / "sample_healthy_gait.npy", healthy_sample)

    print(f"\n✅ Saved to {out}")
    print(f"   sample_pd_gait.npy      — shape {pd_sample.shape}")
    print(f"   sample_healthy_gait.npy — shape {healthy_sample.shape}")
    print("   Safe to commit — public PhysioNet data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gait-dir", required=True,
                        help="Path to PhysioNet gait dataset folder")
    parser.add_argument("--out", required=True,
                        help="Output dir — use ./samples if inside pd-screening-demo")
    args = parser.parse_args()
    export(args.gait_dir, args.out)