"""
Preprocessing utilities — matches paper's pipeline exactly.

Voice: Z-score normalization (µ, σ from training split)
Gait:  Fixed-length windowing (size=256, 50% overlap) + Z-score normalization
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ── UCI Parkinson's Voice Feature Names (in dataset column order) ──────────────
VOICE_FEATURE_NAMES = [
    "MDVP:Fo(Hz)",    "MDVP:Fhi(Hz)",   "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)","MDVP:RAP",
    "MDVP:PPQ",       "Jitter:DDP",      "MDVP:Shimmer",
    "MDVP:Shimmer(dB)","Shimmer:APQ3",   "Shimmer:APQ5",
    "MDVP:APQ",       "Shimmer:DDA",     "NHR",
    "HNR",            "RPDE",            "DFA",
    "spread1",        "spread2",         "D2",
    "PPE",
]

# Intuitive display names for sliders (same order)
VOICE_FEATURE_DISPLAY = [
    "Avg. Vocal Freq (Hz)",        "Max Vocal Freq (Hz)",
    "Min Vocal Freq (Hz)",         "Jitter % (freq variation)",
    "Jitter Abs (freq variation)", "RAP (pitch period variation)",
    "PPQ (pitch period variation)","Jitter DDP",
    "Shimmer (amp variation)",     "Shimmer dB",
    "Shimmer APQ3",                "Shimmer APQ5",
    "Shimmer APQ",                 "Shimmer DDA",
    "Noise-to-Harmonics Ratio",    "Harmonics-to-Noise Ratio",
    "RPDE (complexity)",           "DFA (signal fractal scaling)",
    "Spread 1",                    "Spread 2",
    "D2 (nonlinear dynamics)",     "PPE (pitch entropy)",
]

# Sensible slider ranges derived from UCI dataset statistics
VOICE_FEATURE_RANGES = {
    "MDVP:Fo(Hz)":        (80.0,  270.0, 154.2),
    "MDVP:Fhi(Hz)":       (100.0, 600.0, 197.1),
    "MDVP:Flo(Hz)":       (60.0,  240.0, 116.3),
    "MDVP:Jitter(%)":     (0.001, 0.033, 0.006),
    "MDVP:Jitter(Abs)":   (0.000007, 0.00026, 0.000044),
    "MDVP:RAP":           (0.0003, 0.021, 0.003),
    "MDVP:PPQ":           (0.0005, 0.019, 0.003),
    "Jitter:DDP":         (0.001, 0.063, 0.010),
    "MDVP:Shimmer":       (0.009, 0.119, 0.029),
    "MDVP:Shimmer(dB)":   (0.085, 1.302, 0.282),
    "Shimmer:APQ3":       (0.004, 0.056, 0.015),
    "Shimmer:APQ5":       (0.005, 0.079, 0.018),
    "MDVP:APQ":           (0.007, 0.137, 0.024),
    "Shimmer:DDA":        (0.013, 0.169, 0.044),
    "NHR":                (0.0, 0.315, 0.025),
    "HNR":                (8.0, 33.0, 21.9),
    "RPDE":               (0.25, 0.69, 0.50),
    "DFA":                (0.57, 0.83, 0.72),
    "spread1":            (-7.96, -2.43, -5.68),
    "spread2":            (0.006, 0.451, 0.227),
    "D2":                 (1.42, 3.67, 2.38),
    "PPE":                (0.04, 0.53, 0.21),
}


# ── Voice Preprocessing ────────────────────────────────────────────────────────

def preprocess_voice_csv(df: pd.DataFrame, scaler=None) -> torch.Tensor:
    """
    Accepts a DataFrame with UCI column names (with or without 'name'/'status').
    Returns a (N, 22) float32 tensor, normalized if scaler is provided.
    """
    # Drop non-feature columns if present
    drop_cols = [c for c in ["name", "status"] if c in df.columns]
    features = df.drop(columns=drop_cols)[VOICE_FEATURE_NAMES].values.astype(np.float32)

    if scaler is not None:
        features = scaler.transform(features).astype(np.float32)

    return torch.tensor(features)


def dict_to_voice_tensor(feature_dict: dict, scaler=None) -> torch.Tensor:
    """
    Converts a {feature_name: value} dict (from sliders) to a (1, 22) tensor.
    """
    arr = np.array(
        [feature_dict[f] for f in VOICE_FEATURE_NAMES], dtype=np.float32
    ).reshape(1, -1)

    if scaler is not None:
        arr = scaler.transform(arr).astype(np.float32)

    return torch.tensor(arr)


# ── Gait Preprocessing ─────────────────────────────────────────────────────────

WINDOW_SIZE = 256
OVERLAP     = 0.5

def load_physionet_gait(filepath: str | Path) -> np.ndarray:
    """
    Loads a PhysioNet VGRF .txt file (tab-separated, 19 columns:
    time + 8 left sensors + 8 right sensors + total).
    Returns raw signal array of shape (T, 16), keeping sensors 1-16.
    """
    data = np.loadtxt(filepath)
    # Column 0 = time; columns 1-8 = left foot; 9-16 = right foot; 17 = total
    return data[:, 1:17].astype(np.float32)


def window_gait_signal(
    signal: np.ndarray,
    window_size: int = WINDOW_SIZE,
    overlap: float = OVERLAP,
) -> np.ndarray:
    """
    Segments a (T, C) VGRF signal into overlapping windows.
    Returns (N_windows, C, window_size) — ready for 1D-CNN input.
    """
    step = int(window_size * (1 - overlap))
    T, C = signal.shape
    windows = []
    start = 0
    while start + window_size <= T:
        w = signal[start : start + window_size]   # (window_size, C)
        windows.append(w.T)                        # (C, window_size)
        start += step
    return np.stack(windows, axis=0)              # (N, C, window_size)


def preprocess_gait_file(filepath: str | Path, scaler=None) -> torch.Tensor:
    """
    Full pipeline: load → window → normalize → tensor.
    Returns (N_windows, 16, 256) float32 tensor.
    """
    raw    = load_physionet_gait(filepath)
    windows = window_gait_signal(raw)

    if scaler is not None:
        # scaler fitted on (samples, C*window_size) — reshape, transform, reshape back
        N, C, W = windows.shape
        flat    = windows.reshape(N, -1)
        flat    = scaler.transform(flat).astype(np.float32)
        windows = flat.reshape(N, C, W)

    return torch.tensor(windows)


def preprocess_gait_npy(arr: np.ndarray, scaler=None) -> torch.Tensor:
    """
    For pre-saved .npy sample files (shape: N, 16, 256 or 16, 256).
    """
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]   # add batch dim

    arr = arr.astype(np.float32)

    if scaler is not None:
        N, C, W = arr.shape
        flat = arr.reshape(N, -1)
        flat = scaler.transform(flat).astype(np.float32)
        arr  = flat.reshape(N, C, W)

    return torch.tensor(arr)


# ── Aggregate window-level predictions to subject-level ───────────────────────

def aggregate_window_predictions(
    probs: np.ndarray,
    alpha_v: np.ndarray,
    alpha_g: np.ndarray,
    strategy: str = "mean",
) -> tuple[float, float, float]:
    """
    Aggregates per-window predictions to a single subject-level result.
    Returns (mean_prob, mean_alpha_v, mean_alpha_g).
    """
    return float(probs.mean()), float(alpha_v.mean()), float(alpha_g.mean())
