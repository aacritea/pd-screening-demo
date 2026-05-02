"""
Model loading utilities for the PD Screening Demo.

SETUP INSTRUCTIONS (one-time, after training):
-----------------------------------------------
1. Copy your four .pth weight files into the models/ directory:
       voice_encoder.pth
       gait_encoder.pth
       attention_module.pth
       classifier_head.pth

2. Copy your fitted StandardScaler objects:
       scaler_voice.pkl   — fitted on voice training split
       scaler_gait.pkl    — fitted on gait training split (per-channel or global)

These files are in .gitignore and must never be committed.
"""

from pathlib import Path
import numpy as np
import torch
import streamlit as st

from models.architecture import (
    VoiceEncoder,
    GaitEncoder,
    AttentionFusion,
    ClassificationHead,
    PDMultimodalModel,
)

MODELS_DIR = Path(__file__).parent.parent / "models"
DEVICE = torch.device("cpu")   # HF Spaces free tier — CPU only


def _weights_exist() -> bool:
    required = [
        "voice_encoder.pth",
        "gait_encoder.pth",
        "attention_module.pth",
        "classifier_head.pth",
    ]
    return all((MODELS_DIR / f).exists() for f in required)


@st.cache_resource(show_spinner="Loading model weights…")
def load_full_model() -> PDMultimodalModel | None:
    """
    Loads the full PDMultimodalModel with trained weights.
    Returns None if weight files are not present (demo mode).
    """
    if not _weights_exist():
        return None

    model = PDMultimodalModel()
    model.voice_encoder.load_state_dict(
        torch.load(MODELS_DIR / "voice_encoder.pth", map_location=DEVICE)
    )
    model.gait_encoder.load_state_dict(
        torch.load(MODELS_DIR / "gait_encoder.pth", map_location=DEVICE)
    )
    model.attention_fusion.load_state_dict(
        torch.load(MODELS_DIR / "attention_module.pth", map_location=DEVICE)
    )
    model.classifier.load_state_dict(
        torch.load(MODELS_DIR / "classifier_head.pth", map_location=DEVICE)
    )
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading scalers…")
def load_scalers():
    """
    Returns (voice_scaler, None).
    Voice scaler stored as two plain .npy files: scaler_mean.npy + scaler_scale.npy
    Gait uses per-window normalization — no scaler needed.
    """
    mean_path  = MODELS_DIR / "scaler_mean.npy"
    scale_path = MODELS_DIR / "scaler_scale.npy"
    if not mean_path.exists() or not scale_path.exists():
        return None, None
    scaler = {
        'mean':  np.load(mean_path),
        'scale': np.load(scale_path),
    }
    return scaler, None


def is_demo_mode() -> bool:
    """True when weight files are absent — shows placeholder outputs."""
    return not _weights_exist()