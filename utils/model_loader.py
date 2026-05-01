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
import torch
import joblib
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
    Returns (voice_scaler, gait_scaler).
    Returns (None, None) if scaler files are not present.
    """
    v_path = MODELS_DIR / "scaler_voice.pkl"
    g_path = MODELS_DIR / "scaler_gait.pkl"
    voice_scaler = joblib.load(v_path) if v_path.exists() else None
    gait_scaler  = joblib.load(g_path) if g_path.exists() else None
    return voice_scaler, gait_scaler


def is_demo_mode() -> bool:
    """True when weight files are absent — shows placeholder outputs."""
    return not _weights_exist()
