"""
Inference utilities — wraps model forward passes and returns
clean result dicts for the Streamlit UI to consume.
"""

import numpy as np
import torch
from models.architecture import PDMultimodalModel


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def run_voice_inference(
    model: PDMultimodalModel,
    voice_tensor: torch.Tensor,
) -> dict:
    """
    Returns:
        prob      : float — PD probability
        label     : str   — "Parkinson's Detected" / "Healthy"
        confidence: float — probability of the predicted class
        alpha_v   : float — attention weight for voice (always ~1.0 in voice-only)
        alpha_g   : float — attention weight for gait  (always ~0.0 in voice-only)
    """
    with torch.no_grad():
        prob, alpha_v, alpha_g = model(
            voice_features=voice_tensor, mode="voice"
        )
    prob_val = float(prob.mean())
    return _build_result(prob_val, float(alpha_v.mean()), float(alpha_g.mean()))


def run_gait_inference(
    model: PDMultimodalModel,
    gait_tensor: torch.Tensor,
) -> dict:
    with torch.no_grad():
        prob, alpha_v, alpha_g = model(
            gait_signal=gait_tensor, mode="gait"
        )
    prob_val = float(prob.mean())
    return _build_result(prob_val, float(alpha_v.mean()), float(alpha_g.mean()))


def run_multimodal_inference(
    model: PDMultimodalModel,
    voice_tensor: torch.Tensor,
    gait_tensor:  torch.Tensor,
    mode: str = "both",           # "both" | "voice" | "gait"
) -> dict:
    """
    mode controls modality dropout simulation:
        "both"  — normal multimodal inference
        "voice" — gait embedding zeroed (dropout simulation)
        "gait"  — voice embedding zeroed (dropout simulation)
    """
    # For multimodal: broadcast voice features across all gait windows
    N_windows = gait_tensor.shape[0]
    voice_broadcast = voice_tensor.expand(N_windows, -1)

    with torch.no_grad():
        prob, alpha_v, alpha_g = model(
            voice_features=voice_broadcast,
            gait_signal=gait_tensor,
            mode=mode,
        )

    prob_val  = float(prob.mean())
    av        = float(alpha_v.mean())
    ag        = float(alpha_g.mean())
    result    = _build_result(prob_val, av, ag)
    result["window_probs"]   = _to_numpy(prob)
    result["window_alpha_v"] = _to_numpy(alpha_v)
    result["window_alpha_g"] = _to_numpy(alpha_g)
    return result


def _build_result(prob: float, alpha_v: float, alpha_g: float) -> dict:
    label      = "Parkinson's Indicators Detected" if prob >= 0.5 else "No Indicators Detected"
    confidence = prob if prob >= 0.5 else 1 - prob
    return {
        "prob":       prob,
        "label":      label,
        "confidence": confidence,
        "alpha_v":    alpha_v,
        "alpha_g":    alpha_g,
        "is_pd":      prob >= 0.5,
    }


# ── Demo-mode mock inference (used when weights are absent) ───────────────────

def mock_inference(seed: int = 42) -> dict:
    """Returns plausible-looking placeholder results for demo mode."""
    rng  = np.random.default_rng(seed)
    prob = float(rng.uniform(0.55, 0.90))
    av   = float(rng.uniform(0.45, 0.65))
    ag   = 1.0 - av
    r    = _build_result(prob, av, ag)
    r["window_probs"]   = rng.uniform(0.5, 0.9, size=30).astype(np.float32)
    r["window_alpha_v"] = rng.uniform(0.4, 0.7, size=30).astype(np.float32)
    r["window_alpha_g"] = 1.0 - r["window_alpha_v"]
    return r
