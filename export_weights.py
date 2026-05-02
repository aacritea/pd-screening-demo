"""
export_weights.py
─────────────────────────────────────────────────────────────────────────────
Run this ONCE inside your PRIVATE training repo after training to export
the weights the demo needs. Never commit the output .pth / .pkl files.

IMPORTANT — You have no torch.save() call in your training script right now.
Your best model is only saved in memory as `best_model_state`.
Follow STEP 0 first, then run this script.

─────────────────────────────────────────────────────────────────────────────
STEP 0 — Add this to the END of your training script (after train_multimodal_model):

    torch.save(results['model'].state_dict(), 'best_model.pt')
    print("Model saved to best_model.pt")

─────────────────────────────────────────────────────────────────────────────
STEP 1 — Run this script:

    python /path/to/pd-screening-demo/export_weights.py \\
        --checkpoint ./best_model.pt \\
        --out /path/to/pd-screening-demo/models

    # If you're already inside pd-screening-demo:
    python export_weights.py --checkpoint ~/your-private-repo/best_model.pt --out ./models
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import joblib
import numpy as np


# ── Inline model definitions (copied from training so this script is self-contained)
# These MUST exactly match your training code.

class VoiceMLPEncoder(nn.Module):
    def __init__(self, input_dim=22, embedding_dim=128, hidden_dim=384, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim), nn.ReLU(),
        )
    def forward(self, x): return self.encoder(x)


class GaitCNNEncoder(nn.Module):
    def __init__(self, in_channels=19, embedding_dim=128, dropout=0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.MaxPool1d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, embedding_dim), nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.fc(self.global_pool(self.conv_layers(x)).squeeze(-1))


class AttentionFusion(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=64, modality_dropout=0.0):
        super().__init__()
        self.modality_dropout = modality_dropout
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, v, g):
        w = self.softmax(self.attention_net(torch.cat([v, g], dim=1)))
        return w[:, 0:1] * v + w[:, 1:2] * g, w


class MultimodalClassifier(nn.Module):
    def __init__(self, voice_encoder, gait_encoder,
                 fusion_hidden_dim=64, classifier_hidden_dim=64,
                 dropout=0.3, modality_dropout=0.0):
        super().__init__()
        self.voice_encoder = voice_encoder
        self.gait_encoder  = gait_encoder
        self.fusion        = AttentionFusion(128, fusion_hidden_dim, modality_dropout)
        self.classifier    = nn.Sequential(
            nn.Linear(128, classifier_hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )
    def forward(self, voice, gait, return_attention=False):
        fused, w = self.fusion(self.voice_encoder(voice), self.gait_encoder(gait))
        logits = self.classifier(fused)
        return (logits, w) if return_attention else logits


# ── Export ────────────────────────────────────────────────────────────────────

def export(checkpoint_path: str, out_dir: str, voice_scaler_path: str | None,
           gait_scaler_path: str | None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Reconstruct model with the same args used during training
    voice_enc = VoiceMLPEncoder(input_dim=22, embedding_dim=128, hidden_dim=384, dropout=0.3)
    gait_enc  = GaitCNNEncoder(in_channels=19, embedding_dim=128, dropout=0.3)
    model     = MultimodalClassifier(
        voice_encoder=voice_enc,
        gait_encoder=gait_enc,
        fusion_hidden_dim=64,
        classifier_hidden_dim=64,
        dropout=0.3,
        modality_dropout=0.2,
    )
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ State dict loaded successfully.")

    # Export each submodule separately (matches demo's model_loader.py)
    torch.save(model.voice_encoder.state_dict(), out / "voice_encoder.pth")
    torch.save(model.gait_encoder.state_dict(),  out / "gait_encoder.pth")
    torch.save(model.fusion.state_dict(),         out / "attention_module.pth")
    torch.save(model.classifier.state_dict(),    out / "classifier_head.pth")
    print("✓ Submodule weights saved.")

    # Scalers
    if voice_scaler_path:
        scaler = joblib.load(voice_scaler_path)
        joblib.dump(scaler, out / "scaler_voice.pkl")
        print("✓ Voice scaler saved.")
    else:
        print("⚠️  No --voice-scaler provided. Features won't be normalized in demo.")

    if gait_scaler_path:
        scaler = joblib.load(gait_scaler_path)
        joblib.dump(scaler, out / "scaler_gait.pkl")
        print("✓ Gait scaler saved.")
    else:
        print("⚠️  No --gait-scaler provided. Per-window normalization used in dataset.")

    print(f"\n✅ All files written to: {out}")
    print("   Upload to HF Spaces via the web UI (not git).")
    print("   Run: python export_samples.py --out ./samples  (next step)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True,
                   help="Path to best_model.pt saved from training")
    p.add_argument("--out",          required=True,
                   help="Output dir — use ./models if inside pd-screening-demo")
    p.add_argument("--voice-scaler", default=None,
                   help="Path to voice StandardScaler .pkl (optional)")
    p.add_argument("--gait-scaler",  default=None,
                   help="Path to gait StandardScaler .pkl (optional)")
    args = p.parse_args()
    export(args.checkpoint, args.out, args.voice_scaler, args.gait_scaler)