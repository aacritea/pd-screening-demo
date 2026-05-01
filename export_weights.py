"""
export_weights.py — run this ONCE inside your private training repo.
─────────────────────────────────────────────────────────────────────
It exports everything the demo needs into a folder you then copy
into pd-screening-demo/models/ manually (never via git).

Usage (from your private training repo root):
    python export_weights.py --checkpoint path/to/best_checkpoint.pt \
                             --out /path/to/pd-screening-demo/models

Adjust variable names to match your training code.
"""

import argparse
from pathlib import Path
import torch
import joblib


def export(checkpoint_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load your full checkpoint ─────────────────────────────────────────────
    # Adjust this to match how you saved your checkpoint.
    # Common patterns:
    #   ckpt = torch.load(checkpoint_path)
    #   model.load_state_dict(ckpt['model_state_dict'])
    # or if you saved the full model:
    #   model = torch.load(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # ── Option A: checkpoint saved as state_dict of PDMultimodalModel ─────────
    # Import your training model class here (adjust path as needed):
    # from your_training_module import PDMultimodalModel
    # model = PDMultimodalModel()
    # model.load_state_dict(ckpt['model_state_dict'])

    # ── Option B: checkpoint is the full model object ──────────────────────────
    # model = ckpt  # if you used torch.save(model, ...)

    # After loading, extract each submodule:
    # torch.save(model.voice_encoder.state_dict(),    out / "voice_encoder.pth")
    # torch.save(model.gait_encoder.state_dict(),     out / "gait_encoder.pth")
    # torch.save(model.attention_fusion.state_dict(), out / "attention_module.pth")
    # torch.save(model.classifier.state_dict(),       out / "classifier_head.pth")

    # ── Export scalers ────────────────────────────────────────────────────────
    # Replace with how you stored your scalers in training:
    # joblib.dump(voice_scaler, out / "scaler_voice.pkl")
    # joblib.dump(gait_scaler,  out / "scaler_gait.pkl")

    print(f"✅ Weights exported to {out}")
    print("   Copy this folder to pd-screening-demo/models/")
    print("   Do NOT commit .pth or .pkl files to the demo repo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out",        required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.out)
