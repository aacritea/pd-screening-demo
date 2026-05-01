"""
Parkinson's Disease Multimodal Detection — Model Architecture
Matches exactly the architecture described in:
  "Attention-Based Multimodal Fusion of Voice and Gait for Parkinson's Disease Detection"
  Aakriti Jain, Ujjawal Gaur, Pragya Singh — ICESAIA 2026 (Under Review)

NOTE: This file contains only architecture definitions.
      Trained weights are loaded separately and are NOT part of this repo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Voice Encoder ──────────────────────────────────────────────────────────────
class VoiceEncoder(nn.Module):
    """
    3-layer MLP for tabular UCI Parkinson's acoustic features.
    Architecture: [22 → 384 → 128 → 128] with ReLU, BatchNorm, Dropout(0.3)
    Input: 22 acoustic features (jitter, shimmer, HNR, NHR, nonlinear dynamics)
    Output: 128-dim embedding
    """
    def __init__(self, input_dim: int = 22, embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Gait Encoder ───────────────────────────────────────────────────────────────
class GaitEncoder(nn.Module):
    """
    4-block 1D-CNN for PhysioNet VGRF temporal signals.
    Kernel sizes: 7→5→3→3 | Filters: 64→128→256→256
    Followed by Global Average Pooling → FC projection to embedding_dim.
    Input: (batch, channels, time_steps) — window_size=256, 50% overlap
    Output: 128-dim embedding
    """
    def __init__(self, in_channels: int = 16, embedding_dim: int = 128):
        super().__init__()

        def conv_block(in_ch, out_ch, kernel):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

        self.conv_blocks = nn.Sequential(
            conv_block(in_channels, 64, 7),
            conv_block(64, 128, 5),
            conv_block(128, 256, 3),
            conv_block(256, 256, 3),
        )
        self.projection = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)          # (B, 256, T')
        x = x.mean(dim=-1)               # Global Average Pooling → (B, 256)
        return self.projection(x)        # (B, embedding_dim)


# ── Attention-Based Fusion ─────────────────────────────────────────────────────
class AttentionFusion(nn.Module):
    """
    Learns per-sample attention weights α_v and α_g over the two modalities.
    Hidden attention dim d_a = 16.
    f = α_v * e_v + α_g * e_g
    """
    def __init__(self, embedding_dim: int = 128, attention_dim: int = 16):
        super().__init__()
        self.W_a  = nn.Linear(embedding_dim, attention_dim, bias=True)
        self.w_a  = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, e_v: torch.Tensor, e_g: torch.Tensor):
        """
        Returns:
            fused  : (B, embedding_dim)
            alpha_v: (B,) — attention weight for voice
            alpha_g: (B,) — attention weight for gait
        """
        s_v = self.w_a(torch.tanh(self.W_a(e_v))).squeeze(-1)   # (B,)
        s_g = self.w_a(torch.tanh(self.W_a(e_g))).squeeze(-1)   # (B,)
        scores  = torch.stack([s_v, s_g], dim=1)                 # (B, 2)
        weights = F.softmax(scores, dim=1)                       # (B, 2)
        alpha_v, alpha_g = weights[:, 0], weights[:, 1]
        fused = alpha_v.unsqueeze(1) * e_v + alpha_g.unsqueeze(1) * e_g
        return fused, alpha_v, alpha_g


# ── Classification Head ────────────────────────────────────────────────────────
class ClassificationHead(nn.Module):
    """
    2-layer head: 128 → 64 → 1 with ReLU + Sigmoid.
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


# ── Full Model ─────────────────────────────────────────────────────────────────
class PDMultimodalModel(nn.Module):
    """
    Full attention-based multimodal fusion model.
    Supports three inference modes:
        mode='both'  — voice + gait (default)
        mode='voice' — voice only (gait embedding zeroed)
        mode='gait'  — gait only  (voice embedding zeroed)
    """
    def __init__(
        self,
        voice_input_dim: int = 22,
        gait_in_channels: int = 16,
        embedding_dim: int = 128,
        attention_dim: int = 16,
    ):
        super().__init__()
        self.voice_encoder   = VoiceEncoder(voice_input_dim, embedding_dim)
        self.gait_encoder    = GaitEncoder(gait_in_channels, embedding_dim)
        self.attention_fusion = AttentionFusion(embedding_dim, attention_dim)
        self.classifier      = ClassificationHead(embedding_dim)

    def forward(
        self,
        voice_features: torch.Tensor | None = None,
        gait_signal:    torch.Tensor | None = None,
        mode: str = "both",
    ):
        """
        Returns:
            prob    : (B,) PD probability
            alpha_v : (B,) voice attention weight
            alpha_g : (B,) gait attention weight
        """
        device = next(self.parameters()).device

        if mode in ("both", "voice") and voice_features is not None:
            e_v = self.voice_encoder(voice_features.to(device))
        else:
            # Zero embedding when modality is dropped
            b = gait_signal.shape[0] if gait_signal is not None else 1
            e_v = torch.zeros(b, 128, device=device)

        if mode in ("both", "gait") and gait_signal is not None:
            e_g = self.gait_encoder(gait_signal.to(device))
        else:
            b = voice_features.shape[0] if voice_features is not None else 1
            e_g = torch.zeros(b, 128, device=device)

        # Broadcast voice embedding across gait windows if batch sizes differ
        if e_v.shape[0] != e_g.shape[0]:
            if e_v.shape[0] == 1:
                e_v = e_v.expand(e_g.shape[0], -1)
            elif e_g.shape[0] == 1:
                e_g = e_g.expand(e_v.shape[0], -1)

        fused, alpha_v, alpha_g = self.attention_fusion(e_v, e_g)
        prob = self.classifier(fused)
        return prob, alpha_v, alpha_g
