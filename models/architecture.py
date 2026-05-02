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
    VoiceMLPEncoder — matches training code exactly.
    Architecture: [22 → 384 → ReLU → Drop → 384 → ReLU → Drop → 128 → ReLU]
    No BatchNorm. Submodule name in MultimodalClassifier: model.voice_encoder
    Stored under key: model.voice_encoder.encoder.*
    """
    def __init__(self, input_dim: int = 22, embedding_dim: int = 128,
                 hidden_dim: int = 384, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ── Gait Encoder ───────────────────────────────────────────────────────────────
class GaitEncoder(nn.Module):
    """
    GaitCNNEncoder — matches training code exactly.
    in_channels=19 (PhysioNet has 19 columns after time col).
    Conv blocks: 19→64→128→256→256, kernel 7→5→3→3 + MaxPool(2).
    Global AdaptiveAvgPool → FC [256 → ReLU → Dropout → 128].
    Submodule name: model.gait_encoder
    Stored under keys: model.gait_encoder.conv_layers.* and model.gait_encoder.fc.*
    """
    def __init__(self, in_channels: int = 19, embedding_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)       # (B, 256, T')
        x = self.global_pool(x)       # (B, 256, 1)
        x = x.squeeze(-1)             # (B, 256)
        return self.fc(x)             # (B, 128)


# ── Attention-Based Fusion (matches ImprovedAttentionFusion from training) ─────
class AttentionFusion(nn.Module):
    """
    Joint attention fusion — matches ImprovedAttentionFusion exactly.
    Concatenates both embeddings, passes through attention_net, adds
    learnable modality_bias, applies temperature scaling, then softmax.

    Submodule name in ImprovedMultimodalClassifier: model.fusion
    """
    def __init__(
        self,
        embedding_dim:    int   = 128,
        hidden_dim:       int   = 64,
        modality_dropout: float = 0.0,
        init_temperature: float = 1.0,
        init_bias: list | None  = None,
    ):
        super().__init__()
        self.modality_dropout = modality_dropout

        self.temperature   = nn.Parameter(torch.tensor(init_temperature))
        init_bias          = init_bias or [0.0, 0.0]
        self.modality_bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))

        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, e_v: torch.Tensor, e_g: torch.Tensor):
        """
        Returns:
            fused  : (B, embedding_dim)
            alpha_v: (B,) — attention weight for voice
            alpha_g: (B,) — attention weight for gait
        """
        combined = torch.cat([e_v, e_g], dim=1)             # (B, 256)
        scores   = self.attention_net(combined)              # (B, 2)
        scores   = scores + self.modality_bias
        temp     = self.temperature.clamp(min=0.1, max=10.0)
        scores   = scores / temp
        weights  = self.softmax(scores)                      # (B, 2)
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
    Matches MultimodalClassifier from training exactly.
    Submodules:
        self.voice_encoder  — VoiceEncoder
        self.gait_encoder   — GaitEncoder
        self.fusion         — AttentionFusion
        self.classifier     — nn.Sequential [128→64→ReLU→Drop→1]

    forward(voice, gait, return_attention=False)
        → logits (B,1) [raw, apply sigmoid for probabilities]
        or (logits, attention_weights (B,2)) if return_attention=True

    Call model.eval() before inference. Use torch.sigmoid(logits) for probs.
    """
    def __init__(
        self,
        voice_input_dim:  int   = 22,
        gait_in_channels: int   = 19,
        embedding_dim:    int   = 128,
        fusion_hidden:    int   = 64,
        classifier_hidden:int   = 64,
        dropout:          float = 0.3,
        modality_dropout: float = 0.0,
    ):
        super().__init__()
        self.voice_encoder = VoiceEncoder(voice_input_dim, embedding_dim, dropout=dropout)
        self.gait_encoder  = GaitEncoder(gait_in_channels, embedding_dim, dropout=dropout)
        self.fusion        = AttentionFusion(embedding_dim, fusion_hidden, modality_dropout)
        self.classifier    = nn.Sequential(
            nn.Linear(embedding_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(
        self,
        voice_input: torch.Tensor | None = None,
        gait_input:  torch.Tensor | None = None,
        return_attention: bool = False,
        mode: str = "both",           # "both" | "voice" | "gait"
    ):
        device = next(self.parameters()).device

        if mode in ("both", "voice") and voice_input is not None:
            e_v = self.voice_encoder(voice_input.to(device))
        else:
            b   = gait_input.shape[0] if gait_input is not None else 1
            e_v = torch.zeros(b, 128, device=device)

        if mode in ("both", "gait") and gait_input is not None:
            e_g = self.gait_encoder(gait_input.to(device))
        else:
            b   = voice_input.shape[0] if voice_input is not None else 1
            e_g = torch.zeros(b, 128, device=device)

        # Broadcast if batch sizes differ (voice=1 sample, gait=N windows)
        if e_v.shape[0] != e_g.shape[0]:
            if e_v.shape[0] == 1:
                e_v = e_v.expand(e_g.shape[0], -1)
            elif e_g.shape[0] == 1:
                e_g = e_g.expand(e_v.shape[0], -1)

        fused, alpha_v, alpha_g = self.fusion(e_v, e_g)
        logits = self.classifier(fused)          # (B, 1)

        if return_attention:
            weights = torch.stack([alpha_v, alpha_g], dim=1)   # (B, 2)
            return logits, weights
        return logits

    def predict_proba(self, voice_input, gait_input):
        return torch.sigmoid(self.forward(voice_input, gait_input))