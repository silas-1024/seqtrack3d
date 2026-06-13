"""
RMP-SeqTrack: Reliable Motion Module.

Contains:
  - ReliabilityEstimator: assigns per-step reliability weights to motion deltas
  - MotionEncoder: TransformerEncoder over weighted motion sequence
  - MotionModule: top-level wrapper with forward + diagnostics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Reliability Estimator
# ---------------------------------------------------------------------------
class ReliabilityEstimator(nn.Module):
    """
    Input:  motion sequence  [B, N-1, 4]
    Output: reliability score [B, N-1]  (softmax-normalised, ∈ [0,1])
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, motion_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_seq: [B, N-1, 4]  raw motion deltas
        Returns:
            reliability: [B, N-1]    softmax-normalised scores
        """
        x = F.relu(self.fc1(motion_seq))          # [B, N-1, H]
        x = self.fc2(x).squeeze(-1)               # [B, N-1]
        reliability = x.softmax(dim=-1)            # [B, N-1]
        return reliability


# ---------------------------------------------------------------------------
# 2. Motion Encoder
# ---------------------------------------------------------------------------
class MotionEncoder(nn.Module):
    """
    TransformerEncoder over the weighted motion sequence.
    Input:  [B, N-1, 4]
    Output: [B, hidden_dim]   (mean-pooled over time)
    """
    def __init__(self,
                 input_dim: int = 4,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, weighted_motion: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weighted_motion: [B, N-1, 4]  reliability-weighted motion deltas
        Returns:
            motion_feature:  [B, hidden_dim]
        """
        x = self.input_proj(weighted_motion)       # [B, N-1, hidden_dim]
        x = self.transformer(x)                    # [B, N-1, hidden_dim]
        motion_feature = x.mean(dim=1)             # [B, hidden_dim]  mean pooling
        return motion_feature


# ---------------------------------------------------------------------------
# 3. Motion Module  (top-level wrapper)
# ---------------------------------------------------------------------------
class MotionModule(nn.Module):
    """
    Full motion pipeline:

      historical_boxes  →  motion deltas
                        →  ReliabilityEstimator
                        →  weighted motion
                        →  MotionEncoder  →  motion_feature
                        →  MLP  →  motion_bias (for cross-attention V-gating)

    Returns motion_bias and auxiliary data for loss / analysis.
    """
    def __init__(self,
                 hidden_dim: int = 256,
                 history_length: int = 5,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 enable_reliability: bool = True,
                 motion_scale: float = 128.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.history_length = history_length
        self.enable_reliability = enable_reliability
        self.motion_scale = motion_scale

        self.reliability_estimator = ReliabilityEstimator(
            input_dim=4, hidden_dim=64
        ) if enable_reliability else None

        self.motion_encoder = MotionEncoder(
            input_dim=4,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # MLP: motion_feature -> motion_bias  (256 -> 256 -> 256)
        self.bias_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Gate gain: scales raw motion_bias before sigmoid so that channels
        # start with meaningful variance.  Init = 4 → sigmoid(±4) ∈ [0.02, 0.98].
        self.gate_gain = nn.Parameter(torch.tensor(4.0))

    def _boxes_to_deltas(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert absolute boxes [B, N, 4] to relative motion deltas [B, N-1, 4].
        boxes format: [x, y, w, h]  (normalised or pixel – consistent within a seq)
        """
        # boxes: [B, N, 4]
        deltas = boxes[:, 1:, :] - boxes[:, :-1, :]          # [B, N-1, 4]
        return deltas

    def forward(self,
                historical_boxes: torch.Tensor,
                return_aux: bool = False) -> dict:
        """
        Args:
            historical_boxes: [B, N, 4]  last N boxes in [x,y,w,h] format
            return_aux: if True, return intermediate tensors for loss/viz
        Returns:
            dict with keys:
              'motion_bias':      [B, hidden_dim]   for cross-attention
              'motion_feature':   [B, hidden_dim]   (aux, if return_aux)
              'reliability':      [B, N-1]          (aux, if return_aux)
        """
        B, N, _ = historical_boxes.shape

        # 1. Boxes → deltas, then scaled up by MOTION_SCALE so that signal is
        #    meaningful (∼0.1–1.0 range instead of 0.001–0.005).
        deltas = self._boxes_to_deltas(historical_boxes)     # [B, N-1, 4]
        deltas = deltas * self.motion_scale                  # ← scale fix

        # 2. Reliability estimation
        if self.enable_reliability and self.reliability_estimator is not None:
            reliability = self.reliability_estimator(deltas)  # [B, N-1]
            weighted_motion = deltas * reliability.unsqueeze(-1)  # [B, N-1, 4]
        else:
            reliability = torch.ones(B, N-1, device=deltas.device) / (N-1)
            weighted_motion = deltas

        # 3. Motion encoder
        motion_feature = self.motion_encoder(weighted_motion)  # [B, hidden_dim]

        # 4. MLP -> motion bias, scaled by learnable gain for channel variance
        motion_bias = self.bias_mlp(motion_feature) * self.gate_gain   # [B, hidden_dim]
        gate = motion_bias.sigmoid()                                   # sigmoid(±4) ≈ [0.02,0.98] at init

        out = {'motion_bias': motion_bias}
        if return_aux:
            out['motion_feature']  = motion_feature
            out['reliability']     = reliability
            out['deltas']          = deltas

        return out

    def compute_motion_loss(self, aux: dict) -> torch.Tensor:
        """No auxiliary prototype loss is used in the nodict RMP variant."""
        feat = aux['motion_feature']
        return feat.new_tensor(0.0)
