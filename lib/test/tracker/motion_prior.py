"""
Motion Prior Module — Lightweight MLP Motion Memory for search-crop guidance.

Predicts the next-frame center OFFSET (Δx, Δy) from historical observed
target centers using a tiny learnable MLP.  The predicted center then
SOFTLY guides the search crop — it does NOT hijack it.

=====================================================================
Design Summary
=====================================================================
  observed_center   : model's raw output center (stored in history).
  predicted_center  : MLP forecast for next frame = last_observed + Δ.
  search_center     : final crop center = (1-α_eff)·fallback + α_eff·predicted.

  --- NEVER write predicted_center into history. ---
  --- NEVER post-correct the model's output box.  ---

  map_box_back ALWAYS uses search_center (the actual crop center).

Supports two modes:
  • MOTION_MODEL = "mlp"         :  learnable MLP (this module).
  • MOTION_MODEL = "constant_velocity" / "constant_acceleration" :
                                   rule-based fallback (also included).
=====================================================================

Usage:
    mp = MotionPrior(use=True, model='mlp', alpha=0.1, clip=100,
                     warmup_frames=2, history_len=4, hidden_dim=32,
                     conf_threshold=0.5)
    mp.to(device)                   # move MLP to GPU
    mp.load_state_dict(ckpt)        # (optional) load pre-trained motion weights
    mp.update(cx, cy)               # record observed center of frame t-1
    c_hat = mp.predict()            # predicted center for frame t (or None)
    scx, scy, eff_alpha, info = \
        mp.get_search_center(fallback_cx, fallback_cy, conf_score=0.6,
                             img_W=W, img_H=H)
"""

import math
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
#  Lightweight MLP motion predictor
# ----------------------------------------------------------------------

class MotionMLP(nn.Module):
    """
    Tiny MLP that predicts next-frame center OFFSET (Δx, Δy) in NORMALIZED
    coordinates from a normalized state vector.

    Input  (history_len * 4-dim):  [x_n, y_n, vx_n, vy_n,  ... per frame]
    Output (2-dim):                [dx_norm, dy_norm]

    Normalization convention (internal):
        x_norm = cx / W,   y_norm = cy / H
        vx_norm = (cx_t - cx_{t-1}) / W,  vy_norm = (cy_t - cy_{t-1}) / H

    The output delta is also normalized; callers multiply by (W,H) for pixels.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32,
                 output_dim: int = 2, num_layers: int = 3,
                 activation: str = 'relu'):
        super().__init__()
        layers = []
        in_dim = input_dim
        act = nn.ReLU() if activation == 'relu' else nn.GELU()
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(act)
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

        # Small init: near-zero output so the model starts close to baseline
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init the final layer bias so delta starts at ~0
        last = self.net[-1]
        if isinstance(last, nn.Linear) and last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) normalized state vector.
        Returns:
            (B, 2) normalized delta [dx_norm, dy_norm].
        """
        return self.net(x)


# ----------------------------------------------------------------------
#  MotionPrior — public-facing module used by the tracker
# ----------------------------------------------------------------------

class MotionPrior:
    """
    Motion-based search-crop center SOFT guider for visual tracking.

    Maintains a history of OBSERVED centers (never motion-corrected) and
    predicts the next-frame center.  Supports both a tiny learnable MLP
    and rule-based (constant velocity / acceleration) as fallback.

    Soft blending formula:
        search_center = (1 - effective_alpha) * fallback + effective_alpha * predicted
        effective_alpha = base_alpha × conf_weight × dist_weight

    Attributes:
        use (bool): Master switch. False → always returns fallback.
        model (str): 'mlp', 'constant_velocity', 'constant_acceleration'.
        alpha (float): Base blend weight (small ~0.05–0.2).
        clip (float): Max allowed offset radius (pixels, circular).
        warmup_frames (int): Frames to skip motion prior.
        history_len (int): Number of observed centers to store.
        conf_threshold (float): Confidence below which motion weight decays.
        motion_net (MotionMLP | None): Learnable predictor (model='mlp').
    """

    def __init__(self,
                 use: bool = True,
                 model: str = 'mlp',
                 alpha: float = 0.1,
                 clip: float = 100.0,
                 warmup_frames: int = 2,
                 history_len: int = 4,
                 hidden_dim: int = 32,
                 num_mlp_layers: int = 3,
                 conf_threshold: float = 0.5):
        self.use = use
        self.model = model
        self.alpha = alpha
        self.clip = clip
        self.warmup_frames = warmup_frames
        self.history_len = history_len
        self.conf_threshold = conf_threshold

        # ---- History stores OBSERVED centers (original image px) ----
        # Each entry: (cx, cy).  idx=-1 is the most recent.
        self.history_centers = []
        self.frame_count = 0

        # ---- Build MLP if requested ----
        self.motion_net = None
        if self.use and self.model == 'mlp':
            input_dim = history_len * 4  # [x,y,vx,vy] per frame
            self.motion_net = MotionMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=2,
                num_layers=num_mlp_layers,
                activation='relu',
            )

        # ---- Cached image size for de-normalisation ----
        self._img_W = None
        self._img_H = None

    # ------------------------------------------------------------------
    #  Device / weight management
    # ------------------------------------------------------------------

    def to(self, device):
        """Move MLP (if any) to *device*."""
        if self.motion_net is not None:
            self.motion_net = self.motion_net.to(device)
        return self

    def train(self, mode: bool = True):
        if self.motion_net is not None:
            self.motion_net.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        if self.motion_net is not None:
            return self.motion_net.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if self.motion_net is not None and state_dict:
            self.motion_net.load_state_dict(state_dict)
        return self

    def parameters(self):
        if self.motion_net is not None:
            return self.motion_net.parameters()
        return []

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def update(self, cx: float, cy: float):
        """
        Record an OBSERVED target center (model raw output).

        IMPORTANT: Call this AFTER the tracker has produced the final box
        for the current frame.  NEVER feed motion-corrected centers.
        This prevents compounding prediction error.

        Args:
            cx, cy: target center in original image coordinates.
        """
        if not self.use:
            return
        self.history_centers.append((cx, cy))
        if len(self.history_centers) > self.history_len:
            self.history_centers.pop(0)
        self.frame_count += 1

    def predict(self, img_W: int = None, img_H: int = None):
        """
        Predict next-frame target center in PIXEL coordinates.

        Returns:
            (cx_hat, cy_hat) in original image pixels, or None if disabled
            or history insufficient.
        """
        if not self.use:
            return None
        if self.frame_count < self.warmup_frames:
            return None

        # Update cached image size for normalization
        if img_W is not None:
            self._img_W = img_W
        if img_H is not None:
            self._img_H = img_H

        if self.model == 'mlp':
            return self._predict_mlp()
        elif self.model == 'constant_velocity':
            return self._predict_cv()
        elif self.model == 'constant_acceleration':
            return self._predict_ca()
        else:
            raise ValueError(f"Unknown MOTION_MODEL: {self.model}")

    def get_search_center(self,
                          fallback_cx: float,
                          fallback_cy: float,
                          conf_score: float = None,
                          img_W: int = None,
                          img_H: int = None):
        """
        SOFT blend of fallback and predicted center → search crop center.

        Args:
            fallback_cx, fallback_cy: default center (previous observed center,
                original image pixels).
            conf_score: tracker confidence from PREVIOUS frame (attenuates alpha).
            img_W, img_H: image dimensions (for boundary clipping & normalize).

        Returns:
            (search_cx, search_cy, effective_alpha, info_dict)
        """
        info = {
            'last_center': (fallback_cx, fallback_cy),
            'predicted_center': None,
            'search_center': (fallback_cx, fallback_cy),
            'motion_alpha': self.alpha,
            'motion_weight': 0.0,
            'motion_enabled': False,
            'motion_model': self.model,
            'clip_triggered': False,
            'fallback_triggered': True,
            'conf_attenuated': False,
            'dist_attenuated': False,
        }

        # ---- Predict next center ----
        c_hat = self.predict(img_W=img_W, img_H=img_H)
        if c_hat is None:
            return fallback_cx, fallback_cy, 0.0, info

        info['motion_enabled'] = True
        info['predicted_center'] = (c_hat[0], c_hat[1])

        # ---- Step 1: Compute raw offset & clip (circular) ----
        dx = c_hat[0] - fallback_cx
        dy = c_hat[1] - fallback_cy
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > self.clip and self.clip > 0:
            scale = self.clip / dist
            dx *= scale
            dy *= scale
            info['clip_triggered'] = True

        pred_cx_clipped = fallback_cx + dx
        pred_cy_clipped = fallback_cy + dy
        info['predicted_center'] = (pred_cx_clipped, pred_cy_clipped)

        # ---- Step 2: Effective alpha ----
        effective_alpha = float(self.alpha)

        # 2a. Confidence attenuation
        if conf_score is not None and self.conf_threshold > 0:
            if conf_score < self.conf_threshold:
                conf_ratio = max(0.0, conf_score / self.conf_threshold)
                effective_alpha *= conf_ratio
                info['conf_attenuated'] = True
                if conf_ratio <= 1e-6:
                    return fallback_cx, fallback_cy, 0.0, info

        # 2b. Distance attenuation (beyond 50% of clip radius)
        dist_clipped = math.sqrt(dx * dx + dy * dy)
        if self.clip > 0 and dist_clipped > 0.5 * self.clip:
            dist_ratio = max(0.0, 1.0 - (dist_clipped - 0.5 * self.clip)
                             / (0.5 * self.clip))
            effective_alpha *= dist_ratio
            info['dist_attenuated'] = True

        if effective_alpha <= 1e-6:
            return fallback_cx, fallback_cy, 0.0, info

        # ---- Step 3: Soft blend ----
        search_cx = (1.0 - effective_alpha) * fallback_cx + effective_alpha * pred_cx_clipped
        search_cy = (1.0 - effective_alpha) * fallback_cy + effective_alpha * pred_cy_clipped

        # ---- Step 4: Boundary clamp ----
        if img_W is not None and img_H is not None:
            search_cx = max(0.0, min(search_cx, float(img_W - 1)))
            search_cy = max(0.0, min(search_cy, float(img_H - 1)))

        info['search_center'] = (search_cx, search_cy)
        info['motion_weight'] = effective_alpha
        info['fallback_triggered'] = False
        return search_cx, search_cy, effective_alpha, info

    # ------------------------------------------------------------------
    #  MLP prediction
    # ------------------------------------------------------------------

    def _predict_mlp(self):
        """
        Build normalized feature vector from history, run MLP,
        de-normalize to pixel coordinates.
        """
        if self.motion_net is None:
            return None
        if len(self.history_centers) < 2:
            return None
        w = self._img_W or 1.0
        h = self._img_H or 1.0

        # Build feature: [x_norm, y_norm, vx_norm, vy_norm] per available frame
        feats = []
        for i in range(self.history_len):
            idx = len(self.history_centers) - 1 - i
            if idx >= 0:
                cx, cy = self.history_centers[idx]
                xn = cx / max(w, 1.0)
                yn = cy / max(h, 1.0)
                if idx > 0:
                    cx_p, cy_p = self.history_centers[idx - 1]
                    vx = (cx - cx_p) / max(w, 1.0)
                    vy = (cy - cy_p) / max(h, 1.0)
                else:
                    # Oldest available: zero velocity
                    vx, vy = 0.0, 0.0
                feats.extend([xn, yn, vx, vy])
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0])

        feat_tensor = torch.tensor([feats], dtype=torch.float32,
                                   device=self._device())
        with torch.no_grad():
            delta_norm = self.motion_net(feat_tensor)  # (1, 2)
        dx_norm = delta_norm[0, 0].item()
        dy_norm = delta_norm[0, 1].item()

        # De-normalize to pixels
        last_cx, last_cy = self.history_centers[-1]
        cx_hat = last_cx + dx_norm * w
        cy_hat = last_cy + dy_norm * h
        return (cx_hat, cy_hat)

    # ------------------------------------------------------------------
    #  Rule-based prediction (fallback / backward-compat)
    # ------------------------------------------------------------------

    def _predict_cv(self):
        """Constant-velocity: c_hat = c_t + (c_t - c_{t-1})."""
        if len(self.history_centers) < 2:
            return None
        cx_t, cy_t = self.history_centers[-1]
        cx_t1, cy_t1 = self.history_centers[-2]
        return (2 * cx_t - cx_t1, 2 * cy_t - cy_t1)

    def _predict_ca(self):
        """Constant-acceleration: c_hat = c_t + v_t + 0.5 · a_t."""
        if len(self.history_centers) < 3:
            return self._predict_cv()
        cx_t, cy_t = self.history_centers[-1]
        cx_t1, cy_t1 = self.history_centers[-2]
        cx_t2, cy_t2 = self.history_centers[-3]
        vx = cx_t - cx_t1
        vy = cy_t - cy_t1
        ax = vx - (cx_t1 - cx_t2)
        ay = vy - (cy_t1 - cy_t2)
        return (cx_t + vx + 0.5 * ax, cy_t + vy + 0.5 * ay)

    # ------------------------------------------------------------------
    #  Device helper
    # ------------------------------------------------------------------

    def _device(self):
        if self.motion_net is not None:
            return next(self.motion_net.parameters()).device
        return torch.device('cpu')

    # ------------------------------------------------------------------
    #  Utility
    # ------------------------------------------------------------------

    def reset(self):
        """Clear history (call at start of each video sequence)."""
        self.history_centers = []
        self.frame_count = 0
        self._img_W = None
        self._img_H = None

    def __repr__(self):
        return (f"MotionPrior(use={self.use}, model={self.model}, "
                f"alpha={self.alpha}, clip={self.clip}, "
                f"warmup={self.warmup_frames}, history_len={self.history_len}, "
                f"conf_thresh={self.conf_threshold}, "
                f"history={len(self.history_centers)}, "
                f"has_mlp={self.motion_net is not None})")
