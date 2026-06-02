"""
Motion Prior Module for inference-time search-crop center prediction.

Predicts the next-frame target center from historical observed centers
using a constant-velocity (or constant-acceleration) motion model.
The predicted center is used to shift the search crop region BEFORE
feature extraction — no post-hoc box correction is applied.

Usage:
    mp = MotionPrior(use=True, model='constant_velocity', clip=100, warmup_frames=2)
    mp.update(cx, cy)          # feed observed center after each frame
    c_hat = mp.predict()       # get predicted next center (None if disabled)
    search_cx, search_cy, weight, info = mp.get_search_center(fallback_cx, fallback_cy)
"""

import math


class MotionPrior:
    """
    Motion-based search-crop center predictor for visual tracking.

    Maintains a history of observed target centers (from model outputs,
    NOT motion-corrected) and predicts the next-frame center.

    The predicted center shifts the search crop BEFORE the model runs,
    so the model's feature extraction and box prediction remain unchanged.

    Attributes:
        use (bool): Master switch. When False, always returns fallback.
        model (str): 'constant_velocity' or 'constant_acceleration'.
        clip (float): Maximum allowed center shift (in original-image pixels).
            Predicted offsets beyond this are clipped to this radius.
        warmup_frames (int): Number of frames to skip motion prior after init.
            During warmup, uses fallback center.
        max_history (int): Max number of historical centers to store.
    """

    def __init__(self,
                 use: bool = True,
                 model: str = 'constant_velocity',
                 clip: float = 100.0,
                 warmup_frames: int = 2,
                 max_history: int = 10):
        self.use = use
        self.model = model
        self.clip = clip
        self.warmup_frames = warmup_frames
        self.max_history = max_history

        # History stores observed centers (original image coords) as [(cx,cy), ...]
        self.history_centers = []
        self.frame_count = 0

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def update(self, cx: float, cy: float):
        """
        Record an OBSERVED target center after the tracker finishes a frame.

        IMPORTANT: Always feed the model's raw output center, not a
        motion-corrected one. This avoids compounding prediction error.

        Args:
            cx, cy: Target center in original image coordinates.
        """
        if not self.use:
            return
        self.history_centers.append((cx, cy))
        if len(self.history_centers) > self.max_history:
            self.history_centers.pop(0)
        self.frame_count += 1

    def predict(self):
        """
        Predict the next-frame target center using the configured motion model.

        Returns:
            (cx_hat, cy_hat) in original image coords, or None if:
            - motion prior is disabled, or
            - insufficient history (fewer than 2 frames for CV, 3 for CA).
        """
        if not self.use:
            return None

        if self.frame_count < self.warmup_frames:
            return None

        if self.model == 'constant_velocity':
            return self._predict_cv()
        elif self.model == 'constant_acceleration':
            return self._predict_ca()
        else:
            raise ValueError(f"Unknown motion model: {self.model}")

    def get_search_center(self,
                          fallback_cx: float,
                          fallback_cy: float,
                          conf_score: float = None,
                          img_W: int = None,
                          img_H: int = None):
        """
        Get the center to use for search crop extraction.

        When motion prior is active and a valid prediction exists, returns
        the predicted center (clipped). Otherwise returns the fallback.

        Args:
            fallback_cx, fallback_cy: Default center (usually previous frame's
                observed center) in original image coords.
            conf_score: Optional tracker confidence, for future soft fallback.
            img_W, img_H: Optional image dimensions for boundary clipping.

        Returns:
            tuple: (search_cx, search_cy, motion_weight, info_dict)
                - search_cx, search_cy: center to use for search crop.
                - motion_weight: 1.0 if motion prior is active, 0.0 otherwise.
                - info_dict: diagnostic fields for logging/ablation.
        """
        info = {
            'last_center': (fallback_cx, fallback_cy),
            'predicted_center': None,
            'search_crop_center': (fallback_cx, fallback_cy),
            'motion_weight': 0.0,
            'motion_enabled': self.use and self.frame_count >= self.warmup_frames,
        }

        c_hat = self.predict()
        if c_hat is None:
            info['search_crop_center'] = (fallback_cx, fallback_cy)
            return fallback_cx, fallback_cy, 0.0, info

        # Compute raw offset from fallback
        dx = c_hat[0] - fallback_cx
        dy = c_hat[1] - fallback_cy
        dist = math.sqrt(dx * dx + dy * dy)

        # Clip excessive offset (circular clip to max radius)
        if dist > self.clip and self.clip > 0:
            scale = self.clip / dist
            dx *= scale
            dy *= scale

        search_cx = fallback_cx + dx
        search_cy = fallback_cy + dy

        # Clamp to image boundaries if provided
        if img_W is not None and img_H is not None:
            search_cx = max(0.0, min(search_cx, float(img_W - 1)))
            search_cy = max(0.0, min(search_cy, float(img_H - 1)))

        info['predicted_center'] = (c_hat[0], c_hat[1])
        info['search_crop_center'] = (search_cx, search_cy)
        info['motion_weight'] = 1.0

        return search_cx, search_cy, 1.0, info

    # ------------------------------------------------------------------
    #  Internal prediction methods
    # ------------------------------------------------------------------

    def _predict_cv(self):
        """Constant-velocity: c_hat = c_t + (c_t - c_{t-1})."""
        if len(self.history_centers) < 2:
            return None
        cx_t, cy_t = self.history_centers[-1]
        cx_t1, cy_t1 = self.history_centers[-2]
        return (2 * cx_t - cx_t1, 2 * cy_t - cy_t1)

    def _predict_ca(self):
        """Constant-acceleration: c_hat = c_t + v_t + 0.5 * a_t."""
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
    #  Utility
    # ------------------------------------------------------------------

    def reset(self):
        """Clear history (call at the start of each sequence)."""
        self.history_centers = []
        self.frame_count = 0

    def __repr__(self):
        return (f"MotionPrior(use={self.use}, model={self.model}, "
                f"clip={self.clip}, warmup={self.warmup_frames}, "
                f"history={len(self.history_centers)})")
