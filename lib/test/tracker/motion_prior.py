"""
Motion Prior Module for inference-time search-crop center SOFT guidance.

Predicts the next-frame target center from historical observed centers
using a constant-velocity (or constant-acceleration) motion model.
The predicted center is SOFTLY blended with the fallback center —
it gently biases the search crop, but does NOT hijack it.

Key design:
  - observed_center : model's raw output center (stored in history).
  - predicted_center : motion model forecast for the next frame.
  - search_center    : final crop center = soft_blend(fallback, predicted).
  - map_box_back always uses search_center (the actual crop center).

No post-hoc box correction is applied.

Usage:
    mp = MotionPrior(use=True, model='constant_velocity',
                     alpha=0.1, clip=100, warmup_frames=2,
                     conf_threshold=0.5)
    mp.update(cx, cy)          # feed OBSERVED center after each frame RESULT
    c_hat = mp.predict()       # get predicted next center (None if disabled/warmup)
    search_cx, search_cy, eff_weight, info = \
        mp.get_search_center(fallback_cx, fallback_cy, conf_score=0.6,
                             img_W=W, img_H=H)
"""

import math


class MotionPrior:
    """
    Motion-based search-crop center SOFT guider for visual tracking.

    Maintains a history of OBSERVED target centers (model outputs, never
    motion-corrected) and predicts the next-frame center.

    The predicted center is softly blended with the fallback (previous-frame
    observed center) via:
        search_center = (1 - effective_alpha) * fallback + effective_alpha * predicted
    where effective_alpha = base_alpha * conf_weight * dist_weight.

    This keeps the search crop near the "true" target while allowing a slight
    anticipatory shift — avoiding the degradation caused by full replacement.

    Attributes:
        use (bool): Master switch. When False, always returns fallback.
        model (str): 'constant_velocity' or 'constant_acceleration'.
        alpha (float): Base blend weight for motion prior. Small default (0.1).
        clip (float): Maximum allowed predicted→fallback offset (pixels, circular).
            Offsets beyond this are clipped BEFORE blending.
        warmup_frames (int): Number of frames to skip motion prior after init.
            During warmup, uses fallback center.
        conf_threshold (float): Confidence below which motion weight is reduced.
        max_history (int): Max number of historical centers to store.
    """

    def __init__(self,
                 use: bool = True,
                 model: str = 'constant_velocity',
                 alpha: float = 0.1,
                 clip: float = 100.0,
                 warmup_frames: int = 2,
                 conf_threshold: float = 0.5,
                 max_history: int = 10):
        self.use = use
        self.model = model
        self.alpha = alpha
        self.clip = clip
        self.warmup_frames = warmup_frames
        self.conf_threshold = conf_threshold
        self.max_history = max_history

        # History stores OBSERVED centers (original image coords) as [(cx,cy), ...]
        # NEVER store motion-corrected centers here.
        self.history_centers = []
        self.frame_count = 0

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def update(self, cx: float, cy: float):
        """
        Record an OBSERVED target center after the tracker finishes a frame.

        IMPORTANT: Always feed the model's RAW output center, never a
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
        Get the center to use for search crop extraction via SOFT blending.

        When motion prior is active and a valid prediction exists:
            search_center = (1 - eff_alpha) * fallback + eff_alpha * predicted
        where eff_alpha is the base alpha attenuated by confidence and distance.

        When disabled or warmup not passed: returns fallback directly.

        Args:
            fallback_cx, fallback_cy: Default center in original image coords
                (usually the previous frame's observed center).
            conf_score: Previous frame's tracker confidence, used to attenuate
                motion weight when confidence is low.
            img_W, img_H: Optional image dimensions for boundary clipping.

        Returns:
            tuple: (search_cx, search_cy, effective_alpha, info_dict)
                - search_cx, search_cy: center to use for search crop.
                - effective_alpha: the actual blend weight used (0.0 = baseline).
                - info_dict: diagnostic fields for logging/ablation.
        """
        info = {
            'last_center': (fallback_cx, fallback_cy),
            'predicted_center': None,
            'search_center': (fallback_cx, fallback_cy),
            'motion_weight': 0.0,       # effective alpha after all attenuation
            'base_alpha': self.alpha,
            'motion_enabled': False,
            'clip_triggered': False,
            'fallback_triggered': True,  # True until motion prior actually contributes
            'conf_attenuated': False,
            'dist_attenuated': False,
        }

        c_hat = self.predict()
        if c_hat is None:
            # Warmup not passed, history insufficient, or disabled → pure fallback
            return fallback_cx, fallback_cy, 0.0, info

        info['motion_enabled'] = True

        # ---- Step 1: Compute predicted offset ----
        dx_raw = c_hat[0] - fallback_cx
        dy_raw = c_hat[1] - fallback_cy
        dist_raw = math.sqrt(dx_raw * dx_raw + dy_raw * dy_raw)

        # ---- Step 2: Clip excessive offset (circular clip to max radius) ----
        if dist_raw > self.clip and self.clip > 0:
            scale = self.clip / dist_raw
            dx_raw *= scale
            dy_raw *= scale
            info['clip_triggered'] = True

        # Clipped predicted center
        pred_cx_clipped = fallback_cx + dx_raw
        pred_cy_clipped = fallback_cy + dy_raw

        info['predicted_center'] = (pred_cx_clipped, pred_cy_clipped)

        # ---- Step 3: Compute effective alpha via attenuation chain ----
        effective_alpha = float(self.alpha)

        # 3a. Confidence-based attenuation:
        #     If conf_score is below threshold, scale alpha down linearly.
        #     At conf=0 → alpha → 0; at conf=threshold → alpha unchanged.
        if conf_score is not None and self.conf_threshold > 0:
            if conf_score < self.conf_threshold:
                conf_ratio = max(0.0, conf_score / self.conf_threshold)
                effective_alpha *= conf_ratio
                info['conf_attenuated'] = True
                if conf_ratio <= 1e-6:
                    # Confidence too low: fully fallback to baseline
                    return fallback_cx, fallback_cy, 0.0, info

        # 3b. Distance-based attenuation:
        #     If predicted_center is far from fallback (even after clipping),
        #     further reduce alpha to avoid pulling search crop too far.
        dist_clipped = math.sqrt(dx_raw * dx_raw + dy_raw * dy_raw)
        if self.clip > 0 and dist_clipped > 0.5 * self.clip:
            # Soft decay beyond half the clip radius
            dist_ratio = max(0.0, 1.0 - (dist_clipped - 0.5 * self.clip) / (0.5 * self.clip))
            effective_alpha *= dist_ratio
            info['dist_attenuated'] = True

        if effective_alpha <= 1e-6:
            return fallback_cx, fallback_cy, 0.0, info

        # ---- Step 4: Soft blend ----
        # search_center = (1 - alpha) * fallback + alpha * predicted
        search_cx = (1.0 - effective_alpha) * fallback_cx + effective_alpha * pred_cx_clipped
        search_cy = (1.0 - effective_alpha) * fallback_cy + effective_alpha * pred_cy_clipped

        # ---- Step 5: Clamp to image boundaries ----
        if img_W is not None and img_H is not None:
            search_cx = max(0.0, min(search_cx, float(img_W - 1)))
            search_cy = max(0.0, min(search_cy, float(img_H - 1)))

        info['search_center'] = (search_cx, search_cy)
        info['motion_weight'] = effective_alpha
        info['fallback_triggered'] = False

        return search_cx, search_cy, effective_alpha, info

    # ------------------------------------------------------------------
    #  Internal prediction methods (unchanged from original)
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
                f"alpha={self.alpha}, clip={self.clip}, "
                f"warmup={self.warmup_frames}, conf_thresh={self.conf_threshold}, "
                f"history={len(self.history_centers)})")
