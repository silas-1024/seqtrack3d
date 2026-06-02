"""
Motion Prior Module for inference-time soft center correction.

Implements a constant-velocity motion model that predicts the target center
in the next frame and applies a Gaussian-weighted soft bias to the model's
predicted center.

Usage:
    motion_prior = MotionPrior(use_motion_prior=True, lambda_motion=0.3, sigma=20.0)
    motion_prior.update(cx, cy)   # feed history
    c_hat = motion_prior.predict()  # get prediction (None if insufficient history)
    corrected_center, prior_score, info = motion_prior.apply(center, c_hat)
"""

import numpy as np


class MotionPrior:
    """
    Constant-velocity motion prior for soft center correction.

    Tracks historical target centers, predicts the next center using
    velocity estimation, and applies a Gaussian-weighted blend between
    the model-predicted center and the motion-predicted center.

    Attributes:
        use_motion_prior (bool): Master switch. When False, all methods
            return identity / no-op results.
        lambda_motion (float): Maximum blending weight (0 = no effect,
            1 = fully trust motion prior when distance=0).
        sigma (float): Spatial bandwidth of the Gaussian decay (in
            original image pixels). Larger sigma = wider influence.
        max_history (int): Maximum number of historical centers to store.
    """

    def __init__(self,
                 use_motion_prior: bool = True,
                 lambda_motion: float = 0.3,
                 sigma: float = 20.0,
                 max_history: int = 10):
        self.use_motion_prior = use_motion_prior
        self.lambda_motion = lambda_motion
        self.sigma = sigma
        self.max_history = max_history
        self.history_centers = []  # list of (cx, cy) in original image coords
        self.frame_count = 0

    def update(self, cx: float, cy: float):
        """
        Record the center of the current frame (after tracking is done).

        Args:
            cx, cy: Target center in original image coordinates.
        """
        if not self.use_motion_prior:
            return
        self.history_centers.append((cx, cy))
        if len(self.history_centers) > self.max_history:
            self.history_centers.pop(0)
        self.frame_count += 1

    def predict(self):
        """
        Predict the next-frame center using a constant-velocity model.

        Velocity is estimated from the last two known centers:
            v_t = c_t - c_{t-1}
            c_hat = c_t + v_t

        Returns:
            (cx_hat, cy_hat) in original image coordinates, or None if
            fewer than 2 historical centers are available (falls back to
            no prior).
        """
        if not self.use_motion_prior:
            return None
        if len(self.history_centers) < 2:
            return None

        cx_prev, cy_prev = self.history_centers[-1]
        cx_prev2, cy_prev2 = self.history_centers[-2]

        vx = cx_prev - cx_prev2
        vy = cy_prev - cy_prev2

        cx_hat = cx_prev + vx
        cy_hat = cy_prev + vy

        return (cx_hat, cy_hat)

    def predict_with_acceleration(self):
        """
        (Optional) Predict using constant-acceleration model.

        Uses the last three centers to estimate acceleration:
            a_t = v_t - v_{t-1}
            c_hat = c_t + v_t + 0.5 * a_t

        Returns:
            (cx_hat, cy_hat) or None.
        """
        if not self.use_motion_prior:
            return None
        if len(self.history_centers) < 3:
            return self.predict()  # fall back to constant velocity

        cx_t, cy_t = self.history_centers[-1]
        cx_t1, cy_t1 = self.history_centers[-2]
        cx_t2, cy_t2 = self.history_centers[-3]

        vx = cx_t - cx_t1
        vy = cy_t - cy_t1
        vx_prev = cx_t1 - cx_t2
        vy_prev = cy_t1 - cy_t2

        ax = vx - vx_prev
        ay = vy - vy_prev

        cx_hat = cx_t + vx + 0.5 * ax
        cy_hat = cy_t + vy + 0.5 * ay

        return (cx_hat, cy_hat)

    def apply(self, pred_cx: float, pred_cy: float, c_hat):
        """
        Apply motion-prior soft correction to the predicted center.

        Computes:
            weight = lambda * exp(-dist^2 / (2 * sigma^2))
            corrected = (1 - weight) * pred + weight * c_hat

        Args:
            pred_cx, pred_cy: Model-predicted center (in the same coordinate
                space as c_hat; expected: original image pixels).
            c_hat: Motion-predicted center (cx, cy) or None.

        Returns:
            tuple: (corrected_cx, corrected_cy, prior_weight, info_dict)
                - corrected_cx, corrected_cy: blended center.
                - prior_weight: the actual weight applied (0 if no prior).
                - info_dict: diagnostic information for logging/ablation.
        """
        # Build base info dict (used even when prior is off)
        info = {
            'predicted_center': c_hat,
            'prior_score': 0.0,
            'final_score': 0.0,
            'use_motion_prior': self.use_motion_prior,
            'distance': 0.0,
            'weight': 0.0,
        }

        if (not self.use_motion_prior) or (c_hat is None):
            info['predicted_center'] = None
            return pred_cx, pred_cy, 0.0, info

        # Euclidean distance between model prediction and motion prior
        dx = pred_cx - c_hat[0]
        dy = pred_cy - c_hat[1]
        dist = np.sqrt(dx * dx + dy * dy)

        # Gaussian decay: closer → higher weight
        weight = self.lambda_motion * np.exp(-dist**2 / (2.0 * self.sigma**2))
        weight = float(np.clip(weight, 0.0, self.lambda_motion))

        # Soft blend
        corrected_cx = (1.0 - weight) * pred_cx + weight * c_hat[0]
        corrected_cy = (1.0 - weight) * pred_cy + weight * c_hat[1]

        info['distance'] = float(dist)
        info['weight'] = weight
        info['prior_score'] = weight
        info['predicted_center'] = c_hat
        info['use_motion_prior'] = True

        return corrected_cx, corrected_cy, weight, info

    def reset(self):
        """Clear history (e.g., for a new sequence)."""
        self.history_centers = []
        self.frame_count = 0

    def __repr__(self):
        return (f"MotionPrior(use={self.use_motion_prior}, "
                f"lambda={self.lambda_motion}, sigma={self.sigma}, "
                f"history={len(self.history_centers)})")
