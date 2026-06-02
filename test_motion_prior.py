#!/usr/bin/env python3
"""
Quick smoke test for the MotionPrior module.

Run:
    python lib/test/tracker/motion_prior.py
    or
    cd /media/lisuran/seqtrack3d && python lib/test/tracker/motion_prior.py

Expected output:
    - No errors
    - MotionPrior (disabled) returns identity
    - MotionPrior (enabled) applies soft correction
    - Distance-based weights are in [0, lambda]
"""

import sys
import os

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.test.tracker.motion_prior import MotionPrior


def test_baseline_mode():
    """When use_motion_prior=False, all outputs should be identity."""
    mp = MotionPrior(use_motion_prior=False)
    mp.update(100, 200)
    mp.update(110, 210)
    c_hat = mp.predict()
    assert c_hat is None, "Disabled prior should return None from predict()"
    cx, cy, w, info = mp.apply(120, 220, (130, 230))
    assert cx == 120 and cy == 220, "Disabled prior should return identity center"
    assert w == 0.0, "Disabled prior should return zero weight"
    assert not info['use_motion_prior']
    print("[PASS] Baseline mode (disabled) returns identity.")


def test_constant_velocity():
    """Constant-velocity prediction test."""
    mp = MotionPrior(use_motion_prior=True, lambda_motion=0.5, sigma=30.0)
    mp.update(100, 200)  # frame 1
    mp.update(110, 210)  # frame 2 (v=10,10)
    c_hat = mp.predict()
    assert c_hat is not None
    assert abs(c_hat[0] - 120) < 1e-6, f"Expected cx_hat=120, got {c_hat[0]}"
    assert abs(c_hat[1] - 220) < 1e-6, f"Expected cy_hat=220, got {c_hat[1]}"
    print(f"[PASS] Constant velocity: c_hat = {c_hat}")


def test_soft_blend():
    """Soft blend: closer to motion prior → higher weight."""
    mp = MotionPrior(use_motion_prior=True, lambda_motion=0.5, sigma=30.0)
    mp.update(100, 200)
    mp.update(110, 210)
    c_hat = mp.predict()  # (120, 220)

    # Case 1: prediction exactly matches motion prior → max weight
    cx1, cy1, w1, info1 = mp.apply(120, 220, c_hat)
    assert abs(w1 - 0.5) < 1e-6, f"Expected weight=0.5 at dist=0, got {w1}"
    assert abs(cx1 - 120) < 1e-6, "Center should be unchanged when dist=0"
    print(f"[PASS] Exact match: weight={w1:.4f}, center=({cx1:.1f},{cy1:.1f})")

    # Case 2: prediction far from motion prior → low weight
    cx2, cy2, w2, info2 = mp.apply(200, 300, c_hat)  # dist ≈ 113
    assert w2 < 0.01, f"Expected very low weight for large distance, got {w2}"
    assert abs(cx2 - 200) < 0.1, f"Center should be nearly unchanged for far prediction, got cx2={cx2}"
    print(f"[PASS] Far match: weight={w2:.6f}, center=({cx2:.1f},{cy2:.1f})")

    # Case 3: intermediate distance → partial blend
    cx3, cy3, w3, info3 = mp.apply(135, 235, c_hat)  # dist ≈ 21.2
    expected_w3 = 0.5 * np.exp(-(21.213**2) / (2 * 30**2))
    assert abs(w3 - expected_w3) < 1e-4
    # Check blend
    expected_cx3 = (1 - w3) * 135 + w3 * 120
    assert abs(cx3 - expected_cx3) < 1e-6
    print(f"[PASS] Partial blend: weight={w3:.4f}, center=({cx3:.1f},{cy3:.1f})")


def test_info_dict():
    """Verify that info dict contains all required fields."""
    mp = MotionPrior(use_motion_prior=True)
    mp.update(100, 200)
    mp.update(110, 210)
    c_hat = mp.predict()
    _, _, _, info = mp.apply(120, 220, c_hat)
    required_keys = ['predicted_center', 'prior_score', 'final_score',
                     'use_motion_prior', 'distance', 'weight']
    for k in required_keys:
        assert k in info, f"Missing key '{k}' in info dict"
    print(f"[PASS] Info dict contains all required keys: {list(info.keys())}")


def test_acceleration():
    """Test constant-acceleration model."""
    mp = MotionPrior(use_motion_prior=True)
    mp.update(100, 200)  # frame 1
    mp.update(110, 210)  # frame 2 (v=10,10)
    mp.update(125, 225)  # frame 3 (v=15,15, a=5,5)
    c_hat = mp.predict_with_acceleration()
    # c_hat = c_t + v_t + 0.5*a = (125,225) + (15,15) + 0.5*(5,5) = (142.5, 242.5)
    assert abs(c_hat[0] - 142.5) < 1e-6
    assert abs(c_hat[1] - 242.5) < 1e-6
    print(f"[PASS] Acceleration model: c_hat = {c_hat}")


if __name__ == '__main__':
    import numpy as np  # needed by test_soft_blend
    print("=" * 60)
    print("MotionPrior Smoke Tests")
    print("=" * 60)
    test_baseline_mode()
    test_constant_velocity()
    test_soft_blend()
    test_info_dict()
    test_acceleration()
    print("=" * 60)
    print("All tests passed!")
