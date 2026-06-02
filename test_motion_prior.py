#!/usr/bin/env python3
"""
Smoke test for MotionPrior search-crop center predictor.

Run:
    cd /media/lisuran/seqtrack3d && python test_motion_prior.py

Expected output: all tests pass.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.test.tracker.motion_prior import MotionPrior


def test_disabled():
    """When use=False, always returns fallback."""
    mp = MotionPrior(use=False)
    mp.update(100, 200)
    mp.update(110, 210)
    c_hat = mp.predict()
    assert c_hat is None, "Disabled: predict() should return None"
    cx, cy, w, info = mp.get_search_center(120, 220)
    assert cx == 120 and cy == 220, "Disabled: should return fallback"
    assert w == 0.0, "Disabled: motion_weight should be 0"
    assert not info['motion_enabled']
    print("[PASS] Disabled mode returns fallback.")


def test_warmup():
    """During warmup frames, predict() returns None."""
    mp = MotionPrior(use=True, warmup_frames=3)
    mp.update(100, 200)  # frame 1
    mp.update(110, 210)  # frame 2 (not enough for warmup=3)
    c_hat = mp.predict()
    assert c_hat is None, f"Warmup: predict() should be None at frame {mp.frame_count}"
    cx, cy, w, info = mp.get_search_center(110, 210)
    assert cx == 110 and cy == 210, "Warmup: should return fallback"
    assert w == 0.0
    print("[PASS] Warmup returns fallback.")


def test_constant_velocity():
    """Constant velocity prediction."""
    mp = MotionPrior(use=True, model='constant_velocity', warmup_frames=2)
    mp.update(100, 200)
    mp.update(110, 210)  # v = (10, 10)
    c_hat = mp.predict()
    assert c_hat is not None
    assert abs(c_hat[0] - 120) < 1e-6, f"Expected cx=120, got {c_hat[0]}"
    assert abs(c_hat[1] - 220) < 1e-6, f"Expected cy=220, got {c_hat[1]}"
    print(f"[PASS] CV prediction: c_hat={c_hat}")


def test_constant_acceleration():
    """Constant acceleration prediction."""
    mp = MotionPrior(use=True, model='constant_acceleration', warmup_frames=3)
    mp.update(100, 200)  # frame 1
    mp.update(110, 210)  # frame 2 (v=10,10)
    mp.update(125, 225)  # frame 3 (v=15,15, a=5,5)
    c_hat = mp.predict()
    # c_hat = (125,225) + (15,15) + 0.5*(5,5) = (142.5, 242.5)
    assert abs(c_hat[0] - 142.5) < 1e-6
    assert abs(c_hat[1] - 242.5) < 1e-6
    print(f"[PASS] CA prediction: c_hat={c_hat}")


def test_clip_offset():
    """Large predicted offset should be clipped."""
    mp = MotionPrior(use=True, clip=20.0, warmup_frames=2)
    mp.update(100, 200)
    mp.update(110, 210)  # v=(10,10), c_hat=(120,220)
    cx, cy, w, info = mp.get_search_center(fallback_cx=110, fallback_cy=210)
    # Raw c_hat = (120, 220), dist from fallback (110,210) = sqrt(200) ≈ 14.14
    # clip=20 > 14.14, so no clip
    assert abs(cx - 120) < 1e-6
    assert abs(cy - 220) < 1e-6
    print(f"[PASS] Small offset NOT clipped: ({cx:.1f}, {cy:.1f})")

    # Now test with a huge velocity
    mp2 = MotionPrior(use=True, clip=10.0, warmup_frames=2)
    mp2.update(100, 200)
    mp2.update(100, 200)  # v=(0,0), c_hat=(100,200)
    # Fallback is (200,300), raw offset = (-100,-100), dist=141 > clip=10
    cx2, cy2, w2, info2 = mp2.get_search_center(fallback_cx=200, fallback_cy=300)
    dist = ((cx2 - 200)**2 + (cy2 - 300)**2)**0.5
    assert abs(dist - 10.0) < 0.01, f"Clipped distance should be ~10, got {dist:.2f}"
    print(f"[PASS] Large offset clipped: search_center=({cx2:.1f},{cy2:.1f})")


def test_info_dict():
    """Verify info dict contains required keys."""
    mp = MotionPrior(use=True, warmup_frames=2)
    mp.update(100, 200)
    mp.update(110, 210)
    cx, cy, w, info = mp.get_search_center(110, 210)
    required = ['last_center', 'predicted_center', 'search_crop_center',
                'motion_weight', 'motion_enabled']
    for k in required:
        assert k in info, f"Missing key '{k}'"
    assert info['motion_enabled'] is True
    assert info['motion_weight'] == 1.0
    assert info['predicted_center'] == (120, 220)
    print(f"[PASS] Info dict OK: {list(info.keys())}")


def test_boundary_clip():
    """Search center respects image boundaries."""
    mp = MotionPrior(use=True, clip=500.0, warmup_frames=2)
    mp.update(100, 200)
    mp.update(110, 210)  # c_hat=(120,220)
    # Fallback is near the edge
    cx, cy, w, info = mp.get_search_center(fallback_cx=10, fallback_cy=10,
                                           img_W=200, img_H=150)
    # c_hat=(120,220) - clipped to image bounds
    assert 0 <= cx <= 199, f"cx={cx} out of bounds"
    assert 0 <= cy <= 149, f"cy={cy} out of bounds"
    print(f"[PASS] Boundary clip: ({cx:.1f}, {cy:.1f}) within (200,150)")


def test_history_not_poisoned():
    """History must only contain observed centers, not predicted ones."""
    mp = MotionPrior(use=True, warmup_frames=2)
    mp.update(100, 200)
    mp.update(110, 210)
    mp.update(120, 220)
    # History should be [(100,200), (110,210), (120,220)]
    assert mp.history_centers[-1] == (120, 220)
    # get_search_center does not modify history
    cx, cy, w, info = mp.get_search_center(120, 220)
    assert mp.history_centers[-1] == (120, 220)
    c_hat = mp.predict()
    assert c_hat == (130, 230)  # CV from last two: 2*120-110, 2*220-210
    print("[PASS] History stores only observed centers.")


if __name__ == '__main__':
    print("=" * 60)
    print("MotionPrior (Search-Crop Center) Smoke Tests")
    print("=" * 60)
    test_disabled()
    test_warmup()
    test_constant_velocity()
    test_constant_acceleration()
    test_clip_offset()
    test_info_dict()
    test_boundary_clip()
    test_history_not_poisoned()
    print("=" * 60)
    print("All tests passed!")
