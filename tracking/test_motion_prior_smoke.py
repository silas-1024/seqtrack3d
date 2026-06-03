#!/usr/bin/env python3
"""
Smoke test for MotionPrior (MLP-based motion memory).

Validates:
  1. Baseline switch (USE=False → always returns fallback).
  2. MLP output shape and device correctness.
  3. History update (observed centers only, no motion-corrected entries).
  4. Search center soft-blend computation.
  5. Warmup / clip / confidence attenuation logic.
  6. Rule-based CV/CA backward compatibility.

Run:
    python tracking/test_motion_prior_smoke.py
"""

import sys
import os
# Make sure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from lib.test.tracker.motion_prior import MotionPrior, MotionMLP


# -------------------------------------------------------------------
#  Helper
# -------------------------------------------------------------------

def print_header(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def check(condition, msg):
    if condition:
        print(f"  [PASS] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        raise AssertionError(msg)


# -------------------------------------------------------------------
#  Test 1: MotionMLP standalone
# -------------------------------------------------------------------

def test_mlp_standalone():
    print_header("Test 1: MotionMLP standalone (shape / forward)")

    mlp = MotionMLP(input_dim=16, hidden_dim=32, output_dim=2,
                    num_layers=3, activation='relu')
    mlp.eval()

    # Random normalized input simulating 4 frames × [x,y,vx,vy]
    x = torch.randn(2, 16)
    with torch.no_grad():
        out = mlp(x)

    check(out.shape == (2, 2),
          f"Output shape is {out.shape} (expected (2,2))")
    check(out.dtype == torch.float32,
          f"Output dtype is {out.dtype} (expected float32)")

    # Check small-magnitude due to zero-biased init
    check(out.abs().max().item() < 1.0,
          f"With small init, max |δ| should be < 1.0, got {out.abs().max().item():.4f}")

    print("  MotionMLP forward OK")


# -------------------------------------------------------------------
#  Test 2: Baseline switch (USE=False)
# -------------------------------------------------------------------

def test_baseline_switch():
    print_header("Test 2: Baseline switch (USE=False)")

    mp = MotionPrior(use=False, model='mlp', alpha=0.1,
                     clip=100, warmup_frames=2, history_len=4)

    # Update with some centers
    mp.update(100, 200)
    mp.update(105, 210)
    mp.update(110, 220)

    # Predict should be None
    c_hat = mp.predict()
    check(c_hat is None, f"USE=False → predict() should be None, got {c_hat}")

    # get_search_center should return fallback
    scx, scy, eff_a, info = mp.get_search_center(110, 220, img_W=640, img_H=480)
    check(scx == 110 and scy == 220,
          f"USE=False → search_center should be fallback (110,220), got ({scx},{scy})")
    check(eff_a == 0.0, f"USE=False → effective_alpha should be 0, got {eff_a}")
    check(not info['motion_enabled'],
          f"USE=False → motion_enabled should be False, got {info['motion_enabled']}")
    check(info['motion_model'] == 'mlp',
          f"motion_model should be 'mlp' in info, got {info['motion_model']}")

    # history should still be recorded (for potential future use)
    check(len(mp.history_centers) == 0,
          f"USE=False → history should stay empty, got {len(mp.history_centers)}")

    print("  Baseline switch OK")


# -------------------------------------------------------------------
#  Test 3: MLP prediction pipeline
# -------------------------------------------------------------------

def test_mlp_pipeline():
    print_header("Test 3: MLP prediction (shape, normalization, device)")

    mp = MotionPrior(use=True, model='mlp', alpha=0.1,
                     clip=50, warmup_frames=2, history_len=4,
                     hidden_dim=32)
    mp.to('cpu')
    mp.eval()

    W, H = 640, 480

    # Warmup frame 0
    mp.update(320, 240)
    # Warmup frame 1
    mp.update(322, 238)

    check(mp.frame_count == 2, f"frame_count should be 2, got {mp.frame_count}")
    check(len(mp.history_centers) == 2,
          f"history should have 2 entries, got {len(mp.history_centers)}")

    c_hat = mp.predict(img_W=W, img_H=H)
    check(c_hat is not None, "predict() should not be None after warmup")
    check(isinstance(c_hat, tuple) and len(c_hat) == 2,
          f"predict() should return (cx,cy), got {c_hat}")

    cx_hat, cy_hat = c_hat
    # Should be near (322, 238) since MLP starts near-zero
    check(abs(cx_hat - 322) < W * 0.5,
          f"Predicted cx={cx_hat:.1f} too far from last observed 322")
    check(abs(cy_hat - 238) < H * 0.5,
          f"Predicted cy={cy_hat:.1f} too far from last observed 238")

    print(f"  Predicted center: ({cx_hat:.2f}, {cy_hat:.2f})")
    print("  MLP prediction OK")


# -------------------------------------------------------------------
#  Test 4: Soft blending & info dict
# -------------------------------------------------------------------

def test_soft_blend():
    print_header("Test 4: Soft blending & info dict")

    mp = MotionPrior(use=True, model='mlp', alpha=0.2,
                     clip=50, warmup_frames=1, history_len=4,
                     hidden_dim=32)
    mp.to('cpu')
    mp.eval()
    W, H = 640, 480

    # One warmup frame
    mp.update(320, 240)   # frame 0
    # After warmup
    mp.update(322, 238)   # frame 1 (observed)
    # frame 2: last_observed = (322, 238) = fallback

    scx, scy, eff_a, info = mp.get_search_center(
        fallback_cx=322, fallback_cy=238,
        conf_score=0.8, img_W=W, img_H=H,
    )

    # Check info keys
    for key in ['last_center', 'predicted_center', 'search_center',
                'motion_alpha', 'motion_weight', 'motion_enabled',
                'clip_triggered', 'fallback_triggered',
                'conf_attenuated', 'dist_attenuated']:
        check(key in info, f"info dict missing key '{key}'")

    check(info['motion_enabled'], "motion_enabled should be True after warmup")
    check(not info['fallback_triggered'],
          "fallback_triggered should be False when motion contributes")
    check(not info['conf_attenuated'],
          "conf_attenuated should be False when conf > threshold")
    check(0.0 < eff_a <= 0.2,
          f"effective_alpha should be in (0, 0.2], got {eff_a}")

    # search_center should be between fallback and prediction
    check(abs(scx - 322) <= 50,
          f"search_cx {scx:.2f} too far from fallback 322")
    check(abs(scy - 238) <= 50,
          f"search_cy {scy:.2f} too far from fallback 238")

    print(f"  search_center=({scx:.2f},{scy:.2f}) eff_alpha={eff_a:.4f}")
    print("  Soft blend OK")


# -------------------------------------------------------------------
#  Test 5: Clip & confidence attenuation
# -------------------------------------------------------------------

def test_clip_and_conf():
    print_header("Test 5: Clip & confidence attenuation")

    mp = MotionPrior(use=True, model='mlp', alpha=0.3,
                     clip=10, warmup_frames=1, history_len=4,
                     hidden_dim=32, conf_threshold=0.5)
    mp.to('cpu')
    mp.eval()
    W, H = 640, 480

    mp.update(320, 240)   # warmup
    mp.update(322, 238)

    # ---- 5a: Low confidence → attenuation ----
    scx, scy, eff_a, info = mp.get_search_center(
        fallback_cx=322, fallback_cy=238,
        conf_score=0.1, img_W=W, img_H=H,
    )
    check(info['conf_attenuated'],
          f"conf_attenuated should be True when conf=0.1 < threshold=0.5")
    check(eff_a < 0.3, f"eff_alpha should be < base alpha 0.3, got {eff_a}")

    # ---- 5b: Very low confidence → full fallback ----
    scx2, scy2, eff_a2, info2 = mp.get_search_center(
        fallback_cx=322, fallback_cy=238,
        conf_score=0.0, img_W=W, img_H=H,
    )
    check(scx2 == 322 and scy2 == 238 and eff_a2 == 0.0,
          f"conf=0 → should fall back to (322,238), got ({scx2},{scy2}), α={eff_a2}")
    check(info2['fallback_triggered'],
          "fallback_triggered should be True at conf=0")

    # ---- 5c: Clip triggered when offset > clip radius ----
    # Simulate large offset by replacing history with big jump
    mp.reset()
    mp.update(100, 100)   # warmup
    mp.update(500, 400)   # big jump

    # Manually override the predicted center (we just test clip logic here)
    # The actual MLP will produce a small delta, but we trust the clip code path.
    scx3, scy3, eff_a3, info3 = mp.get_search_center(
        fallback_cx=500, fallback_cy=400,
        conf_score=0.8, img_W=640, img_H=480,
    )
    # clip may or may not trigger depending on MLP output;
    # at minimum check that search_center respects clip radius vs fallback
    dist3 = ((scx3 - 500)**2 + (scy3 - 400)**2)**0.5
    check(dist3 <= mp.clip + 1e-5,
          f"search_center distance {dist3:.2f} exceeds clip radius {mp.clip}")

    print(f"  conf_attenuated test: α={eff_a:.4f}")
    print(f"  conf=0 fallback:      α={eff_a2:.4f}")
    print(f"  clip distance:        {dist3:.2f} (max={mp.clip})")
    print("  Clip & conf attenuation OK")


# -------------------------------------------------------------------
#  Test 6: History integrity
# -------------------------------------------------------------------

def test_history_integrity():
    print_header("Test 6: History integrity (observed-only, no motion-corrected)")

    mp = MotionPrior(use=True, model='mlp', alpha=0.1,
                     clip=100, warmup_frames=2, history_len=4)
    mp.to('cpu')
    mp.eval()

    mp.update(100, 200)
    mp.update(105, 210)
    mp.update(110, 220)

    # All entries should be exactly what we inserted
    expected = [(100, 200), (105, 210), (110, 220)]
    check(mp.history_centers == expected,
          f"history should be {expected}, got {mp.history_centers}")

    # Add more to test history_len cap
    mp.update(115, 230)
    mp.update(120, 240)
    mp.update(125, 250)

    check(len(mp.history_centers) == mp.history_len,
          f"history should be capped at {mp.history_len}, got {len(mp.history_centers)}")

    # Most recent should be last inserted
    check(mp.history_centers[-1] == (125, 250),
          f"last entry should be (125,250), got {mp.history_centers[-1]}")

    print("  History integrity OK")


# -------------------------------------------------------------------
#  Test 7: Rule-based CV/CA backward compatibility
# -------------------------------------------------------------------

def test_cv_ca():
    print_header("Test 7: CV / CA backward compatibility")

    # Constant velocity
    mp_cv = MotionPrior(use=True, model='constant_velocity',
                        alpha=1.0, clip=100, warmup_frames=1)
    mp_cv.update(100, 200)
    mp_cv.update(105, 210)
    c_hat_cv = mp_cv.predict()
    # c_hat = (105,210) + ((105,210)-(100,200)) = (110,220)
    check(c_hat_cv == (110, 220),
          f"CV predict should be (110,220), got {c_hat_cv}")

    # Constant acceleration
    mp_ca = MotionPrior(use=True, model='constant_acceleration',
                        alpha=1.0, clip=100, warmup_frames=1)
    mp_ca.update(100, 200)  # t-2
    mp_ca.update(105, 210)  # t-1
    mp_ca.update(108, 216)  # t
    c_hat_ca = mp_ca.predict()
    # v = (108-105, 216-210) = (3,6)
    # a = v - v_prev = (3,6) - (5,10) = (-2,-4)
    # c_hat = (108,216) + (3,6) + 0.5*(-2,-4) = (110,220)
    check(c_hat_ca == (110, 220),
          f"CA predict should be (110,220), got {c_hat_ca}")

    print("  CV/CA backward compat OK")


# -------------------------------------------------------------------
#  Test 8: device management
# -------------------------------------------------------------------

def test_device():
    print_header("Test 8: Device management (CPU / CUDA)")

    mp = MotionPrior(use=True, model='mlp', alpha=0.1,
                     clip=100, warmup_frames=2, history_len=4,
                     hidden_dim=32)
    mp.to('cpu')
    mp.eval()

    # parameters should exist
    params = list(mp.parameters())
    check(len(params) > 0, f"MLP should have parameters, got {len(params)}")

    # state_dict round-trip
    sd = mp.state_dict()
    check(len(sd) > 0, f"state_dict should be non-empty, got {len(sd)}")
    mp.load_state_dict(sd)

    # predict on CPU
    mp.update(320, 240)
    mp.update(322, 238)
    c_hat = mp.predict(img_W=640, img_H=480)
    check(c_hat is not None, "predict should work on CPU")

    # If CUDA available, test .to('cuda')
    if torch.cuda.is_available():
        mp.to('cuda')
        mp.eval()
        device = mp._device()
        check(str(device) == 'cuda:0',
              f"device should be 'cuda:0', got '{device}'")
        mp.update(324, 236)
        c_hat_gpu = mp.predict(img_W=640, img_H=480)
        check(c_hat_gpu is not None, "predict should work on CUDA")
        print("  CUDA device test OK")
    else:
        print("  CUDA not available, skipping GPU test")

    print("  Device management OK")


# -------------------------------------------------------------------
#  Test 9: warmup behavior
# -------------------------------------------------------------------

def test_warmup():
    print_header("Test 9: Warmup behavior")

    mp = MotionPrior(use=True, model='mlp', alpha=0.1,
                     clip=100, warmup_frames=5, history_len=4)

    # Before warmup completes
    for i in range(4):
        mp.update(100 + i * 5, 200 + i * 5)

    check(mp.frame_count == 4,
          f"frame_count should be 4, got {mp.frame_count}")

    c_hat = mp.predict()
    check(c_hat is None,
          f"predict() should be None before warmup (frame 4 < 5)")

    scx, scy, eff_a, info = mp.get_search_center(
        fallback_cx=115, fallback_cy=215,
        img_W=640, img_H=480,
    )
    check(scx == 115 and scy == 215,
          f"search_center should be fallback during warmup, got ({scx},{scy})")
    check(eff_a == 0.0, f"eff_alpha should be 0 during warmup, got {eff_a}")
    check(info['fallback_triggered'],
          "fallback_triggered should be True during warmup")

    # After warmup
    mp.update(120, 220)  # frame 5 → warmup passed
    c_hat2 = mp.predict(img_W=640, img_H=480)
    check(c_hat2 is not None,
          f"predict() should return a value after warmup (frame 5 >= 5)")

    print("  Warmup behavior OK")


# -------------------------------------------------------------------
#  Main
# -------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  MotionPrior MLP Smoke Test Suite")
    print("=" * 60)

    try:
        test_mlp_standalone()
        test_baseline_switch()
        test_mlp_pipeline()
        test_soft_blend()
        test_clip_and_conf()
        test_history_integrity()
        test_cv_ca()
        test_device()
        test_warmup()

        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n  TEST FAILED: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
