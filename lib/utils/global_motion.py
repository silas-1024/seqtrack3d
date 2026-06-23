"""Traditional affine ego-motion estimation for RMP motion preprocessing."""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


IDENTITY_AFFINE = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]], dtype=np.float32)


def _to_gray(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.shape[2] == 1:
        return image[..., 0].astype(np.uint8, copy=False)
    return cv2.cvtColor(image.astype(np.uint8, copy=False), cv2.COLOR_RGB2GRAY)


def _background_mask(image_shape, bbox=None, expansion=1.5):
    mask = np.full(image_shape[:2], 255, dtype=np.uint8)
    if bbox is None:
        return mask

    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0:
        return mask
    cx, cy = x + 0.5 * w, y + 0.5 * h
    half_w, half_h = 0.5 * w * expansion, 0.5 * h * expansion
    x1 = max(0, int(np.floor(cx - half_w)))
    y1 = max(0, int(np.floor(cy - half_h)))
    x2 = min(image_shape[1], int(np.ceil(cx + half_w)))
    y2 = min(image_shape[0], int(np.ceil(cy + half_h)))
    mask[y1:y2, x1:x2] = 0
    return mask


def _fallback_stats(reason):
    return {
        "success": 0.0,
        "fallback": 1.0,
        "inlier_ratio": 0.0,
        "reprojection_error": 0.0,
        "num_matches": 0,
        "reason": reason,
    }


def estimate_affine_motion(prev_image, curr_image, prev_bbox=None, curr_bbox=None,
                           max_features=2000, ratio_threshold=0.75):
    """Estimate an affine transform from the previous frame to the current frame.

    Target regions are excluded from feature detection when boxes are available.
    Any unavailable dependency or unstable estimate returns an identity transform.
    """
    if cv2 is None:
        return IDENTITY_AFFINE.copy(), _fallback_stats("opencv_unavailable")

    try:
        prev_gray = _to_gray(prev_image)
        curr_gray = _to_gray(curr_image)
        prev_mask = _background_mask(prev_gray.shape, prev_bbox)
        curr_mask = _background_mask(curr_gray.shape, curr_bbox)

        detector = cv2.ORB_create(nfeatures=max_features, fastThreshold=7)
        keypoints_prev, descriptors_prev = detector.detectAndCompute(prev_gray, prev_mask)
        keypoints_curr, descriptors_curr = detector.detectAndCompute(curr_gray, curr_mask)
        if descriptors_prev is None or descriptors_curr is None:
            return IDENTITY_AFFINE.copy(), _fallback_stats("missing_descriptors")

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn_matches = matcher.knnMatch(descriptors_prev, descriptors_curr, k=2)
        matches = [first for first, second in knn_matches
                   if first.distance < ratio_threshold * second.distance]
        if len(matches) < 6:
            stats = _fallback_stats("insufficient_matches")
            stats["num_matches"] = len(matches)
            return IDENTITY_AFFINE.copy(), stats

        points_prev = np.float32(
            [keypoints_prev[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
        points_curr = np.float32(
            [keypoints_curr[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
        affine, inliers = cv2.estimateAffine2D(
            points_prev,
            points_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        if affine is None or inliers is None or not np.isfinite(affine).all():
            stats = _fallback_stats("ransac_failed")
            stats["num_matches"] = len(matches)
            return IDENTITY_AFFINE.copy(), stats

        inlier_mask = inliers.reshape(-1).astype(bool)
        if inlier_mask.sum() < 3:
            stats = _fallback_stats("insufficient_inliers")
            stats["num_matches"] = len(matches)
            return IDENTITY_AFFINE.copy(), stats

        affine = affine.astype(np.float32)
        projected = cv2.transform(points_prev[inlier_mask], affine)
        errors = np.linalg.norm(
            projected.reshape(-1, 2) - points_curr[inlier_mask].reshape(-1, 2), axis=1)
        stats = {
            "success": 1.0,
            "fallback": 0.0,
            "inlier_ratio": float(inlier_mask.mean()),
            "reprojection_error": float(errors.mean()) if errors.size else 0.0,
            "num_matches": len(matches),
            "reason": "ok",
        }
        return affine, stats
    except Exception as exc:
        return IDENTITY_AFFINE.copy(), _fallback_stats(type(exc).__name__)


def transform_xywh_box(box, affine, image_size=None):
    """Transform all four corners of a pixel-space top-left xywh box."""
    x, y, w, h = [float(v) for v in box]
    corners = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
    ], dtype=np.float32)
    homogeneous = np.concatenate(
        [corners, np.ones((4, 1), dtype=np.float32)], axis=1)
    transformed = homogeneous @ np.asarray(affine, dtype=np.float32).T
    x1, y1 = transformed.min(axis=0)
    x2, y2 = transformed.max(axis=0)
    if image_size is not None:
        width, height = image_size
        x1, x2 = np.clip([x1, x2], 0.0, float(width))
        y1, y2 = np.clip([y1, y2], 0.0, float(height))
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
