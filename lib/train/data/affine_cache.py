"""Read-only affine cache used by E2 motion preprocessing."""

from collections import OrderedDict
from dataclasses import dataclass
import glob
import os
import warnings

import numpy as np


IDENTITY_AFFINE = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


@dataclass
class AffineLookup:
    affine: np.ndarray
    cache_hit: bool
    valid: bool
    fallback_identity: bool
    inlier_ratio: float
    reproj_error: float


class AffineCache:
    """Lazy per-sequence NPZ loader with a small process-local LRU."""

    def __init__(self, root, dataset_name=None, enabled=True, max_sequences=32,
                 fallback="identity"):
        if fallback != "identity":
            raise ValueError("Only AFFINE_CACHE_FALLBACK=identity is supported")
        self.root = os.path.expanduser(str(root or ""))
        self.dataset_name = str(dataset_name or "")
        self.enabled = bool(enabled)
        self.max_sequences = max(1, int(max_sequences))
        self._cache = OrderedDict()
        self._warned = set()

    def _identity(self, cache_hit=False):
        return AffineLookup(
            IDENTITY_AFFINE.copy(), cache_hit, False, True, 0.0, 0.0)

    @staticmethod
    def _safe_name(name):
        return str(name).replace("\\", "/").strip("/")

    def _candidate_paths(self, seq_name):
        seq_name = self._safe_name(seq_name)
        names = [self.dataset_name]
        lower = self.dataset_name.lower()
        if lower not in names:
            names.append(lower)
        candidates = [
            os.path.join(self.root, name, seq_name + ".npz")
            for name in names if name
        ]
        candidates.append(os.path.join(self.root, seq_name + ".npz"))
        recursive = glob.glob(
            os.path.join(self.root, "*", seq_name + ".npz"))
        if len(recursive) == 1:
            candidates.extend(recursive)
        return candidates

    def _warn_once(self, key, message):
        if key not in self._warned:
            warnings.warn(message, RuntimeWarning, stacklevel=3)
            self._warned.add(key)

    def _load(self, seq_name):
        key = self._safe_name(seq_name)
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value

        path = next((p for p in self._candidate_paths(key) if os.path.isfile(p)), None)
        if path is None:
            self._warn_once(
                ("missing", key),
                f"Affine cache missing for dataset={self.dataset_name!r}, "
                f"sequence={key!r}; using identity.")
            return None

        try:
            with np.load(path, allow_pickle=False) as data:
                entry = {
                    "affines": np.asarray(data["affines"], dtype=np.float32),
                    "valid": np.asarray(data["valid"], dtype=bool),
                    "inlier_ratio": np.asarray(data["inlier_ratio"], dtype=np.float32),
                    "reproj_error": np.asarray(data["reproj_error"], dtype=np.float32),
                    "fallback_identity": np.asarray(
                        data["fallback_identity"], dtype=bool),
                    "frame_ids": np.asarray(data["frame_ids"]),
                }
            entry["index"] = {
                self._frame_key(frame_id): idx
                for idx, frame_id in enumerate(entry["frame_ids"])
            }
            if entry["affines"].shape != (len(entry["frame_ids"]) - 1, 2, 3):
                raise ValueError("affines shape does not match frame_ids")
        except Exception as exc:
            self._warn_once(
                ("invalid", key),
                f"Invalid affine cache {path}: {exc}; using identity.")
            return None

        self._cache[key] = entry
        while len(self._cache) > self.max_sequences:
            self._cache.popitem(last=False)
        return entry

    @staticmethod
    def _frame_key(frame_id):
        if isinstance(frame_id, np.generic):
            frame_id = frame_id.item()
        return str(frame_id)

    @staticmethod
    def _homogeneous(affine):
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2] = affine
        return matrix

    def get_affine(self, seq_name, frame_a, frame_b):
        if not self.enabled:
            return self._identity()
        entry = self._load(seq_name)
        if entry is None:
            return self._identity()

        index = entry["index"]
        try:
            idx_a = index[self._frame_key(frame_a)]
            idx_b = index[self._frame_key(frame_b)]
        except KeyError:
            self._warn_once(
                ("frame", self._safe_name(seq_name), frame_a, frame_b),
                f"Frame id not found in affine cache for {seq_name}: "
                f"{frame_a}->{frame_b}; using identity.")
            return self._identity()

        if idx_a == idx_b:
            return AffineLookup(
                IDENTITY_AFFINE.copy(), True, True, False, 1.0, 0.0)

        try:
            reverse = idx_b < idx_a
            lo, hi = sorted((idx_a, idx_b))
            matrices = entry["affines"][lo:hi]
            valid = entry["valid"][lo:hi]
            fallback = entry["fallback_identity"][lo:hi]
            inlier = entry["inlier_ratio"][lo:hi]
            reproj = entry["reproj_error"][lo:hi]

            composed = np.eye(3, dtype=np.float64)
            for affine in matrices:
                composed = self._homogeneous(affine) @ composed
            if reverse:
                composed = np.linalg.inv(composed)
            if not np.isfinite(composed).all():
                raise ValueError("non-finite composed transform")

            degraded = bool((~valid).any() or fallback.any())
            good = valid & ~fallback
            return AffineLookup(
                composed[:2].astype(np.float32),
                True,
                bool(valid.all()),
                degraded,
                float(inlier[good].mean()) if good.any() else 0.0,
                float(reproj[good].mean()) if good.any() else 0.0,
            )
        except Exception as exc:
            self._warn_once(
                ("compose", self._safe_name(seq_name), frame_a, frame_b),
                f"Affine compose failed for {seq_name} {frame_a}->{frame_b}: "
                f"{exc}; using identity.")
            return self._identity(cache_hit=True)


def get_affine(cache, seq_name, frame_a, frame_b):
    """Convenience API matching the requested cache lookup signature."""
    return cache.get_affine(seq_name, frame_a, frame_b)
