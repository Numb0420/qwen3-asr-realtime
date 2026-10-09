"""WebSocket pre-roll helpers."""
from __future__ import annotations

import numpy as np


def append_pre_roll(ring: np.ndarray, chunk: np.ndarray, max_samples: int) -> np.ndarray:
    """Unconditionally keep the latest max_samples of raw PCM."""
    samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return np.asarray(ring, dtype=np.float32).reshape(-1)
    if max_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    base = np.asarray(ring, dtype=np.float32).reshape(-1)
    out = samples if base.size == 0 else np.concatenate([base, samples])
    if out.size > max_samples:
        out = out[-max_samples:]
    return out


def unused_pre_roll(ring: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    """Ring already contains `chunk` at the end; return only samples not in this packet."""
    extra = int(np.asarray(ring).size) - int(np.asarray(chunk).size)
    if extra <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(ring, dtype=np.float32).reshape(-1)[:extra].copy()
