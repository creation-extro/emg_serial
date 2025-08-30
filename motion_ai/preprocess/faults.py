from __future__ import annotations

import numpy as np
from typing import Dict, Any


def inject_faults(arr: np.ndarray, fs: float, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Inject synthetic faults for resilience testing.
    cfg options (all optional):
      - enabled: bool (default False)
      - noise_sigma: float (additive Gaussian noise amplitude, default 0.0)
      - sample_dropout_ratio: float in [0..1] random samples set to zero (default 0.0)
      - channel_dropout_prob: float in [0..1] probability to zero an entire channel (default 0.0)
      - roll_jitter_max_samples: int max |shift| samples to roll per channel (default 0)
    """
    if not cfg or not bool(cfg.get("enabled", False)):
        return arr

    x = np.array(arr, dtype=float, copy=True)
    # x may be 1D or flattened; this function expects 1D or (n_channels, n_samples)
    n = x.size

    # If channels and samples are provided, try to reshape
    n_channels = int(cfg.get("n_channels", 0) or 0)
    n_samples = int(cfg.get("n_samples", 0) or 0)
    if n_channels > 0 and n_samples > 0 and (n_channels * n_samples == n):
        X = x.reshape(n_channels, n_samples)
    else:
        # Treat as 1D single-channel
        X = x.reshape(1, -1)

    # 1) Additive Gaussian noise
    sigma = float(cfg.get("noise_sigma", 0.0) or 0.0)
    if sigma > 0.0:
        X = X + np.random.normal(0.0, sigma, size=X.shape)

    # 2) Random sample dropout
    drop_ratio = float(cfg.get("sample_dropout_ratio", 0.0) or 0.0)
    if drop_ratio > 0.0:
        m = X.shape[1]
        k = int(drop_ratio * m)
        if k > 0:
            idx = np.random.choice(m, size=k, replace=False)
            X[:, idx] = 0.0

    # 3) Channel dropout
    ch_drop_p = float(cfg.get("channel_dropout_prob", 0.0) or 0.0)
    if ch_drop_p > 0.0:
        for i in range(X.shape[0]):
            if np.random.rand() < ch_drop_p:
                X[i, :] = 0.0

    # 4) Jitter via circular roll
    max_roll = int(cfg.get("roll_jitter_max_samples", 0) or 0)
    if max_roll > 0:
        for i in range(X.shape[0]):
            r = np.random.randint(-max_roll, max_roll + 1)
            if r != 0:
                X[i, :] = np.roll(X[i, :], r)

    # Return flattened to original shape
    return X.reshape(-1)