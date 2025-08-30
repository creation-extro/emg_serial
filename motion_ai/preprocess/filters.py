from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1.0:
        high = 0.999
    if low <= 0:
        low = 1e-6
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass(x: np.ndarray, fs: float, low: float = 20.0, high: float = 450.0, order: int = 4) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter to a 1D or 2D signal.
    - x: shape (..., n_samples). If 2D, it is (n_channels, n_samples)
    - fs: sampling frequency (Hz)
    - low: low cutoff (Hz)
    - high: high cutoff (Hz)
    - order: filter order
    """
    b, a = butter_bandpass(low, high, fs, order)
    if x.ndim == 1:
        return filtfilt(b, a, x)
    elif x.ndim == 2:
        return np.vstack([filtfilt(b, a, ch) for ch in x])
    else:
        raise ValueError("x must be 1D or 2D")


def apply_notch(x: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """
    Apply an IIR notch filter at the specified mains frequency (50/60 Hz).
    - x: 1D or 2D array (..., n_samples)
    - fs: sampling frequency
    - freq: notch frequency (50.0 or 60.0 typical)
    - q: quality factor (higher -> narrower notch)
    """
    w0 = freq / (fs / 2.0)
    if w0 >= 1.0:
        # If fs too low relative to freq, skip notch
        return x.copy()
    b, a = iirnotch(w0, q)
    if x.ndim == 1:
        return filtfilt(b, a, x)
    elif x.ndim == 2:
        return np.vstack([filtfilt(b, a, ch) for ch in x])
    else:
        raise ValueError("x must be 1D or 2D")


def clean_emg(
    x: np.ndarray,
    fs: float,
    mains_freq: float = 50.0,
    band_low: float = 20.0,
    band_high: float = 450.0,
    band_order: int = 4,
    notch_q: float = 30.0,
) -> np.ndarray:
    """
    Convenience: apply bandpass then notch.
    x may be shape (n_channels, n_samples) or (n_samples,).
    """
    y = apply_bandpass(x, fs, band_low, band_high, band_order)
    y = apply_notch(y, fs, mains_freq, notch_q)
    return y
