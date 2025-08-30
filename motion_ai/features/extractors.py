from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List


def rms(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.mean(np.square(x))))


def mav(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.mean(np.abs(x)))


def fft_peak_power(x: np.ndarray, fs: float) -> float:
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.squeeze()
    # Hann window to reduce spectral leakage
    w = np.hanning(len(x))
    X = np.fft.rfft((x - np.mean(x)) * w)
    Pxx = (np.abs(X) ** 2) / np.sum(w**2)
    peak = float(np.max(Pxx)) if len(Pxx) > 0 else 0.0
    return peak


def accel_peak_count(accel_axis: np.ndarray, threshold: float = None) -> int:
    x = np.asarray(accel_axis)
    if threshold is None:
        threshold = float(np.mean(np.abs(x)) + 2 * np.std(x))
    peaks = np.where(np.abs(x) > threshold)[0]
    return int(len(peaks))


def gyro_mean(gyro_axis: np.ndarray) -> float:
    return float(np.mean(gyro_axis))


EMG_COLS_CANDIDATES: List[List[str]] = [
    ["ch1", "ch2", "ch3"],
    ["raw_emg1", "raw_emg2"],
    ["emg1_clean", "emg2_clean", "emg3_clean"],
]

ACCEL_COLS_CANDIDATES: List[List[str]] = [
    ["accel_x", "accel_y", "accel_z"],
]

GYRO_COLS_CANDIDATES: List[List[str]] = [
    ["gyro_x", "gyro_y", "gyro_z"],
]


def select_first_present(df: pd.DataFrame, candidates: List[List[str]]) -> List[str]:
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols
    return []


def extract_features_from_window(df_window: pd.DataFrame, fs: float) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    emg_cols = select_first_present(df_window, EMG_COLS_CANDIDATES)
    if not emg_cols:
        raise ValueError("No EMG columns found; expected one of candidates")

    # EMG features per channel
    for c in emg_cols:
        x = df_window[c].values
        feats[f"{c}_rms"] = rms(x)
        feats[f"{c}_mav"] = mav(x)
        feats[f"{c}_fft_peak"] = fft_peak_power(x, fs)

    # Optional IMU features
    accel_cols = select_first_present(df_window, ACCEL_COLS_CANDIDATES)
    if accel_cols:
        for c in accel_cols:
            feats[f"{c}_peaks"] = float(accel_peak_count(df_window[c].values))

    gyro_cols = select_first_present(df_window, GYRO_COLS_CANDIDATES)
    if gyro_cols:
        for c in gyro_cols:
            feats[f"{c}_mean"] = gyro_mean(df_window[c].values)

    return feats
