from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd

from motion_ai.classifiers.svm_baseline import TrainConfig, build_feature_table, train_svm
from motion_ai.features.extractors import extract_features_from_window


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluation for SVM baseline")
    parser.add_argument("--csv", nargs="+", required=True, help="CSV files")
    parser.add_argument("--model_out", default="svm_baseline.joblib")
    parser.add_argument("--fs", type=float, default=1000.0)
    parser.add_argument("--window_ms", type=int, default=200)
    parser.add_argument("--hop_ms", type=int, default=100)
    args = parser.parse_args()

    cfg = TrainConfig(fs=args.fs, window_ms=args.window_ms, hop_ms=args.hop_ms)

    # Train and report accuracy
    metrics = train_svm(args.csv, cfg, args.model_out)
    print("ACCURACY:", metrics["accuracy"])
    print(metrics["report"])

    # Latency test: compute features + predict for a sample window
    bundle = joblib.load(args.model_out)
    df = pd.read_csv(args.csv[0])
    n = 500
    # Build a simple window with first n samples and required EMG columns guessed by extractor via ch1,ch2,ch3
    cols = [c for c in df.columns if c in ("ch1", "ch2", "ch3", "raw_emg1", "raw_emg2", "emg1_clean", "emg2_clean", "emg3_clean")]
    if not cols:
        # fall back to first numeric columns
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:3]
    df_window = df[cols].iloc[:n].copy()
    df_window.columns = [f"ch{i+1}" for i in range(len(df_window.columns))]

    start = time.perf_counter()
    feats = extract_features_from_window(df_window, fs=args.fs)
    X = np.array([[feats.get(f, 0.0) for f in bundle["features"]]])
    bundle["scaler"].transform(X)
    latency_ms = (time.perf_counter() - start) * 1000.0
    print(f"LATENCY_ms: {latency_ms:.2f}")
