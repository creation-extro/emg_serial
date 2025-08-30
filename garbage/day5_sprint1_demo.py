#!/usr/bin/env python3
"""
Day 5 — Sprint-1 Demo

End-to-end demo pipeline:
  CSV stream -> classifier -> policy+safety -> actuator sim + haptic triggers
  - Live console updates (optional live plot)
  - Logs saved to out_dir (CSV + JSON summary)
  - Reproducible plots: predictions over time, motor angles, latency histogram, safety events

Usage example:
  python day5_sprint1_demo.py \
    --csv data/combined_emg_data.csv \
    --model motion_models/mlp_light.joblib \
    --out_dir demo_logs \
    --fs 1000 --window_ms 200 --hop_ms 100 --confidence_threshold 0.6

This script runs in-process without HTTP for robustness and reproducibility.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse light MLP trainer utils for consistent features
from motion_ai.classifiers.mlp_light import (
    TrainConfig,
    window_indices,
    load_dataset,
    infer_emg_matrix,
    infer_label_series,
    predict_window_with_confidence,
)

# Policy + safety layer
from motion_ai.control.safety_layer import (
    SafetyGuard,
    SafetyConfig,
    map_gesture_to_commands,
    build_haptic_alerts,
)


@dataclass
class DemoConfig:
    fs: float = 1000.0
    window_ms: int = 200
    hop_ms: int = 100
    confidence_threshold: float = 0.6
    out_dir: str = "demo_logs"
    live_plot: bool = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def label_to_index(label: str) -> int:
    order = ["rest", "open", "fist", "step", "lean"]
    lab = (label or "").lower().strip().replace("-", "_").replace(" ", "_")
    try:
        return order.index(lab)
    except ValueError:
        order.append(lab)
        return len(order) - 1


def run_demo(csv_paths: List[str], model_path: str, cfg: DemoConfig) -> Dict[str, Any]:
    # Prepare output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.out_dir, f"day5_run_{ts}")
    ensure_dir(run_dir)

    # Load dataset and model bundle
    df = load_dataset(csv_paths)
    emg = infer_emg_matrix(df)  # shape: (n_channels, n_samples)
    labels = None
    try:
        labels = infer_label_series(df)  # per-sample if available
    except Exception:
        labels = None

    bundle = joblib.load(model_path)
    # Validate expected keys
    for k in ("clf", "scaler", "features"):
        if k not in bundle:
            raise RuntimeError(f"Model bundle missing key: {k}")

    # Windowing
    win = int(cfg.fs * cfg.window_ms / 1000.0)
    hop = int(cfg.fs * cfg.hop_ms / 1000.0)
    idxs = window_indices(emg.shape[1], win, hop)

    # Safety guard (stateful) and actuator sim state
    guard = SafetyGuard(SafetyConfig())
    actuators: Dict[str, Dict[str, Any]] = {}

    def update_actuator_state(cmds: List[Dict[str, Any]]) -> None:
        for c in cmds:
            aid = c.get("actuator_id", "unknown")
            st = actuators.setdefault(aid, {
                "angles": [],
                "forces": [],
            })
            st["angles"].append(c.get("angle"))
            st["forces"].append(c.get("force"))

    # Logging structures
    rows: List[Dict[str, Any]] = []
    safety_counts = {
        "deadzone_applied": 0,
        "hysteresis_applied": 0,
        "rate_clamped": 0,
        "haptic_alerts": 0,
    }

    # Optional live plotting
    if cfg.live_plot:
        plt.ion()
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        pred_line, = axes[0].plot([], [], label="pred")
        exp_line, = axes[0].plot([], [], label="expected")
        axes[0].set_title("Predictions over windows")
        axes[0].legend()
        lat_line, = axes[1].plot([], [], label="latency_ms")
        axes[1].set_title("Latency (ms)")
        axes[1].legend()
        angle_ax = axes[2]
        angle_ax.set_title("Motor angles (deg)")
        fig.tight_layout()

    pred_idx: List[int] = []
    exp_idx: List[int] = []
    latency_ms_list: List[float] = []
    window_ids: List[int] = []

    # Stream windows
    for w_id, (a, b) in enumerate(idxs, start=1):
        # Prepare df_window for feature extraction used by MLP bundle
        n_ch = emg.shape[0]
        df_window = pd.DataFrame({f"ch{i+1}": emg[i, a:b] for i in range(n_ch)})

        # Expected label via window majority, if available
        expected = None
        if labels is not None and len(labels) == emg.shape[1]:
            try:
                mode = labels.iloc[a:b].mode()
                expected = str(mode.iloc[0]) if len(mode) > 0 else None
            except Exception:
                expected = None

        # Classification + policy + safety
        t0 = time.perf_counter()
        gesture, confidence = predict_window_with_confidence(model_path, df_window, cfg.fs)
        effective_gesture = gesture if confidence >= cfg.confidence_threshold else "rest"

        # Build intent-like object with optional synthetic haptic triggers for demo
        # Simulate a missed grip 1 in 25 windows when confident fist
        features: Dict[str, Any] = {
            "raw_gesture": gesture,
            "window_start": a,
            "window_end": b,
        }
        if (effective_gesture.lower() in ("fist",) and confidence >= 0.6 and (w_id % 25 == 0)):
            features["grip_contact"] = False  # triggers missed_grip haptic

        raw_specs = map_gesture_to_commands({"gesture": effective_gesture, "features": features})
        haptic_specs = build_haptic_alerts({"gesture": effective_gesture, "confidence": confidence, "features": features})
        all_specs = raw_specs + haptic_specs

        safe_cmds: List[Dict[str, Any]] = []
        now = time.time()
        for spec in all_specs:
            safe_spec = guard.apply_to_spec(spec, now=now)
            safe_cmds.append(safe_spec)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Update sim state
        update_actuator_state(safe_cmds)

        # Aggregate safety flags
        for c in safe_cmds:
            flags = c.get("safety_flags", {}) or {}
            if flags.get("deadzone_applied"):
                safety_counts["deadzone_applied"] += 1
            if flags.get("hysteresis_applied"):
                safety_counts["hysteresis_applied"] += 1
            if c.get("rate_clamped"):
                safety_counts["rate_clamped"] += 1
            if c.get("haptic_alert") is not None:
                safety_counts["haptic_alerts"] += 1

        # Log a primary row
        row: Dict[str, Any] = {
            "window_id": w_id,
            "start_idx": a,
            "end_idx": b,
            "expected": expected,
            "predicted": gesture,
            "effective_gesture": effective_gesture,
            "confidence": confidence,
            "latency_ms": elapsed_ms,
        }
        # Include first motor command summary (if any)
        if safe_cmds:
            c0 = safe_cmds[0]
            row.update({
                "actuator_id": c0.get("actuator_id"),
                "angle": c0.get("angle"),
                "force": c0.get("force"),
                "rate_clamped": bool(c0.get("rate_clamped")),
                "deadzone_applied": bool((c0.get("safety_flags") or {}).get("deadzone_applied")),
                "hysteresis_applied": bool((c0.get("safety_flags") or {}).get("hysteresis_applied")),
                "haptic_alert": c0.get("haptic_alert"),
            })
        rows.append(row)

        # Metrics arrays
        pred_idx.append(label_to_index(effective_gesture))
        exp_idx.append(label_to_index(expected) if expected is not None else -1)
        latency_ms_list.append(elapsed_ms)
        window_ids.append(w_id)

        # Live plot update
        if cfg.live_plot and (w_id % 5 == 0 or w_id == 1):
            pred_line.set_data(window_ids, pred_idx)
            if any(x >= 0 for x in exp_idx):
                exp_vals = [x if x >= 0 else np.nan for x in exp_idx]
                exp_line.set_data(window_ids, exp_vals)
                axes = plt.gcf().axes
                axes[0].relim(); axes[0].autoscale_view()
            lat_line.set_data(window_ids, latency_ms_list)
            axes = plt.gcf().axes
            axes[1].relim(); axes[1].autoscale_view()
            # angles: plot last 200 windows for each actuator
            axes[2].clear()
            for aid, st in actuators.items():
                angles = [x if isinstance(x, (int, float)) else np.nan for x in st["angles"]]
                axes[2].plot(range(len(angles)), angles, label=aid)
            axes[2].legend()
            plt.pause(0.001)

    # Save logs
    df_log = pd.DataFrame(rows)
    csv_path = os.path.join(run_dir, "demo_log.csv")
    df_log.to_csv(csv_path, index=False)

    # Compute summary
    n = len(df_log)
    if n == 0:
        raise RuntimeError("No windows processed")

    # Accuracy
    if df_log["expected"].notna().any():
        cmp = df_log.dropna(subset=["expected"]).copy()
        cmp["correct"] = (cmp["effective_gesture"].astype(str).str.lower() == cmp["expected"].astype(str).str.lower())
        accuracy = float(cmp["correct"].mean()) if len(cmp) else None
    else:
        accuracy = None

    # Latency stats
    lat = df_log["latency_ms"].values
    latency_stats = {
        "mean_ms": float(np.mean(lat)),
        "median_ms": float(np.median(lat)),
        "p90_ms": float(np.percentile(lat, 90)),
        "p95_ms": float(np.percentile(lat, 95)),
        "max_ms": float(np.max(lat)),
    }

    # Safety summary
    safety_summary = dict(safety_counts)

    # Save summary JSON
    summary = {
        "n_windows": n,
        "accuracy": accuracy,
        "latency": latency_stats,
        "safety": safety_summary,
        "confidence_threshold": cfg.confidence_threshold,
        "fs": cfg.fs,
        "window_ms": cfg.window_ms,
        "hop_ms": cfg.hop_ms,
        "model_path": os.path.abspath(model_path),
        "csv_paths": [os.path.abspath(p) for p in csv_paths],
        "run_dir": os.path.abspath(run_dir),
        "timestamp": ts,
    }
    json_path = os.path.join(run_dir, "sprint1_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots (reproducible)
    try:
        # Predictions over time
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(window_ids, pred_idx, label="pred")
        if any(x >= 0 for x in exp_idx):
            exp_vals = [x if x >= 0 else np.nan for x in exp_idx]
            ax1.plot(window_ids, exp_vals, label="expected", alpha=0.6)
        ax1.set_title("Predictions over windows")
        ax1.set_xlabel("window id")
        ax1.set_ylabel("class idx")
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig(os.path.join(run_dir, "predictions_over_time.png"))
        plt.close(fig1)

        # Motor angles
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        for aid, st in actuators.items():
            angles = [x if isinstance(x, (int, float)) else np.nan for x in st["angles"]]
            ax2.plot(range(len(angles)), angles, label=aid)
        ax2.set_title("Motor angles over time")
        ax2.set_xlabel("window index")
        ax2.set_ylabel("angle (deg)")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(os.path.join(run_dir, "motor_angles.png"))
        plt.close(fig2)

        # Latency histogram
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.hist(latency_ms_list, bins=30, color="#5DADE2", alpha=0.9)
        ax3.set_title("Latency (ms)")
        ax3.set_xlabel("ms")
        ax3.set_ylabel("count")
        fig3.tight_layout()
        fig3.savefig(os.path.join(run_dir, "latency_hist.png"))
        plt.close(fig3)

        # Safety events bar
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        keys = list(safety_summary.keys())
        vals = [safety_summary[k] for k in keys]
        ax4.bar(keys, vals, color="#58D68D")
        ax4.set_title("Safety events")
        for i, v in enumerate(vals):
            ax4.text(i, v + max(1, 0.02 * max(vals + [1])), str(v), ha='center', va='bottom')
        fig4.tight_layout()
        fig4.savefig(os.path.join(run_dir, "safety_events.png"))
        plt.close(fig4)
    except Exception as e:
        print(f"Plotting error (skipped): {e}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 5 Sprint-1 Demo: CSV -> classifier -> policy+safety -> actuator sim")
    parser.add_argument("--csv", nargs="+", required=True, help="CSV file paths")
    parser.add_argument("--model", required=True, help="Model bundle path (joblib)")
    parser.add_argument("--out_dir", default="demo_logs", help="Output_dir for logs and plots")
    parser.add_argument("--fs", type=float, default=1000.0)
    parser.add_argument("--window_ms", type=int, default=200)
    parser.add_argument("--hop_ms", type=int, default=100)
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--live_plot", action="store_true", help="Enable live matplotlib plots during run")
    args = parser.parse_args()

    cfg = DemoConfig(
        fs=args.fs,
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        confidence_threshold=args.confidence_threshold,
        out_dir=args.out_dir,
        live_plot=args.live_plot,
    )

    summary = run_demo(args.csv, args.model, cfg)

    # Print textual report
    print("\n===== Sprint-1 Report =====")
    print(f"Windows: {summary['n_windows']}")
    print(f"Accuracy: {summary['accuracy'] if summary['accuracy'] is not None else 'N/A (no labels)'}")
    lat = summary['latency']
    print(f"Latency (ms): mean={lat['mean_ms']:.2f}, p95={lat['p95_ms']:.2f}, max={lat['max_ms']:.2f}")
    saf = summary['safety']
    print(f"Safety events: deadzone={saf['deadzone_applied']}, hysteresis={saf['hysteresis_applied']}, rate_clamped={saf['rate_clamped']}, haptic={saf['haptic_alerts']}")
    print(f"Artifacts: {summary['run_dir']}")


if __name__ == "__main__":
    main()
