#!/usr/bin/env python3
"""
Motion AI Demo Script

This script demonstrates the Motion AI system by processing EMG data from a CSV file
and showing the classification results, safety features, and adaptation mechanisms.

Usage:
  python demo/run_motion_ai.py --csv demo/sample.csv
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

# Import motion_ai components
from motion_ai.classifiers.mlp_light import (
    TrainConfig,
    window_indices,
    load_dataset,
    infer_emg_matrix,
    infer_label_series,
    predict_window_with_confidence,
)

# Import safety layer
from motion_ai.control.safety_layer import (
    SafetyGuard,
    SafetyConfig,
    map_gesture_to_commands,
    build_haptic_alerts,
)

# Import adaptation components
from motion_ai.preprocess.adaptation import (
    OnlineRMSAdapter,
    SimpleDriftDetector,
    AdaptationLogger,
)

# Import fault injection
from motion_ai.preprocess.faults import inject_faults


@dataclass
class DemoConfig:
    fs: float = 1000.0  # Sample rate in Hz
    window_ms: int = 200  # Window size in ms
    hop_ms: int = 100  # Hop size in ms
    confidence_threshold: float = 0.6  # Confidence threshold for classification
    out_dir: str = "demo_logs"  # Output directory for logs
    live_plot: bool = False  # Enable live plotting
    inject_faults: bool = False  # Enable fault injection
    fault_config: Dict[str, Any] = None  # Fault injection configuration


def ensure_dir(path: str) -> None:
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def label_to_index(label: Optional[str]) -> int:
    """Convert gesture label to numeric index for plotting"""
    if label is None:
        return -1
    
    # Map of gesture names to indices
    gesture_map = {
        "rest": 0,
        "open": 1,
        "fist": 2,
        "pinch": 3,
        "point": 4,
        "four": 5,
        "five": 6,
        "peace": 7,
        "thumbs_up": 8,
        "hook_grip": 9,
        "flat_palm": 10,
        "ok_sign": 11,
        # Add mappings for dataset-specific labels
        "RELAX": 0,
        "0-OPEN": 1,
        "1-CLOSE": 2,
        "2-PINCH": 3,
        "3-POINT": 4,
        "4-FOUR": 5,
        "5-FIVE": 6,
        "6-PEACE": 7,
        "7-THUMBS_UP": 8,
        "8-HOOK_GRIP": 9,
        "9-FLAT_PALM": 10,
        "10-OK_SIGN": 11,
    }
    return gesture_map.get(label, -1)


def run_demo(csv_paths: List[str], model_path: str, cfg: DemoConfig) -> Dict[str, Any]:
    # Prepare output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.out_dir, f"motion_ai_run_{ts}")
    ensure_dir(run_dir)

    # Load dataset and model bundle
    df = load_dataset(csv_paths)
    emg = infer_emg_matrix(df)  # shape: (n_channels, n_samples)
    labels = None
    try:
        labels = infer_label_series(df)  # per-sample if available
    except Exception:
        labels = None

    # Load model bundle
    try:
        bundle = joblib.load(model_path)
        # Validate expected keys
        for k in ("clf", "scaler", "features"):
            if k not in bundle:
                raise RuntimeError(f"Model bundle missing key: {k}")
    except Exception as e:
        # If model not found, try to find a model in the best_model directory
        print(f"Error loading model {model_path}: {e}")
        print("Looking for alternative models...")
        alt_models = [
            "best_model/ar_random_forest_model.pkl",
            "best_model/improved_deep_learning_emg_model.h5",
        ]
        for alt_model in alt_models:
            if os.path.exists(alt_model):
                print(f"Found alternative model: {alt_model}")
                model_path = alt_model
                bundle = joblib.load(model_path)
                break
        else:
            raise RuntimeError(f"No suitable model found. Please train a model first.")

    # Windowing
    win = int(cfg.fs * cfg.window_ms / 1000.0)
    hop = int(cfg.fs * cfg.hop_ms / 1000.0)
    idxs = window_indices(emg.shape[1], win, hop)

    # Initialize adaptation components
    adapter = OnlineRMSAdapter()
    drift_detector = SimpleDriftDetector()
    adaptation_logger = AdaptationLogger(os.path.join(run_dir, "adaptation_events.csv"))

    # Safety guard (stateful) and actuator sim state
    guard = SafetyGuard(SafetyConfig())
    actuators: Dict[str, Dict[str, Any]] = {}

    # Logging structures
    rows: List[Dict[str, Any]] = []
    safety_counts = {
        "deadzone_applied": 0,
        "hysteresis_applied": 0,
        "rate_clamped": 0,
        "haptic_alerts": 0,
        "drift_detected": 0,
        "baseline_updates": 0,
        "fault_injections": 0,
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
        
        # Apply fault injection if enabled
        window_emg = emg[:, a:b].copy()
        if cfg.inject_faults and cfg.fault_config:
            window_emg = inject_faults(window_emg, cfg.fault_config)
            safety_counts["fault_injections"] += 1
            
        df_window = pd.DataFrame({f"ch{i+1}": window_emg[i, :] for i in range(n_ch)})

        # Expected label via window majority, if available
        expected = None
        if labels is not None and len(labels) == emg.shape[1]:
            try:
                mode = labels.iloc[a:b].mode()
                expected = str(mode.iloc[0]) if len(mode) > 0 else None
            except Exception:
                expected = None

        # Process window through adaptation
        rms_values = np.sqrt(np.mean(np.square(window_emg), axis=1))
        mean_rms = np.mean(rms_values)
        
        # Convert numpy array to dictionary for adapter
        rms_dict = {f'ch{i}': float(val) for i, val in enumerate(rms_values)}
        
        # Update RMS baseline and get adaptive cutoffs
        adapt_info = adapter.update(rms_dict)
        
        # Check for drift
        drift_active, z, drift_change = drift_detector.update(mean_rms)
        if drift_active:
            safety_counts["drift_detected"] += 1
        
        # Log adaptation events
        if adapt_info.get("updated", False):
            safety_counts["baseline_updates"] += 1
            adaptation_logger.log_event(
                "baseline_update",
                {
                    "mean_baseline": adapt_info.get("mean_baseline", 0.0),
                    "low_cutoff_mean": adapt_info.get("low_cutoff_mean", 0.0),
                    "high_cutoff_mean": adapt_info.get("high_cutoff_mean", 0.0),
                    "n_updates": adapt_info.get("n_updates", 0),
                }
            )
        
        if drift_change is not None:
            adaptation_logger.log_event(
                "drift_" + drift_change,
                {
                    "z_score": z,
                    "mean_rms": mean_rms,
                }
            )

        # Measure latency
        t0 = time.perf_counter()

        # Predict gesture with confidence
        gesture, confidence = predict_window_with_confidence(model_path, df_window, cfg.fs)

        # Apply confidence threshold
        effective_gesture = gesture if confidence >= cfg.confidence_threshold else "rest"

        # Map gesture to motor commands
        cmds = map_gesture_to_commands(effective_gesture)

        # Apply safety layer
        safe_cmds = [guard.apply_to_spec(cmd, time.time()) for cmd in cmds]

        # Update actuator state
        for cmd in safe_cmds:
            actuator_id = cmd.get("actuator_id")
            if actuator_id:
                if actuator_id not in actuators:
                    actuators[actuator_id] = {}
                actuators[actuator_id].update({
                    "angle": cmd.get("angle"),
                    "force": cmd.get("force"),
                })

        # Measure elapsed time
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Count safety events
        flags = {}
        if safe_cmds:
            flags = safe_cmds[0].get("safety_flags", {})
            if flags.get("deadzone_applied"):
                safety_counts["deadzone_applied"] += 1
            if flags.get("hysteresis_applied"):
                safety_counts["hysteresis_applied"] += 1
            if safe_cmds[0].get("rate_clamped"):
                safety_counts["rate_clamped"] += 1
            if safe_cmds[0].get("haptic_alert") is not None:
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
            "drift_active": drift_active,
            "drift_z_score": z,
            "mean_rms": mean_rms,
            "rms_baseline": adapt_info.get("mean_baseline", 0.0),
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

        # Live console update
        if w_id % 10 == 0 or w_id == len(idxs):
            print(f"Window {w_id}/{len(idxs)}: {gesture} ({confidence:.2f}) -> {effective_gesture}")
            if drift_active:
                print(f"  ⚠️ DRIFT DETECTED: z-score = {z:.2f}")
            if adapt_info.get("updated", False):
                print(f"  📊 Baseline updated: {adapt_info.get('mean_baseline', 0.0):.6f}")

        # Optional live plot update
        if cfg.live_plot and (w_id % 5 == 0 or w_id == len(idxs)):
            pred_line.set_data(window_ids, pred_idx)
            if any(x >= 0 for x in exp_idx):
                exp_vals = [x if x >= 0 else np.nan for x in exp_idx]
                exp_line.set_data(window_ids, exp_vals)
            lat_line.set_data(window_ids, latency_ms_list)

            for ax in (axes[0], axes[1]):
                ax.relim()
                ax.autoscale_view()

            # Plot actuator angles
            angle_ax.clear()
            angle_ax.set_title("Motor angles (deg)")
            angles = [actuators.get(aid, {}).get("angle", 0) for aid in sorted(actuators.keys())]
            if angles:
                angle_ax.bar(range(len(angles)), angles)
                angle_ax.set_xticks(range(len(angles)))
                angle_ax.set_xticklabels(sorted(actuators.keys()))
            plt.pause(0.01)

    # Save logs
    df_log = pd.DataFrame(rows)
    log_path = os.path.join(run_dir, "motion_ai_log.csv")
    df_log.to_csv(log_path, index=False)

    # Calculate accuracy if labels available
    n = len(pred_idx)
    accuracy = None
    if labels is not None and any(x >= 0 for x in exp_idx):
        valid_idx = [i for i, x in enumerate(exp_idx) if x >= 0]
        if valid_idx:
            accuracy = sum(pred_idx[i] == exp_idx[i] for i in valid_idx) / len(valid_idx)

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
    json_path = os.path.join(run_dir, "motion_ai_report.json")
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
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "predictions.png"))

        # Latency histogram
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.hist(lat, bins=30)
        ax2.set_title("Latency distribution")
        ax2.set_xlabel("latency (ms)")
        ax2.set_ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "latency.png"))

        # Motor angles
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        angles = [actuators.get(aid, {}).get("angle", 0) for aid in sorted(actuators.keys())]
        if angles:
            ax3.bar(range(len(angles)), angles)
            ax3.set_xticks(range(len(angles)))
            ax3.set_xticklabels(sorted(actuators.keys()))
        ax3.set_title("Final motor angles")
        ax3.set_ylabel("angle (deg)")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "motor_angles.png"))

        # Adaptation metrics
        if "mean_rms" in df_log.columns and "rms_baseline" in df_log.columns:
            fig4, ax4 = plt.subplots(figsize=(10, 3))
            ax4.plot(df_log["window_id"], df_log["mean_rms"], label="Mean RMS")
            ax4.plot(df_log["window_id"], df_log["rms_baseline"], label="RMS Baseline")
            if "drift_z_score" in df_log.columns:
                ax4_twin = ax4.twinx()
                ax4_twin.plot(df_log["window_id"], df_log["drift_z_score"], 'r-', label="Z-Score")
                ax4_twin.set_ylabel("Z-Score")
            ax4.set_title("Adaptation Metrics")
            ax4.set_xlabel("Window ID")
            ax4.set_ylabel("RMS Value")
            ax4.legend(loc="upper left")
            if "drift_z_score" in df_log.columns:
                ax4_twin.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "adaptation.png"))

    except Exception as e:
        print(f"Error generating plots: {e}")

    return summary


def generate_metrics_card(summary: Dict[str, Any], output_path: str) -> None:
    """Generate a metrics card with key performance indicators"""
    metrics_md = f"""# Motion AI Metrics Card

## Classification Performance

- **Accuracy**: {summary['accuracy'] if summary['accuracy'] is not None else 'N/A (no labels)'}
- **Confidence Threshold**: {summary['confidence_threshold']}

## Latency

- **Mean**: {summary['latency']['mean_ms']:.2f} ms
- **Median**: {summary['latency']['median_ms']:.2f} ms
- **P95**: {summary['latency']['p95_ms']:.2f} ms
- **Max**: {summary['latency']['max_ms']:.2f} ms

## Safety Coverage

- **Deadzone Applied**: {summary['safety']['deadzone_applied']} times
- **Hysteresis Applied**: {summary['safety']['hysteresis_applied']} times
- **Rate Limiting**: {summary['safety']['rate_clamped']} times
- **Haptic Alerts**: {summary['safety']['haptic_alerts']} times

## Adaptation & Resilience

- **Baseline Updates**: {summary['safety']['baseline_updates']} times
- **Drift Detected**: {summary['safety']['drift_detected']} times
- **Fault Injections**: {summary['safety']['fault_injections']} times

## Configuration

- **Window Size**: {summary['window_ms']} ms
- **Hop Size**: {summary['hop_ms']} ms
- **Sample Rate**: {summary['fs']} Hz
- **Model**: {os.path.basename(summary['model_path'])}
- **Run Timestamp**: {summary['timestamp']}
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(metrics_md)


def main() -> None:
    parser = argparse.ArgumentParser(description="Motion AI Demo: CSV -> classifier -> policy+safety -> metrics")
    parser.add_argument("--csv", nargs="+", required=True, help="CSV file paths")
    parser.add_argument("--model", default="best_model/ar_random_forest_model.pkl", help="Model bundle path (joblib)")
    parser.add_argument("--out_dir", default="demo_logs", help="Output_dir for logs and plots")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sample rate in Hz")
    parser.add_argument("--window_ms", type=int, default=200, help="Window size in ms")
    parser.add_argument("--hop_ms", type=int, default=100, help="Hop size in ms")
    parser.add_argument("--confidence_threshold", type=float, default=0.6, help="Confidence threshold for classification")
    parser.add_argument("--live_plot", action="store_true", help="Enable live matplotlib plots during run")
    parser.add_argument("--inject_faults", action="store_true", help="Enable fault injection")
    args = parser.parse_args()

    # Configure fault injection if enabled
    fault_config = None
    if args.inject_faults:
        fault_config = {
            "noise_sigma": 0.05,  # Add noise with 0.05 standard deviation
            "sample_dropout_ratio": 0.02,  # Drop 2% of samples
            "channel_dropout_prob": 0.01,  # 1% chance of channel dropout
            "roll_jitter_max_samples": 2,  # Add jitter of up to 2 samples
        }

    cfg = DemoConfig(
        fs=args.fs,
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        confidence_threshold=args.confidence_threshold,
        out_dir=args.out_dir,
        live_plot=args.live_plot,
        inject_faults=args.inject_faults,
        fault_config=fault_config,
    )

    # Ensure output directory exists
    ensure_dir(cfg.out_dir)

    # Run the demo
    summary = run_demo(args.csv, args.model, cfg)

    # Generate metrics card
    metrics_path = os.path.join(summary["run_dir"], "metrics_card.md")
    generate_metrics_card(summary, metrics_path)

    # Print textual report
    print("\n===== Motion AI Report =====")
    print(f"Windows: {summary['n_windows']}")
    print(f"Accuracy: {summary['accuracy'] if summary['accuracy'] is not None else 'N/A (no labels)'}")
    lat = summary['latency']
    print(f"Latency (ms): mean={lat['mean_ms']:.2f}, p95={lat['p95_ms']:.2f}, max={lat['max_ms']:.2f}")
    saf = summary['safety']
    print(f"Safety events: deadzone={saf['deadzone_applied']}, hysteresis={saf['hysteresis_applied']}, rate_clamped={saf['rate_clamped']}, haptic={saf['haptic_alerts']}")
    print(f"Adaptation: baseline_updates={saf['baseline_updates']}, drift_detected={saf['drift_detected']}")
    print(f"Artifacts: {summary['run_dir']}")
    print(f"Metrics Card: {metrics_path}")


if __name__ == "__main__":
    main()