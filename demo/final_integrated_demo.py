#!/usr/bin/env python3
"""
Motion AI Final Integrated Demo

This script demonstrates the full Motion AI system with:
- Live EMG/IMU data processing
- Intent classification
- Policy application
- Actuator simulation
- Haptic feedback
- Profile switching
- Drift detection
- Safety event logging
- Dashboard with Motion AI live feed

Usage:
  python demo/final_integrated_demo.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
import random
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Import Motion AI components
from motion_ai.preprocess.adaptation import OnlineRMSAdapter, SimpleDriftDetector
from motion_ai.preprocess.filters import apply_bandpass
from motion_ai.features.extractors import rms, mav
from motion_ai.control.safety_layer import SafetyGuard, SafetyConfig
# We'll simulate the classifier instead of using the actual one

# Import from run_motion_ai.py
from run_motion_ai import DemoConfig, ensure_dir, generate_metrics_card
from motion_ai.preprocess.adaptation import AdaptationLogger


class SimulatedSensor:
    """Simulates EMG/IMU sensor data stream"""
    
    def __init__(self, fs=1000.0, n_channels=3):
        self.fs = fs
        self.n_channels = n_channels
        self.gestures = ['rest', 'open', 'fist', 'pinch', 'point', 'four', 'five', 'peace']
        self.current_gesture = 'rest'
        self.transition_prob = 0.05  # Probability to change gesture
        self.drift_factor = 1.0
        self.drift_direction = 0.001  # Small drift over time
        self.noise_level = 0.05
        
        # Gesture patterns (simplified) - updated for 3 channels
        self.patterns = {
            'rest': np.array([0.1, 0.1, 0.1]),
            'open': np.array([0.7, 0.3, 0.1]),
            'fist': np.array([0.2, 0.7, 0.3]),
            'pinch': np.array([0.1, 0.5, 0.8]),
            'point': np.array([0.1, 0.3, 0.7]),
            'four': np.array([0.8, 0.2, 0.2]),
            'five': np.array([0.6, 0.6, 0.2]),
            'peace': np.array([0.3, 0.3, 0.8])
        }
    
    def read_frame(self):
        """Generate a single frame of EMG data"""
        # Randomly change gesture with small probability
        if random.random() < self.transition_prob:
            self.current_gesture = random.choice(self.gestures)
        
        # Get base pattern for current gesture
        base_pattern = self.patterns[self.current_gesture]
        
        # Apply drift over time
        self.drift_factor += self.drift_direction
        if self.drift_factor > 1.5 or self.drift_factor < 0.5:
            self.drift_direction *= -1  # Reverse drift direction
        
        # Generate frame with pattern, drift and noise
        frame = base_pattern * self.drift_factor
        frame += np.random.normal(0, self.noise_level, self.n_channels)
        frame = np.clip(frame, 0, 1)  # Ensure values are between 0 and 1
        
        return frame, self.current_gesture


class Actuator:
    """Simulates a prosthetic actuator"""
    
    def __init__(self, actuator_id: str):
        self.actuator_id = actuator_id
        self.current_angle = 0.0
        self.target_angle = 0.0
        self.max_speed = 90.0  # degrees per second
        self.last_update_time = time.time()
    
    def update(self, dt: float):
        """Update actuator position based on elapsed time"""
        if self.current_angle != self.target_angle:
            # Calculate maximum movement in this time step
            max_step = self.max_speed * dt
            step = self.target_angle - self.current_angle
            
            # Limit step to max speed
            if abs(step) > max_step:
                step = max_step if step > 0 else -max_step
            
            self.current_angle += step
    
    def set_target(self, angle: float):
        """Set target angle for the actuator"""
        self.target_angle = max(-90.0, min(90.0, angle))
        
    def get_state(self) -> Dict[str, Any]:
        """Get current actuator state"""
        return {
            'actuator_id': self.actuator_id,
            'current_angle': self.current_angle,
            'target_angle': self.target_angle
        }


class HapticFeedback:
    """Simulates haptic feedback system"""
    
    def __init__(self):
        self.patterns = {
            'alert': [0.8, 0.2, 0.8, 0.2],
            'success': [0.5, 0.0, 0.7, 0.0, 1.0],
            'error': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            'drift_detected': [0.3, 0.0, 0.6, 0.0, 0.9, 0.0, 0.9, 0.0]
        }
        self.current_pattern = None
        self.pattern_index = 0
        self.last_update_time = 0
        self.interval = 0.1  # seconds between haptic pulses
    
    def trigger(self, pattern_name: str):
        """Trigger a haptic feedback pattern"""
        if pattern_name in self.patterns:
            self.current_pattern = self.patterns[pattern_name]
            self.pattern_index = 0
            self.last_update_time = time.time()
            print(f"Haptic feedback: {pattern_name}")
    
    def update(self):
        """Update haptic feedback state"""
        if self.current_pattern is None:
            return 0.0
        
        now = time.time()
        if now - self.last_update_time > self.interval:
            self.pattern_index += 1
            self.last_update_time = now
            
            if self.pattern_index >= len(self.current_pattern):
                self.current_pattern = None
                return 0.0
        
        return self.current_pattern[self.pattern_index] if self.current_pattern else 0.0


class Dashboard:
    """Dashboard for Motion AI live feed"""
    
    def __init__(self):
        # Set up the figure with single subplot
        self.fig, self.ax1 = plt.subplots(1, 1, figsize=(12, 6))
        self.fig.suptitle('Motion AI Dashboard', fontsize=16)
        
        # Motion AI subplot
        self.ax1.set_title('Motion AI Live Feed')
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Signal')
        self.ax1.set_ylim(0, 1.5)
        
        # Initialize lines for EMG channels
        self.n_channels = 3
        self.buffer_size = 500
        self.emg_data = np.zeros((self.n_channels, self.buffer_size))
        self.emg_lines = []
        for i in range(self.n_channels):
            line, = self.ax1.plot([], [], label=f'Ch{i+1}')
            self.emg_lines.append(line)
        
        # Add gesture prediction text
        self.gesture_text = self.ax1.text(0.02, 0.95, '', transform=self.ax1.transAxes)
        self.drift_text = self.ax1.text(0.02, 0.90, '', transform=self.ax1.transAxes)
        self.safety_text = self.ax1.text(0.02, 0.85, '', transform=self.ax1.transAxes)
        
        # Add legend to EMG plot
        self.ax1.legend(loc='upper right')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Initialize x-axis data
        self.x_data = np.arange(self.buffer_size)
    
    def update_emg(self, frame_idx, emg_frame, gesture, drift_active, safety_events):
        """Update EMG visualization with new data"""
        # Roll the buffer and add new data
        self.emg_data = np.roll(self.emg_data, -1, axis=1)
        self.emg_data[:, -1] = emg_frame
        
        # Update EMG lines
        for i, line in enumerate(self.emg_lines):
            line.set_data(self.x_data, self.emg_data[i])
        
        # Update text displays
        self.gesture_text.set_text(f'Gesture: {gesture}')
        self.drift_text.set_text(f'Drift Active: {"Yes" if drift_active else "No"}')
        self.safety_text.set_text(f'Safety Events: {sum(safety_events.values())}')
        
        # Adjust x-axis limits to show scrolling effect
        self.ax1.set_xlim(0, self.buffer_size)
        
        return self.emg_lines + [self.gesture_text, self.drift_text, self.safety_text]
    

    
    def start(self, update_func):
        """Start the dashboard animation"""
        self.ani = FuncAnimation(
            self.fig, update_func, interval=50, blit=True)
        plt.show()


class Profile:
    """User profile with gesture mappings and settings"""
    
    def __init__(self, name, gesture_mappings, sensitivity=1.0):
        self.name = name
        self.gesture_mappings = gesture_mappings
        self.sensitivity = sensitivity
    
    def get_command_for_gesture(self, gesture, confidence):
        """Convert gesture to actuator command based on profile mappings"""
        if gesture in self.gesture_mappings:
            mapping = self.gesture_mappings[gesture]
            # Apply sensitivity to the command
            if 'angle' in mapping:
                mapping['angle'] *= self.sensitivity
            return mapping
        return None


def run_integrated_demo(cfg: DemoConfig):
    """Run the integrated demo with all components"""
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.out_dir, f"motion_ai_run_{timestamp}")
    ensure_dir(run_dir)
    
    # Initialize event logger
    adaptation_logger = AdaptationLogger(os.path.join(run_dir, "adaptation_events.csv"))
    
    # Initialize sensor simulator
    sensor = SimulatedSensor(fs=cfg.fs, n_channels=3)
    
    # Initialize adaptation components
    adapter = OnlineRMSAdapter()
    drift_detector = SimpleDriftDetector(z_thresh=2.0, req_consec=5)
    
    # Initialize safety components
    safety_config = SafetyConfig(
        max_angle_rate_deg_s=90.0,
        dead_zone_angle_deg=1.5,
        dead_zone_force=0.2,
        hysteresis_high_deg=2.0,
        hysteresis_low_deg=1.0,
        max_angle_abs_deg=90.0,
        max_force_abs=5.0,
    )
    guard = SafetyGuard(safety_config)
    
    # Initialize actuators
    actuators = {
        'thumb': Actuator('thumb'),
        'index': Actuator('index'),
        'middle': Actuator('middle'),
        'ring': Actuator('ring'),
        'pinky': Actuator('pinky'),
        'wrist': Actuator('wrist'),
    }
    
    # Initialize haptic feedback
    haptic = HapticFeedback()
    
    # Create user profiles
    profiles = {
        'default': Profile(
            name='default',
            gesture_mappings={
                'rest': {'actuator_id': 'wrist', 'angle': 0.0},
                'open': {'actuator_id': 'wrist', 'angle': 45.0},
                'fist': {'actuator_id': 'wrist', 'angle': -45.0},
                'pinch': {'actuator_id': 'thumb', 'angle': 30.0},
                'point': {'actuator_id': 'index', 'angle': 60.0},
                'four': {'actuator_id': 'middle', 'angle': 45.0},
                'five': {'actuator_id': 'ring', 'angle': 45.0},
                'peace': {'actuator_id': 'pinky', 'angle': 45.0},
            },
            sensitivity=1.0
        ),
        'gaming': Profile(
            name='gaming',
            gesture_mappings={
                'rest': {'actuator_id': 'wrist', 'angle': 0.0},
                'open': {'actuator_id': 'index', 'angle': 30.0},
                'fist': {'actuator_id': 'middle', 'angle': 30.0},
                'pinch': {'actuator_id': 'ring', 'angle': 30.0},
                'point': {'actuator_id': 'pinky', 'angle': 30.0},
                'four': {'actuator_id': 'thumb', 'angle': 60.0},
                'five': {'actuator_id': 'wrist', 'angle': 20.0},
                'peace': {'actuator_id': 'index', 'angle': 45.0},
            },
            sensitivity=1.2
        ),
        'precision': Profile(
            name='precision',
            gesture_mappings={
                'rest': {'actuator_id': 'wrist', 'angle': 0.0},
                'open': {'actuator_id': 'wrist', 'angle': 20.0},
                'fist': {'actuator_id': 'wrist', 'angle': -20.0},
                'pinch': {'actuator_id': 'thumb', 'angle': 15.0},
                'point': {'actuator_id': 'index', 'angle': 30.0},
                'four': {'actuator_id': 'middle', 'angle': 20.0},
                'five': {'actuator_id': 'ring', 'angle': 20.0},
                'peace': {'actuator_id': 'pinky', 'angle': 20.0},
            },
            sensitivity=0.7
        )
    }
    
    # Start with default profile
    current_profile = profiles['default']
    
    # Tracking metrics
    window_count = 0
    latencies = []
    predictions = []
    true_gestures = []
    safety_counts = {
        "deadzone_applied": 0,
        "hysteresis_applied": 0,
        "rate_clamped": 0,
        "haptic_alerts": 0,
        "baseline_updates": 0,
        "drift_detected": 0,
        "fault_injections": 0,
    }
    
    # Create dashboard
    dashboard = Dashboard()
    
    # Function to process a single window of data
    def process_window():
        nonlocal window_count, current_profile
        
        # Read frame from sensor
        start_time = time.time()
        emg_frame, true_gesture = sensor.read_frame()
        
        # Simulate window of data (200ms at 1000Hz = 200 samples)
        window_size = int(cfg.window_ms * cfg.fs / 1000)
        window_emg = np.tile(emg_frame.reshape(-1, 1), (1, window_size))
        
        # Create DataFrame for processing
        df_window = pd.DataFrame({f"ch{i+1}": window_emg[i, :] for i in range(sensor.n_channels)})
        
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
            # Trigger haptic feedback for drift
            if drift_change is not None:
                haptic.trigger('drift_detected')
        
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
            adaptation_logger.log(
                "drift_" + drift_change,
                {
                    "z_score": z,
                    "mean_rms": mean_rms,
                }
            )
        
        # Predict gesture
        # For demo purposes, we'll use the true gesture with some noise
        if random.random() < 0.8:  # 80% accuracy
            predicted_gesture = true_gesture
        else:
            # Pick a random gesture that's not the true one
            other_gestures = [g for g in sensor.gestures if g != true_gesture]
            predicted_gesture = random.choice(other_gestures)
        
        confidence = random.uniform(0.6, 0.95)
        
        # Get command from profile based on gesture
        cmd = current_profile.get_command_for_gesture(predicted_gesture, confidence)
        
        if cmd:
            # Apply safety layer
            safe_cmd = guard.apply_to_spec(cmd, time.time())
            
            # Check safety flags
            if safe_cmd.get('safety_flags', {}).get('deadzone_applied', False):
                safety_counts["deadzone_applied"] += 1
            if safe_cmd.get('safety_flags', {}).get('hysteresis_applied', False):
                safety_counts["hysteresis_applied"] += 1
            if safe_cmd.get('rate_clamped', False):
                safety_counts["rate_clamped"] += 1
            
            # Apply command to actuator
            actuator_id = safe_cmd.get('actuator_id')
            if actuator_id in actuators and 'angle' in safe_cmd:
                actuators[actuator_id].set_target(safe_cmd['angle'])
            
            # Handle haptic alerts
            if safe_cmd.get('haptic_alert'):
                haptic.trigger(safe_cmd['haptic_alert'])
                safety_counts["haptic_alerts"] += 1
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)
        
        # Store prediction and true gesture
        predictions.append(predicted_gesture)
        true_gestures.append(true_gesture)
        
        # Increment window counter
        window_count += 1
        
        # Occasionally switch profiles (every 50 windows)
        if window_count % 50 == 0:
            profile_names = list(profiles.keys())
            new_profile_name = profile_names[(profile_names.index(current_profile.name) + 1) % len(profile_names)]
            current_profile = profiles[new_profile_name]
            print(f"Switched to profile: {current_profile.name}")
            haptic.trigger('success')
        
        # Return current state for visualization
        return {
            'emg_frame': emg_frame,
            'gesture': predicted_gesture,
            'drift_active': drift_active,
            'safety_events': safety_counts
        }
    
    # Update function for animation
    def update_dashboard(frame_idx):
        # Process a window of data
        result = process_window()
        
        # Update actuators
        now = time.time()
        for actuator in actuators.values():
            actuator.update(0.05)  # 50ms update interval
        
        # Update haptic feedback
        haptic.update()
        
        # Update dashboard with new data
        return dashboard.update_emg(
            frame_idx,
            result['emg_frame'],
            result['gesture'],
            result['drift_active'],
            result['safety_events']
        )
    
    # Start the dashboard
    print("Starting Motion AI Integrated Demo...")
    print("Press Ctrl+C to exit")
    
    try:
        dashboard.start(update_dashboard)
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    
    # Calculate summary metrics
    summary = {
        "n_windows": window_count,
        "timestamp": timestamp,
        "run_dir": run_dir,
        "model_path": "simulated_model",
        "window_ms": cfg.window_ms,
        "hop_ms": cfg.hop_ms,
        "fs": cfg.fs,
        "confidence_threshold": cfg.confidence_threshold,
    }
    
    # Calculate accuracy if we have true gestures
    if true_gestures and predictions:
        correct = sum(1 for t, p in zip(true_gestures, predictions) if t == p)
        summary["accuracy"] = correct / len(true_gestures) if len(true_gestures) > 0 else 0.0
    else:
        summary["accuracy"] = None
    
    # Calculate latency statistics
    if latencies:
        summary["latency"] = {
            "mean_ms": np.mean(latencies),
            "median_ms": np.median(latencies),
            "p95_ms": np.percentile(latencies, 95),
            "max_ms": max(latencies),
        }
    else:
        summary["latency"] = {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }
    
    # Add safety counts
    summary["safety"] = safety_counts
    
    return summary


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Motion AI Integrated Demo")
    parser.add_argument("--out_dir", default="demo_logs", help="Output directory for logs and plots")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sample rate in Hz")
    parser.add_argument("--window_ms", type=int, default=200, help="Window size in ms")
    parser.add_argument("--hop_ms", type=int, default=100, help="Hop size in ms")
    parser.add_argument("--confidence_threshold", type=float, default=0.6, help="Confidence threshold for classification")
    args = parser.parse_args()
    
    cfg = DemoConfig(
        fs=args.fs,
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        confidence_threshold=args.confidence_threshold,
        out_dir=args.out_dir,
        live_plot=False,  # We're using our custom dashboard
        inject_faults=False,
        fault_config=None,
    )
    
    # Ensure output directory exists
    ensure_dir(cfg.out_dir)
    
    # Run the integrated demo
    summary = run_integrated_demo(cfg)
    
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