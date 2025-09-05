"""
Comprehensive Performance Evaluation System

Evaluates model performance, system metrics, and real-time performance tracking.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class MetricType(Enum):
    """Types of evaluation metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    CONFUSION_MATRIX = "confusion_matrix"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CONFIDENCE = "confidence"
    STABILITY = "stability"
    SAFETY = "safety"


class EvaluationMode(Enum):
    """Evaluation execution modes"""
    REALTIME = "realtime"
    BATCH = "batch"
    CONTINUOUS = "continuous"
    OFFLINE = "offline"


@dataclass
class EvaluationMetrics:
    """Container for evaluation results"""
    
    # Classification metrics
    accuracy: float = 0.0
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    f1_score: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    throughput_hz: float = 0.0
    
    # Confidence metrics
    avg_confidence: float = 0.0
    confidence_std: float = 0.0
    low_confidence_rate: float = 0.0
    
    # Stability metrics
    prediction_stability: float = 0.0
    gesture_transition_rate: float = 0.0
    
    # Safety metrics
    safety_violations: int = 0
    false_positive_rate: float = 0.0
    emergency_activations: int = 0
    
    # System metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    prediction_count: int = 0
    evaluation_duration_s: float = 0.0
    
    # Additional metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """Configuration for performance evaluation"""
    
    # Evaluation settings
    evaluation_mode: EvaluationMode = EvaluationMode.CONTINUOUS
    evaluation_window_size: int = Field(default=100, ge=10, le=10000)
    evaluation_interval_s: float = Field(default=30.0, ge=1.0, le=3600.0)
    
    # Metric settings
    enabled_metrics: List[MetricType] = Field(default_factory=lambda: list(MetricType))
    confidence_threshold: float = Field(default=0.6, ge=0.1, le=0.95)
    latency_threshold_ms: float = Field(default=100.0, ge=1.0, le=5000.0)
    
    # Alert settings
    enable_alerts: bool = Field(default=True)
    accuracy_alert_threshold: float = Field(default=0.7, ge=0.1, le=1.0)
    latency_alert_threshold_ms: float = Field(default=200.0, ge=1.0, le=5000.0)
    stability_alert_threshold: float = Field(default=0.8, ge=0.1, le=1.0)
    
    # Storage settings
    save_results: bool = Field(default=True)
    results_dir: str = Field(default="evaluation_results")
    max_history_size: int = Field(default=10000, ge=100, le=100000)
    
    # System monitoring
    monitor_system_resources: bool = Field(default=True)
    system_check_interval_s: float = Field(default=5.0, ge=1.0, le=60.0)


class BaseMetricCalculator(ABC):
    """Abstract base class for metric calculators"""
    
    @abstractmethod
    def calculate(self, predictions: List[Dict[str, Any]], ground_truth: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate metrics from predictions"""
        pass
    
    @abstractmethod
    def get_metric_names(self) -> List[str]:
        """Get list of metric names this calculator provides"""
        pass


class ClassificationMetricsCalculator(BaseMetricCalculator):
    """Calculator for classification metrics"""
    
    def __init__(self, gestures: List[str]):
        self.gestures = gestures
    
    def calculate(self, predictions: List[Dict[str, Any]], ground_truth: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate classification metrics"""
        
        if not ground_truth:
            # Can only calculate confidence-based metrics without ground truth
            confidences = [p.get("confidence", 0.0) for p in predictions]
            return {
                "avg_confidence": np.mean(confidences) if confidences else 0.0,
                "confidence_std": np.std(confidences) if confidences else 0.0,
                "low_confidence_rate": sum(1 for c in confidences if c < 0.6) / len(confidences) if confidences else 0.0
            }
        
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        pred_gestures = [p.get("gesture", "unknown") for p in predictions]
        
        # Calculate confusion matrix
        confusion = defaultdict(lambda: defaultdict(int))
        for pred, true in zip(pred_gestures, ground_truth):
            confusion[true][pred] += 1
        
        # Calculate per-class metrics
        precision = {}
        recall = {}
        f1 = {}
        
        for gesture in self.gestures:
            # True positives, false positives, false negatives
            tp = confusion[gesture][gesture]
            fp = sum(confusion[other][gesture] for other in self.gestures if other != gesture)
            fn = sum(confusion[gesture][other] for other in self.gestures if other != gesture)
            
            # Precision
            precision[gesture] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Recall
            recall[gesture] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # F1 Score
            p, r = precision[gesture], recall[gesture]
            f1[gesture] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        # Overall accuracy
        total_correct = sum(confusion[g][g] for g in self.gestures)
        total_predictions = len(predictions)
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
        
        # Macro averages
        macro_precision = np.mean(list(precision.values())) if precision else 0.0
        macro_recall = np.mean(list(recall.values())) if recall else 0.0
        macro_f1 = np.mean(list(f1.values())) if f1 else 0.0
        
        return {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            **{f"precision_{g}": precision[g] for g in self.gestures},
            **{f"recall_{g}": recall[g] for g in self.gestures},
            **{f"f1_{g}": f1[g] for g in self.gestures}
        }
    
    def get_metric_names(self) -> List[str]:
        """Get list of metric names"""
        base_metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
        per_class_metrics = []
        
        for gesture in self.gestures:
            per_class_metrics.extend([
                f"precision_{gesture}",
                f"recall_{gesture}",
                f"f1_{gesture}"
            ])
        
        return base_metrics + per_class_metrics


class LatencyMetricsCalculator(BaseMetricCalculator):
    """Calculator for latency and throughput metrics"""
    
    def calculate(self, predictions: List[Dict[str, Any]], ground_truth: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate latency metrics"""
        
        latencies = []
        timestamps = []
        
        for p in predictions:
            if "latency_ms" in p:
                latencies.append(p["latency_ms"])
            if "timestamp" in p:
                timestamps.append(p["timestamp"])
        
        if not latencies:
            return {}
        
        metrics = {
            "avg_latency_ms": np.mean(latencies),
            "max_latency_ms": np.max(latencies),
            "min_latency_ms": np.min(latencies),
            "std_latency_ms": np.std(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99)
        }
        
        # Calculate throughput if timestamps available
        if len(timestamps) >= 2:
            time_span = max(timestamps) - min(timestamps)
            if time_span > 0:
                metrics["throughput_hz"] = len(timestamps) / time_span
        
        return metrics
    
    def get_metric_names(self) -> List[str]:
        return [
            "avg_latency_ms", "max_latency_ms", "min_latency_ms", 
            "std_latency_ms", "p95_latency_ms", "p99_latency_ms", "throughput_hz"
        ]


class StabilityMetricsCalculator(BaseMetricCalculator):
    """Calculator for prediction stability metrics"""
    
    def calculate(self, predictions: List[Dict[str, Any]], ground_truth: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate stability metrics"""
        
        if len(predictions) < 2:
            return {}
        
        gestures = [p.get("gesture", "unknown") for p in predictions]
        confidences = [p.get("confidence", 0.0) for p in predictions]
        
        # Prediction stability (how often predictions change)
        transitions = sum(1 for i in range(1, len(gestures)) if gestures[i] != gestures[i-1])
        stability = 1.0 - (transitions / (len(gestures) - 1))
        
        # Gesture transition rate
        transition_rate = transitions / len(gestures)
        
        # Confidence stability
        confidence_changes = [abs(confidences[i] - confidences[i-1]) for i in range(1, len(confidences))]
        avg_confidence_change = np.mean(confidence_changes) if confidence_changes else 0.0
        
        return {
            "prediction_stability": stability,
            "gesture_transition_rate": transition_rate,
            "avg_confidence_change": avg_confidence_change,
            "confidence_volatility": np.std(confidence_changes) if confidence_changes else 0.0
        }
    
    def get_metric_names(self) -> List[str]:
        return [
            "prediction_stability", "gesture_transition_rate", 
            "avg_confidence_change", "confidence_volatility"
        ]


class PerformanceEvaluator:
    """
    Comprehensive performance evaluation system with real-time monitoring
    """
    
    def __init__(self, config: EvaluationConfig, gestures: List[str]):
        self.config = config
        self.gestures = gestures
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Prediction history
        self.prediction_history: deque = deque(maxlen=config.max_history_size)
        self.ground_truth_history: deque = deque(maxlen=config.max_history_size)
        
        # Evaluation state
        self.evaluation_results: List[EvaluationMetrics] = []
        self.current_window_start = time.time()
        self.alert_history: List[Dict[str, Any]] = []
        
        # Metric calculators
        self.metric_calculators = {
            MetricType.ACCURACY: ClassificationMetricsCalculator(gestures),
            MetricType.LATENCY: LatencyMetricsCalculator(),
            MetricType.STABILITY: StabilityMetricsCalculator()
        }
        
        # System monitoring
        self.system_metrics = {}
        self.last_system_check = 0.0
    
    def add_prediction(
        self, 
        gesture: str, 
        confidence: float, 
        latency_ms: float,
        timestamp: Optional[float] = None,
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a prediction for evaluation"""
        
        prediction = {
            "gesture": gesture,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "timestamp": timestamp or time.time(),
            "metadata": metadata or {}
        }
        
        self.prediction_history.append(prediction)
        
        if ground_truth:
            self.ground_truth_history.append(ground_truth)
        
        # Check if we should run evaluation
        if self.config.evaluation_mode == EvaluationMode.CONTINUOUS:
            self._check_evaluation_trigger()
    
    def _check_evaluation_trigger(self):
        """Check if evaluation should be triggered"""
        
        current_time = time.time()
        
        # Time-based trigger
        if current_time - self.current_window_start >= self.config.evaluation_interval_s:
            self.evaluate_current_window()
        
        # Size-based trigger
        elif len(self.prediction_history) >= self.config.evaluation_window_size:
            self.evaluate_current_window()
    
    def evaluate_current_window(self) -> EvaluationMetrics:
        """Evaluate the current prediction window"""
        
        if not self.prediction_history:
            return EvaluationMetrics()
        
        start_time = time.time()
        
        # Get recent predictions
        window_size = min(self.config.evaluation_window_size, len(self.prediction_history))
        recent_predictions = list(self.prediction_history)[-window_size:]
        recent_ground_truth = None
        
        if self.ground_truth_history:
            recent_ground_truth = list(self.ground_truth_history)[-window_size:]
        
        # Calculate metrics
        metrics = EvaluationMetrics()
        metrics.prediction_count = len(recent_predictions)
        metrics.evaluation_duration_s = time.time() - start_time
        
        # Run enabled metric calculators
        for metric_type, calculator in self.metric_calculators.items():
            if metric_type in self.config.enabled_metrics:
                try:
                    calculated_metrics = calculator.calculate(recent_predictions, recent_ground_truth)
                    self._update_metrics_object(metrics, calculated_metrics)
                except Exception as e:
                    self.logger.error(f"Failed to calculate {metric_type}: {e}")
        
        # Update system metrics
        if self.config.monitor_system_resources:
            self._update_system_metrics(metrics)
        
        # Store results
        self.evaluation_results.append(metrics)
        
        # Check for alerts
        if self.config.enable_alerts:
            self._check_alerts(metrics)
        
        # Save results
        if self.config.save_results:
            self._save_evaluation_results()
        
        # Reset window
        self.current_window_start = time.time()
        
        self.logger.info(f"Evaluation completed: accuracy={metrics.accuracy:.3f}, latency={metrics.avg_latency_ms:.1f}ms")
        
        return metrics
    
    def _update_metrics_object(self, metrics: EvaluationMetrics, calculated: Dict[str, float]):
        """Update metrics object with calculated values"""
        
        # Map calculated metrics to EvaluationMetrics fields
        field_mapping = {
            "accuracy": "accuracy",
            "avg_latency_ms": "avg_latency_ms",
            "max_latency_ms": "max_latency_ms",
            "min_latency_ms": "min_latency_ms",
            "p95_latency_ms": "p95_latency_ms",
            "throughput_hz": "throughput_hz",
            "avg_confidence": "avg_confidence",
            "confidence_std": "confidence_std",
            "low_confidence_rate": "low_confidence_rate",
            "prediction_stability": "prediction_stability",
            "gesture_transition_rate": "gesture_transition_rate"
        }
        
        for calc_name, calc_value in calculated.items():
            if calc_name in field_mapping:
                setattr(metrics, field_mapping[calc_name], calc_value)
            else:
                # Store in custom metrics
                metrics.custom_metrics[calc_name] = calc_value
    
    def _update_system_metrics(self, metrics: EvaluationMetrics):
        """Update system resource metrics"""
        
        current_time = time.time()
        
        if current_time - self.last_system_check >= self.config.system_check_interval_s:
            try:
                import psutil
                process = psutil.Process()
                
                metrics.cpu_usage_percent = process.cpu_percent()
                metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                
                self.last_system_check = current_time
                
            except ImportError:
                # psutil not available
                pass
            except Exception as e:
                self.logger.warning(f"Failed to get system metrics: {e}")
    
    def _check_alerts(self, metrics: EvaluationMetrics):
        """Check for performance alerts"""
        
        alerts = []
        
        # Accuracy alert
        if metrics.accuracy < self.config.accuracy_alert_threshold:
            alerts.append({
                "type": "accuracy",
                "message": f"Accuracy dropped to {metrics.accuracy:.3f}",
                "severity": "warning",
                "threshold": self.config.accuracy_alert_threshold,
                "value": metrics.accuracy
            })
        
        # Latency alert
        if metrics.avg_latency_ms > self.config.latency_alert_threshold_ms:
            alerts.append({
                "type": "latency",
                "message": f"Average latency increased to {metrics.avg_latency_ms:.1f}ms",
                "severity": "warning",
                "threshold": self.config.latency_alert_threshold_ms,
                "value": metrics.avg_latency_ms
            })
        
        # Stability alert
        if metrics.prediction_stability < self.config.stability_alert_threshold:
            alerts.append({
                "type": "stability",
                "message": f"Prediction stability dropped to {metrics.prediction_stability:.3f}",
                "severity": "warning",
                "threshold": self.config.stability_alert_threshold,
                "value": metrics.prediction_stability
            })
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(f"Performance Alert: {alert['message']}")
            alert["timestamp"] = time.time()
            self.alert_history.append(alert)
    
    def _save_evaluation_results(self):
        """Save evaluation results to disk"""
        
        try:
            timestamp = int(time.time())
            results_file = self.results_dir / f"evaluation_{timestamp}.json"
            
            # Convert metrics to serializable format
            serializable_results = []
            for metrics in self.evaluation_results[-10:]:  # Save last 10 results
                result_dict = {
                    "accuracy": metrics.accuracy,
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "max_latency_ms": metrics.max_latency_ms,
                    "p95_latency_ms": metrics.p95_latency_ms,
                    "throughput_hz": metrics.throughput_hz,
                    "avg_confidence": metrics.avg_confidence,
                    "prediction_stability": metrics.prediction_stability,
                    "prediction_count": metrics.prediction_count,
                    "custom_metrics": metrics.custom_metrics,
                    "timestamp": time.time()
                }
                serializable_results.append(result_dict)
            
            with open(results_file, 'w') as f:
                json.dump({
                    "results": serializable_results,
                    "config": self.config.dict(),
                    "gestures": self.gestures
                }, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save evaluation results: {e}")
    
    def get_latest_metrics(self) -> Optional[EvaluationMetrics]:
        """Get the latest evaluation metrics"""
        return self.evaluation_results[-1] if self.evaluation_results else None
    
    def get_metrics_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """Get summary of recent metrics"""
        
        if not self.evaluation_results:
            return {}
        
        recent_results = self.evaluation_results[-last_n:]
        
        accuracies = [r.accuracy for r in recent_results if r.accuracy > 0]
        latencies = [r.avg_latency_ms for r in recent_results if r.avg_latency_ms > 0]
        confidences = [r.avg_confidence for r in recent_results if r.avg_confidence > 0]
        
        summary = {
            "evaluation_count": len(recent_results),
            "total_predictions": sum(r.prediction_count for r in recent_results),
            "time_span_minutes": (time.time() - self.current_window_start) / 60.0
        }
        
        if accuracies:
            summary.update({
                "avg_accuracy": np.mean(accuracies),
                "min_accuracy": np.min(accuracies),
                "max_accuracy": np.max(accuracies)
            })
        
        if latencies:
            summary.update({
                "avg_latency_ms": np.mean(latencies),
                "min_latency_ms": np.min(latencies),
                "max_latency_ms": np.max(latencies)
            })
        
        if confidences:
            summary.update({
                "avg_confidence": np.mean(confidences),
                "min_confidence": np.min(confidences),
                "max_confidence": np.max(confidences)
            })
        
        return summary
    
    def get_alerts(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return self.alert_history[-last_n:] if self.alert_history else []
    
    def reset_evaluation(self):
        """Reset evaluation state"""
        self.prediction_history.clear()
        self.ground_truth_history.clear()
        self.evaluation_results.clear()
        self.alert_history.clear()
        self.current_window_start = time.time()
        
        self.logger.info("Evaluation state reset")
    
    def export_results(self, filepath: str) -> bool:
        """Export all evaluation results to a file"""
        
        try:
            export_data = {
                "config": self.config.dict(),
                "gestures": self.gestures,
                "evaluation_results": [],
                "alert_history": self.alert_history,
                "summary": self.get_metrics_summary(len(self.evaluation_results))
            }
            
            # Convert metrics to dict format
            for metrics in self.evaluation_results:
                result_dict = {
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "max_latency_ms": metrics.max_latency_ms,
                    "p95_latency_ms": metrics.p95_latency_ms,
                    "throughput_hz": metrics.throughput_hz,
                    "avg_confidence": metrics.avg_confidence,
                    "prediction_stability": metrics.prediction_stability,
                    "safety_violations": metrics.safety_violations,
                    "prediction_count": metrics.prediction_count,
                    "custom_metrics": metrics.custom_metrics
                }
                export_data["evaluation_results"].append(result_dict)
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported evaluation results to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            return False