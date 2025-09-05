"""
Advanced Model Management System for Motion AI

Handles model loading, switching, ensembles, and performance monitoring.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import logging
import time
import numpy as np
import joblib
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class ModelType(Enum):
    """Types of classification models"""
    SVM = "svm"
    MLP = "mlp"
    RANDOM_FOREST = "random_forest"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"


class ModelStatus(Enum):
    """Model loading and operational status"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetadata:
    """Metadata for a trained model"""
    model_id: str
    model_type: ModelType
    version: str
    creation_date: str
    training_samples: int
    gestures: List[str]
    accuracy: float
    feature_names: List[str]
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    validation_scores: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


class ModelConfig(BaseModel):
    """Configuration for model management"""
    
    # Model selection
    default_model_id: Optional[str] = None
    fallback_model_id: Optional[str] = None
    ensemble_models: List[str] = Field(default_factory=list)
    
    # Performance settings
    confidence_threshold: float = Field(default=0.6, ge=0.1, le=0.95)
    performance_window_size: int = Field(default=100, ge=10, le=1000)
    auto_model_switching: bool = Field(default=True)
    switch_threshold_drop: float = Field(default=0.1, ge=0.05, le=0.3)
    
    # Caching settings
    enable_model_cache: bool = Field(default=True)
    max_cached_models: int = Field(default=3, ge=1, le=10)
    cache_timeout_minutes: int = Field(default=60, ge=5, le=480)
    
    # Ensemble settings
    enable_ensemble: bool = Field(default=False)
    ensemble_voting: str = Field(default="soft")  # "hard", "soft", "weighted"
    ensemble_weights: Dict[str, float] = Field(default_factory=dict)
    
    # Monitoring settings
    track_predictions: bool = Field(default=True)
    prediction_history_size: int = Field(default=1000, ge=100, le=10000)
    alert_on_performance_drop: bool = Field(default=True)


class BaseClassifier(ABC):
    """Abstract base class for all classifiers"""
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict gesture and confidence from features"""
        pass
    
    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities for all classes"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        pass
    
    @abstractmethod
    def get_classes(self) -> List[str]:
        """Get list of gesture classes"""
        pass


class SklearnClassifierWrapper(BaseClassifier):
    """Wrapper for scikit-learn classifiers"""
    
    def __init__(self, model_bundle: Dict[str, Any]):
        self.clf = model_bundle["clf"]
        self.scaler = model_bundle.get("scaler")
        self.feature_names = model_bundle.get("features", [])
        self.classes = model_bundle.get("classes_", getattr(self.clf, "classes_", []))
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict gesture and confidence"""
        if self.scaler:
            features = self.scaler.transform(features)
        
        if hasattr(self.clf, "predict_proba"):
            proba = self.clf.predict_proba(features)[0]
            idx = np.argmax(proba)
            gesture = str(self.classes[idx])
            confidence = float(np.max(proba))
        else:
            gesture = str(self.clf.predict(features)[0])
            confidence = 1.0
        
        return gesture, confidence
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.scaler:
            features = self.scaler.transform(features)
        
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(features)
        else:
            # For non-probabilistic classifiers, return one-hot
            pred = self.clf.predict(features)
            proba = np.zeros((len(pred), len(self.classes)))
            for i, p in enumerate(pred):
                if p in self.classes:
                    proba[i, list(self.classes).index(p)] = 1.0
            return proba
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        importance = {}
        
        if hasattr(self.clf, "feature_importances_"):
            # Tree-based models
            for i, score in enumerate(self.clf.feature_importances_):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                importance[feature_name] = float(score)
        elif hasattr(self.clf, "coef_"):
            # Linear models
            coef = self.clf.coef_
            if len(coef.shape) > 1:
                coef = np.mean(np.abs(coef), axis=0)
            for i, score in enumerate(coef):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                importance[feature_name] = float(abs(score))
        
        return importance
    
    def get_classes(self) -> List[str]:
        """Get list of gesture classes"""
        return [str(c) for c in self.classes]


class EnsembleClassifier(BaseClassifier):
    """Ensemble classifier combining multiple models"""
    
    def __init__(self, classifiers: List[BaseClassifier], weights: Optional[List[float]] = None):
        self.classifiers = classifiers
        self.weights = weights or [1.0] * len(classifiers)
        self.classes = self._get_unified_classes()
    
    def _get_unified_classes(self) -> List[str]:
        """Get unified class list from all classifiers"""
        all_classes = set()
        for clf in self.classifiers:
            all_classes.update(clf.get_classes())
        return sorted(list(all_classes))
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict using ensemble voting"""
        all_probas = []
        
        for clf, weight in zip(self.classifiers, self.weights):
            try:
                proba = clf.predict_proba(features)[0]
                # Align probabilities to unified class list
                aligned_proba = np.zeros(len(self.classes))
                clf_classes = clf.get_classes()
                for i, cls_name in enumerate(clf_classes):
                    if cls_name in self.classes:
                        idx = self.classes.index(cls_name)
                        aligned_proba[idx] = proba[i]
                
                all_probas.append(aligned_proba * weight)
            except Exception as e:
                logging.warning(f"Classifier failed in ensemble: {e}")
                continue
        
        if not all_probas:
            return "unknown", 0.0
        
        # Average weighted probabilities
        avg_proba = np.mean(all_probas, axis=0)
        idx = np.argmax(avg_proba)
        gesture = self.classes[idx]
        confidence = float(avg_proba[idx])
        
        return gesture, confidence
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities"""
        all_probas = []
        
        for clf, weight in zip(self.classifiers, self.weights):
            try:
                proba = clf.predict_proba(features)
                aligned_proba = np.zeros((proba.shape[0], len(self.classes)))
                clf_classes = clf.get_classes()
                for i, cls_name in enumerate(clf_classes):
                    if cls_name in self.classes:
                        idx = self.classes.index(cls_name)
                        aligned_proba[:, idx] = proba[:, i]
                
                all_probas.append(aligned_proba * weight)
            except Exception:
                continue
        
        if not all_probas:
            return np.zeros((features.shape[0], len(self.classes)))
        
        return np.mean(all_probas, axis=0)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get averaged feature importance from ensemble"""
        all_importance = []
        
        for clf in self.classifiers:
            try:
                importance = clf.get_feature_importance()
                all_importance.append(importance)
            except Exception:
                continue
        
        if not all_importance:
            return {}
        
        # Average importance scores
        combined = {}
        for importance in all_importance:
            for feature, score in importance.items():
                if feature not in combined:
                    combined[feature] = []
                combined[feature].append(score)
        
        return {feature: np.mean(scores) for feature, scores in combined.items()}
    
    def get_classes(self) -> List[str]:
        """Get unified class list"""
        return self.classes


class ModelManager:
    """
    Advanced model management system with caching, ensembles, and performance monitoring
    """
    
    def __init__(self, config: ModelConfig, models_dir: str = "models"):
        self.config = config
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Model registry and cache
        self.model_metadata: Dict[str, ModelMetadata] = {}
        self.model_cache: Dict[str, Tuple[BaseClassifier, float]] = {}  # model_id -> (classifier, cache_time)
        self.model_status: Dict[str, ModelStatus] = {}
        
        # Performance tracking
        self.prediction_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        
        # Current model state
        self.active_model_id: Optional[str] = None
        self.ensemble_classifier: Optional[EnsembleClassifier] = None
        
        # Load model registry
        self._load_model_registry()
        
        # Initialize default model
        if self.config.default_model_id:
            self.load_model(self.config.default_model_id)
    
    def _load_model_registry(self):
        """Load model metadata registry"""
        registry_file = self.models_dir / "registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                    
                for model_id, metadata_dict in registry_data.items():
                    self.model_metadata[model_id] = ModelMetadata(**metadata_dict)
                    self.model_status[model_id] = ModelStatus.UNLOADED
                
                self.logger.info(f"Loaded {len(self.model_metadata)} models from registry")
            except Exception as e:
                self.logger.error(f"Failed to load model registry: {e}")
    
    def register_model(
        self, 
        model_path: str, 
        metadata: ModelMetadata,
        make_default: bool = False
    ) -> bool:
        """Register a new model in the system"""
        
        try:
            # Verify model file exists
            if not Path(model_path).exists():
                self.logger.error(f"Model file not found: {model_path}")
                return False
            
            # Add to registry
            self.model_metadata[metadata.model_id] = metadata
            self.model_status[metadata.model_id] = ModelStatus.UNLOADED
            
            # Copy model to models directory if needed
            target_path = self.models_dir / f"{metadata.model_id}.pkl"
            if Path(model_path) != target_path:
                import shutil
                shutil.copy2(model_path, target_path)
            
            # Update registry file
            self._save_model_registry()
            
            # Set as default if requested
            if make_default:
                self.config.default_model_id = metadata.model_id
            
            self.logger.info(f"Registered model: {metadata.model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register model {metadata.model_id}: {e}")
            return False
    
    def _save_model_registry(self):
        """Save model metadata registry"""
        registry_file = self.models_dir / "registry.json"
        
        try:
            registry_data = {}
            for model_id, metadata in self.model_metadata.items():
                registry_data[model_id] = {
                    "model_id": metadata.model_id,
                    "model_type": metadata.model_type.value,
                    "version": metadata.version,
                    "creation_date": metadata.creation_date,
                    "training_samples": metadata.training_samples,
                    "gestures": metadata.gestures,
                    "accuracy": metadata.accuracy,
                    "feature_names": metadata.feature_names,
                    "preprocessing_config": metadata.preprocessing_config,
                    "validation_scores": metadata.validation_scores,
                    "hyperparameters": metadata.hyperparameters,
                    "description": metadata.description
                }
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")
    
    def load_model(self, model_id: str) -> bool:
        """Load a model into memory"""
        
        if model_id not in self.model_metadata:
            self.logger.error(f"Model not found in registry: {model_id}")
            return False
        
        # Check if already loaded and cached
        if model_id in self.model_cache:
            classifier, cache_time = self.model_cache[model_id]
            if time.time() - cache_time < self.config.cache_timeout_minutes * 60:
                self.active_model_id = model_id
                self.model_status[model_id] = ModelStatus.READY
                return True
        
        try:
            self.model_status[model_id] = ModelStatus.LOADING
            
            # Load model file
            model_path = self.models_dir / f"{model_id}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model_bundle = joblib.load(model_path)
            
            # Create classifier wrapper
            metadata = self.model_metadata[model_id]
            if metadata.model_type in [ModelType.SVM, ModelType.MLP, ModelType.RANDOM_FOREST]:
                classifier = SklearnClassifierWrapper(model_bundle)
            else:
                raise ValueError(f"Unsupported model type: {metadata.model_type}")
            
            # Cache the model
            self._cache_model(model_id, classifier)
            
            self.active_model_id = model_id
            self.model_status[model_id] = ModelStatus.READY
            
            self.logger.info(f"Loaded model: {model_id}")
            return True
            
        except Exception as e:
            self.model_status[model_id] = ModelStatus.ERROR
            self.logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    def _cache_model(self, model_id: str, classifier: BaseClassifier):
        """Cache a loaded model"""
        
        if not self.config.enable_model_cache:
            return
        
        # Remove oldest cached models if cache is full
        while len(self.model_cache) >= self.config.max_cached_models:
            oldest_id = min(self.model_cache.keys(), key=lambda k: self.model_cache[k][1])
            del self.model_cache[oldest_id]
            self.logger.info(f"Evicted cached model: {oldest_id}")
        
        self.model_cache[model_id] = (classifier, time.time())
    
    def predict(
        self, 
        features: np.ndarray, 
        model_id: Optional[str] = None
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Make a prediction using the specified or active model
        
        Returns:
            Tuple of (gesture, confidence, explanation_data)
        """
        
        target_model_id = model_id or self.active_model_id
        
        if not target_model_id:
            return "unknown", 0.0, {"error": "No active model"}
        
        # Ensure model is loaded
        if target_model_id not in self.model_cache:
            if not self.load_model(target_model_id):
                return "unknown", 0.0, {"error": f"Failed to load model {target_model_id}"}
        
        try:
            classifier, _ = self.model_cache[target_model_id]
            
            # Make prediction
            start_time = time.time()
            gesture, confidence = classifier.predict(features)
            prediction_time = (time.time() - start_time) * 1000  # ms
            
            # Create explanation data
            explanation = {
                "model_id": target_model_id,
                "model_type": self.model_metadata[target_model_id].model_type.value,
                "prediction_time_ms": round(prediction_time, 2),
                "feature_count": features.shape[1],
                "confidence_threshold": self.config.confidence_threshold,
                "above_threshold": confidence >= self.config.confidence_threshold
            }
            
            # Add feature importance if available
            try:
                importance = classifier.get_feature_importance()
                if importance:
                    # Get top 5 most important features
                    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    explanation["top_features"] = dict(sorted_importance[:5])
            except Exception:
                pass
            
            # Record prediction for monitoring
            self._record_prediction(target_model_id, gesture, confidence, features, explanation)
            
            return gesture, confidence, explanation
            
        except Exception as e:
            self.logger.error(f"Prediction failed for model {target_model_id}: {e}")
            return "unknown", 0.0, {"error": str(e)}
    
    def predict_ensemble(self, features: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """Make prediction using ensemble of models"""
        
        if not self.config.enable_ensemble or not self.config.ensemble_models:
            return self.predict(features)
        
        try:
            # Load ensemble models
            classifiers = []
            weights = []
            
            for model_id in self.config.ensemble_models:
                if model_id not in self.model_cache:
                    if not self.load_model(model_id):
                        continue
                
                classifier, _ = self.model_cache[model_id]
                classifiers.append(classifier)
                weights.append(self.config.ensemble_weights.get(model_id, 1.0))
            
            if not classifiers:
                return "unknown", 0.0, {"error": "No ensemble models available"}
            
            # Create ensemble classifier
            ensemble = EnsembleClassifier(classifiers, weights)
            
            # Make prediction
            start_time = time.time()
            gesture, confidence = ensemble.predict(features)
            prediction_time = (time.time() - start_time) * 1000
            
            explanation = {
                "ensemble_models": self.config.ensemble_models,
                "ensemble_weights": weights,
                "prediction_time_ms": round(prediction_time, 2),
                "voting_method": self.config.ensemble_voting,
                "model_count": len(classifiers)
            }
            
            return gesture, confidence, explanation
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            return "unknown", 0.0, {"error": str(e)}
    
    def _record_prediction(
        self, 
        model_id: str, 
        gesture: str, 
        confidence: float,
        features: np.ndarray,
        explanation: Dict[str, Any]
    ):
        """Record prediction for performance monitoring"""
        
        if not self.config.track_predictions:
            return
        
        record = {
            "timestamp": time.time(),
            "model_id": model_id,
            "gesture": gesture,
            "confidence": confidence,
            "feature_count": features.shape[1],
            "explanation": explanation
        }
        
        self.prediction_history.append(record)
        
        # Maintain history size limit
        if len(self.prediction_history) > self.config.prediction_history_size:
            self.prediction_history.pop(0)
        
        # Update performance metrics
        self._update_performance_metrics(model_id, confidence)
    
    def _update_performance_metrics(self, model_id: str, confidence: float):
        """Update rolling performance metrics"""
        
        if model_id not in self.performance_metrics:
            self.performance_metrics[model_id] = {
                "avg_confidence": confidence,
                "prediction_count": 1,
                "confidence_sum": confidence
            }
        else:
            metrics = self.performance_metrics[model_id]
            metrics["prediction_count"] += 1
            metrics["confidence_sum"] += confidence
            metrics["avg_confidence"] = metrics["confidence_sum"] / metrics["prediction_count"]
            
            # Apply window limit
            if metrics["prediction_count"] > self.config.performance_window_size:
                # Approximate moving average
                metrics["avg_confidence"] = (
                    metrics["avg_confidence"] * 0.95 + confidence * 0.05
                )
    
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get performance statistics for a model"""
        
        if model_id not in self.performance_metrics:
            return {}
        
        metrics = self.performance_metrics[model_id].copy()
        
        # Add recent prediction statistics
        recent_predictions = [
            record for record in self.prediction_history[-100:]  # Last 100 predictions
            if record["model_id"] == model_id
        ]
        
        if recent_predictions:
            recent_confidences = [r["confidence"] for r in recent_predictions]
            metrics.update({
                "recent_avg_confidence": np.mean(recent_confidences),
                "recent_min_confidence": np.min(recent_confidences),
                "recent_max_confidence": np.max(recent_confidences),
                "recent_std_confidence": np.std(recent_confidences),
                "recent_prediction_count": len(recent_predictions)
            })
        
        return metrics
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models with their status"""
        
        models = {}
        for model_id, metadata in self.model_metadata.items():
            models[model_id] = {
                "metadata": {
                    "model_type": metadata.model_type.value,
                    "version": metadata.version,
                    "accuracy": metadata.accuracy,
                    "gestures": metadata.gestures,
                    "description": metadata.description
                },
                "status": self.model_status[model_id].value,
                "is_active": model_id == self.active_model_id,
                "is_cached": model_id in self.model_cache,
                "performance": self.get_model_performance(model_id)
            }
        
        return models
    
    def switch_model(self, model_id: str) -> bool:
        """Switch to a different model"""
        
        if model_id not in self.model_metadata:
            self.logger.error(f"Model not found: {model_id}")
            return False
        
        if self.load_model(model_id):
            old_model = self.active_model_id
            self.active_model_id = model_id
            self.logger.info(f"Switched from {old_model} to {model_id}")
            return True
        
        return False
    
    def get_model_explanation(self, model_id: str) -> Dict[str, Any]:
        """Get detailed explanation about a model"""
        
        if model_id not in self.model_metadata:
            return {}
        
        metadata = self.model_metadata[model_id]
        explanation = {
            "model_info": {
                "id": metadata.model_id,
                "type": metadata.model_type.value,
                "version": metadata.version,
                "accuracy": metadata.accuracy,
                "training_samples": metadata.training_samples,
                "gestures": metadata.gestures,
                "description": metadata.description
            },
            "status": self.model_status[model_id].value,
            "performance": self.get_model_performance(model_id)
        }
        
        # Add feature importance if model is loaded
        if model_id in self.model_cache:
            try:
                classifier, _ = self.model_cache[model_id]
                importance = classifier.get_feature_importance()
                if importance:
                    explanation["feature_importance"] = importance
            except Exception:
                pass
        
        return explanation