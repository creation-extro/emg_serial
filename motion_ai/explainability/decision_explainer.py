"""
Decision Explainability System for Motion AI

Provides interpretable explanations for gesture recognition decisions, feature importance,
and action mapping rationale.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import logging
import numpy as np
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class ExplanationType(Enum):
    """Types of explanations available"""
    FEATURE_IMPORTANCE = "feature_importance"
    DECISION_PATH = "decision_path"
    CONFIDENCE_BREAKDOWN = "confidence_breakdown"
    COUNTERFACTUAL = "counterfactual"
    ATTENTION_WEIGHTS = "attention_weights"
    RULE_BASED = "rule_based"
    SIMILARITY = "similarity"


class ExplanationLevel(Enum):
    """Levels of explanation detail"""
    SIMPLE = "simple"      # Basic explanation for end users
    DETAILED = "detailed"  # Detailed explanation for experts
    TECHNICAL = "technical" # Full technical details for developers
    DEBUG = "debug"        # Debug-level information


@dataclass 
class FeatureExplanation:
    """Explanation for individual feature contributions"""
    feature_name: str
    importance_score: float
    value: float
    contribution: float  # Positive or negative contribution to prediction
    percentile: float   # Where this value falls in the distribution
    description: str = ""
    human_readable: str = ""


@dataclass
class ConfidenceBreakdown:
    """Breakdown of factors affecting confidence"""
    base_confidence: float
    feature_quality_factor: float
    model_certainty: float
    consistency_factor: float
    noise_factor: float
    final_confidence: float
    explanation: str = ""


@dataclass
class DecisionPath:
    """Path through decision logic"""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    branch_points: List[Dict[str, Any]] = field(default_factory=list)
    final_decision: str = ""
    decision_factors: List[str] = field(default_factory=list)


@dataclass
class ExplanationResult:
    """Complete explanation result"""
    
    # Core prediction info
    predicted_gesture: str
    confidence: float
    timestamp: float
    
    # Feature explanations
    feature_explanations: List[FeatureExplanation] = field(default_factory=list)
    top_features: List[str] = field(default_factory=list)
    
    # Confidence breakdown
    confidence_breakdown: Optional[ConfidenceBreakdown] = None
    
    # Decision path
    decision_path: Optional[DecisionPath] = None
    
    # Alternative predictions
    alternative_predictions: List[Tuple[str, float]] = field(default_factory=list)
    
    # Context information
    model_info: Dict[str, Any] = field(default_factory=dict)
    signal_quality: Dict[str, float] = field(default_factory=dict)
    
    # Human-readable explanations
    simple_explanation: str = ""
    detailed_explanation: str = ""
    technical_explanation: str = ""
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ExplanationConfig(BaseModel):
    """Configuration for explanation generation"""
    
    # Explanation settings
    explanation_level: ExplanationLevel = ExplanationLevel.DETAILED
    include_feature_importance: bool = Field(default=True)
    include_confidence_breakdown: bool = Field(default=True)
    include_decision_path: bool = Field(default=True)
    include_alternatives: bool = Field(default=True)
    
    # Feature analysis
    max_features_to_explain: int = Field(default=10, ge=1, le=50)
    min_importance_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    
    # Confidence analysis
    confidence_threshold_warning: float = Field(default=0.6, ge=0.1, le=0.95)
    low_confidence_explanation: bool = Field(default=True)
    
    # Comparison settings
    include_similar_gestures: bool = Field(default=True)
    max_similar_gestures: int = Field(default=3, ge=1, le=5)
    
    # Output format
    generate_human_readable: bool = Field(default=True)
    include_recommendations: bool = Field(default=True)
    language: str = Field(default="en")


class BaseExplainer(ABC):
    """Abstract base class for explainer components"""
    
    @abstractmethod
    def explain(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for prediction data"""
        pass
    
    @abstractmethod
    def get_explanation_types(self) -> List[ExplanationType]:
        """Get supported explanation types"""
        pass


class FeatureImportanceExplainer(BaseExplainer):
    """Explains predictions based on feature importance"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.feature_descriptions = self._get_feature_descriptions()
    
    def _get_feature_descriptions(self) -> Dict[str, str]:
        """Get human-readable descriptions for features"""
        descriptions = {}
        
        for feature in self.feature_names:
            if "rms" in feature.lower():
                descriptions[feature] = f"Signal strength in {feature.replace('_rms', '').replace('_', ' ')}"
            elif "mean" in feature.lower():
                descriptions[feature] = f"Average signal level in {feature.replace('_mean', '').replace('_', ' ')}"
            elif "std" in feature.lower():
                descriptions[feature] = f"Signal variability in {feature.replace('_std', '').replace('_', ' ')}"
            elif "energy" in feature.lower():
                descriptions[feature] = f"Signal energy in {feature.replace('_energy', '').replace('_', ' ')}"
            elif "freq" in feature.lower():
                descriptions[feature] = f"Frequency content in {feature.replace('_freq', '').replace('_', ' ')}"
            else:
                descriptions[feature] = feature.replace('_', ' ').title()
        
        return descriptions
    
    def explain(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate feature importance explanation"""
        
        features = prediction_data.get("features", {})
        feature_importance = prediction_data.get("feature_importance", {})
        
        if not feature_importance:
            return {"feature_explanations": []}
        
        # Create feature explanations
        explanations = []
        for feature_name, importance in feature_importance.items():
            if feature_name in features:
                value = features[feature_name]
                
                # Calculate contribution (simplified)
                contribution = importance * value
                
                # Estimate percentile (would need historical data in real implementation)
                percentile = min(95, max(5, 50 + (value - 0.5) * 50))
                
                explanation = FeatureExplanation(
                    feature_name=feature_name,
                    importance_score=importance,
                    value=value,
                    contribution=contribution,
                    percentile=percentile,
                    description=self.feature_descriptions.get(feature_name, feature_name),
                    human_readable=self._generate_human_readable_feature_explanation(
                        feature_name, value, importance, contribution
                    )
                )
                explanations.append(explanation)
        
        # Sort by importance
        explanations.sort(key=lambda x: abs(x.importance_score), reverse=True)
        
        return {
            "feature_explanations": explanations,
            "top_features": [exp.feature_name for exp in explanations[:5]]
        }
    
    def _generate_human_readable_feature_explanation(
        self, 
        feature_name: str, 
        value: float, 
        importance: float,
        contribution: float
    ) -> str:
        """Generate human-readable explanation for a feature"""
        
        description = self.feature_descriptions.get(feature_name, feature_name)
        
        # Determine signal strength
        if value < 0.2:
            strength = "very low"
        elif value < 0.4:
            strength = "low"
        elif value < 0.6:
            strength = "moderate"
        elif value < 0.8:
            strength = "high"
        else:
            strength = "very high"
        
        # Determine influence
        if abs(contribution) > 0.1:
            influence = "strongly"
        elif abs(contribution) > 0.05:
            influence = "moderately"
        else:
            influence = "weakly"
        
        direction = "supports" if contribution > 0 else "opposes"
        
        return f"{description} is {strength} and {influence} {direction} this prediction"
    
    def get_explanation_types(self) -> List[ExplanationType]:
        return [ExplanationType.FEATURE_IMPORTANCE]


class ConfidenceExplainer(BaseExplainer):
    """Explains confidence scores and their components"""
    
    def explain(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate confidence breakdown explanation"""
        
        confidence = prediction_data.get("confidence", 0.0)
        features = prediction_data.get("features", {})
        
        # Analyze signal quality
        signal_quality = self._analyze_signal_quality(features)
        
        # Break down confidence factors
        base_confidence = confidence
        
        # Feature quality factor (based on signal strength)
        avg_signal_strength = np.mean([v for k, v in features.items() if "rms" in k.lower()])
        feature_quality_factor = min(1.0, avg_signal_strength * 2.0) if avg_signal_strength else 0.5
        
        # Model certainty (how decisive the prediction is)
        model_certainty = confidence  # Simplified - would use prediction probabilities
        
        # Consistency factor (would need historical data)
        consistency_factor = 0.8  # Placeholder
        
        # Noise factor (based on signal variability)
        noise_levels = [v for k, v in features.items() if "std" in k.lower()]
        noise_factor = max(0.1, 1.0 - np.mean(noise_levels)) if noise_levels else 0.8
        
        breakdown = ConfidenceBreakdown(
            base_confidence=base_confidence,
            feature_quality_factor=feature_quality_factor,
            model_certainty=model_certainty,
            consistency_factor=consistency_factor,
            noise_factor=noise_factor,
            final_confidence=confidence,
            explanation=self._generate_confidence_explanation(
                confidence, feature_quality_factor, noise_factor
            )
        )
        
        return {
            "confidence_breakdown": breakdown,
            "signal_quality": signal_quality
        }
    
    def _analyze_signal_quality(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Analyze signal quality metrics"""
        
        quality = {}
        
        # Signal strength
        rms_features = [v for k, v in features.items() if "rms" in k.lower()]
        if rms_features:
            quality["signal_strength"] = np.mean(rms_features)
        
        # Signal stability
        std_features = [v for k, v in features.items() if "std" in k.lower()]
        if std_features:
            quality["stability"] = 1.0 - min(1.0, np.mean(std_features))
        
        # Overall quality score
        if quality:
            quality["overall"] = np.mean(list(quality.values()))
        
        return quality
    
    def _generate_confidence_explanation(
        self, 
        confidence: float, 
        feature_quality: float,
        noise_factor: float
    ) -> str:
        """Generate human-readable confidence explanation"""
        
        if confidence >= 0.8:
            base_msg = "Very confident prediction"
        elif confidence >= 0.6:
            base_msg = "Moderately confident prediction"
        elif confidence >= 0.4:
            base_msg = "Low confidence prediction"
        else:
            base_msg = "Very uncertain prediction"
        
        factors = []
        
        if feature_quality < 0.5:
            factors.append("weak signal quality")
        elif feature_quality > 0.8:
            factors.append("strong signal quality")
        
        if noise_factor < 0.6:
            factors.append("high noise levels")
        elif noise_factor > 0.8:
            factors.append("clean signal")
        
        if factors:
            return f"{base_msg} due to {', '.join(factors)}"
        else:
            return base_msg
    
    def get_explanation_types(self) -> List[ExplanationType]:
        return [ExplanationType.CONFIDENCE_BREAKDOWN]


class DecisionPathExplainer(BaseExplainer):
    """Explains the decision-making path through the system"""
    
    def explain(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate decision path explanation"""
        
        gesture = prediction_data.get("gesture", "unknown")
        confidence = prediction_data.get("confidence", 0.0)
        features = prediction_data.get("features", {})
        model_type = prediction_data.get("model_type", "unknown")
        
        # Create decision path
        path = DecisionPath()
        
        # Step 1: Signal preprocessing
        path.steps.append({
            "step": "signal_preprocessing",
            "description": "EMG signal filtered and windowed",
            "details": "Applied bandpass filter and extracted 200ms window"
        })
        
        # Step 2: Feature extraction
        feature_count = len(features)
        path.steps.append({
            "step": "feature_extraction",
            "description": f"Extracted {feature_count} features from signal",
            "details": f"Computed time-domain and frequency-domain features"
        })
        
        # Step 3: Classification
        path.steps.append({
            "step": "classification",
            "description": f"Applied {model_type} classifier",
            "details": f"Predicted '{gesture}' with {confidence:.3f} confidence"
        })
        
        # Step 4: Confidence thresholding
        threshold = 0.6  # Would get from config
        if confidence >= threshold:
            path.steps.append({
                "step": "threshold_check",
                "description": "Confidence above threshold - prediction accepted",
                "details": f"Confidence {confidence:.3f} >= threshold {threshold}"
            })
            path.final_decision = gesture
        else:
            path.steps.append({
                "step": "threshold_check",
                "description": "Confidence below threshold - defaulting to 'rest'",
                "details": f"Confidence {confidence:.3f} < threshold {threshold}"
            })
            path.final_decision = "rest"
        
        # Add decision factors
        path.decision_factors = [
            f"Model prediction: {gesture}",
            f"Confidence level: {confidence:.3f}",
            f"Feature quality assessment",
            f"Threshold comparison: {threshold}"
        ]
        
        return {"decision_path": path}
    
    def get_explanation_types(self) -> List[ExplanationType]:
        return [ExplanationType.DECISION_PATH]


class DecisionExplainer:
    """
    Main explainer class that coordinates different explanation components
    """
    
    def __init__(self, config: ExplanationConfig, feature_names: List[str]):
        self.config = config
        self.feature_names = feature_names
        self.logger = logging.getLogger(__name__)
        
        # Initialize explainer components
        self.explainers = {
            ExplanationType.FEATURE_IMPORTANCE: FeatureImportanceExplainer(feature_names),
            ExplanationType.CONFIDENCE_BREAKDOWN: ConfidenceExplainer(),
            ExplanationType.DECISION_PATH: DecisionPathExplainer()
        }
        
        # Gesture similarity data (would be loaded from training data)
        self.gesture_similarities = self._initialize_gesture_similarities()
    
    def _initialize_gesture_similarities(self) -> Dict[str, List[Tuple[str, float]]]:
        """Initialize gesture similarity mappings"""
        
        # Simplified similarity mappings - would be computed from actual data
        similarities = {
            "fist": [("pinch", 0.7), ("four", 0.5), ("five", 0.3)],
            "open": [("five", 0.8), ("four", 0.6), ("peace", 0.4)],
            "pinch": [("fist", 0.7), ("point", 0.6), ("peace", 0.4)],
            "point": [("pinch", 0.6), ("peace", 0.5), ("fist", 0.3)],
            "four": [("five", 0.9), ("open", 0.6), ("fist", 0.5)],
            "five": [("four", 0.9), ("open", 0.8), ("peace", 0.4)],
            "peace": [("point", 0.5), ("pinch", 0.4), ("open", 0.4)],
            "rest": [("open", 0.2), ("five", 0.1), ("four", 0.1)]
        }
        
        return similarities
    
    def explain_prediction(
        self,
        gesture: str,
        confidence: float,
        features: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None,
        model_info: Optional[Dict[str, Any]] = None,
        alternatives: Optional[List[Tuple[str, float]]] = None
    ) -> ExplanationResult:
        """
        Generate comprehensive explanation for a prediction
        
        Args:
            gesture: Predicted gesture
            confidence: Prediction confidence
            features: Extracted features
            feature_importance: Feature importance scores
            model_info: Information about the model used
            alternatives: Alternative predictions with scores
            
        Returns:
            Complete explanation result
        """
        
        # Prepare prediction data
        prediction_data = {
            "gesture": gesture,
            "confidence": confidence,
            "features": features,
            "feature_importance": feature_importance or {},
            "model_info": model_info or {},
            "alternatives": alternatives or [],
            "model_type": model_info.get("type", "unknown") if model_info else "unknown"
        }
        
        # Initialize result
        result = ExplanationResult(
            predicted_gesture=gesture,
            confidence=confidence,
            timestamp=time.time(),
            model_info=model_info or {},
            alternative_predictions=alternatives or []
        )
        
        # Generate feature importance explanation
        if self.config.include_feature_importance and feature_importance:
            try:
                feature_explanation = self.explainers[ExplanationType.FEATURE_IMPORTANCE].explain(prediction_data)
                result.feature_explanations = feature_explanation.get("feature_explanations", [])
                result.top_features = feature_explanation.get("top_features", [])
            except Exception as e:
                self.logger.warning(f"Failed to generate feature importance explanation: {e}")
        
        # Generate confidence breakdown
        if self.config.include_confidence_breakdown:
            try:
                confidence_explanation = self.explainers[ExplanationType.CONFIDENCE_BREAKDOWN].explain(prediction_data)
                result.confidence_breakdown = confidence_explanation.get("confidence_breakdown")
                result.signal_quality = confidence_explanation.get("signal_quality", {})
            except Exception as e:
                self.logger.warning(f"Failed to generate confidence explanation: {e}")
        
        # Generate decision path
        if self.config.include_decision_path:
            try:
                path_explanation = self.explainers[ExplanationType.DECISION_PATH].explain(prediction_data)
                result.decision_path = path_explanation.get("decision_path")
            except Exception as e:
                self.logger.warning(f"Failed to generate decision path explanation: {e}")
        
        # Generate human-readable explanations
        if self.config.generate_human_readable:
            result.simple_explanation = self._generate_simple_explanation(result)
            result.detailed_explanation = self._generate_detailed_explanation(result)
            if self.config.explanation_level == ExplanationLevel.TECHNICAL:
                result.technical_explanation = self._generate_technical_explanation(result)
        
        # Generate recommendations and warnings
        if self.config.include_recommendations:
            result.recommendations = self._generate_recommendations(result)
            result.warnings = self._generate_warnings(result)
        
        return result
    
    def _generate_simple_explanation(self, result: ExplanationResult) -> str:
        """Generate simple, user-friendly explanation"""
        
        gesture = result.predicted_gesture
        confidence = result.confidence
        
        if confidence >= 0.8:
            certainty = "very confident"
        elif confidence >= 0.6:
            certainty = "confident"
        elif confidence >= 0.4:
            certainty = "somewhat confident"
        else:
            certainty = "uncertain"
        
        explanation = f"I am {certainty} that you made a '{gesture}' gesture."
        
        # Add top reason if available
        if result.feature_explanations:
            top_feature = result.feature_explanations[0]
            explanation += f" This is mainly because {top_feature.human_readable.lower()}."
        
        return explanation
    
    def _generate_detailed_explanation(self, result: ExplanationResult) -> str:
        """Generate detailed explanation"""
        
        parts = []
        
        # Basic prediction info
        parts.append(f"Predicted gesture: {result.predicted_gesture} (confidence: {result.confidence:.3f})")
        
        # Feature importance
        if result.feature_explanations:
            parts.append("Key factors in this decision:")
            for i, feature_exp in enumerate(result.feature_explanations[:3]):
                parts.append(f"  {i+1}. {feature_exp.human_readable}")
        
        # Confidence factors
        if result.confidence_breakdown:
            cb = result.confidence_breakdown
            parts.append(f"Confidence based on: signal quality ({cb.feature_quality_factor:.2f}), "
                        f"model certainty ({cb.model_certainty:.2f}), noise level ({cb.noise_factor:.2f})")
        
        # Alternative predictions
        if result.alternative_predictions:
            alts = [f"{gesture} ({conf:.3f})" for gesture, conf in result.alternative_predictions[:2]]
            parts.append(f"Other possibilities considered: {', '.join(alts)}")
        
        return " ".join(parts)
    
    def _generate_technical_explanation(self, result: ExplanationResult) -> str:
        """Generate technical explanation with full details"""
        
        technical = {
            "prediction": {
                "gesture": result.predicted_gesture,
                "confidence": result.confidence,
                "model": result.model_info.get("type", "unknown")
            },
            "features": [
                {
                    "name": fe.feature_name,
                    "importance": fe.importance_score,
                    "value": fe.value,
                    "contribution": fe.contribution
                } for fe in result.feature_explanations
            ],
            "decision_path": result.decision_path.steps if result.decision_path else [],
            "signal_quality": result.signal_quality,
            "alternatives": result.alternative_predictions
        }
        
        return json.dumps(technical, indent=2)
    
    def _generate_recommendations(self, result: ExplanationResult) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Low confidence recommendations
        if result.confidence < self.config.confidence_threshold_warning:
            recommendations.append("Try repositioning the EMG sensors for better signal quality")
            recommendations.append("Ensure the gesture is held clearly and consistently")
            
            if result.signal_quality.get("signal_strength", 0) < 0.3:
                recommendations.append("Check sensor contact - signal strength is low")
        
        # High noise recommendations
        if result.signal_quality.get("stability", 1.0) < 0.6:
            recommendations.append("Reduce muscle tension in non-target muscles")
            recommendations.append("Ensure stable arm positioning")
        
        # Alternative gesture suggestions
        if result.alternative_predictions:
            top_alt = result.alternative_predictions[0]
            if top_alt[1] > 0.4:  # Close alternative
                recommendations.append(f"Be careful to distinguish from '{top_alt[0]}' gesture")
        
        return recommendations
    
    def _generate_warnings(self, result: ExplanationResult) -> List[str]:
        """Generate warnings about potential issues"""
        
        warnings = []
        
        # Low confidence warning
        if result.confidence < 0.4:
            warnings.append("Very low confidence - prediction may be unreliable")
        
        # Signal quality warnings
        if result.signal_quality.get("signal_strength", 1.0) < 0.2:
            warnings.append("Signal strength is very low")
        
        if result.signal_quality.get("stability", 1.0) < 0.4:
            warnings.append("Signal is unstable - high noise detected")
        
        # Model warnings
        if result.model_info.get("type") == "unknown":
            warnings.append("Model information unavailable")
        
        return warnings
    
    def explain_comparison(
        self, 
        prediction1: ExplanationResult, 
        prediction2: ExplanationResult
    ) -> Dict[str, Any]:
        """Compare two predictions and explain the differences"""
        
        comparison = {
            "gesture_change": {
                "from": prediction1.predicted_gesture,
                "to": prediction2.predicted_gesture,
                "changed": prediction1.predicted_gesture != prediction2.predicted_gesture
            },
            "confidence_change": {
                "from": prediction1.confidence,
                "to": prediction2.confidence,
                "delta": prediction2.confidence - prediction1.confidence
            },
            "feature_changes": [],
            "explanation": ""
        }
        
        # Analyze feature changes
        if prediction1.feature_explanations and prediction2.feature_explanations:
            features1 = {fe.feature_name: fe for fe in prediction1.feature_explanations}
            features2 = {fe.feature_name: fe for fe in prediction2.feature_explanations}
            
            for feature_name in features1:
                if feature_name in features2:
                    fe1, fe2 = features1[feature_name], features2[feature_name]
                    value_change = fe2.value - fe1.value
                    importance_change = fe2.importance_score - fe1.importance_score
                    
                    if abs(value_change) > 0.1 or abs(importance_change) > 0.1:
                        comparison["feature_changes"].append({
                            "feature": feature_name,
                            "value_change": value_change,
                            "importance_change": importance_change
                        })
        
        # Generate explanation
        if comparison["gesture_change"]["changed"]:
            comparison["explanation"] = (
                f"Gesture changed from '{prediction1.predicted_gesture}' to "
                f"'{prediction2.predicted_gesture}' due to changes in signal characteristics"
            )
        else:
            conf_change = comparison["confidence_change"]["delta"]
            if abs(conf_change) > 0.1:
                direction = "increased" if conf_change > 0 else "decreased"
                comparison["explanation"] = f"Confidence {direction} by {abs(conf_change):.3f}"
            else:
                comparison["explanation"] = "Prediction remained stable"
        
        return comparison
    
    def get_explanation_capability(self) -> Dict[str, Any]:
        """Get information about explanation capabilities"""
        
        return {
            "supported_explanation_types": [et.value for et in ExplanationType],
            "available_explainers": list(self.explainers.keys()),
            "feature_count": len(self.feature_names),
            "config": self.config.dict(),
            "gesture_similarities": self.gesture_similarities
        }