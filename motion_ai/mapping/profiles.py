"""
Profile Management System for Motion AI

Handles user profiles, adaptive preferences, and configuration-driven behavior.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import logging

from pydantic import BaseModel, Field


class UserType(Enum):
    """Types of users with different needs and capabilities"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    CLINICAL = "clinical"
    RESEARCH = "research"


class ControlStyle(Enum):
    """Different control style preferences"""
    CONSERVATIVE = "conservative"  # High confidence thresholds, safe actions
    BALANCED = "balanced"         # Standard thresholds and response
    AGGRESSIVE = "aggressive"     # Lower thresholds, faster response
    ADAPTIVE = "adaptive"         # Learns and adapts to user behavior


@dataclass
class UserPreferences:
    """User-specific preferences and settings"""
    confidence_sensitivity: float = 0.0  # Adjustment to base confidence thresholds
    response_speed: float = 1.0          # Speed multiplier for actions
    gesture_priority: Dict[str, float] = field(default_factory=dict)
    action_preferences: Dict[str, float] = field(default_factory=dict)
    safety_level: float = 1.0            # Safety setting multiplier
    learning_enabled: bool = True
    feedback_frequency: str = "normal"   # "minimal", "normal", "verbose"


class ProfileConfig(BaseModel):
    """Configuration for a user profile"""
    
    # Basic profile info
    profile_id: str
    user_type: UserType
    control_style: ControlStyle
    
    # Core settings
    base_confidence_threshold: float = Field(default=0.6, ge=0.1, le=0.95)
    confidence_adaptation_rate: float = Field(default=0.05, ge=0.0, le=0.3)
    action_timeout_ms: int = Field(default=500, ge=100, le=5000)
    
    # Safety settings
    safety_multiplier: float = Field(default=1.0, ge=0.5, le=2.0)
    enable_emergency_gestures: bool = Field(default=True)
    require_confirmation: List[str] = Field(default_factory=list)
    
    # Adaptation settings
    enable_learning: bool = Field(default=True)
    learning_rate: float = Field(default=0.02, ge=0.0, le=0.1)
    adaptation_window_size: int = Field(default=50, ge=10, le=200)
    
    # Gesture settings
    gesture_weights: Dict[str, float] = Field(default_factory=dict)
    disabled_gestures: List[str] = Field(default_factory=list)
    custom_gestures: Dict[str, Any] = Field(default_factory=dict)
    
    # Interface settings
    feedback_level: str = Field(default="normal")  # "minimal", "normal", "verbose"
    audio_feedback: bool = Field(default=True)
    haptic_feedback: bool = Field(default=True)
    visual_feedback: bool = Field(default=True)
    
    # Performance tracking
    track_performance: bool = Field(default=True)
    performance_history_size: int = Field(default=1000, ge=100, le=10000)


class ProfileManager:
    """
    Manages user profiles and configuration-driven behavior adaptation
    """
    
    def __init__(self, profiles_dir: str = "profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Active profiles and states
        self.profiles: Dict[str, ProfileConfig] = {}
        self.current_profile: Optional[str] = None
        self.profile_states: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Load existing profiles
        self._load_profiles()
        
        # Create default profiles if none exist
        if not self.profiles:
            self._create_default_profiles()
    
    def _load_profiles(self):
        """Load all profiles from disk"""
        
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                    profile = ProfileConfig(**data)
                    self.profiles[profile.profile_id] = profile
                    self.logger.info(f"Loaded profile: {profile.profile_id}")
            except Exception as e:
                self.logger.error(f"Failed to load profile {profile_file}: {e}")
    
    def _create_default_profiles(self):
        """Create default user profiles"""
        
        default_profiles = [
            ProfileConfig(
                profile_id="beginner_conservative",
                user_type=UserType.BEGINNER,
                control_style=ControlStyle.CONSERVATIVE,
                base_confidence_threshold=0.8,
                safety_multiplier=1.5,
                action_timeout_ms=1000,
                require_confirmation=["emergency_stop", "strong_grip"],
                feedback_level="verbose"
            ),
            ProfileConfig(
                profile_id="intermediate_balanced",
                user_type=UserType.INTERMEDIATE,
                control_style=ControlStyle.BALANCED,
                base_confidence_threshold=0.65,
                safety_multiplier=1.0,
                action_timeout_ms=500,
                enable_learning=True,
                feedback_level="normal"
            ),
            ProfileConfig(
                profile_id="advanced_aggressive",
                user_type=UserType.ADVANCED,
                control_style=ControlStyle.AGGRESSIVE,
                base_confidence_threshold=0.5,
                safety_multiplier=0.8,
                action_timeout_ms=250,
                learning_rate=0.05,
                feedback_level="minimal"
            ),
            ProfileConfig(
                profile_id="adaptive_learner",
                user_type=UserType.INTERMEDIATE,
                control_style=ControlStyle.ADAPTIVE,
                base_confidence_threshold=0.6,
                confidence_adaptation_rate=0.1,
                enable_learning=True,
                learning_rate=0.03,
                adaptation_window_size=30
            ),
            ProfileConfig(
                profile_id="clinical_standard",
                user_type=UserType.CLINICAL,
                control_style=ControlStyle.BALANCED,
                base_confidence_threshold=0.7,
                safety_multiplier=1.2,
                track_performance=True,
                performance_history_size=5000,
                feedback_level="verbose"
            ),
            ProfileConfig(
                profile_id="research_detailed",
                user_type=UserType.RESEARCH,
                control_style=ControlStyle.ADAPTIVE,
                base_confidence_threshold=0.55,
                enable_learning=True,
                track_performance=True,
                performance_history_size=10000,
                feedback_level="verbose"
            )
        ]
        
        for profile in default_profiles:
            self.save_profile(profile)
            self.logger.info(f"Created default profile: {profile.profile_id}")
    
    def get_profile(self, profile_id: str) -> Optional[ProfileConfig]:
        """Get a profile by ID"""
        return self.profiles.get(profile_id)
    
    def set_active_profile(self, profile_id: str) -> bool:
        """Set the active profile"""
        if profile_id not in self.profiles:
            self.logger.error(f"Profile not found: {profile_id}")
            return False
        
        self.current_profile = profile_id
        self.logger.info(f"Activated profile: {profile_id}")
        return True
    
    def get_active_profile(self) -> Optional[ProfileConfig]:
        """Get the currently active profile"""
        if self.current_profile:
            return self.profiles.get(self.current_profile)
        return None
    
    def create_profile(self, profile_config: ProfileConfig) -> bool:
        """Create a new user profile"""
        try:
            self.profiles[profile_config.profile_id] = profile_config
            self.save_profile(profile_config)
            self.logger.info(f"Created profile: {profile_config.profile_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create profile: {e}")
            return False
    
    def save_profile(self, profile: ProfileConfig):
        """Save a profile to disk"""
        try:
            profile_file = self.profiles_dir / f"{profile.profile_id}.json"
            with open(profile_file, 'w') as f:
                json.dump(profile.dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save profile {profile.profile_id}: {e}")
    
    def update_profile(self, profile_id: str, updates: Dict[str, Any]) -> bool:
        """Update a profile with new settings"""
        if profile_id not in self.profiles:
            return False
        
        try:
            profile = self.profiles[profile_id]
            profile_dict = profile.dict()
            profile_dict.update(updates)
            
            updated_profile = ProfileConfig(**profile_dict)
            self.profiles[profile_id] = updated_profile
            self.save_profile(updated_profile)
            
            self.logger.info(f"Updated profile: {profile_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update profile {profile_id}: {e}")
            return False
    
    def adapt_confidence_threshold(
        self, 
        profile_id: str, 
        gesture: str, 
        success: bool,
        current_confidence: float
    ):
        """Adapt confidence threshold based on performance feedback"""
        
        profile = self.get_profile(profile_id)
        if not profile or not profile.enable_learning:
            return
        
        # Initialize profile state if needed
        if profile_id not in self.profile_states:
            self.profile_states[profile_id] = {
                "gesture_performance": {},
                "adaptation_counts": {}
            }
        
        state = self.profile_states[profile_id]
        
        # Track gesture performance
        if gesture not in state["gesture_performance"]:
            state["gesture_performance"][gesture] = {
                "successes": 0,
                "attempts": 0,
                "confidence_sum": 0.0
            }
        
        perf = state["gesture_performance"][gesture]
        perf["attempts"] += 1
        perf["confidence_sum"] += current_confidence
        
        if success:
            perf["successes"] += 1
        
        # Calculate success rate and average confidence
        success_rate = perf["successes"] / perf["attempts"]
        avg_confidence = perf["confidence_sum"] / perf["attempts"]
        
        # Adapt threshold if we have enough data
        if perf["attempts"] >= profile.adaptation_window_size:
            target_success_rate = 0.85  # Target 85% success rate
            
            if success_rate < target_success_rate - 0.1:
                # Increase threshold (be more conservative)
                adjustment = profile.learning_rate
            elif success_rate > target_success_rate + 0.1:
                # Decrease threshold (be more aggressive)
                adjustment = -profile.learning_rate
            else:
                adjustment = 0.0
            
            if adjustment != 0.0:
                current_weights = profile.gesture_weights.copy()
                current_weights[gesture] = current_weights.get(gesture, 0.0) + adjustment
                
                # Clamp the adjustment
                current_weights[gesture] = max(-0.3, min(0.3, current_weights[gesture]))
                
                self.update_profile(profile_id, {"gesture_weights": current_weights})
                
                self.logger.info(
                    f"Adapted {gesture} threshold for {profile_id}: "
                    f"success_rate={success_rate:.3f}, adjustment={adjustment:.3f}"
                )
    
    def get_effective_confidence_threshold(
        self, 
        profile_id: str, 
        gesture: str, 
        base_threshold: float
    ) -> float:
        """Get the effective confidence threshold for a gesture and profile"""
        
        profile = self.get_profile(profile_id)
        if not profile:
            return base_threshold
        
        # Start with profile base threshold
        threshold = profile.base_confidence_threshold
        
        # Apply gesture-specific weight
        gesture_weight = profile.gesture_weights.get(gesture, 0.0)
        threshold += gesture_weight
        
        # Apply control style modification
        style_modifiers = {
            ControlStyle.CONSERVATIVE: 0.15,
            ControlStyle.BALANCED: 0.0,
            ControlStyle.AGGRESSIVE: -0.15,
            ControlStyle.ADAPTIVE: 0.0  # Handled by learning
        }
        
        threshold += style_modifiers.get(profile.control_style, 0.0)
        
        # Clamp to valid range
        return max(0.1, min(0.95, threshold))
    
    def is_gesture_enabled(self, profile_id: str, gesture: str) -> bool:
        """Check if a gesture is enabled for a profile"""
        profile = self.get_profile(profile_id)
        if not profile:
            return True
        
        return gesture not in profile.disabled_gestures
    
    def should_require_confirmation(self, profile_id: str, action_id: str) -> bool:
        """Check if an action requires confirmation for a profile"""
        profile = self.get_profile(profile_id)
        if not profile:
            return False
        
        return action_id in profile.require_confirmation
    
    def record_performance(
        self, 
        profile_id: str, 
        gesture: str, 
        action_id: str,
        confidence: float,
        success: bool,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record performance data for analysis"""
        
        profile = self.get_profile(profile_id)
        if not profile or not profile.track_performance:
            return
        
        if profile_id not in self.performance_history:
            self.performance_history[profile_id] = []
        
        record = {
            "timestamp": time.time(),
            "gesture": gesture,
            "action_id": action_id,
            "confidence": confidence,
            "success": success,
            "latency_ms": latency_ms,
            "metadata": metadata or {}
        }
        
        history = self.performance_history[profile_id]
        history.append(record)
        
        # Maintain history size limit
        if len(history) > profile.performance_history_size:
            history.pop(0)
    
    def get_profile_statistics(self, profile_id: str) -> Dict[str, Any]:
        """Get performance statistics for a profile"""
        
        if profile_id not in self.performance_history:
            return {}
        
        history = self.performance_history[profile_id]
        if not history:
            return {}
        
        # Calculate basic statistics
        total_attempts = len(history)
        successes = sum(1 for record in history if record["success"])
        success_rate = successes / total_attempts if total_attempts > 0 else 0.0
        
        avg_confidence = sum(record["confidence"] for record in history) / total_attempts
        avg_latency = sum(record["latency_ms"] for record in history) / total_attempts
        
        # Per-gesture statistics
        gesture_stats = {}
        for record in history:
            gesture = record["gesture"]
            if gesture not in gesture_stats:
                gesture_stats[gesture] = {"attempts": 0, "successes": 0, "confidence_sum": 0.0}
            
            stats = gesture_stats[gesture]
            stats["attempts"] += 1
            stats["confidence_sum"] += record["confidence"]
            if record["success"]:
                stats["successes"] += 1
        
        # Convert to rates
        for gesture, stats in gesture_stats.items():
            stats["success_rate"] = stats["successes"] / stats["attempts"]
            stats["avg_confidence"] = stats["confidence_sum"] / stats["attempts"]
            del stats["confidence_sum"]
        
        return {
            "total_attempts": total_attempts,
            "overall_success_rate": success_rate,
            "average_confidence": avg_confidence,
            "average_latency_ms": avg_latency,
            "gesture_statistics": gesture_stats
        }
    
    def list_profiles(self) -> List[str]:
        """List all available profile IDs"""
        return list(self.profiles.keys())
    
    def export_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Export a profile configuration"""
        profile = self.get_profile(profile_id)
        if not profile:
            return None
        
        return {
            "profile": profile.dict(),
            "statistics": self.get_profile_statistics(profile_id),
            "state": self.profile_states.get(profile_id, {})
        }
    
    def import_profile(self, profile_data: Dict[str, Any]) -> bool:
        """Import a profile configuration"""
        try:
            profile_config = ProfileConfig(**profile_data["profile"])
            self.create_profile(profile_config)
            
            # Import state if available
            if "state" in profile_data:
                self.profile_states[profile_config.profile_id] = profile_data["state"]
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to import profile: {e}")
            return False