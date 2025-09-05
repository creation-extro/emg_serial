"""
Advanced Gesture-to-Action Mapping Layer

This module provides sophisticated mapping between recognized gestures and action commands,
with support for context-aware adaptation, user profiles, and explainable mappings.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class ActionType(Enum):
    """Types of actions that can be triggered by gestures"""
    MOTOR_CONTROL = "motor_control"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    COMMUNICATION = "communication"
    SYSTEM = "system"


class Priority(Enum):
    """Priority levels for action execution"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ActionSpec:
    """Specification for an action that can be triggered by a gesture"""
    action_id: str
    action_type: ActionType
    priority: Priority
    params: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    safety_checks: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.6
    timeout_ms: Optional[int] = None
    description: str = ""


class MappingConfig(BaseModel):
    """Configuration for gesture mapping behavior"""
    
    # Base mapping settings
    default_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    priority_boost_factor: float = Field(default=0.1, ge=0.0, le=0.5)
    context_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Adaptation settings
    enable_adaptive_mapping: bool = Field(default=True)
    adaptation_rate: float = Field(default=0.05, ge=0.0, le=0.5)
    min_confidence_delta: float = Field(default=0.05, ge=0.0, le=0.3)
    
    # Safety settings
    require_safety_checks: bool = Field(default=True)
    max_concurrent_actions: int = Field(default=3, ge=1, le=10)
    action_cooldown_ms: int = Field(default=250, ge=0, le=5000)
    
    # Profile settings
    enable_user_profiles: bool = Field(default=True)
    profile_learning_rate: float = Field(default=0.02, ge=0.0, le=0.1)


class GestureMapper:
    """
    Advanced gesture-to-action mapping with context awareness and adaptation
    """
    
    def __init__(self, config: MappingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core mapping data
        self.gesture_mappings: Dict[str, List[ActionSpec]] = {}
        self.context_mappings: Dict[str, Dict[str, List[ActionSpec]]] = {}
        self.active_actions: Dict[str, float] = {}  # action_id -> timestamp
        
        # Adaptation state
        self.confidence_history: Dict[str, List[float]] = {}
        self.success_rates: Dict[str, float] = {}
        self.adaptation_weights: Dict[str, float] = {}
        
        # Initialize default mappings
        self._initialize_default_mappings()
    
    def _initialize_default_mappings(self):
        """Initialize default gesture-to-action mappings"""
        
        # Hand gesture mappings
        self.gesture_mappings = {
            "rest": [
                ActionSpec(
                    action_id="system_idle",
                    action_type=ActionType.SYSTEM,
                    priority=Priority.LOW,
                    description="System idle state"
                )
            ],
            "fist": [
                ActionSpec(
                    action_id="grip_activate",
                    action_type=ActionType.MANIPULATION,
                    priority=Priority.HIGH,
                    params={"force": 0.8, "grip_type": "power"},
                    confidence_threshold=0.7,
                    description="Activate power grip"
                ),
                ActionSpec(
                    action_id="emergency_stop",
                    action_type=ActionType.SYSTEM,
                    priority=Priority.CRITICAL,
                    params={"reason": "emergency_gesture"},
                    confidence_threshold=0.9,
                    description="Emergency stop all motors"
                )
            ],
            "open": [
                ActionSpec(
                    action_id="grip_release",
                    action_type=ActionType.MANIPULATION,
                    priority=Priority.HIGH,
                    params={"release_speed": "gradual"},
                    description="Release current grip"
                ),
                ActionSpec(
                    action_id="hand_open",
                    action_type=ActionType.MOTOR_CONTROL,
                    priority=Priority.MEDIUM,
                    params={"target_position": "fully_open"},
                    description="Open prosthetic hand"
                )
            ],
            "pinch": [
                ActionSpec(
                    action_id="precision_grip",
                    action_type=ActionType.MANIPULATION,
                    priority=Priority.HIGH,
                    params={"force": 0.3, "grip_type": "precision"},
                    confidence_threshold=0.65,
                    description="Activate precision grip"
                )
            ],
            "point": [
                ActionSpec(
                    action_id="pointing_gesture",
                    action_type=ActionType.COMMUNICATION,
                    priority=Priority.MEDIUM,
                    params={"gesture_type": "directional"},
                    description="Pointing/directional gesture"
                ),
                ActionSpec(
                    action_id="index_extend",
                    action_type=ActionType.MOTOR_CONTROL,
                    priority=Priority.MEDIUM,
                    params={"finger": "index", "position": "extended"},
                    description="Extend index finger"
                )
            ]
        }
        
        # Context-specific mappings
        self.context_mappings = {
            "manipulation_mode": {
                "fist": [
                    ActionSpec(
                        action_id="strong_grip",
                        action_type=ActionType.MANIPULATION,
                        priority=Priority.HIGH,
                        params={"force": 1.0, "grip_type": "power"},
                        confidence_threshold=0.75,
                        description="Maximum force grip for heavy objects"
                    )
                ]
            },
            "precision_mode": {
                "pinch": [
                    ActionSpec(
                        action_id="micro_pinch",
                        action_type=ActionType.MANIPULATION,
                        priority=Priority.HIGH,
                        params={"force": 0.1, "precision": "high"},
                        confidence_threshold=0.8,
                        description="Ultra-precise pinch for delicate objects"
                    )
                ]
            },
            "navigation_mode": {
                "point": [
                    ActionSpec(
                        action_id="direction_command",
                        action_type=ActionType.NAVIGATION,
                        priority=Priority.HIGH,
                        params={"command_type": "directional"},
                        description="Navigation direction command"
                    )
                ]
            }
        }
    
    def map_gesture_to_actions(
        self, 
        gesture: str, 
        confidence: float,
        context: Optional[str] = None,
        user_profile: Optional[str] = None
    ) -> Tuple[List[ActionSpec], Dict[str, Any]]:
        """
        Map a gesture to action specifications with context and confidence consideration
        
        Returns:
            Tuple of (selected_actions, explanation_data)
        """
        
        explanation = {
            "gesture": gesture,
            "confidence": confidence,
            "context": context,
            "user_profile": user_profile,
            "decision_factors": [],
            "rejected_actions": [],
            "confidence_adjustments": {}
        }
        
        # Get candidate actions
        candidates = self._get_candidate_actions(gesture, context)
        
        if not candidates:
            explanation["decision_factors"].append("No actions mapped for gesture")
            return [], explanation
        
        # Filter by confidence and safety
        valid_actions = []
        for action in candidates:
            adjusted_confidence = self._adjust_confidence_for_context(
                confidence, action, context, user_profile
            )
            
            explanation["confidence_adjustments"][action.action_id] = {
                "original": confidence,
                "adjusted": adjusted_confidence,
                "threshold": action.confidence_threshold
            }
            
            if adjusted_confidence >= action.confidence_threshold:
                if self._check_safety_constraints(action):
                    valid_actions.append(action)
                    explanation["decision_factors"].append(
                        f"Action {action.action_id} passed confidence and safety checks"
                    )
                else:
                    explanation["rejected_actions"].append({
                        "action_id": action.action_id,
                        "reason": "Failed safety constraints"
                    })
            else:
                explanation["rejected_actions"].append({
                    "action_id": action.action_id,
                    "reason": f"Confidence {adjusted_confidence:.3f} below threshold {action.confidence_threshold}"
                })
        
        # Prioritize actions
        valid_actions.sort(key=lambda a: (a.priority.value, -confidence))
        
        # Apply concurrent action limits
        if len(valid_actions) > self.config.max_concurrent_actions:
            explanation["decision_factors"].append(
                f"Limited to {self.config.max_concurrent_actions} concurrent actions"
            )
            valid_actions = valid_actions[:self.config.max_concurrent_actions]
        
        return valid_actions, explanation
    
    def _get_candidate_actions(self, gesture: str, context: Optional[str]) -> List[ActionSpec]:
        """Get all candidate actions for a gesture and context"""
        candidates = []
        
        # Base gesture mappings
        if gesture in self.gesture_mappings:
            candidates.extend(self.gesture_mappings[gesture])
        
        # Context-specific mappings
        if context and context in self.context_mappings:
            if gesture in self.context_mappings[context]:
                candidates.extend(self.context_mappings[context][gesture])
        
        return candidates
    
    def _adjust_confidence_for_context(
        self, 
        base_confidence: float, 
        action: ActionSpec,
        context: Optional[str],
        user_profile: Optional[str]
    ) -> float:
        """Adjust confidence based on context and user profile"""
        
        adjusted = base_confidence
        
        # Context boost
        if context and self.config.context_weight > 0:
            context_boost = self._get_context_boost(action, context)
            adjusted += context_boost * self.config.context_weight
        
        # User profile adjustment
        if user_profile and self.config.enable_user_profiles:
            profile_adjustment = self._get_profile_adjustment(action, user_profile)
            adjusted += profile_adjustment
        
        # Adaptive adjustment based on historical success
        if self.config.enable_adaptive_mapping:
            adaptation_adjustment = self._get_adaptation_adjustment(action)
            adjusted += adaptation_adjustment
        
        return min(1.0, max(0.0, adjusted))
    
    def _get_context_boost(self, action: ActionSpec, context: str) -> float:
        """Calculate confidence boost based on context relevance"""
        
        context_relevance = {
            ("manipulation_mode", ActionType.MANIPULATION): 0.15,
            ("precision_mode", ActionType.MANIPULATION): 0.20,
            ("navigation_mode", ActionType.NAVIGATION): 0.18,
            ("communication_mode", ActionType.COMMUNICATION): 0.12,
        }
        
        return context_relevance.get((context, action.action_type), 0.0)
    
    def _get_profile_adjustment(self, action: ActionSpec, user_profile: str) -> float:
        """Get user profile-based confidence adjustment"""
        
        # This would be loaded from user profile data
        profile_preferences = {
            "power_user": {
                ActionType.MANIPULATION: 0.1,
                ActionType.MOTOR_CONTROL: 0.05
            },
            "precision_user": {
                ActionType.MANIPULATION: 0.15,
                ActionType.COMMUNICATION: 0.05
            },
            "adaptive_user": {
                ActionType.SYSTEM: 0.05
            }
        }
        
        return profile_preferences.get(user_profile, {}).get(action.action_type, 0.0)
    
    def _get_adaptation_adjustment(self, action: ActionSpec) -> float:
        """Get adaptive confidence adjustment based on historical performance"""
        
        action_id = action.action_id
        
        if action_id not in self.success_rates:
            return 0.0
        
        success_rate = self.success_rates[action_id]
        baseline_rate = 0.8  # Expected success rate
        
        # Boost or reduce confidence based on historical success
        delta = (success_rate - baseline_rate) * self.config.adaptation_rate
        return max(-0.2, min(0.2, delta))  # Clamp adjustment
    
    def _check_safety_constraints(self, action: ActionSpec) -> bool:
        """Check if action meets safety constraints"""
        
        if not self.config.require_safety_checks:
            return True
        
        # Check action cooldown
        import time
        now = time.time() * 1000  # Convert to ms
        
        if action.action_id in self.active_actions:
            last_execution = self.active_actions[action.action_id]
            if now - last_execution < self.config.action_cooldown_ms:
                return False
        
        # Check safety preconditions (would be implemented based on specific requirements)
        for precondition in action.preconditions:
            if not self._check_precondition(precondition):
                return False
        
        return True
    
    def _check_precondition(self, precondition: str) -> bool:
        """Check a specific safety precondition"""
        
        # Example precondition checks
        precondition_checks = {
            "system_ready": True,  # Would check actual system state
            "no_collision_risk": True,  # Would check collision detection
            "motor_operational": True,  # Would check motor status
            "user_authorized": True,  # Would check user permissions
        }
        
        return precondition_checks.get(precondition, True)
    
    def update_success_rate(self, action_id: str, success: bool):
        """Update the success rate for an action based on execution outcome"""
        
        if action_id not in self.success_rates:
            self.success_rates[action_id] = 0.8  # Start with baseline
        
        # Exponential moving average
        current_rate = self.success_rates[action_id]
        new_rate = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
        self.success_rates[action_id] = new_rate
    
    def add_custom_mapping(self, gesture: str, action: ActionSpec, context: Optional[str] = None):
        """Add a custom gesture-to-action mapping"""
        
        if context:
            if context not in self.context_mappings:
                self.context_mappings[context] = {}
            if gesture not in self.context_mappings[context]:
                self.context_mappings[context][gesture] = []
            self.context_mappings[context][gesture].append(action)
        else:
            if gesture not in self.gesture_mappings:
                self.gesture_mappings[gesture] = []
            self.gesture_mappings[gesture].append(action)
        
        self.logger.info(f"Added custom mapping: {gesture} -> {action.action_id} (context: {context})")
    
    def get_mapping_statistics(self) -> Dict[str, Any]:
        """Get statistics about current mappings and performance"""
        
        total_mappings = sum(len(actions) for actions in self.gesture_mappings.values())
        context_mappings = sum(
            sum(len(actions) for actions in context.values()) 
            for context in self.context_mappings.values()
        )
        
        return {
            "total_base_mappings": total_mappings,
            "total_context_mappings": context_mappings,
            "gestures_mapped": len(self.gesture_mappings),
            "contexts_available": len(self.context_mappings),
            "success_rates": dict(self.success_rates),
            "config": self.config.dict()
        }