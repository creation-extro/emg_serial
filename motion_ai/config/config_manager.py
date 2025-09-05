"""
Advanced Configuration Management System

Handles config-driven behavior, profile management, and adaptive configuration.
"""

from typing import Dict, List, Any, Optional, Union, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
import os
from pathlib import Path
import logging
import time
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, validator


class ConfigLevel(Enum):
    """Configuration hierarchy levels"""
    SYSTEM = "system"        # System-wide defaults
    PROFILE = "profile"      # User profile settings
    SESSION = "session"      # Session-specific overrides
    RUNTIME = "runtime"      # Runtime dynamic adjustments


class ConfigScope(Enum):
    """Configuration scope definitions"""
    GLOBAL = "global"                    # Affects entire system
    MODEL = "model"                      # Model-specific settings
    PREPROCESSING = "preprocessing"      # Signal preprocessing
    CLASSIFICATION = "classification"    # Classification behavior
    MAPPING = "mapping"                  # Gesture-to-action mapping
    SAFETY = "safety"                    # Safety system
    UI = "ui"                           # User interface
    MONITORING = "monitoring"            # Performance monitoring


@dataclass
class ConfigValidationRule:
    """Rule for validating configuration values"""
    field_path: str
    validator_func: Callable[[Any], bool]
    error_message: str
    severity: str = "error"  # "error", "warning", "info"


class SystemConfig(BaseModel):
    """Core system configuration"""
    
    # System identification
    system_id: str = Field(default="motion_ai_default")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")  # development, staging, production
    
    # Model configuration
    default_model_id: Optional[str] = None
    model_selection_strategy: str = Field(default="confidence")  # confidence, performance, ensemble
    enable_model_switching: bool = Field(default=True)
    model_switch_threshold: float = Field(default=0.1, ge=0.0, le=0.5)
    
    # Processing configuration
    sampling_rate_hz: float = Field(default=1000.0, ge=100.0, le=10000.0)
    window_size_ms: float = Field(default=200.0, ge=50.0, le=2000.0)
    overlap_ratio: float = Field(default=0.5, ge=0.0, le=0.9)
    
    # Classification thresholds
    confidence_threshold: float = Field(default=0.6, ge=0.1, le=0.95)
    uncertainty_threshold: float = Field(default=0.4, ge=0.1, le=0.8)
    stability_threshold: float = Field(default=0.8, ge=0.1, le=1.0)
    
    # Safety settings
    enable_safety_layer: bool = Field(default=True)
    emergency_stop_gesture: str = Field(default="fist")
    safety_timeout_ms: int = Field(default=5000, ge=1000, le=30000)
    max_action_rate_hz: float = Field(default=10.0, ge=1.0, le=100.0)
    
    # Performance settings
    enable_performance_monitoring: bool = Field(default=True)
    performance_window_size: int = Field(default=100, ge=10, le=1000)
    enable_adaptation: bool = Field(default=True)
    adaptation_rate: float = Field(default=0.05, ge=0.0, le=0.5)
    
    # Logging and debugging
    log_level: str = Field(default="INFO")
    enable_prediction_logging: bool = Field(default=True)
    enable_feature_logging: bool = Field(default=False)
    log_rotation_size_mb: int = Field(default=100, ge=10, le=1000)
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f'Environment must be one of {allowed}')
        return v


class ProfileConfig(BaseModel):
    """User profile configuration"""
    
    # Profile metadata
    profile_id: str
    profile_name: str
    user_type: str = Field(default="standard")  # beginner, standard, advanced, expert
    created_timestamp: float = Field(default_factory=time.time)
    last_updated: float = Field(default_factory=time.time)
    
    # User preferences
    preferred_gestures: List[str] = Field(default_factory=list)
    disabled_gestures: List[str] = Field(default_factory=list)
    gesture_sensitivity: Dict[str, float] = Field(default_factory=dict)
    
    # Adaptation settings
    enable_learning: bool = Field(default=True)
    learning_rate: float = Field(default=0.02, ge=0.0, le=0.1)
    confidence_adaptation: bool = Field(default=True)
    threshold_adaptation: bool = Field(default=True)
    
    # Interface preferences
    feedback_level: str = Field(default="normal")  # minimal, normal, detailed, verbose
    enable_audio_feedback: bool = Field(default=True)
    enable_haptic_feedback: bool = Field(default=True)
    enable_visual_feedback: bool = Field(default=True)
    
    # Safety preferences
    safety_level: str = Field(default="standard")  # relaxed, standard, strict, maximum
    require_confirmation: List[str] = Field(default_factory=list)
    enable_emergency_gestures: bool = Field(default=True)
    
    # Performance preferences
    prioritize_accuracy: bool = Field(default=True)
    prioritize_speed: bool = Field(default=False)
    enable_advanced_features: bool = Field(default=False)
    
    # Custom overrides
    custom_thresholds: Dict[str, float] = Field(default_factory=dict)
    custom_mappings: Dict[str, Any] = Field(default_factory=dict)
    custom_features: Dict[str, Any] = Field(default_factory=dict)


class ConfigProfile(BaseModel):
    """Complete configuration profile combining system and user settings"""
    
    # Profile metadata
    profile_id: str
    profile_name: str
    base_profile: Optional[str] = None  # Inherits from another profile
    
    # Configuration sections
    system: SystemConfig
    user: ProfileConfig
    
    # Runtime state
    is_active: bool = Field(default=False)
    last_loaded: float = Field(default=0.0)
    load_count: int = Field(default=0)
    
    # Validation and metadata
    schema_version: str = Field(default="1.0")
    checksum: Optional[str] = None
    
    def get_effective_value(self, config_path: str, default: Any = None) -> Any:
        """Get effective configuration value following precedence rules"""
        
        # Parse config path (e.g., "system.confidence_threshold" or "user.learning_rate")
        parts = config_path.split('.')
        
        if len(parts) < 2:
            return default
        
        section = parts[0]
        field_path = '.'.join(parts[1:])
        
        # Get from appropriate section
        if section == "system":
            config_obj = self.system
        elif section == "user":
            config_obj = self.user
        else:
            return default
        
        # Navigate through nested fields
        current = config_obj
        for field in field_path.split('.'):
            if hasattr(current, field):
                current = getattr(current, field)
            elif isinstance(current, dict) and field in current:
                current = current[field]
            else:
                return default
        
        return current


class ConfigManager:
    """
    Advanced configuration management with profiles, validation, and adaptation
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Configuration state
        self.profiles: Dict[str, ConfigProfile] = {}
        self.active_profile: Optional[str] = None
        self.config_history: List[Dict[str, Any]] = []
        
        # Validation rules
        self.validation_rules: List[ConfigValidationRule] = []
        self.validation_enabled = True
        
        # Change tracking
        self.change_listeners: Dict[str, List[Callable]] = {}
        self.last_change_time = time.time()
        
        # Load existing profiles
        self._load_existing_profiles()
        
        # Create default profile if none exist
        if not self.profiles:
            self._create_default_profiles()
        
        # Setup validation rules
        self._setup_validation_rules()
    
    def _load_existing_profiles(self):
        """Load all existing configuration profiles"""
        
        for config_file in self.config_dir.glob("*.json"):
            try:
                self._load_profile_from_file(config_file)
            except Exception as e:
                self.logger.error(f"Failed to load profile {config_file}: {e}")
        
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                self._load_profile_from_file(config_file)
            except Exception as e:
                self.logger.error(f"Failed to load profile {config_file}: {e}")
    
    def _load_profile_from_file(self, config_file: Path):
        """Load a profile from a configuration file"""
        
        with open(config_file, 'r') as f:
            if config_file.suffix == '.json':
                data = json.load(f)
            elif config_file.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_file.suffix}")
        
        profile = ConfigProfile(**data)
        self.profiles[profile.profile_id] = profile
        
        self.logger.info(f"Loaded profile: {profile.profile_id} from {config_file}")
    
    def _create_default_profiles(self):
        """Create default configuration profiles"""
        
        # Default system profile
        default_system = SystemConfig()
        default_user = ProfileConfig(
            profile_id="default_user",
            profile_name="Default User Profile",
            user_type="standard"
        )
        
        default_profile = ConfigProfile(
            profile_id="default",
            profile_name="Default Configuration",
            system=default_system,
            user=default_user
        )
        
        self.add_profile(default_profile)
        
        # Beginner profile
        beginner_system = SystemConfig(
            confidence_threshold=0.8,
            enable_safety_layer=True,
            safety_timeout_ms=3000
        )
        beginner_user = ProfileConfig(
            profile_id="beginner_user",
            profile_name="Beginner User Profile",
            user_type="beginner",
            learning_rate=0.01,
            safety_level="strict",
            feedback_level="verbose",
            require_confirmation=["emergency_stop", "high_force_action"]
        )
        
        beginner_profile = ConfigProfile(
            profile_id="beginner",
            profile_name="Beginner Configuration",
            system=beginner_system,
            user=beginner_user
        )
        
        self.add_profile(beginner_profile)
        
        # Expert profile
        expert_system = SystemConfig(
            confidence_threshold=0.5,
            enable_model_switching=True,
            model_switch_threshold=0.05,
            enable_adaptation=True,
            adaptation_rate=0.1
        )
        expert_user = ProfileConfig(
            profile_id="expert_user",
            profile_name="Expert User Profile",
            user_type="expert",
            learning_rate=0.05,
            safety_level="relaxed",
            feedback_level="minimal",
            enable_advanced_features=True,
            prioritize_speed=True
        )
        
        expert_profile = ConfigProfile(
            profile_id="expert",
            profile_name="Expert Configuration",
            system=expert_system,
            user=expert_user
        )
        
        self.add_profile(expert_profile)
        
        # Clinical profile
        clinical_system = SystemConfig(
            confidence_threshold=0.7,
            enable_performance_monitoring=True,
            enable_prediction_logging=True,
            enable_feature_logging=True,
            safety_timeout_ms=2000
        )
        clinical_user = ProfileConfig(
            profile_id="clinical_user",
            profile_name="Clinical User Profile",
            user_type="clinical",
            learning_rate=0.03,
            safety_level="maximum",
            feedback_level="detailed",
            enable_learning=True,
            enable_advanced_features=True
        )
        
        clinical_profile = ConfigProfile(
            profile_id="clinical",
            profile_name="Clinical Configuration",
            system=clinical_system,
            user=clinical_user
        )
        
        self.add_profile(clinical_profile)
        
        self.logger.info("Created default configuration profiles")
    
    def _setup_validation_rules(self):
        """Setup configuration validation rules"""
        
        self.validation_rules = [
            ConfigValidationRule(
                field_path="system.confidence_threshold",
                validator_func=lambda x: 0.1 <= x <= 0.95,
                error_message="Confidence threshold must be between 0.1 and 0.95"
            ),
            ConfigValidationRule(
                field_path="system.sampling_rate_hz",
                validator_func=lambda x: 100.0 <= x <= 10000.0,
                error_message="Sampling rate must be between 100 and 10000 Hz"
            ),
            ConfigValidationRule(
                field_path="system.window_size_ms",
                validator_func=lambda x: 50.0 <= x <= 2000.0,
                error_message="Window size must be between 50 and 2000 ms"
            ),
            ConfigValidationRule(
                field_path="user.learning_rate",
                validator_func=lambda x: 0.0 <= x <= 0.1,
                error_message="Learning rate must be between 0.0 and 0.1"
            ),
            ConfigValidationRule(
                field_path="system.max_action_rate_hz",
                validator_func=lambda x: 1.0 <= x <= 100.0,
                error_message="Action rate must be between 1 and 100 Hz",
                severity="warning"
            )
        ]
    
    def add_profile(self, profile: ConfigProfile, save: bool = True) -> bool:
        """Add a new configuration profile"""
        
        try:
            # Validate profile
            if self.validation_enabled:
                validation_result = self.validate_profile(profile)
                if not validation_result["valid"]:
                    errors = [err for err in validation_result["errors"] if err["severity"] == "error"]
                    if errors:
                        self.logger.error(f"Profile validation failed: {errors}")
                        return False
            
            # Add to registry
            self.profiles[profile.profile_id] = profile
            
            # Save to disk
            if save:
                self.save_profile(profile)
            
            self.logger.info(f"Added profile: {profile.profile_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add profile {profile.profile_id}: {e}")
            return False
    
    def load_profile(self, profile_id: str) -> bool:
        """Load and activate a configuration profile"""
        
        if profile_id not in self.profiles:
            self.logger.error(f"Profile not found: {profile_id}")
            return False
        
        try:
            # Deactivate current profile
            if self.active_profile:
                self.profiles[self.active_profile].is_active = False
            
            # Activate new profile
            profile = self.profiles[profile_id]
            profile.is_active = True
            profile.last_loaded = time.time()
            profile.load_count += 1
            
            self.active_profile = profile_id
            
            # Notify listeners
            self._notify_change_listeners("profile_changed", {
                "old_profile": self.active_profile,
                "new_profile": profile_id,
                "profile": profile
            })
            
            self.logger.info(f"Loaded profile: {profile_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load profile {profile_id}: {e}")
            return False
    
    def get_active_profile(self) -> Optional[ConfigProfile]:
        """Get the currently active configuration profile"""
        
        if self.active_profile and self.active_profile in self.profiles:
            return self.profiles[self.active_profile]
        return None
    
    def get_config_value(
        self, 
        config_path: str, 
        default: Any = None,
        profile_id: Optional[str] = None
    ) -> Any:
        """Get a configuration value by path"""
        
        target_profile_id = profile_id or self.active_profile
        
        if not target_profile_id or target_profile_id not in self.profiles:
            return default
        
        profile = self.profiles[target_profile_id]
        return profile.get_effective_value(config_path, default)
    
    def set_config_value(
        self, 
        config_path: str, 
        value: Any,
        profile_id: Optional[str] = None,
        save: bool = True
    ) -> bool:
        """Set a configuration value by path"""
        
        target_profile_id = profile_id or self.active_profile
        
        if not target_profile_id or target_profile_id not in self.profiles:
            self.logger.error(f"No profile available to set config: {config_path}")
            return False
        
        try:
            profile = self.profiles[target_profile_id]
            
            # Parse config path
            parts = config_path.split('.')
            if len(parts) < 2:
                return False
            
            section = parts[0]
            field_path = '.'.join(parts[1:])
            
            # Get target section
            if section == "system":
                config_obj = profile.system
            elif section == "user":
                config_obj = profile.user
            else:
                self.logger.error(f"Unknown config section: {section}")
                return False
            
            # Navigate to parent of target field
            field_parts = field_path.split('.')
            current = config_obj
            
            for field in field_parts[:-1]:
                if hasattr(current, field):
                    current = getattr(current, field)
                elif isinstance(current, dict):
                    if field not in current:
                        current[field] = {}
                    current = current[field]
                else:
                    self.logger.error(f"Cannot navigate to field: {field}")
                    return False
            
            # Set the value
            final_field = field_parts[-1]
            if hasattr(current, final_field):
                setattr(current, final_field, value)
            elif isinstance(current, dict):
                current[final_field] = value
            else:
                self.logger.error(f"Cannot set field: {final_field}")
                return False
            
            # Update metadata
            profile.user.last_updated = time.time()
            self.last_change_time = time.time()
            
            # Validate change
            if self.validation_enabled:
                validation_result = self.validate_config_value(config_path, value)
                if not validation_result["valid"]:
                    # Log warnings but don't fail for warnings
                    for error in validation_result["errors"]:
                        if error["severity"] == "error":
                            self.logger.error(f"Config validation error: {error['message']}")
                            return False
                        else:
                            self.logger.warning(f"Config validation warning: {error['message']}")
            
            # Save profile
            if save:
                self.save_profile(profile)
            
            # Notify listeners
            self._notify_change_listeners("config_changed", {
                "path": config_path,
                "value": value,
                "profile_id": target_profile_id
            })
            
            self.logger.info(f"Set config {config_path} = {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set config {config_path}: {e}")
            return False
    
    def validate_profile(self, profile: ConfigProfile) -> Dict[str, Any]:
        """Validate a complete configuration profile"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate each rule
        for rule in self.validation_rules:
            try:
                value = profile.get_effective_value(rule.field_path)
                if value is not None and not rule.validator_func(value):
                    error = {
                        "field": rule.field_path,
                        "message": rule.error_message,
                        "severity": rule.severity,
                        "value": value
                    }
                    
                    if rule.severity == "error":
                        validation_result["valid"] = False
                        validation_result["errors"].append(error)
                    else:
                        validation_result["warnings"].append(error)
                        
            except Exception as e:
                self.logger.warning(f"Validation rule failed for {rule.field_path}: {e}")
        
        return validation_result
    
    def validate_config_value(self, config_path: str, value: Any) -> Dict[str, Any]:
        """Validate a single configuration value"""
        
        validation_result = {
            "valid": True,
            "errors": []
        }
        
        # Find applicable validation rules
        for rule in self.validation_rules:
            if rule.field_path == config_path:
                try:
                    if not rule.validator_func(value):
                        error = {
                            "field": config_path,
                            "message": rule.error_message,
                            "severity": rule.severity,
                            "value": value
                        }
                        
                        if rule.severity == "error":
                            validation_result["valid"] = False
                        
                        validation_result["errors"].append(error)
                        
                except Exception as e:
                    self.logger.warning(f"Validation failed for {config_path}: {e}")
        
        return validation_result
    
    def save_profile(self, profile: ConfigProfile, format: str = "json") -> bool:
        """Save a configuration profile to disk"""
        
        try:
            if format == "json":
                profile_file = self.config_dir / f"{profile.profile_id}.json"
                with open(profile_file, 'w') as f:
                    json.dump(profile.dict(), f, indent=2)
            elif format == "yaml":
                profile_file = self.config_dir / f"{profile.profile_id}.yaml"
                with open(profile_file, 'w') as f:
                    yaml.dump(profile.dict(), f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved profile: {profile.profile_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save profile {profile.profile_id}: {e}")
            return False
    
    def list_profiles(self) -> Dict[str, Dict[str, Any]]:
        """List all available configuration profiles"""
        
        profiles_info = {}
        
        for profile_id, profile in self.profiles.items():
            profiles_info[profile_id] = {
                "name": profile.profile_name,
                "user_type": profile.user.user_type,
                "is_active": profile.is_active,
                "last_loaded": profile.last_loaded,
                "load_count": profile.load_count,
                "created": profile.user.created_timestamp,
                "last_updated": profile.user.last_updated
            }
        
        return profiles_info
    
    def duplicate_profile(
        self, 
        source_profile_id: str, 
        new_profile_id: str,
        new_profile_name: str
    ) -> bool:
        """Create a duplicate of an existing profile"""
        
        if source_profile_id not in self.profiles:
            self.logger.error(f"Source profile not found: {source_profile_id}")
            return False
        
        if new_profile_id in self.profiles:
            self.logger.error(f"Profile already exists: {new_profile_id}")
            return False
        
        try:
            source_profile = self.profiles[source_profile_id]
            
            # Create new profile with updated IDs
            new_profile_data = source_profile.dict()
            new_profile_data["profile_id"] = new_profile_id
            new_profile_data["profile_name"] = new_profile_name
            new_profile_data["user"]["profile_id"] = f"{new_profile_id}_user"
            new_profile_data["user"]["profile_name"] = f"{new_profile_name} User"
            new_profile_data["user"]["created_timestamp"] = time.time()
            new_profile_data["base_profile"] = source_profile_id
            new_profile_data["is_active"] = False
            new_profile_data["load_count"] = 0
            
            new_profile = ConfigProfile(**new_profile_data)
            return self.add_profile(new_profile)
            
        except Exception as e:
            self.logger.error(f"Failed to duplicate profile: {e}")
            return False
    
    def delete_profile(self, profile_id: str) -> bool:
        """Delete a configuration profile"""
        
        if profile_id not in self.profiles:
            self.logger.error(f"Profile not found: {profile_id}")
            return False
        
        if profile_id == self.active_profile:
            self.logger.error("Cannot delete active profile")
            return False
        
        try:
            # Remove from memory
            del self.profiles[profile_id]
            
            # Remove from disk
            for suffix in ['.json', '.yaml']:
                profile_file = self.config_dir / f"{profile_id}{suffix}"
                if profile_file.exists():
                    profile_file.unlink()
            
            self.logger.info(f"Deleted profile: {profile_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete profile {profile_id}: {e}")
            return False
    
    def add_change_listener(self, event_type: str, callback: Callable):
        """Add a listener for configuration changes"""
        
        if event_type not in self.change_listeners:
            self.change_listeners[event_type] = []
        
        self.change_listeners[event_type].append(callback)
    
    def _notify_change_listeners(self, event_type: str, event_data: Dict[str, Any]):
        """Notify registered change listeners"""
        
        if event_type in self.change_listeners:
            for callback in self.change_listeners[event_type]:
                try:
                    callback(event_data)
                except Exception as e:
                    self.logger.warning(f"Change listener failed: {e}")
    
    def export_profile(self, profile_id: str, filepath: str) -> bool:
        """Export a profile to a specific file"""
        
        if profile_id not in self.profiles:
            return False
        
        try:
            profile = self.profiles[profile_id]
            
            with open(filepath, 'w') as f:
                if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    yaml.dump(profile.dict(), f, default_flow_style=False)
                else:
                    json.dump(profile.dict(), f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export profile: {e}")
            return False
    
    def import_profile(self, filepath: str) -> bool:
        """Import a profile from a file"""
        
        try:
            with open(filepath, 'r') as f:
                if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            profile = ConfigProfile(**data)
            return self.add_profile(profile)
            
        except Exception as e:
            self.logger.error(f"Failed to import profile: {e}")
            return False
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration state"""
        
        active_profile = self.get_active_profile()
        
        summary = {
            "active_profile": self.active_profile,
            "total_profiles": len(self.profiles),
            "profiles": list(self.profiles.keys()),
            "last_change": self.last_change_time,
            "validation_enabled": self.validation_enabled,
            "validation_rules_count": len(self.validation_rules)
        }
        
        if active_profile:
            summary["active_config"] = {
                "confidence_threshold": active_profile.get_effective_value("system.confidence_threshold"),
                "learning_enabled": active_profile.get_effective_value("user.enable_learning"),
                "safety_level": active_profile.get_effective_value("user.safety_level"),
                "user_type": active_profile.get_effective_value("user.user_type"),
                "feedback_level": active_profile.get_effective_value("user.feedback_level")
            }
        
        return summary