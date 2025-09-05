# Mapping layer initialization
from .gesture_mapper import GestureMapper, MappingConfig, ActionSpec
from .profiles import ProfileManager, ProfileConfig
from .adapters import ContextAdapter, AdaptationStrategy

__all__ = [
    'GestureMapper',
    'MappingConfig', 
    'ActionSpec',
    'ProfileManager',
    'ProfileConfig',
    'ContextAdapter',
    'AdaptationStrategy'
]