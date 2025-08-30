from typing import Any, Dict, List, Optional
import math
import time


class SafetyConfig:
    """
    Configuration for safety clamps and thresholds.
    - max_angle_rate_deg_s: maximum allowed change in angle per second
    - dead_zone_angle_deg: absolute angle below which angle is treated as zero
    - dead_zone_force: absolute force below which force is treated as zero
    - hysteresis_high_deg / hysteresis_low_deg: thresholds to avoid chatter near zero
    - max_angle_abs_deg: hard absolute clamp for angles
    - max_force_abs: hard absolute clamp for force
    """

    def __init__(
        self,
        max_angle_rate_deg_s: float = 90.0,
        dead_zone_angle_deg: float = 1.5,
        dead_zone_force: float = 0.2,
        hysteresis_high_deg: float = 2.0,
        hysteresis_low_deg: float = 1.0,
        max_angle_abs_deg: float = 90.0,
        max_force_abs: float = 5.0,
    ) -> None:
        self.max_angle_rate_deg_s = max_angle_rate_deg_s
        self.dead_zone_angle_deg = dead_zone_angle_deg
        self.dead_zone_force = dead_zone_force
        self.hysteresis_high_deg = hysteresis_high_deg
        self.hysteresis_low_deg = hysteresis_low_deg
        self.max_angle_abs_deg = max_angle_abs_deg
        self.max_force_abs = max_force_abs


class SafetyGuard:
    """
    Safety wrapper for motor command specs. Maintains per-actuator state to apply
    rate limiting, dead-zones, and hysteresis to angle/force commands.

    Command spec format (dict):
      {
        'actuator_id': str,
        'angle': Optional[float],
        'force': Optional[float],
        'safety_flags': Dict[str, Any],
        'haptic_alert': Optional[str]
      }
    """

    def __init__(self, config: Optional[SafetyConfig] = None) -> None:
        self.config = config or SafetyConfig()
        # state per actuator_id
        self._state: Dict[str, Dict[str, Any]] = {}

    def _state_for(self, actuator_id: str) -> Dict[str, Any]:
        if actuator_id not in self._state:
            self._state[actuator_id] = {
                'last_angle': 0.0,
                'last_time': None,  # type: Optional[float]
                'active': False,    # for hysteresis around zero
            }
        return self._state[actuator_id]

    def apply_to_spec(self, spec: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
        spec = dict(spec)  # shallow copy
        flags: Dict[str, Any] = dict(spec.get('safety_flags', {}))
        actuator_id: str = spec.get('actuator_id', 'unknown')

        rate_clamped = False
        hysteresis_applied = False
        deadzone_applied = False
        is_safe = True

        st = self._state_for(actuator_id)

        # Angle pipeline
        if spec.get('angle') is not None:
            a = float(spec['angle'])
            # Hard absolute clamp
            a = max(-self.config.max_angle_abs_deg, min(self.config.max_angle_abs_deg, a))

            # Dead-zone
            if abs(a) < self.config.dead_zone_angle_deg:
                a = 0.0
                deadzone_applied = True

            # Rate limit (prevent overshoot)
            if st['last_time'] is not None and now is not None:
                dt = max(0.0, float(now) - float(st['last_time']))
                max_step = self.config.max_angle_rate_deg_s * dt
                desired_step = a - float(st['last_angle'])
                if abs(desired_step) > max_step:
                    a = float(st['last_angle']) + math.copysign(max_step, desired_step)
                    rate_clamped = True

            # Hysteresis around zero to reduce jitter
            abs_a = abs(a)
            if not st['active']:
                # Not active yet; require crossing the high threshold
                if abs_a < self.config.hysteresis_high_deg:
                    a = 0.0
                    hysteresis_applied = True
                else:
                    st['active'] = True
            else:
                # Active; drop to zero only when below low threshold
                if abs_a < self.config.hysteresis_low_deg:
                    a = 0.0
                    st['active'] = False
                    hysteresis_applied = True

            # Update state
            st['last_angle'] = a
            st['last_time'] = now
            spec['angle'] = a

        # Force pipeline
        if spec.get('force') is not None:
            f = float(spec['force'])
            # Dead-zone
            if abs(f) < self.config.dead_zone_force:
                f = 0.0
                deadzone_applied = True
            # Hard absolute clamp
            f = max(-self.config.max_force_abs, min(self.config.max_force_abs, f))
            spec['force'] = f

        flags.update({
            'deadzone_applied': deadzone_applied,
            'hysteresis_applied': hysteresis_applied,
        })
        spec['safety_flags'] = flags
        spec['is_safe'] = is_safe
        spec['rate_clamped'] = rate_clamped
        # 'haptic_alert' key is preserved if present
        return spec


# =====================
# Policy / Haptic Rules
# =====================

def _to_features(intent: Any) -> Dict[str, Any]:
    if isinstance(intent, dict):
        return dict(intent.get('features', {}))
    return dict(getattr(intent, 'features', {}) or {})


def _get_attr(intent: Any, key: str, default: Any = None) -> Any:
    if isinstance(intent, dict):
        return intent.get(key, default)
    return getattr(intent, key, default)


def normalize_gesture(gesture: Optional[str]) -> str:
    g = (gesture or '').lower().strip()
    # Unify common forms
    g = g.replace('-', '_').replace(' ', '_')
    # Drop leading index like "1_close"
    parts = g.split('_', 1)
    if parts and parts[0].isdigit() and len(parts) > 1:
        g = parts[1]

    mappings = {
        'close': 'fist',
        'clench': 'fist',
        'grip': 'fist',
        'fist': 'fist',
        'step': 'step',
        'walk': 'step',
        'gait': 'step',
        'lean': 'lean',
        'tilt': 'lean',
    }
    return mappings.get(g, g)


def map_gesture_to_commands(intent: Any) -> List[Dict[str, Any]]:
    """
    Policy mapping table (gesture -> command spec list):
      - Fist -> close hand servos
      - Step -> plantarflex ankle servo
      - Lean -> spine actuator adjust (uses features.lean_deg if present)
    """
    gesture = _get_attr(intent, 'gesture', '')
    features = _to_features(intent)
    g = normalize_gesture(gesture)

    commands: List[Dict[str, Any]] = []

    if g == 'fist':
        commands.append({
            'actuator_id': 'hand_servo',
            'angle': 45.0,  # close hand
            'force': None,
            'safety_flags': {'policy': 'close_hand'},
        })

    elif g == 'step':
        commands.append({
            'actuator_id': 'ankle_servo',
            'angle': 15.0,  # plantarflex
            'force': None,
            'safety_flags': {'policy': 'plantarflex'},
        })

    elif g == 'lean':
        # use feature override if provided
        try:
            lean_deg = float(features.get('lean_deg', 5.0))
        except Exception:
            lean_deg = 5.0
        lean_deg = max(-10.0, min(10.0, lean_deg))
        commands.append({
            'actuator_id': 'spine_servo',
            'angle': lean_deg,
            'force': None,
            'safety_flags': {'policy': 'spine_adjust'},
        })

    return commands


def build_haptic_alerts(intent: Any) -> List[Dict[str, Any]]:
    """
    Haptic feedback rules:
      - Unsafe foot tilt -> vibrate left/right actuator
      - Missed grip -> vibration pulse
    Expects optional fields in intent.features:
      - foot_tilt_deg (or foot_roll_deg), foot_tilt_dir ('left'|'right' or sign via foot_roll_sign)
      - grip_contact (bool) or grip_force (float [0..1] or Newtons)
    """
    features = _to_features(intent)
    gesture = _get_attr(intent, 'gesture', '')
    confidence = float(_get_attr(intent, 'confidence', 0.0) or 0.0)

    haptics: List[Dict[str, Any]] = []

    # 1) Unsafe foot tilt
    tilt_deg_val = features.get('foot_tilt_deg', features.get('foot_roll_deg'))
    tilt_dir = features.get('foot_tilt_dir')
    tilt_sign = features.get('foot_roll_sign')

    unsafe_thresh = 15.0  # degrees
    if tilt_deg_val is not None:
        try:
            t = float(tilt_deg_val)
            if abs(t) >= unsafe_thresh:
                side: Optional[str] = None
                if isinstance(tilt_dir, str):
                    d = tilt_dir.lower().strip()
                    if d in ('left', 'l'):
                        side = 'left'
                    elif d in ('right', 'r'):
                        side = 'right'
                if side is None:
                    try:
                        sgn = float(tilt_sign) if tilt_sign is not None else t
                        side = 'right' if sgn > 0 else 'left'
                    except Exception:
                        side = 'right' if t > 0 else 'left'

                actuator = 'haptic_right' if side == 'right' else 'haptic_left'
                intensity = min(1.0, abs(t) / 30.0)  # scale 0..1 by severity
                haptics.append({
                    'actuator_id': actuator,
                    'angle': None,
                    'force': 1.0 * intensity,  # vibration intensity proxy
                    'safety_flags': {'haptic': 'unsafe_foot_tilt', 'tilt_deg': t, 'side': side},
                    'haptic_alert': 'unsafe_foot_tilt',
                })
        except Exception:
            pass

    # 2) Missed grip pulse
    gnorm = normalize_gesture(gesture)
    grip_contact = features.get('grip_contact')
    grip_force = features.get('grip_force')

    missed = False
    if gnorm == 'fist' and confidence >= 0.6:
        if grip_contact is False:
            missed = True
        elif grip_force is not None:
            try:
                gf = float(grip_force)
                # Treat normalized [0..1] or small Newtons as low force
                missed = gf < 0.2
            except Exception:
                missed = False

    if missed:
        haptics.append({
            'actuator_id': 'haptic_wrist',
            'angle': None,
            'force': 1.0,  # strong pulse
            'safety_flags': {'haptic': 'missed_grip', 'pattern': 'pulse'},
            'haptic_alert': 'missed_grip',
        })

    return haptics


__all__ = [
    'SafetyConfig',
    'SafetyGuard',
    'normalize_gesture',
    'map_gesture_to_commands',
    'build_haptic_alerts',
]
