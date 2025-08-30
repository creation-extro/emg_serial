from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field
import time
import math
import numpy as np
import pandas as pd
import joblib
import os

# Import safety and policy layer
try:
    from motion_ai.control.safety_layer import (
        SafetyGuard,
        SafetyConfig,
        map_gesture_to_commands,
        build_haptic_alerts,
        normalize_gesture,
    )
except Exception:  # Fallback for relative import contexts
    from ..control.safety_layer import (
        SafetyGuard,
        SafetyConfig,
        map_gesture_to_commands,
        build_haptic_alerts,
        normalize_gesture,
    )


# ==========================
# Shared Schemas (Contracts)
# ==========================
class SignalFrame(BaseModel):
    """
    SignalFrame represents a single EMG capture window ready for classification.
    - timestamp: unix epoch seconds (float)
    - channels: flattened numeric readings per channel (order fixed by preprocessing)
    - metadata: optional, includes device_id, window_size_ms, subject_id, etc.
    """

    timestamp: float = Field(..., description="Unix epoch time in seconds")
    channels: List[float] = Field(
        ..., description="Array of EMG channel readings in a fixed order"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntentFrame(BaseModel):
    """
    IntentFrame represents the recognized intent/gesture.
    - gesture: semantic label, e.g., "open_hand", "pinch", "unknown"
    - confidence: [0..1]
    - features: optional dictionary of feature values used for transparency/debug
    - soft_priority: advisory priority [0..1] (e.g., from external /intent)
    - design_candidates: optional UI-only candidates for dashboard (not used in control)
    """

    gesture: str
    confidence: float = Field(ge=0.0, le=1.0)
    features: Dict[str, Any] = Field(default_factory=dict)
    soft_priority: float = Field(default=0.0, ge=0.0, le=1.0)
    design_candidates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Optional visualization payload for dashboard; not consumed in control loop"
        ),
    )


class MotorCmd(BaseModel):
    """
    MotorCmd represents a low-level command to an actuator.
    - actuator_id: unique actuator identifier, e.g., "wrist_flexor"
    - angle: optional degrees for rotary actuators
    - force: optional Newtons for linear/force actuators
    - safety_flags: bag of safety annotations (e.g., {"clamped": true})
    - is_safe: end-to-end safety assessment
    - rate_clamped: whether rate limiting adjusted the command
    - haptic_alert: optional haptic reason tag if this command is a haptic output

    Exactly one of [angle, force] is typically set by the policy; both may be None for noop.
    """

    actuator_id: str
    angle: Optional[float] = None
    force: Optional[float] = None
    safety_flags: Dict[str, Any] = Field(default_factory=dict)
    # Safety flags extension
    is_safe: bool = True
    rate_clamped: bool = False
    haptic_alert: Optional[str] = None


# ============
# Router setup
# ============
router = APIRouter()

# Single safety guard instance to preserve hysteresis/rate state per actuator
_safety = SafetyGuard(SafetyConfig())

# Day 7: Online Adaptation & Drift Detection singletons
try:
    from motion_ai.preprocess.adaptation import OnlineRMSAdapter, SimpleDriftDetector, AdaptationLogger
except Exception:
    from ..preprocess.adaptation import OnlineRMSAdapter, SimpleDriftDetector, AdaptationLogger

_adapter = OnlineRMSAdapter()
_drift = SimpleDriftDetector()
_logger = AdaptationLogger(path=os.path.join(".qodo", "adaptation_events.csv"))


@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/classify", response_model=IntentFrame)
async def classify(signal: SignalFrame) -> IntentFrame:
    """
    Stub classifier that always returns gesture="unknown" with 0 confidence.
    - Echoes back minimal features for traceability (e.g., number of channels)
    - If the client sets metadata["include_design_candidates"] == true, returns empty
      design_candidates array for UI plumbing.
    """
    include_candidates = bool(signal.metadata.get("include_design_candidates", False))

    features = {
        "channels_len": len(signal.channels),
        "timestamp": signal.timestamp,
        # add other feature transparencies as needed
    }

    return IntentFrame(
        gesture="unknown",
        confidence=0.0,
        features=features,
        design_candidates=([] if include_candidates else None),
    )


@router.post("/v1/policy", response_model=List[MotorCmd])
async def plan_policy(intent: IntentFrame) -> List[MotorCmd]:
    """
    Policy mapping with safety wrapper and haptic rules.
    - Maps gestures to actuator commands (fist/step/lean)
    - Applies safety clamps: rate limit, dead-zone, hysteresis
    - Adds haptic alerts for unsafe tilt and missed grip
    """
    now = time.time()

    # Build primary commands from intent
    raw_cmd_specs: List[Dict[str, Any]] = map_gesture_to_commands(intent)

    # Haptic alerts
    haptic_specs = build_haptic_alerts(intent)

    all_specs = raw_cmd_specs + haptic_specs

    # Apply safety wrapper to each spec and coerce into MotorCmd
    safe_cmds: List[MotorCmd] = []
    for spec in all_specs:
        safe_spec = _safety.apply_to_spec(spec, now=now)
        safe_cmds.append(MotorCmd(**safe_spec))

    # Fallback noop if nothing to do
    if not safe_cmds:
        safe_cmds.append(
            MotorCmd(
                actuator_id="noop",
                angle=None,
                force=None,
                safety_flags={"reason": "no_policy_match", "from_gesture": intent.gesture},
                is_safe=True,
                rate_clamped=False,
                haptic_alert=None,
            )
        )

    return safe_cmds


# ==============================
# Day 4: Hybrid + Latency route
# ==============================
@router.post("/v1/hybrid", response_model=List[MotorCmd])
async def hybrid(signal: SignalFrame) -> List[MotorCmd]:
    """
    Day 4: Probability-capable classifier (e.g., MLP) -> gesture + confidence.
    Fallback to 'rest' if confidence < threshold, then map to MotorCmds with safety + haptics.
    Configure via SignalFrame.metadata:
      - model_path: str (required)
      - fs: float (default 1000.0)
      - confidence_threshold: float (default 0.6)
      - n_channels: int (optional)
      - n_samples: int (optional)
    Returns MotorCmd[] and annotates latency_ms and confidence in the first command's safety_flags.
    """
    t0 = time.perf_counter()

    md = signal.metadata or {}
    model_path = md.get("model_path")
    fs = float(md.get("fs", 1000.0))
    threshold = float(md.get("confidence_threshold", 0.6))

    # If no model provided, return safe default quickly
    if not model_path:
        intent = IntentFrame(
            gesture="rest",
            confidence=0.0,
            features={"reason": "no_model_path", "channels_len": len(signal.channels), "timestamp": signal.timestamp},
        )
        cmds = await plan_policy(intent)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if cmds:
            flags = dict(cmds[0].safety_flags)
            flags.update({"latency_ms": round(elapsed_ms, 2), "confidence": 0.0, "raw_gesture": "unknown"})
            cmds[0].safety_flags = flags
        return cmds

    from motion_ai.features.extractors import extract_features_from_window

    bundle = joblib.load(model_path)
    clf = bundle.get("clf")
    scaler = bundle.get("scaler")
    feat_names = bundle.get("features", [])

    # Build window DataFrame based on provided channel layout or fallback
    arr = np.asarray(signal.channels, dtype=float)

    # Day 8: Fault Injection (optional)
    try:
        from motion_ai.preprocess.faults import inject_faults
    except Exception:
        from ..preprocess.faults import inject_faults
    fault_cfg = md.get("fault_injection") or {}
    if isinstance(fault_cfg, dict) and fault_cfg.get("enabled"):
        # Provide optional reshape hints for injection
        fi_cfg = dict(fault_cfg)
        if "n_channels" not in fi_cfg:
            if int(md.get("n_channels", 0) or 0) > 0:
                fi_cfg["n_channels"] = int(md.get("n_channels", 0) or 0)
        if "n_samples" not in fi_cfg:
            if int(md.get("n_samples", 0) or 0) > 0:
                fi_cfg["n_samples"] = int(md.get("n_samples", 0) or 0)
        arr = inject_faults(arr, fs=fs, cfg=fi_cfg)
    n_channels = int(md.get("n_channels", 0) or 0)
    n_samples = int(md.get("n_samples", 0) or 0)

    if n_channels > 0 and n_samples > 0 and n_channels * n_samples == len(arr):
        emg_mat = arr.reshape(n_channels, n_samples)
        cols = {f"ch{i+1}": emg_mat[i] for i in range(min(n_channels, 3))}
        while len(cols) < 3 and len(cols) > 0:
            first_key = list(cols.keys())[0]
            cols[f"ch{len(cols)+1}"] = cols[first_key]
        df_window = pd.DataFrame(cols if cols else {"ch1": arr})
    else:
        df_window = pd.DataFrame({
            "ch1": arr,
            "ch2": arr,
            "ch3": arr,
        })

    feats = extract_features_from_window(df_window, fs=fs)

    # Day 7: Online adaptation and drift detection
    rms_vals = []
    for k, v in feats.items():
        if k.endswith("_rms"):
            try:
                rms_vals.append(float(v))
            except Exception:
                pass
    mean_rms = float(sum(rms_vals) / len(rms_vals)) if rms_vals else 0.0
    rms_by_ch = {k.replace("_rms", ""): float(v) for k, v in feats.items() if k.endswith("_rms")}
    adapt_info = _adapter.update(rms_by_ch)
    drift_active, z, drift_changed = _drift.update(mean_rms)
    if adapt_info.get("changed", 0.0) >= 1.0:
        _logger.log("baseline_update", {
            "mean_baseline": round(adapt_info.get("mean_baseline", 0.0), 6),
            "low_cutoff_mean": round(adapt_info.get("low_cutoff_mean", 0.0), 6),
            "high_cutoff_mean": round(adapt_info.get("high_cutoff_mean", 0.0), 6),
            "n_updates": int(adapt_info.get("n_updates", 0.0)),
        })
    if drift_changed is not None:
        _logger.log("drift_" + drift_changed, {"z": round(z, 3), "mean_rms": round(mean_rms, 6)})
    X = np.array([[feats.get(f, 0.0) for f in feat_names]], dtype=float)
    if scaler is not None:
        X = scaler.transform(X)

    # Confidence-based prediction
    gesture = "unknown"
    confidence = 0.0
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        classes = getattr(clf, "classes_", None)
        if classes is None and "classes_" in bundle:
            classes = np.array(bundle["classes_"])
        gesture = str(classes[idx]) if classes is not None else str(clf.predict(X)[0])
        confidence = float(np.max(proba))
    else:
        gesture = str(clf.predict(X)[0])
        confidence = 1.0

    # Day 8: Dropout-based fallback
    # Detect prolonged "silence" (RMS below small threshold) and force 'rest'
    silence_rms_thresh = float(md.get("silence_rms_thresh", 1e-6))
    max_dropout_ms = float(md.get("max_dropout_ms", 250.0))
    window_ms = 1000.0 * (len(arr) / max(1.0, fs))
    is_silent = False
    try:
        # Use per-channel RMS if available
        rms_vals = [float(v) for k, v in feats.items() if k.endswith("_rms")]
        if rms_vals:
            is_silent = all(r <= silence_rms_thresh for r in rms_vals)
    except Exception:
        is_silent = False

    # Maintain a simple silence accumulator (in ms) in metadata cache-like field
    # NOTE: stateless API; emulate via rolling window size
    silent_ms = window_ms if is_silent else 0.0
    force_rest = silent_ms >= max_dropout_ms

    # Fallback to 'rest' if below threshold or forced by dropout
    effective_gesture = gesture if (confidence >= threshold and not force_rest) else "rest"

    # Build intent and plan
    intent = IntentFrame(
        gesture=effective_gesture,
        confidence=confidence,
        features={"raw_gesture": gesture, "channels_len": len(signal.channels), "timestamp": signal.timestamp},
    )
    cmds = await plan_policy(intent)

    # Annotate latency and confidence for logging/inspection
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if cmds:
        flags = dict(cmds[0].safety_flags)
        flags.update({
            "latency_ms": round(elapsed_ms, 2),
            "confidence": round(confidence, 3),
            "raw_gesture": gesture,
            # Day 7 annotations
            "adapt_mean_baseline": round(adapt_info.get("mean_baseline", 0.0), 6),
            "adapt_low_cutoff": round(adapt_info.get("low_cutoff_mean", 0.0), 6),
            "adapt_high_cutoff": round(adapt_info.get("high_cutoff_mean", 0.0), 6),
            "drift_active": bool(drift_active),
            "drift_z": round(z, 3),
            # Day 8 annotations
            "fault_injection": bool(fault_cfg.get("enabled", False)) if isinstance(fault_cfg, dict) else False,
            "dropout_force_rest": bool(force_rest),
            "window_ms": round(window_ms, 1),
            "silence_rms_thresh": silence_rms_thresh,
        })
        cmds[0].safety_flags = flags

    return cmds


# ==============================
# Day 6: Intent Fusion + Shaping
# ==============================
class FusionRequest(BaseModel):
    """
    Intent fusion request combining EMG/IMU intent with external advisory intent.
    Profiles: "grip_priority" or "balance_priority".
    """
    emg_intent: Optional[IntentFrame] = None
    external_intent: Optional[IntentFrame] = None
    profile: str = Field(default="grip_priority")
    weight_emg: float = Field(default=0.6, ge=0.0, le=1.0)
    weight_external: float = Field(default=0.4, ge=0.0, le=1.0)


def _profile_tuning(profile: str) -> Dict[str, float]:
    p = (profile or "").strip().lower()
    if p in ("balance", "balance_priority", "balance-priority"):
        return {"w_boost_external": 0.2, "w_boost_emg": 0.0, "hand_mult": 0.8, "balance_mult": 1.1, "energy_base": 0.80}
    return {"w_boost_external": 0.0, "w_boost_emg": 0.2, "hand_mult": 1.0, "balance_mult": 0.85, "energy_base": 0.85}


def _instability_from_features(features: Dict[str, Any]) -> float:
    if not features:
        return 0.0
    val = features.get("foot_tilt_deg", features.get("foot_roll_deg"))
    try:
        t = abs(float(val)) if val is not None else 0.0
    except Exception:
        t = 0.0
    return max(0.0, min(1.0, t / 30.0))


def _is_balance_gesture(g: str) -> bool:
    return normalize_gesture(g) in ("step", "lean")


def _is_hand_gesture(g: str) -> bool:
    return normalize_gesture(g) in ("fist", "pinch", "open", "open_hand")


def _fuse_intents(emg: Optional[IntentFrame], ext: Optional[IntentFrame], profile: str, w_emg: float, w_ext: float) -> IntentFrame:
    if emg is None and ext is None:
        return IntentFrame(gesture="rest", confidence=0.0, features={"reason": "no_inputs"})
    if emg is None:
        return ext  # type: ignore[return-value]
    if ext is None:
        return emg

    tune = _profile_tuning(profile)
    w_emg = max(0.0, min(1.0, w_emg + tune["w_boost_emg"]))
    w_ext = max(0.0, min(1.0, w_ext + tune["w_boost_external"]))

    # Apply advisory soft priority
    pri = float(getattr(ext, "soft_priority", 0.0) or 0.0)
    w_ext *= (1.0 + 0.5 * max(0.0, min(1.0, pri)))

    # Renormalize weights
    s = w_emg + w_ext
    if s <= 0:
        w_emg = w_ext = 0.5
    else:
        w_emg, w_ext = w_emg / s, w_ext / s

    g_emg = normalize_gesture(emg.gesture)
    g_ext = normalize_gesture(ext.gesture)
    instab = _instability_from_features(emg.features or {})

    # If both agree, combine confidence
    if g_emg == g_ext:
        conf = max(0.0, min(1.0, w_emg * emg.confidence + w_ext * ext.confidence))
        return IntentFrame(gesture=g_emg, confidence=conf, features={"src": "fusion", "agreement": True, "g_emg": emg.gesture, "g_ext": ext.gesture, "w_emg": round(w_emg,3), "w_ext": round(w_ext,3), "instability": instab})

    # Disagree: compute weighted scores with profile-aware boosts
    s_emg = w_emg * emg.confidence
    s_ext = w_ext * ext.confidence

    # Hand vs balance bias
    if _is_hand_gesture(g_emg):
        s_emg *= (1.1 if tune["hand_mult"] >= 1.0 else tune["hand_mult"])  # slight favor in grip profile
    if _is_balance_gesture(g_ext):
        s_ext *= max(1.0, tune["balance_mult"]) * (1.0 + 0.5 * instab)  # favor external when unstable

    # If instability is high, further damp hand intent unless external also hand
    if instab >= 0.7 and _is_hand_gesture(g_emg) and not _is_hand_gesture(g_ext):
        s_emg *= 0.6

    if s_ext >= s_emg:
        chosen, conf = g_ext, max(0.0, min(1.0, s_ext))
    else:
        chosen, conf = g_emg, max(0.0, min(1.0, s_emg))

    return IntentFrame(gesture=chosen, confidence=conf, features={"src": "fusion", "agreement": False, "g_emg": emg.gesture, "g_ext": ext.gesture, "w_emg": round(w_emg,3), "w_ext": round(w_ext,3), "instability": instab, "s_emg": round(s_emg,3), "s_ext": round(s_ext,3)})


def _shape_command_specs(specs: List[Dict[str, Any]], fused: IntentFrame, profile: str) -> List[Dict[str, Any]]:
    # Energy conservation scales amplitude by confidence and profile baseline
    tune = _profile_tuning(profile)
    conf = float(fused.confidence or 0.0)
    energy_scale = max(0.5, min(1.2, tune["energy_base"] + 0.3 * conf))

    instab = _instability_from_features(fused.features or {})

    shaped: List[Dict[str, Any]] = []
    for spec in specs:
        s = dict(spec)
        act = str(s.get("actuator_id", "")).lower()
        angle = s.get("angle")
        force = s.get("force")
        mult = 1.0

        if act == "hand_servo":
            # Favor hand in grip profile; damp when unstable in balance profile
            if profile.startswith("grip"):
                mult *= tune["hand_mult"]
            else:
                mult *= (0.6 if instab >= 0.7 else 0.8)
        elif act in ("ankle_servo", "spine_servo"):
            # Favor balance actuators, especially when unstable
            mult *= (tune["balance_mult"] * (1.0 + 0.3 * instab))
        else:
            mult *= 1.0

        if angle is not None:
            try:
                a = float(angle)
                s["angle"] = math.copysign(abs(a) * mult * energy_scale, a)
            except Exception:
                pass
        if force is not None:
            try:
                f = float(force)
                s["force"] = math.copysign(abs(f) * mult * energy_scale, f)
            except Exception:
                pass
        flags = dict(s.get("safety_flags", {}))
        flags.update({
            "policy_shaping": True,
            "profile": profile,
            "energy_scale": round(energy_scale, 3),
            "instability": round(instab, 3),
            "mult": round(mult, 3),
        })
        s["safety_flags"] = flags
        shaped.append(s)

    return shaped


@router.post("/v1/intent/fuse", response_model=List[MotorCmd])
async def fuse_and_plan(req: FusionRequest) -> List[MotorCmd]:
    now = time.time()

    fused = _fuse_intents(req.emg_intent, req.external_intent, req.profile, req.weight_emg, req.weight_external)

    # Map to command specs
    raw_cmd_specs: List[Dict[str, Any]] = map_gesture_to_commands(fused)

    # Add haptic alerts based on fused state (e.g., instability warnings)
    haptic_specs = build_haptic_alerts(fused)

    # Apply policy shaping
    shaped_specs = _shape_command_specs(raw_cmd_specs + haptic_specs, fused, req.profile)

    # Safety wrapper
    out: List[MotorCmd] = []
    for spec in shaped_specs:
        safe_spec = _safety.apply_to_spec(spec, now=now)
        out.append(MotorCmd(**safe_spec))

    if not out:
        out.append(
            MotorCmd(
                actuator_id="noop",
                angle=None,
                force=None,
                safety_flags={"reason": "no_policy_match", "from_gesture": fused.gesture, "profile": req.profile},
                is_safe=True,
                rate_clamped=False,
                haptic_alert=None,
            )
        )

    return out


# ==============
# Standalone app
# ==============
# This allows running directly: `uvicorn motion_ai.api.router:app --reload`
app = FastAPI(title="motion_ai - Policy + Safety + Hybrid (Day 6)", version="0.6.0")
app.include_router(router)
