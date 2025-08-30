from __future__ import annotations

import os
import math
import time
from typing import Dict, Optional, Tuple


class AdaptationLogger:
    """Lightweight CSV logger for adaptation/drift events."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("timestamp,event,details\n")

    def log(self, event: str, details: Dict[str, object]) -> None:
        ts = time.time()
        flat = ";".join(f"{k}={details[k]}" for k in details)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{ts},{event},{flat}\n")


class OnlineRMSAdapter:
    """
    Tracks per-channel RMS baseline with EMA and provides adaptive cutoffs.
    - baseline EMA stabilizes over time as signal drifts (fatigue/electrode shift)
    - returns low/high cutoffs derived from baseline
    """

    def __init__(
        self,
        alpha: float = 0.02,              # EMA smoothing factor for baseline
        low_mult: float = 0.7,            # low cutoff multiplier vs baseline
        high_mult: float = 1.6,           # high cutoff multiplier vs baseline
        hysteresis: float = 0.05,         # relative change to log updates
        min_updates_before_adapt: int = 20,
    ) -> None:
        self.alpha = float(alpha)
        self.low_mult = float(low_mult)
        self.high_mult = float(high_mult)
        self.hysteresis = float(hysteresis)
        self.min_updates_before_adapt = int(min_updates_before_adapt)

        self._baseline: Dict[str, float] = {}
        self._n_updates: int = 0
        self._last_logged: Dict[str, float] = {}

    @staticmethod
    def _ema(prev: Optional[float], x: float, alpha: float) -> float:
        if prev is None:
            return float(x)
        return float(alpha * x + (1.0 - alpha) * prev)

    def update(self, rms_by_ch: Dict[str, float]) -> Dict[str, float]:
        self._n_updates += 1
        changed = False
        for ch, v in rms_by_ch.items():
            prev = self._baseline.get(ch)
            self._baseline[ch] = self._ema(prev, float(v), self.alpha)
            # detect meaningful change from last logged
            last = self._last_logged.get(ch)
            if last is None:
                self._last_logged[ch] = self._baseline[ch]
                changed = True
            else:
                if last > 0 and abs(self._baseline[ch] - last) / last >= self.hysteresis:
                    self._last_logged[ch] = self._baseline[ch]
                    changed = True

        # Compute aggregate mean/thresholds
        if self._baseline:
            mean_baseline = sum(self._baseline.values()) / len(self._baseline)
        else:
            mean_baseline = 0.0
        low_cutoff_mean = self.low_mult * mean_baseline
        high_cutoff_mean = self.high_mult * mean_baseline

        out: Dict[str, float] = {
            **{f"baseline_{ch}": v for ch, v in self._baseline.items()},
            **{f"low_cutoff_{ch}": self.low_mult * v for ch, v in self._baseline.items()},
            **{f"high_cutoff_{ch}": self.high_mult * v for ch, v in self._baseline.items()},
            "mean_baseline": mean_baseline,
            "low_cutoff_mean": low_cutoff_mean,
            "high_cutoff_mean": high_cutoff_mean,
            "n_updates": float(self._n_updates),
            "changed": float(1.0 if (changed and self._n_updates >= self.min_updates_before_adapt) else 0.0),
        }
        return out


class SimpleDriftDetector:
    """
    Running-mean drift detector on RMS magnitude (aggregated across channels).
    Uses Welford updates to track mean/std; flags drift when z-score > z_thresh
    for k_consecutive windows, with hysteresis for clearing the flag.
    """

    def __init__(
        self,
        z_thresh: float = 3.0,
        z_clear: float = 1.5,
        req_consec: int = 5,
        clear_consec: int = 5,
    ) -> None:
        self.z_thresh = float(z_thresh)
        self.z_clear = float(z_clear)
        self.req_consec = int(req_consec)
        self.clear_consec = int(clear_consec)

        # Welford state
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

        self._drift_active = False
        self._cnt_over = 0
        self._cnt_under = 0

    def _update_welford(self, x: float) -> None:
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2 += delta * delta2

    def _std(self) -> float:
        return math.sqrt(self._M2 / (self._n - 1)) if self._n > 1 else 0.0

    def update(self, value: float) -> Tuple[bool, float, Optional[str]]:
        # compute current z-score vs reference (before updating stats)
        ref_mean = self._mean
        ref_std = self._std()
        z = 0.0
        if ref_std > 1e-12:
            z = abs((value - ref_mean) / ref_std)
        # update stats
        self._update_welford(value)

        changed: Optional[str] = None
        if self._drift_active:
            # check for clearing condition
            if z <= self.z_clear:
                self._cnt_under += 1
                self._cnt_over = 0
                if self._cnt_under >= self.clear_consec:
                    self._drift_active = False
                    self._cnt_under = 0
                    changed = "end"
            else:
                self._cnt_under = 0
        else:
            if z >= self.z_thresh:
                self._cnt_over += 1
                self._cnt_under = 0
                if self._cnt_over >= self.req_consec:
                    self._drift_active = True
                    self._cnt_over = 0
                    changed = "start"
            else:
                self._cnt_over = 0

        return self._drift_active, z, changed
