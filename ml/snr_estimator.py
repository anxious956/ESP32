"""On-device SNR estimator (Phase 3, the hardest component).

Two estimators that share ONE SNR definition with ml/noise.py:
    SNR_dB = 10*log10(P_signal / P_noise),   P = mean(x**2)

Key idea (works without a fragile 'silence' gate, robust at low SNR):
  * The whole-window mean power of the noisy signal is P_signal + P_noise
    (signal and noise are independent, powers add).
  * The NOISE FLOOR is the low envelope of short-frame powers — estimated by a
    low percentile (single window) or a min-statistics sliding minimum (stream).
  * Then  P_signal = P_total - P_noise  and the SNR follows.

`bias` corrects the percentile/minimum's known underestimate of the true noise
power; it is calibrated against injected SNR in demo_snr.py (gate #1).

Both estimators are deliberately simple (frame energies + a running minimum) so
they port directly to fixed-point C on the ESP32.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# frame energies                                                              #
# --------------------------------------------------------------------------- #
def frame_powers(x: np.ndarray, frame_len: int = 256, hop: int = 128) -> np.ndarray:
    """Per-frame average power (mean of squares)."""
    x = np.asarray(x, dtype=np.float64)
    n = 1 + max(0, (len(x) - frame_len) // hop)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = i * hop
        f = x[s:s + frame_len]
        out[i] = np.mean(f * f)
    return out


# --------------------------------------------------------------------------- #
# single-window estimator (pairs with per-window inference / Phase 4)         #
# --------------------------------------------------------------------------- #
def estimate_snr_db(x: np.ndarray, frame_len: int = 256, hop: int = 128,
                    noise_pct: float = 15.0, bias: float = 1.10,
                    eps: float = 1e-12) -> float:
    """Estimate window SNR (dB) from one noisy window.

    P_noise = bias * percentile(frame_powers, noise_pct)   (the noise floor)
    P_total = mean(x**2)                                    (= P_signal+P_noise)
    """
    x = np.asarray(x, dtype=np.float64)
    fp = frame_powers(x, frame_len, hop)
    p_total = float(np.mean(x * x))
    p_noise = float(np.percentile(fp, noise_pct)) * bias
    p_noise = max(p_noise, eps)
    p_signal = max(p_total - p_noise, eps)
    return 10.0 * np.log10(p_signal / p_noise)


# --------------------------------------------------------------------------- #
# streaming estimator for the firmware (min-statistics + hysteresis VAD)      #
# --------------------------------------------------------------------------- #
class StreamingSNR:
    """Continuous noise-floor tracker + event detector.

    Feed one frame power at a time. The noise floor is the minimum of recent
    sub-window minima (min-statistics) — it survives ongoing events because it
    only needs occasional low-energy gaps, unlike a threshold-gated 'silence'
    average. Hysteresis (enter > exit) keeps the event flag from chattering.

    This is the reference for the C port; all state is scalar/ring-buffer.
    """

    def __init__(self, sub_len: int = 15, n_sub: int = 10, ema: float = 0.6,
                 bias: float = 1.10, enter_ratio: float = 4.0,
                 exit_ratio: float = 2.0, eps: float = 1e-12):
        self.sub_len = sub_len
        self.n_sub = n_sub
        self.ema = ema
        self.bias = bias
        self.enter = enter_ratio
        self.exit = exit_ratio
        self.eps = eps

        self._p = None                      # EMA-smoothed power
        self._cur_min = np.inf              # current sub-window minimum
        self._cur_cnt = 0
        self._mins: list[float] = []        # ring of past sub-window minima
        self._in_event = False

    def noise_floor(self) -> float:
        pool = self._mins + ([self._cur_min] if np.isfinite(self._cur_min) else [])
        nf = min(pool) if pool else self.eps
        return max(nf * self.bias, self.eps)

    def update(self, frame_power: float):
        """Return (snr_db, in_event) after consuming one frame power."""
        p = frame_power if self._p is None else (
            self.ema * self._p + (1 - self.ema) * frame_power)
        self._p = p

        # advance the sliding minimum
        self._cur_min = min(self._cur_min, p)
        self._cur_cnt += 1
        if self._cur_cnt >= self.sub_len:
            self._mins.append(self._cur_min)
            if len(self._mins) > self.n_sub:
                self._mins.pop(0)
            self._cur_min = np.inf
            self._cur_cnt = 0

        nf = self.noise_floor()
        # hysteresis event flag
        if self._in_event:
            if p < self.exit * nf:
                self._in_event = False
        else:
            if p > self.enter * nf:
                self._in_event = True

        p_signal = max(p - nf, self.eps)
        snr_db = 10.0 * np.log10(p_signal / nf)
        return snr_db, self._in_event
