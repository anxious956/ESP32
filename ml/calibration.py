"""Calibration core — temperature scaling, per-SNR temperature scaling, ECE,
reliability diagrams.

This is the audio-domain reproduction of the method in
"Calibrating Deep AMC Classifiers with SNR-Adaptive Temperature Scaling":
a single softmax temperature is poorly suited when miscalibration varies with
SNR, so we fit a separate temperature per SNR bucket and interpolate T(SNR).

Correctness notes that MUST hold when this is ported to the ESP32 firmware:
  * SNR definition must be IDENTICAL here and on-device (see noise.py): we use
    SNR_dB = 10*log10(P_signal / P_noise), power = mean(x**2).
  * Temperature is fit on the SAME logits that are deployed. On hardware that
    means fitting on the int8/quantized model's logits, not the float model's.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


# --------------------------------------------------------------------------- #
# softmax / NLL                                                               #
# --------------------------------------------------------------------------- #
def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Numerically stable temperature-scaled softmax. logits: (N, K)."""
    z = np.asarray(logits, dtype=np.float64) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, T: float = 1.0) -> float:
    """Mean negative log-likelihood at temperature T."""
    p = softmax(logits, T)
    idx = np.arange(len(labels))
    return float(-np.mean(np.log(p[idx, labels] + 1e-12)))


# --------------------------------------------------------------------------- #
# temperature fitting                                                         #
# --------------------------------------------------------------------------- #
def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    bounds: tuple[float, float] = (0.05, 10.0)) -> float:
    """Fit a single scalar temperature by minimizing NLL on (logits, labels)."""
    res = minimize_scalar(lambda T: nll(logits, labels, T),
                          bounds=bounds, method="bounded")
    return float(res.x)


def fit_temperature_per_snr(logits: np.ndarray, labels: np.ndarray,
                            snrs: np.ndarray, snr_levels: np.ndarray,
                            min_count: int = 30,
                            bounds: tuple[float, float] = (0.05, 10.0)):
    """Fit one temperature per SNR level.

    Returns (levels, temps) arrays defining the T(SNR) curve. Levels with too
    few samples fall back to the global temperature so the curve stays defined.
    """
    snrs = np.asarray(snrs, dtype=np.float64)
    snr_levels = np.asarray(snr_levels, dtype=np.float64)
    global_T = fit_temperature(logits, labels, bounds)

    temps = []
    for lvl in snr_levels:
        mask = np.isclose(snrs, lvl)
        if mask.sum() >= min_count:
            temps.append(fit_temperature(logits[mask], labels[mask], bounds))
        else:
            temps.append(global_T)
    return snr_levels, np.asarray(temps, dtype=np.float64)


def isotonic_decreasing(y: np.ndarray) -> np.ndarray:
    """Project onto non-increasing sequences (pool-adjacent-violators).

    Temperatures should not increase with SNR (higher SNR -> less over-confident
    -> smaller T). Enforcing this removes noisy non-monotonic per-bucket fits so
    the curve generalizes from validation to test.
    """
    vals, wts = [], []
    for v in map(float, y):
        vals.append(v); wts.append(1.0)
        while len(vals) > 1 and vals[-2] < vals[-1]:   # violation of non-increasing
            v2, w2 = vals.pop(), wts.pop()
            v1, w1 = vals.pop(), wts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2)
    out = []
    for v, w in zip(vals, wts):
        out.extend([v] * int(round(w)))
    return np.asarray(out, dtype=np.float64)


def apply_per_snr_temperature(logits: np.ndarray, snrs: np.ndarray,
                              levels: np.ndarray, temps: np.ndarray) -> np.ndarray:
    """Apply T(SNR) per sample (linear interpolation, clamped at the ends).

    This is exactly what the firmware does: estimate SNR -> look up T -> divide
    logits by T -> softmax.
    """
    T_per_sample = np.interp(np.asarray(snrs, dtype=np.float64), levels, temps)
    z = np.asarray(logits, dtype=np.float64) / T_per_sample[:, None]
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Expected Calibration Error                                                  #
# --------------------------------------------------------------------------- #
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15):
    """Equal-width ECE on the top-1 confidence.

    Returns (ece, bin_stats) where bin_stats is a list of
    (conf_mean, acc_mean, count) per non-empty bin — ready for a reliability
    diagram.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    N = len(labels)
    ece = 0.0
    stats = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:  # include the left edge in the first bin
            mask |= confidences == lo
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        conf = float(confidences[mask].mean())
        acc = float(correct[mask].mean())
        ece += (cnt / N) * abs(acc - conf)
        stats.append((conf, acc, cnt))
    return float(ece), stats
