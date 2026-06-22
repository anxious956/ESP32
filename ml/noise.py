"""Controlled noise injection at a known SNR (Phase 4 core).

SNR is *controlled by injection*, never claimed to be physically measured from a
real mic recording. We add white Gaussian noise in the TIME domain (before the
FFT / feature pipeline) — the true RadioML analog, where noise is added to the
IQ samples, not to the features.

SNR definition (must match the on-device estimator):
    SNR_dB = 10 * log10(P_signal / P_noise),   P = mean(x**2)
"""
from __future__ import annotations

import numpy as np


def signal_power(x: np.ndarray) -> float:
    """Average power = mean of squares."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x ** 2))


def add_awgn(signal: np.ndarray, snr_db: float,
             rng: np.random.Generator | None = None) -> np.ndarray:
    """Add white Gaussian noise so the result has the requested SNR (dB).

    Power is measured from `signal`; noise variance is set to
    P_signal / 10**(snr_db/10).
    """
    rng = rng or np.random.default_rng()
    signal = np.asarray(signal, dtype=np.float64)
    p_sig = signal_power(signal)
    if p_sig <= 0:
        return signal.copy()
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(p_noise), size=signal.shape)
    return signal + noise


def measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Sanity check: recover SNR from a clean/noisy pair (used in tests)."""
    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(noisy, dtype=np.float64) - clean
    p_noise = signal_power(noise)
    if p_noise <= 0:
        return float("inf")
    return 10.0 * np.log10(signal_power(clean) / p_noise)
