"""Synthetic audio dataset for the end-to-end pipeline test (no hardware).

Physical noise model: a FIXED noise floor (room noise), with the EVENT amplitude
scaled to hit a target SNR — so SNR is set the same way as ml/noise.py:
    window-mean signal power / noise power,  P = mean(x**2).

Classes: clap (decaying noise burst), whistle (enveloped tone), snap (short
click), silence (noise only). Features = log-mel summary stats (mean & std over
time per mel band) -> compact input for the int8-MLP fallback model.
"""
from __future__ import annotations

import numpy as np

from features import log_mel_spectrogram

SR = 16000
N = SR // 2                 # 0.5 s window
CLASSES = ["clap", "whistle", "snap", "silence"]
N_MELS = 32


# ----------------------------- event shapes ------------------------------- #
def _clap(rng):
    x = np.zeros(N)
    L = int(0.12 * SR)
    off = rng.integers(0, N - L)
    t = np.arange(L) / SR
    x[off:off + L] = rng.normal(0, 1, L) * np.exp(-t / 0.04)
    return x


def _snap(rng):
    x = np.zeros(N)
    L = int(0.01 * SR)
    off = rng.integers(0, N - L)
    t = np.arange(L) / SR
    x[off:off + L] = rng.normal(0, 1, L) * np.exp(-t / 0.002)
    return x


def _whistle(rng):
    x = np.zeros(N)
    L = int(0.30 * SR)
    off = rng.integers(0, N - L)
    t = np.arange(L) / SR
    f = rng.uniform(1500, 3000)
    env = np.minimum(t / 0.03, 1) * np.minimum((t[-1] - t) / 0.03, 1)
    x[off:off + L] = np.sin(2 * np.pi * f * t) * env
    return x


_MAKERS = {"clap": _clap, "whistle": _whistle, "snap": _snap}


# ----------------------------- feature vector ----------------------------- #
def features_of(window):
    """Log-mel -> per-band mean & std over time -> 2*N_MELS vector."""
    lm = log_mel_spectrogram(window, sr=SR, n_fft=512, hop=160, n_mels=N_MELS)
    return np.concatenate([lm.mean(axis=0), lm.std(axis=0)])


# ----------------------------- dataset builder ---------------------------- #
def build_dataset(n_per_class, rng, noise_std=0.03,
                  snr_range=(-5.0, 20.0)):
    """Return (X, y, snr_true, snr_est).

    snr_true is the injected SNR (events) / -20 dB sentinel (silence).
    snr_est is the on-device estimator's reading on the noisy window — this is
    what we index T(SNR) with (gate #1).
    """
    from snr_estimator import estimate_snr_db  # local import avoids cycle

    noise_power = noise_std ** 2
    X, y, snr_true, snr_est = [], [], [], []
    for ci, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            if cls == "silence":
                window = rng.normal(0, noise_std, N)
                strue = -20.0
            else:
                ev = _MAKERS[cls](rng)
                snr = rng.uniform(*snr_range)
                ev_p = np.mean(ev ** 2)
                ev *= np.sqrt((noise_power * 10 ** (snr / 10)) / max(ev_p, 1e-12))
                window = ev + rng.normal(0, noise_std, N)
                strue = snr
            X.append(features_of(window))
            y.append(ci)
            snr_true.append(strue)
            snr_est.append(estimate_snr_db(window))
    return (np.asarray(X), np.asarray(y),
            np.asarray(snr_true), np.asarray(snr_est))
