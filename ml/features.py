"""Log-mel feature extraction (Phase 2).

Pure numpy/scipy so it runs anywhere and is easy to mirror in fixed-point C on
the ESP32 later. Defaults target the flagship: 16 kHz mono, ~0.5 s windows.

Pipeline: frame -> Hann window -> rFFT -> power spectrum -> mel filterbank ->
log. The same parameters must be used for training and on-device inference.
"""
from __future__ import annotations

import numpy as np


def hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int,
                   fmin: float = 50.0, fmax: float | None = None) -> np.ndarray:
    """Triangular mel filterbank: (n_mels, n_fft//2 + 1)."""
    fmax = fmax or sr / 2.0
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_bins)

    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.searchsorted(fft_freqs, hz_pts)

    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        for k in range(n_bins):
            f = fft_freqs[k]
            if left <= f <= center and center > left:
                fb[m - 1, k] = (f - left) / (center - left)
            elif center <= f <= right and right > center:
                fb[m - 1, k] = (right - f) / (right - center)
    return fb


def log_mel_spectrogram(signal: np.ndarray, sr: int = 16000,
                        n_fft: int = 512, hop: int = 160, n_mels: int = 32,
                        fmin: float = 50.0, fmax: float | None = None,
                        eps: float = 1e-6) -> np.ndarray:
    """Return a (n_frames, n_mels) log-mel spectrogram."""
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)))

    window = np.hanning(n_fft)
    fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)

    frames = []
    for start in range(0, len(signal) - n_fft + 1, hop):
        frame = signal[start:start + n_fft] * window
        spec = np.abs(np.fft.rfft(frame)) ** 2          # power spectrum
        mel = fb @ spec                                  # (n_mels,)
        frames.append(np.log(mel + eps))
    return np.asarray(frames, dtype=np.float64)
