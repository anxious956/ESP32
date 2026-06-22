"""Reliability-diagram plotting (matplotlib, headless-safe)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibration import expected_calibration_error


def reliability_diagram(probs: np.ndarray, labels: np.ndarray, path: str,
                        title: str = "Reliability diagram", n_bins: int = 15):
    """Save a reliability diagram (accuracy vs confidence per bin) to `path`."""
    ece, stats = expected_calibration_error(probs, labels, n_bins)
    confs = np.array([s[0] for s in stats])
    accs = np.array([s[1] for s in stats])

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    ax.bar(confs, accs, width=1.0 / n_bins, alpha=0.75, edgecolor="black",
           color="#3b7dd8", label="model")
    # gap between confidence and accuracy
    ax.bar(confs, confs - accs, bottom=accs, width=1.0 / n_bins, alpha=0.3,
           color="red", edgecolor="red", label="gap")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return ece


def temperature_curve(levels: np.ndarray, temps: np.ndarray, path: str):
    """Save the fitted T(SNR) curve."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(levels, temps, "o-", color="#d8643b")
    ax.axhline(1.0, ls="--", color="gray", lw=1, label="T = 1 (no scaling)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("fitted temperature T")
    ax.set_title("Per-SNR temperature  T(SNR)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
