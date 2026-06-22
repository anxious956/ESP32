"""Demo: reproduce the SNR-vs-ECE calibration result on SYNTHETIC data.

No hardware required. We fabricate a classifier whose confidence is inflated
(over-confident), and whose over-confidence grows as SNR drops — exactly the
regime where a single global temperature is not enough and per-SNR temperature
scaling wins. This mirrors the AMC paper's thesis in the audio class layout
(clap / whistle / snap / silence) before the real ESP32 data exists.

Run:  python ml/demo_calibration.py
Outputs: ml/outputs/*.png + a printed ECE table.
"""
from __future__ import annotations

import os

import numpy as np

from calibration import (expected_calibration_error, fit_temperature,
                         fit_temperature_per_snr, apply_per_snr_temperature,
                         softmax)
from plots import reliability_diagram, temperature_curve

CLASSES = ["clap", "whistle", "snap", "silence"]
K = len(CLASSES)
SNR_LEVELS = np.array([-10, -5, 0, 5, 10, 15, 20], dtype=float)


def make_synthetic(n_per_snr: int, rng: np.random.Generator):
    """Generate (logits, labels, snrs).

    quality q(SNR) in (0,1): high SNR -> well separated true class (accurate).
    over-confidence inflation grows as SNR drops -> miscalibration concentrated
    at low SNR, so per-SNR T differs across SNR.
    """
    logits, labels, snrs = [], [], []
    for snr in SNR_LEVELS:
        q = 1.0 / (1.0 + np.exp(-(snr - 2.0) / 5.0))      # sigmoid in [0,1]
        separation = 4.0 * q                               # accuracy driver
        inflation = 1.0 + 3.0 * (1.0 - q)                  # over-confidence driver
        for _ in range(n_per_snr):
            y = rng.integers(K)
            z = rng.normal(0.0, 1.0, size=K)
            z[y] += separation                             # boost true class
            z *= inflation                                 # sharpen => over-confident
            logits.append(z)
            labels.append(y)
            snrs.append(snr)
    return (np.asarray(logits), np.asarray(labels), np.asarray(snrs, dtype=float))


def ece_by_snr(probs, labels, snrs):
    rows = []
    for snr in SNR_LEVELS:
        m = snrs == snr
        ece, _ = expected_calibration_error(probs[m], labels[m])
        acc = (probs[m].argmax(1) == labels[m]).mean()
        rows.append((snr, acc, ece))
    return rows


def main():
    out = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)

    # calibration (fit) split and held-out test split
    fit_logits, fit_labels, fit_snrs = make_synthetic(600, rng)
    test_logits, test_labels, test_snrs = make_synthetic(600, rng)

    # ----- fit temperatures on the fit split -----
    global_T = fit_temperature(fit_logits, fit_labels)
    levels, temps = fit_temperature_per_snr(fit_logits, fit_labels, fit_snrs,
                                            SNR_LEVELS)

    # ----- evaluate on the test split -----
    p_uncal = softmax(test_logits, 1.0)
    p_globalT = softmax(test_logits, global_T)
    p_perSNR = apply_per_snr_temperature(test_logits, test_snrs, levels, temps)

    e_uncal, _ = expected_calibration_error(p_uncal, test_labels)
    e_global, _ = expected_calibration_error(p_globalT, test_labels)
    e_perSNR, _ = expected_calibration_error(p_perSNR, test_labels)

    acc = (p_uncal.argmax(1) == test_labels).mean()
    print("\n=== Calibrated Edge Classifier — synthetic calibration demo ===")
    print(f"classes: {CLASSES}   test samples: {len(test_labels)}")
    print(f"overall accuracy (unchanged by scaling): {acc:.3f}\n")
    print(f"global temperature T          = {global_T:.3f}")
    print("per-SNR T(SNR):")
    for s, t in zip(levels, temps):
        print(f"    SNR {int(s):+3d} dB  ->  T = {t:.3f}")

    print("\nExpected Calibration Error (lower = better):")
    print(f"    uncalibrated (T=1)        : {e_uncal:.4f}")
    print(f"    global temperature        : {e_global:.4f}  "
          f"({100*(e_uncal-e_global)/e_uncal:+.1f}% vs uncal)")
    print(f"    per-SNR temperature       : {e_perSNR:.4f}  "
          f"({100*(e_uncal-e_perSNR)/e_uncal:+.1f}% vs uncal)")

    print("\nECE by SNR (uncalibrated):")
    for snr, a, e in ece_by_snr(p_uncal, test_labels, test_snrs):
        print(f"    SNR {int(snr):+3d} dB   acc={a:.2f}   ECE={e:.4f}")

    # ----- figures -----
    reliability_diagram(p_uncal, test_labels, os.path.join(out, "reliability_uncalibrated.png"),
                        title="Uncalibrated (T=1)")
    reliability_diagram(p_perSNR, test_labels, os.path.join(out, "reliability_perSNR.png"),
                        title="Per-SNR temperature scaling")
    temperature_curve(levels, temps, os.path.join(out, "temperature_curve.png"))
    print(f"\nsaved figures to {out}/")


if __name__ == "__main__":
    main()
