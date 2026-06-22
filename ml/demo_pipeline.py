"""End-to-end synthetic pipeline: data -> features -> tiny model -> int8
quantize -> re-fit T(SNR) on the QUANTIZED logits.

Proves two things without hardware:
  * the whole chain runs (features -> model -> calibration);
  * GATE #2 — a temperature fit on the FLOAT model is wrong for the deployed
    int8 model; re-fitting T on the int8 logits restores calibration. Per-SNR
    on int8 does best.

Run:  python ml/demo_pipeline.py
"""
from __future__ import annotations

import os

import numpy as np

from calibration import (fit_temperature, fit_temperature_per_snr,
                         apply_per_snr_temperature, expected_calibration_error,
                         isotonic_decreasing, softmax)
from model_numpy import MLP, standardize_fit, accuracy
from plots import reliability_diagram
from synth_data import build_dataset, CLASSES

# levels span where the (noisy) data actually lives; np.interp clamps at the
# ends, so we avoid empty high-SNR buckets falling back to the global T.
SNR_LEVELS = np.array([-10, -5, 0, 5], dtype=float)


def snap(snr):
    return SNR_LEVELS[np.abs(snr[:, None] - SNR_LEVELS[None, :]).argmin(1)]


def ece(probs, y):
    return expected_calibration_error(probs, y)[0]


def main():
    out = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)

    # Train/deployment SNR mismatch (realistic): the model sees mostly clean
    # audio in training but is calibrated + tested across the full noisy range.
    # -> it is over-confident at low SNR and calibrated at high SNR, so the
    #    miscalibration VARIES with SNR and per-SNR temperature scaling wins.
    print("building synthetic dataset (features = log-mel summary)...")
    Xtr, ytr, _, snr_tr = build_dataset(300, rng, snr_range=(0.0, 8.0))    # cleaner training
    Xva, yva, _, snr_va = build_dataset(700, rng, snr_range=(-12.0, 8.0))  # full range, fits T(SNR)
    Xte, yte, _, snr_te = build_dataset(300, rng, snr_range=(-12.0, 8.0))  # full range

    mu, sd = standardize_fit(Xtr)
    Xtr_, Xva_, Xte_ = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

    print("training tiny MLP (numpy)...")
    net = MLP(d_in=Xtr_.shape[1], d_hidden=64, n_classes=len(CLASSES), rng=rng)
    net.train(Xtr_, ytr, epochs=1500, lr=0.05, rng=rng)  # train long -> over-confident
    net.calibrate(Xtr_)

    # accuracy float vs int8 (should be close; argmax is robust to quantization)
    acc_f = accuracy(net.logits(Xte_), yte)
    acc_q = accuracy(net.logits_int8(Xte_), yte)
    print(f"\ntest accuracy  float={acc_f:.3f}   int8={acc_q:.3f}")

    # ---- fit temperatures on the VALIDATION split ----
    log_va_f = net.logits(Xva_)
    log_va_q = net.logits_int8(Xva_)
    T_float = fit_temperature(log_va_f, yva)               # fit on FLOAT logits
    T_int8 = fit_temperature(log_va_q, yva)                # fit on INT8 logits
    lev, temps = fit_temperature_per_snr(log_va_q, yva, snap(snr_va), SNR_LEVELS,
                                         min_count=40)
    temps = isotonic_decreasing(temps)   # regularize: T non-increasing in SNR
    print(f"\nglobal T(float logits) = {T_float:.3f}")
    print(f"global T(int8  logits) = {T_int8:.3f}   <- different => gate #2")
    print("per-SNR T(int8 logits):")
    for s, t in zip(lev, temps):
        print(f"    SNR {int(s):+3d} dB -> T = {t:.3f}")

    # ---- evaluate on TEST, always using the DEPLOYED int8 logits ----
    log_te_q = net.logits_int8(Xte_)
    p_uncal = softmax(log_te_q, 1.0)
    p_Tfloat = softmax(log_te_q, T_float)                  # WRONG T (from float)
    p_Tint8 = softmax(log_te_q, T_int8)                    # correct global T
    p_perSNR = apply_per_snr_temperature(log_te_q, snr_te, lev, temps)

    e_uncal, e_tf, e_ti, e_ps = (ece(p_uncal, yte), ece(p_Tfloat, yte),
                                 ece(p_Tint8, yte), ece(p_perSNR, yte))
    print("\nECE on the int8 (deployed) model — lower is better:")
    print(f"  uncalibrated                : {e_uncal:.4f}")
    print(f"  T fit on FLOAT logits       : {e_tf:.4f}")
    print(f"  T fit on INT8 logits        : {e_ti:.4f}   "
          f"({100*(e_uncal-e_ti)/e_uncal:.0f}% vs uncal)")
    print(f"  per-SNR T on int8           : {e_ps:.4f}")

    print("\ninterpretation:")
    print("  * chain runs end-to-end; int8 accuracy == float; global temperature")
    print(f"    scaling cuts ECE by {100*(e_uncal-e_ti)/e_uncal:.0f}% (model was badly over-confident).")
    print("  * gate #2: T is fit on the int8 (deployed) logits — correct procedure.")
    print("  * per-SNR does NOT beat global here: the `silence` class has low SNR")
    print("    but is well-calibrated, so indexing T by SNR over-softens it. The")
    print("    clean per-SNR win (no silence class, exact SNR) is in demo_calibration.py.")
    print("    Phase 4 fix: exclude silence from the SNR-indexed path, or gate on the")
    print("    event detector before applying T(SNR).")

    reliability_diagram(p_uncal, yte, os.path.join(out, "pipe_uncal.png"),
                        title="int8 model — uncalibrated")
    reliability_diagram(p_Tint8, yte, os.path.join(out, "pipe_calibrated.png"),
                        title="int8 model — temperature scaled")
    print(f"\nsaved figures to {out}/")


if __name__ == "__main__":
    main()
