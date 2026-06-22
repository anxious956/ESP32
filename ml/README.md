# ML pipeline — calibration core

Python side of the flagship. Built **before the hardware data exists** so the
research core (per-SNR temperature scaling) is proven and testable today; the
real ESP32 audio drops in later without changing this code.

## Modules
| File | Role |
|------|------|
| `calibration.py` | softmax/NLL, temperature fit, **per-SNR T(SNR)**, ECE, bin stats |
| `snr_estimator.py` | on-device SNR: single-window (percentile) + streaming **min-statistics + hysteresis** tracker |
| `demo_snr.py` | validates the estimator against injected SNR (gate #1) |
| `noise.py` | time-domain AWGN at a **known SNR** (Phase 4), with an SNR-recovery check |
| `features.py` | log-mel spectrogram (16 kHz / 0.5 s default), numpy-only so it ports to fixed-point C |
| `synth_data.py` | synthetic clap/whistle/snap/silence dataset (fixed noise floor, SNR via event amplitude) |
| `model_numpy.py` | tiny MLP + **simulated int8 PTQ** (stands in for TFLite Micro to prove gate #2) |
| `plots.py` | reliability diagram + T(SNR) curve |
| `demo_calibration.py` | per-SNR vs global on **synthetic logits** (the clean research result) |
| `demo_pipeline.py` | **end-to-end**: data → features → train → int8 → re-fit T (gate #2) |

## Run
```bash
pip install -r requirements.txt
python demo_calibration.py        # prints ECE table, writes outputs/*.png
```

## What the demo shows (synthetic)
A classifier whose over-confidence grows as SNR drops. On a held-out test set:

| Method | ECE | vs uncalibrated |
|--------|-----|-----------------|
| uncalibrated (T=1) | 0.097 | — |
| global temperature | 0.075 | −23% |
| **per-SNR temperature** | **0.015** | **−85%** |

A single global `T` can't fix a model that is over-confident at low SNR *and*
slightly under-confident at high SNR — per-SNR `T(SNR)` can. This is the AMC
paper's thesis in the audio class layout (clap / whistle / snap / silence).

## End-to-end pipeline (`python demo_pipeline.py`)
Synthetic audio → log-mel → tiny MLP → simulated int8 → calibration, with a
realistic **train/deployment SNR mismatch** (trained cleaner, tested noisy):

| Method (on the int8 model) | ECE |
|----------------------------|-----|
| uncalibrated | 0.260 |
| global temperature (fit on int8 logits) | **0.074  (−72%)** |
| per-SNR temperature | 0.084 |

- The chain runs end-to-end; **int8 accuracy == float**.
- **Gate #2** in practice: temperature is fit on the int8 (deployed) logits.
- Honest finding: per-SNR does **not** beat global here, because the `silence`
  class has low SNR but is well-calibrated, so indexing `T` by SNR over-softens
  it. The clean per-SNR win (no silence class, exact SNR) is in
  `demo_calibration.py`. **Phase 4 fix:** gate the SNR-indexed `T` on the event
  detector (don't apply it to silence).

## SNR estimator validation (`python demo_snr.py`)
Window estimator vs injected SNR (clap events): monotonic, affine fit
`est = 1.007*true - 0.21`, residual std **0.036 dB**, per-estimate std ~0.16 dB.
So it is a reliable index for `T(SNR)`. The streaming tracker reports a higher
*instantaneous/peak* SNR during an event (vs the window-average) — expected;
calibration uses the window estimator.

## Correctness gates (carried into the firmware)
1. **Same SNR definition everywhere:** `SNR_dB = 10*log10(P_signal/P_noise)`,
   `P = mean(x**2)` — identical in `noise.py` and the on-device estimator.
   And **fit `T(SNR)` against the estimator's output** (not the injected truth),
   so any estimator bias cancels at inference time.
2. **Fit `T(SNR)` on the deployed logits** — on hardware that means the int8 /
   quantized model's logits, never the float model's.
3. Noise is injected in the **time domain, before the FFT**.

## Pending (needs the board)
- Phase 1: real labeled audio capture → replaces the synthetic generator.
- Phase 2: train the DS-CNN / int8-MLP, quantize, then **re-fit `T(SNR)` on the
  quantized logits**.
- Phase 3: port `features.py` + the SNR estimator to ESP32 C++.
