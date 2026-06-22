# ML pipeline — calibration core

Python side of the flagship. Built **before the hardware data exists** so the
research core (per-SNR temperature scaling) is proven and testable today; the
real ESP32 audio drops in later without changing this code.

## Modules
| File | Role |
|------|------|
| `calibration.py` | softmax/NLL, temperature fit, **per-SNR T(SNR)**, ECE, bin stats |
| `noise.py` | time-domain AWGN at a **known SNR** (Phase 4), with an SNR-recovery check |
| `features.py` | log-mel spectrogram (16 kHz / 0.5 s default), numpy-only so it ports to fixed-point C |
| `plots.py` | reliability diagram + T(SNR) curve |
| `demo_calibration.py` | end-to-end proof on **synthetic** data, no hardware |

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

## Correctness gates (carried into the firmware)
1. **Same SNR definition everywhere:** `SNR_dB = 10*log10(P_signal/P_noise)`,
   `P = mean(x**2)` — identical in `noise.py` and the on-device estimator.
2. **Fit `T(SNR)` on the deployed logits** — on hardware that means the int8 /
   quantized model's logits, never the float model's.
3. Noise is injected in the **time domain, before the FFT**.

## Pending (needs the board)
- Phase 1: real labeled audio capture → replaces the synthetic generator.
- Phase 2: train the DS-CNN / int8-MLP, quantize, then **re-fit `T(SNR)` on the
  quantized logits**.
- Phase 3: port `features.py` + the SNR estimator to ESP32 C++.
