# Calibrated Edge Classifier on ESP32

> On-device neural audio-event classifier with **per-SNR temperature scaling** —
> a hardware reproduction of published model-calibration research, running on a $5
> microcontroller.

A low-cost analog mic feeds an **ESP32**, which extracts log-mel features, runs a
quantized neural network to label sound events (**clap / whistle / snap / silence**),
estimates the instantaneous SNR, and applies **per-SNR temperature scaling** so the
reported confidence is *calibrated under noise* — not a raw softmax. Inference, SNR
estimation, and calibration **all run on the MCU**. Predictions stream to a
serverless AWS pipeline with a live reliability dashboard.

This is a hardware extension of: *"Calibrating Deep AMC Classifiers with SNR-Adaptive
Temperature Scaling"* (per-SNR temperature scaling, Expected Calibration Error / ECE,
reliability diagrams, RadioML). The RF→audio swap is deliberate: the cheap mic's noise
is exactly the low-SNR regime the calibration method is built for.

---

## How hard is this, honestly?

**Overall: intermediate → advanced embedded-ML.** No single block is exotic. The
difficulty is that the *headline result* — "calibration measurably reduces ECE,
on-device" — only survives if several subtle things are **all** correct at the same
time. Getting the firmware to blink is easy; getting the calibration to actually
reproduce is the hard, interview-worthy part.

| Phase | What | Difficulty | Why |
|------:|------|:----------:|-----|
| 0 | Mic check (`analogRead`, GPIO34) | 🟢 Easy | A few lines; just confirms the ADC sees sound. |
| 1 | Stable sampling (timer-ISR / I2S-ADC) + labeled data capture | 🟡 Medium | Jitter-free pacing matters or the FFT smears. ADC-via-DMA is version-fragile (see note). |
| 2 | Log-mel features + train tiny NN + int8 quantize + **fit `T(SNR)`** | 🟡 Medium | Standard Keras, but the calibration fit must be on the *quantized* logits. |
| 3 | On-device inference + **noise-floor SNR estimator** + apply `T(SNR)` | 🔴 Hard | The estimator is weakest exactly at low SNR; fixed-point feature pipeline on the MCU. |
| 4 | Noise-injection sweep → SNR-vs-ECE + reliability diagrams | 🟡 Medium | Pure Python, but the SNR definition must match the firmware's. |
| 5 | AWS IoT → Lambda → DynamoDB + dashboard | 🟢 Easy* | *Easy for this author specifically — reuses existing serverless strength. |

**Hardest risks, ranked:**
1. **Calibration-correctness chain** — same SNR definition on both paths, and fit `T`
   on int8 logits (not float). Silently wrong if missed; nothing crashes, the numbers
   just lie.
2. **SNR estimator at low SNR** — a threshold-gated "silence" tracker fails when event
   and noise overlap, i.e. the regime that matters. Needs a min-statistics / percentile
   tracker with hysteresis.
3. **On-device feature extraction** within the MCU's compute + RAM budget (fixed-point
   FFT, mel filterbank, log).
4. **TFLite Micro on ESP32** — tensor-arena sizing and op support; int8-MLP fallback if
   the tooling fights back.

**Empirical unknowns that only board data can answer** (not solvable by more planning):
does the KY-038 + ESP32 ADC produce *spectrally separable* events? Does the model
miscalibrate *enough* under noise for `T(SNR)` to visibly help?

---

## What model do we need?

A **tiny audio-event classifier**, sized to run int8 inside the ESP32's usable SRAM.

**Input features.** 16 kHz mono, ~0.5 s window per event → ~32 mel bins × ~32 frames
log-mel "image" (≈1 k values). Short events (clap/snap) can use a shorter window.

**Primary model — small depthwise-separable CNN** (the proven shape for tiny keyword /
sound-event spotting, e.g. MLPerf-Tiny DS-CNN):
- 1 standard conv stem + 2–3 depthwise-separable conv blocks → global pool → dense(4).
- ~10–40 k params; **int8 weights ≈ 15–40 KB in flash**, **tensor arena ≈ 30–60 KB SRAM**.
- Expressive enough to become *confident* (and therefore *miscalibrated* under noise) —
  which is the whole point: a model that's never overconfident has nothing to calibrate.

**Fallback model — int8 MLP on summary features** (the doc's "hand-written int8 MLP"):
- Input = MFCC summary stats (e.g. 13 MFCC × {mean, std} = 26 values).
- 2 hidden layers → dense(4). ~5–10 KB. Trivial to hand-roll in C if TFLM misbehaves.
- Weaker, but removes the TFLite-Micro dependency entirely for Phase 3.

**Budget reality check (ESP32):** ~520 KB SRAM total, ~290–320 KB usable with WiFi up;
4 MB flash typical. Both models fit comfortably; the CNN is the target, the MLP is the
safety net.

**Calibration "model".** Not a network — a fitted scalar function `T(SNR)`: minimize NLL
on a held-out validation set *per SNR bucket*, on the quantized logits, producing a small
lookup/curve the firmware indexes with its on-device SNR estimate.

---

## Hard constraints (current)

- **No motor battery / bench supply** → all motor/L298N projects on hold. This build
  needs none of them.
- **No decoupling caps** → relevant only to the nRF24 side-project, not this one.
- Mic is low-quality analog → good for *distinct events* (clap/whistle/snap), not
  keyword spotting. That limitation is by design here.

## Implementation note — sampling path (Phase 1)

The legacy **I2S built-in-ADC** mode (`i2s_adc_enable` / `I2S_ADC_BUILTIN`) was
**removed in ESP-IDF 5 / arduino-esp32 core 3.x** (replaced by `esp_adc/adc_continuous`).
So:
- **arduino-esp32 core 2.x** → I2S-ADC available.
- **core 3.x** → use `adc_continuous` DMA driver, or a **hardware-timer ISR +
  `adc1_get_raw()`** (most portable — recommended default).

Tell me your **arduino-esp32 core version** before Phase 1 so the sampling sketch targets
the right API. Phase 0 uses plain `analogRead` and is unaffected.

---

## Wiring (Phase 0 — mic)

| Mic pin | ESP32 | Note |
|--------|-------|------|
| `AO` | **GPIO34** | ADC1, input-only, safe with WiFi |
| `VCC` | **3V3** | **not 5V** — AO can exceed 3.3 V and damage the ADC |
| `GND` | GND | |

Blue pot on the sensor board = analog gain; tune after the first run.

---

## Roadmap

`Phase 0` mic check · `Phase 1` sampling + data capture · `Phase 2` features + train +
quantize + fit `T(SNR)` · `Phase 3` on-device inference + SNR estimator + calibration ·
`Phase 4` SNR-vs-ECE evaluation · `Phase 5` AWS dashboard.

**Status: Phase 1** (capture firmware + logger ready; ML calibration core proven on
synthetic data — waiting on a USB **data** cable to collect real audio).
