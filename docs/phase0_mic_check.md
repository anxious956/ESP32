# Phase 0 — Mic check

Goal: prove the ESP32 ADC actually responds to sound on the KY-038 `AO` pin, and
tune the gain pot, **before** building the full pipeline. ~10 minutes.

## Wiring

| Mic pin | ESP32 | Note |
|--------|-------|------|
| `AO`  | **GPIO34** | ADC1_CH6, input-only, safe with WiFi |
| `VCC` | **3V3** | **not 5V** — AO can exceed 3.3 V and damage the ADC |
| `GND` | GND | |

Power the ESP32 from USB (PlatformIO). Leave `D0` unconnected — we use the analog `AO`.

## Build & flash (PlatformIO in VS Code)

- Click **PlatformIO: Upload** (→ arrow in the bottom bar), or run `pio run -t upload`.
- If `board` in `platformio.ini` doesn't match yours, change it (pins are unchanged).

## Run the live monitor

Find the port first: PlatformIO **Devices**, or `pio device list`.

```bash
pip install pyserial            # once
python tools/serial_logger.py --port /dev/ttyUSB0     # Linux/Mac
python tools/serial_logger.py --port COM5             # Windows
```

> Close the PlatformIO Serial Monitor before running the logger — only one program
> can hold the port at a time.

## Pass / fail

| Condition | Expected | Verdict |
|----------|----------|---------|
| Quiet room | `rms` low and roughly steady; `floor` settles | baseline OK |
| Clap / whistle / snap ~30 cm away | `rms` and `p2p` jump clearly, `EVENT` prints | ✅ **PASS** |
| `p2p` pinned near 4095 while quiet | input clipping | turn blue pot **down** |
| Nothing moves on a clap | no signal | pot **up**, or check wiring / that VCC is on **3V3** |
| `mean` ~0 or ~4095 always | pin floating / mis-wired | recheck `AO`→GPIO34 |

**PASS = a clap is clearly separable from the quiet floor.** That's all Phase 0 needs.
Note the rough `rms` floor vs clap values — it sets the starting point for the Phase 1
sampling thresholds and later the SNR estimator.

## Notes
- `rms_ac` is the RMS after removing the DC bias — i.e. the "loudness", which is what
  we threshold on. `p2p` is a quick clipping indicator.
- Phase 0 uses plain `analogRead` (timing not critical). Stable, jitter-free sampling
  arrives in Phase 1 (timer-ISR / I2S-ADC).
