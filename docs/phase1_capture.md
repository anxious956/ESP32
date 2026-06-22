# Phase 1 — labeled audio capture

Collect the training set: short labeled windows of **clap / whistle / snap /
silence**, ~50–100 each. Each window is 0.5 s @ 16 kHz (8000 samples).

## Order of operations (when the cable arrives)
1. **Verify the mic first** with Phase 0:
   - PlatformIO toolbar → select environment **`phase0_miccheck`** → Upload.
   - `python tools/serial_logger.py --port COMxx` → clap should make `rms` jump.
2. Then switch to capture:
   - Select environment **`phase1_capture`** → Upload.
   - Close the PlatformIO Serial Monitor (frees the port).

## Collect
```bash
pip install pyserial
python tools/phase1_collect.py --port COM5 --label clap
```
- Press **Enter** to capture one window (clap right after pressing).
- Type a label name (`whistle`, `snap`, `silence`, `clap`) then Enter to **switch class**.
- `q` to quit. Progress count is shown per class.

Each capture appends a row to `data/<label>.csv`. After every clap the tool prints
the window `rms` — if it doesn't rise on a clap, re-check the pot / wiring before
collecting a whole class of bad data.

## Tips for a dataset that makes calibration visible
- Record in the **noisy regime too** — the calibration story only shows up under
  noise. Vary distance, add background sound for some windows. (We also inject
  controlled noise digitally in Phase 4; real variation helps the model generalize.)
- Keep `silence` genuinely varied (room tone, fan, distant talk) — it is the
  noise-floor reference for the SNR estimator later.
- ~50 windows/class is enough to start; 100+ is better.

## Format
`data/clap.csv` etc. — each row is one window of 8000 integer ADC samples
(0–4095), 16 kHz. Loaded directly in Phase 2 (`ml/features.py` → log-mel).

## Notes
- Serial is **921600 baud** in this phase (fast bulk transfer); the tool sets it
  automatically.
- Data CSVs are gitignored (can get large); `data/.gitkeep` keeps the folder.
