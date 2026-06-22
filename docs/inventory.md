# Parts inventory & status

Full list of parts on hand, with readiness for the current build.
Legend: ✅ ready · ⚠️ needs confirming · ⛔ blocked (missing supporting part)

## Microcontrollers
| Part | Qty | Status | Notes |
|------|:--:|:--:|------|
| ESP32 dev board (WiFi + BLE) | 1 | ✅ | The star of this build. USB-powered, no battery needed. |
| Arduino Uno — ELEGOO UNO R3 | 1 | ✅ | Spare / nRF side-project TX or RX. |
| Arduino Uno — blue clone (CH340) | 1 | ✅ | Needs CH340 driver on host. |

## Motion / actuation
| Part | Qty | Status | Notes |
|------|:--:|:--:|------|
| TT gear motors (yellow, ~3–6 V) + wheels | 2 | ⛔ | Need 6–12 V motor supply (battery holder) — not on hand. |
| SG90 micro servo | 1 | ✅ | Usable now (low current; can run off 5 V USB rail with care). |

## Drivers
| Part | Qty | Status | Notes |
|------|:--:|:--:|------|
| L298N dual H-bridge | 1 | ⛔ | Needs separate 6–12 V motor supply — on hold with the motors. |

## Sensors
| Part | Qty | Status | Notes |
|------|:--:|:--:|------|
| HC-SR04 ultrasonic | 1 | ⚠️ | **ECHO is 5 V** → voltage divider (1k/2k) or level shifter before ESP32 GPIO. |
| IR line-tracking sensor ("Tracker Sensor V2.1", digital out) | 1 | ✅ | Digital out; fine on 3.3 V logic in. |
| Analog sound/mic sensor (KY-038 style, LM393 + blue gain pot) | 1 | ✅ | **Flagship sensor.** AO→GPIO34, VCC→3V3 (not 5V), GND→GND. Low quality → good for sound *events*, not keyword spotting. |

## Comms / RF
| Part | Qty | Status | Notes |
|------|:--:|:--:|------|
| nRF24L01 2.4 GHz transceiver (SPI) | 2 | ⚠️ | **3.3 V only.** No decoupling cap on hand → may brown out; short wires, clean 3.3 V, lower TX power/data rate if unstable. |
| HC-05 Bluetooth (classic SPP) | 1 | ✅ | Usable; 5 V VCC, but RX needs divider from 5 V logic. |

## Discrete / breadboard kit
| Part | Qty | Status | Notes |
|------|:--:|:--:|------|
| Breadboard | 1 | ✅ | |
| Jumper — male–female | some | ✅ | Confirmed on hand. |
| Jumper — male–male | some | ✅ | **Confirmed** — used for breadboard hole-to-hole (mic ↔ ESP32). |
| Jumper — female–female | ? | ⚠️ | Count unconfirmed (not needed for current wiring). |
| 74xx logic ICs (AND / NAND / XOR) | ? | ⚠️ | **Read exact part numbers off the chips** before planning the logic project. |
| LEDs (red) | ? | ⚠️ | Quantity to confirm. |
| Resistors | ? | ⚠️ | **Confirm values/quantity** (LED current-limit + input pull-downs). |
| Potentiometer (standalone, panel-mount) | 1 | ✅ | Reserved for live PID tuning when the rover is built. |
| Buzzer or transistor (black component) | 1 | ⚠️ | **Confirm which it is.** |
| White bar component | 1 | ⚠️ | **Confirm:** breadboard strip vs 7-segment display. |
| USB cable | 1 | ✅ | ESP32 power + flashing. |

## Hard constraints (cannot buy parts right now)
- **No motor battery / bench supply (6–12 V)** → all TT-motor / L298N projects on hold.
- **No extra capacitors** (e.g. 10 µF) → nRF24 may reset/brown out without decoupling.
- **Mic is low-quality analog** → distinct sound events only (clap/whistle/snap), not yes/no keywords.

## Readiness by project
| Project | Buildable now? | Missing |
|---------|:--:|------|
| ⭐ Calibrated Edge Classifier (flagship) | ✅ **Yes** | nothing — ESP32 + mic + breadboard + male-male jumpers all on hand |
| A. nRF24 link characterization | ⚠️ Mostly | no decoupling cap (workarounds noted) |
| B. 74xx logic + Verilog | ⚠️ | confirm chip part numbers + resistor values |
| C. ESP32 sensor-fusion IoT node | ✅ Yes | (HC-SR04 needs ECHO divider) |
| D. PID rover + IoT capstone | ⛔ | motor battery pack |

## TODO (user to confirm)
- [ ] Read markings on the 74xx chips (exact part numbers).
- [ ] Confirm resistor values + quantities.
- [ ] Identify the black component (buzzer vs transistor).
- [ ] Identify the white bar (breadboard strip vs 7-segment).
- [ ] Confirm female–female jumper count (optional).
