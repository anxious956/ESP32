#!/usr/bin/env python3
"""Phase 0 serial logger / mic check.

Reads the ESP32 Phase-0 CSV stream (min,max,mean,rms_ac,p2p) and shows a live
one-line dashboard with an RMS bar, a slow noise-floor estimate, a session peak,
and an EVENT flag — enough to confirm the mic works and to tune the gain pot.

Pass/fail (see docs/phase0_mic_check.md):
  - Quiet room      -> rms_ac low and roughly steady (a clean noise floor).
  - Clap/whistle/snap ~30 cm away -> rms_ac and p2p jump clearly above the floor.
  - p2p stuck near 4095 while quiet -> gain too high, turn the blue pot DOWN.
  - Nothing moves on a clap        -> gain too low (pot UP) or check wiring / 3V3.

Usage:
  python tools/serial_logger.py --port /dev/ttyUSB0
  python tools/serial_logger.py --port COM5 --csv session.csv
"""
import argparse
import sys
import time

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial not installed. Run:  pip install pyserial")


def main() -> None:
    ap = argparse.ArgumentParser(description="ESP32 Phase 0 mic-check serial logger")
    ap.add_argument("--port", required=True, help="serial port (e.g. /dev/ttyUSB0, COM5)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--csv", help="optional path to append raw CSV rows")
    args = ap.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=2)
    except serial.SerialException as e:
        sys.exit(f"Could not open {args.port}: {e}")
    time.sleep(0.3)

    peak_rms = 0.0
    floor = None
    csvf = open(args.csv, "a") if args.csv else None
    if csvf:
        csvf.write("# min,max,mean,rms_ac,p2p\n")

    print(f"Listening on {args.port} @ {args.baud}. Ctrl-C to stop.\n")
    try:
        while True:
            raw = ser.readline().decode("utf-8", "replace").strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split(",")
            if len(parts) != 5:
                continue
            try:
                vmin, vmax, mean, rms, p2p = (float(x) for x in parts)
            except ValueError:
                continue

            if csvf:
                csvf.write(raw + "\n")

            peak_rms = max(peak_rms, rms)
            # noise floor = low envelope: snap to any lower RMS, leak up slowly
            floor = rms if floor is None else min(rms, floor + 0.5)
            event = rms > max(floor * 3.0, floor + 30.0)

            scale = max(peak_rms, 1.0)
            n = int(40 * min(rms / scale, 1.0))
            bar = "#" * n + "-" * (40 - n)
            flag = "  <== EVENT" if event else ""
            sys.stdout.write(
                f"\rrms={rms:7.1f} floor={floor:6.1f} p2p={p2p:5.0f} "
                f"peak={peak_rms:7.1f} [{bar}]{flag}   "
            )
            sys.stdout.flush()
            if event:
                sys.stdout.write("\n")  # keep a permanent line for each event
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if csvf:
            csvf.close()
        ser.close()


if __name__ == "__main__":
    main()
