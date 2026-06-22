#!/usr/bin/env python3
"""Phase 1 — labeled audio capture.

Drives the phase1_capture firmware: press Enter to grab one 0.5 s window for the
current label; each window is appended to data/<label>.csv (one row = one window
of N int samples @ 16 kHz). Aim for ~50-100 windows per class.

Usage:
  python tools/phase1_collect.py --port COM5 --label clap
  python tools/phase1_collect.py --port /dev/ttyUSB0 --label silence

Interactive keys (then Enter):
  <Enter>                  capture one window for the current label
  clap|whistle|snap|silence   switch label
  q                        quit
"""
import argparse
import csv
import os
import sys
import time

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial not installed. Run:  pip install pyserial")

LABELS = ["clap", "whistle", "snap", "silence"]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def read_window(ser, timeout_s=10):
    """Send 'c', parse BEGIN/<data>/END. Returns (samples, sample_rate) or (None, None)."""
    ser.reset_input_buffer()
    ser.write(b"c\n")
    n = sr = None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", "replace").strip()
        if line.startswith("BEGIN"):
            parts = line.split()
            n, sr = int(parts[1]), int(parts[2])
            break
    if n is None:
        return None, None
    data_line = ser.readline().decode("utf-8", "replace").strip()
    ser.readline()  # consume END
    vals = [int(x) for x in data_line.split(",") if x]
    if len(vals) != n:
        return None, sr
    return vals, sr


def rms_ac(vals):
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5


def count_rows(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def main():
    ap = argparse.ArgumentParser(description="Phase 1 labeled audio capture")
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--label", default="clap", choices=LABELS)
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        ser = serial.Serial(args.port, args.baud, timeout=3)
    except serial.SerialException as e:
        sys.exit(f"Could not open {args.port}: {e}")
    time.sleep(0.3)

    label = args.label
    print(f"Port {args.port} @ {args.baud}. Saving to {DATA_DIR}/")
    print("Enter=capture | type a label to switch | q=quit")
    while True:
        path = os.path.join(DATA_DIR, f"{label}.csv")
        try:
            cmd = input(f"[{label}: {count_rows(path)} saved] Enter to capture > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if cmd == "q":
            break
        if cmd in LABELS:
            label = cmd
            continue
        if cmd != "":
            print(f"  unknown command; valid labels: {LABELS}")
            continue

        vals, sr = read_window(ser)
        if vals is None:
            print("  capture FAILED (no/short data) — is the firmware flashed & port free?")
            continue
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(vals)
        print(f"  saved 1 window  (sr={sr}Hz, rms={rms_ac(vals):.1f})")

    ser.close()


if __name__ == "__main__":
    main()
