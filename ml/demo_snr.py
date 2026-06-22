"""Validate the on-device SNR estimator against KNOWN injected SNR (gate #1).

What this proves before we touch hardware:
  1. The window estimator is monotonic and low-variance vs the true injected SNR
     -> it is a reliable *index* for T(SNR).
  2. The crucial design rule: fit T(SNR) against the ESTIMATOR'S output (not the
     injected truth). Any constant bias then cancels, because the same estimator
     produces the index at inference time. We also report the affine est->true
     map for human-readable dB.
  3. The streaming min-statistics tracker follows the noise floor through events
     without a fragile silence gate.

Run:  python ml/demo_snr.py   ->  ml/outputs/snr_*.png + printed report
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from noise import add_awgn
from snr_estimator import estimate_snr_db, frame_powers, StreamingSNR

SR = 16000
N = SR // 2                      # 0.5 s window
SNR_TRUE = np.arange(-10, 21, 2.5)


# --------------------------------------------------------------------------- #
# synthetic events (unit-ish; add_awgn measures power from the window)        #
# --------------------------------------------------------------------------- #
def make_clap(rng, offset=None):
    x = np.zeros(N)
    L = int(0.12 * SR)
    off = offset if offset is not None else rng.integers(0, N - L)
    t = np.arange(L) / SR
    x[off:off + L] = rng.normal(0, 1, L) * np.exp(-t / 0.04)
    return x


def make_whistle(rng, offset=None):
    x = np.zeros(N)
    L = int(0.30 * SR)
    off = offset if offset is not None else rng.integers(0, N - L)
    t = np.arange(L) / SR
    env = np.minimum(t / 0.03, 1) * np.minimum((t[-1] - t) / 0.03, 1)
    x[off:off + L] = np.sin(2 * np.pi * 2200 * t) * env
    return x


# --------------------------------------------------------------------------- #
# (1) window estimator vs injected SNR                                        #
# --------------------------------------------------------------------------- #
def validate_window(out, rng, n_trials=300):
    means, stds = [], []
    for snr in SNR_TRUE:
        ests = []
        for _ in range(n_trials):
            ev = make_clap(rng)
            ests.append(estimate_snr_db(add_awgn(ev, snr, rng)))
        ests = np.asarray(ests)
        means.append(ests.mean())
        stds.append(ests.std())
    means, stds = np.asarray(means), np.asarray(stds)

    # monotonicity + affine fit (est ~ a*true + b) over the usable range
    usable = SNR_TRUE >= -5
    a, b = np.polyfit(SNR_TRUE[usable], means[usable], 1)
    resid = means[usable] - (a * SNR_TRUE[usable] + b)
    mono = np.all(np.diff(means) > 0)

    print("\n=== (1) Window SNR estimator vs injected SNR ===")
    print(f"  monotonic over full sweep : {mono}")
    print(f"  affine fit (true>=-5 dB)   : est = {a:.3f}*true + {b:.3f}")
    print(f"  residual std around fit    : {resid.std():.3f} dB")
    print(f"  estimator noise (avg std)  : {stds[usable].mean():.3f} dB")
    print("   true_dB   est_mean   est_std")
    for s, m, sd in zip(SNR_TRUE, means, stds):
        print(f"   {s:6.1f}   {m:7.2f}   {sd:6.2f}")

    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(SNR_TRUE, SNR_TRUE, "--", color="gray", label="ideal (est = true)")
    ax.errorbar(SNR_TRUE, means, yerr=stds, fmt="o-", color="#3b7dd8",
                capsize=3, label="estimator (mean ± std)")
    ax.set_xlabel("true injected SNR (dB)")
    ax.set_ylabel("estimated SNR (dB)")
    ax.set_title("SNR estimator validation (clap events)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "snr_validation.png"), dpi=130)
    plt.close(fig)
    return mono, resid.std()


# --------------------------------------------------------------------------- #
# (2) streaming tracker through events                                        #
# --------------------------------------------------------------------------- #
def demo_stream(out, rng, snr_db=10.0):
    # silence | clap | silence | whistle | silence
    segs = [np.zeros(N), make_clap(rng, offset=int(0.15 * N)), np.zeros(N // 2),
            make_whistle(rng, offset=int(0.1 * N)), np.zeros(N)]
    clean = np.concatenate(segs)
    stream = add_awgn(clean, snr_db, rng)

    fp = frame_powers(stream, 256, 128)
    tr = StreamingSNR()
    nf, snr, flag = [], [], []
    for p in fp:
        s, ev = tr.update(p)
        nf.append(tr.noise_floor()); snr.append(s); flag.append(ev)
    nf, snr, flag = map(np.asarray, (nf, snr, flag))
    t = np.arange(len(fp)) * 128 / SR

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    a1.semilogy(t, fp, color="#888", lw=0.8, label="frame power")
    a1.semilogy(t, nf, color="#d8643b", lw=1.5, label="noise floor (min-stats)")
    a1.fill_between(t, fp.min(), fp.max(), where=flag, color="#3b7dd8", alpha=0.15,
                    label="event flag")
    a1.set_ylabel("power"); a1.legend(fontsize=8, loc="upper right")
    a1.set_title(f"Streaming noise-floor tracker (stream @ {snr_db:.0f} dB)")
    a2.plot(t, snr, color="#3b7dd8"); a2.axhline(snr_db, ls="--", color="gray",
             lw=1, label=f"injected {snr_db:.0f} dB")
    a2.fill_between(t, snr.min(), snr.max(), where=flag, color="#3b7dd8", alpha=0.15)
    a2.set_ylabel("estimated SNR (dB)"); a2.set_xlabel("time (s)")
    a2.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "snr_stream.png"), dpi=130)
    plt.close(fig)

    in_event_snr = snr[flag].mean() if flag.any() else float("nan")
    print("\n=== (2) Streaming tracker ===")
    print(f"  mean estimated SNR while event flag is high: {in_event_snr:.2f} dB "
          f"(injected {snr_db:.0f} dB)")


def main():
    out = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    validate_window(out, rng)
    demo_stream(out, rng)
    print(f"\nsaved figures to {out}/")
    print("\nGATE #1 takeaway: fit T(SNR) against the ESTIMATOR's output on noisy\n"
          "calibration windows (not the injected truth) — then the bias cancels.")


if __name__ == "__main__":
    main()
