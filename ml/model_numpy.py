"""Tiny MLP in numpy + simulated int8 post-training quantization.

This stands in for the TFLite-Micro int8 model so the end-to-end chain and the
calibration gate #2 ("fit T on the DEPLOYED/quantized logits") can be proven
without a TensorFlow dependency. The real int8 TFLite export happens in Phase 2
on the recorded data; the *logic* — quantization shifts the logits, so the
temperature must be re-fit on them — is identical and demonstrated here.

Model: d_in -> d_hidden (ReLU) -> n_classes (logits).
Quantization: per-tensor symmetric int8 fake-quant on inputs, weights, and the
hidden activation (ranges from a calibration pass).
"""
from __future__ import annotations

import numpy as np


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fake_quant(x, scale):
    """Symmetric int8 fake-quant: round to the int8 grid, then dequantize."""
    if scale <= 0:
        return np.asarray(x, dtype=np.float64)
    q = np.clip(np.round(x / scale), -127, 127)
    return q * scale


class MLP:
    def __init__(self, d_in, d_hidden, n_classes, rng):
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_in), (d_in, d_hidden))
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / d_hidden), (d_hidden, n_classes))
        self.b2 = np.zeros(n_classes)
        self.q = None  # quantization scales, set by calibrate()

    # ----------------------------- float path ---------------------------- #
    def logits(self, X):
        a1 = np.maximum(X @ self.W1 + self.b1, 0.0)
        return a1 @ self.W2 + self.b2

    def train(self, X, y, epochs=400, lr=0.05, batch=128, rng=None):
        rng = rng or np.random.default_rng(0)
        n, K = len(X), self.W2.shape[1]
        Y = np.eye(K)[y]
        vW1 = vW2 = vb1 = vb2 = 0.0  # momentum
        mom = 0.9
        for ep in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                xb, yb = X[bi], Y[bi]
                z1 = xb @ self.W1 + self.b1
                a1 = np.maximum(z1, 0.0)
                p = _softmax(a1 @ self.W2 + self.b2)
                # gradients (softmax cross-entropy)
                dz2 = (p - yb) / len(bi)
                gW2 = a1.T @ dz2
                gb2 = dz2.sum(0)
                da1 = dz2 @ self.W2.T
                dz1 = da1 * (z1 > 0)
                gW1 = xb.T @ dz1
                gb1 = dz1.sum(0)
                vW1 = mom * vW1 - lr * gW1; self.W1 += vW1
                vb1 = mom * vb1 - lr * gb1; self.b1 += vb1
                vW2 = mom * vW2 - lr * gW2; self.W2 += vW2
                vb2 = mom * vb2 - lr * gb2; self.b2 += vb2
        return self

    # --------------------------- quantized path -------------------------- #
    def calibrate(self, X):
        """Set int8 scales from a calibration batch."""
        a1 = np.maximum(X @ self.W1 + self.b1, 0.0)
        out = a1 @ self.W2 + self.b2
        self.q = {
            "x": np.abs(X).max() / 127.0,
            "W1": np.abs(self.W1).max() / 127.0,
            "a1": np.abs(a1).max() / 127.0,
            "W2": np.abs(self.W2).max() / 127.0,
            "out": np.abs(out).max() / 127.0,   # TFLite int8 quantizes the output too
        }
        return self

    def logits_int8(self, X):
        """Forward pass with simulated int8 weights/activations/output.

        The OUTPUT quantization is what makes the deployed logits genuinely
        differ from the float logits, so the temperature must be re-fit on them
        (gate #2).
        """
        assert self.q is not None, "call calibrate() first"
        Xq = _fake_quant(X, self.q["x"])
        W1q = _fake_quant(self.W1, self.q["W1"])
        a1 = np.maximum(Xq @ W1q + self.b1, 0.0)
        a1q = _fake_quant(a1, self.q["a1"])
        W2q = _fake_quant(self.W2, self.q["W2"])
        out = a1q @ W2q + self.b2
        return _fake_quant(out, self.q["out"])


def standardize_fit(X):
    mu = X.mean(0); sd = X.std(0) + 1e-8
    return mu, sd


def accuracy(logits, y):
    return float((logits.argmax(1) == y).mean())
