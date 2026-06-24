# Copyright (C) 2025 Senior AI Architect.
# Licensed under AGPLv3. For commercial licensing, contact the author.
#
# Deterministic dataset generators for the KSDZ benchmark suite.
#
# Evaluation protocol (documented, not assumed):
# Each generator produces a float signal, then quantizes it to uint8 via
# min/max normalization to [0, 255]. The uint8 byte stream is the canonical
# artifact every codec sees. This matches KSDZ's native domain (it operates on
# uint8 signals), keeps all error/spectral metrics finite for both lossy and
# lossless codecs, and makes the comparison fair. The reproducibility hash is
# SHA256 over those exact uint8 bytes, so a (name, N, seed) triple pins one
# byte-for-byte input.

import hashlib
import numpy as np


def quantize_uint8(signal):
    """Min/max normalize a float signal into uint8 [0, 255]."""
    signal = np.asarray(signal, dtype=np.float64).ravel()
    s_min = float(np.min(signal))
    s_max = float(np.max(signal))
    if s_max > s_min:
        normalized = 255.0 * (signal - s_min) / (s_max - s_min)
    else:
        normalized = np.zeros_like(signal)
    return np.clip(np.round(normalized), 0, 255).astype(np.uint8)


def sinusoidal_mixture(N, seed):
    """Sum of a few sinusoids with seed-chosen frequencies/phases. Near-periodic
    (best case for spectral-keep-top-k codecs)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 100.0, N)
    freqs = rng.uniform(0.5, 8.0, size=5)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=5)
    amps = rng.uniform(0.5, 1.0, size=5)
    sig = np.zeros(N)
    for a, f, p in zip(amps, freqs, phases):
        sig += a * np.sin(2.0 * np.pi * f * t / 100.0 + p)
    return sig


def gaussian_noise(N, seed):
    """White Gaussian noise stream (high-entropy worst case)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(N)


def lorenz_trajectory(N, seed):
    """x-coordinate of a Lorenz attractor integrated with RK4.
    Deterministic given (N, seed); seed perturbs the initial condition."""
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    dt = 0.01
    rng = np.random.default_rng(seed)
    state = np.array([1.0, 1.0, 1.0]) + rng.uniform(-0.01, 0.01, size=3)

    def deriv(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    out = np.empty(N)
    for i in range(N):
        out[i] = state[0]
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def random_walk(N, seed):
    """Cumulative sum of Gaussian increments (non-stationary, smooth-ish)."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(N))


# Registry: name -> float-signal generator(N, seed)
GENERATORS = {
    "sinusoidal_mixture": sinusoidal_mixture,
    "gaussian_noise": gaussian_noise,
    "lorenz_trajectory": lorenz_trajectory,
    "random_walk": random_walk,
}


class Dataset:
    def __init__(self, name, N, seed, array_uint8):
        self.name = name
        self.N = int(N)
        self.seed = int(seed)
        self.array = array_uint8                     # uint8 ndarray
        self.data = array_uint8.tobytes()            # canonical bytes
        self.sha256 = hashlib.sha256(self.data).hexdigest()
        self.original_size = len(self.data)


def build_dataset(name, N, seed):
    if name not in GENERATORS:
        raise KeyError(f"unknown dataset '{name}'; have {sorted(GENERATORS)}")
    signal = GENERATORS[name](N, seed)
    return Dataset(name, N, seed, quantize_uint8(signal))


def dataset_names():
    return sorted(GENERATORS)
