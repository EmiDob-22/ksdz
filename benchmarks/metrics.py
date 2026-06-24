# Copyright (C) 2025 Senior AI Architect.
# Licensed under AGPLv3. For commercial licensing, contact the author.
#
# Formal error / distortion metrics for the KSDZ benchmark suite. All metrics
# operate on the uint8 canonical signal cast to float, so results are finite and
# comparable across lossy and lossless codecs.

import numpy as np


def compression_ratio(original_size, compressed_size):
    """CR = 1 - compressed/original. Higher is smaller output.
    Negative means the codec expanded the data."""
    if original_size <= 0:
        return float("nan")
    return 1.0 - (compressed_size / original_size)


def _as_float(arr_or_bytes):
    if isinstance(arr_or_bytes, (bytes, bytearray)):
        return np.frombuffer(bytes(arr_or_bytes), dtype=np.uint8).astype(np.float64)
    return np.asarray(arr_or_bytes, dtype=np.float64).ravel()


def _aligned(a, b):
    """Truncate to common length so a codec returning a different-length
    reconstruction still yields a finite, defined error."""
    a = _as_float(a)
    b = _as_float(b)
    n = min(a.size, b.size)
    return a[:n], b[:n]


def l2_error(x, x_hat):
    a, b = _aligned(x, x_hat)
    if a.size == 0:
        return float("nan")
    return float(np.linalg.norm(a - b))


def linf_error(x, x_hat):
    a, b = _aligned(x, x_hat)
    if a.size == 0:
        return float("nan")
    return float(np.max(np.abs(a - b)))


_EPS = 1e-12


def rmse(x, x_hat):
    """Root mean squared error — length-independent, unlike raw L2."""
    a, b = _aligned(x, x_hat)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def nrmse(x, x_hat):
    """RMSE normalized by the dynamic range of x, so it is comparable across
    datasets: nrmse = rmse / (max(x) - min(x) + eps)."""
    a, _ = _aligned(x, x_hat)
    if a.size == 0:
        return float("nan")
    rng = float(np.max(a) - np.min(a))
    return rmse(x, x_hat) / (rng + _EPS)


def spectral_distortion(x, x_hat):
    """L2 distance between FFT magnitude spectra: || |FFT(x)| - |FFT(x_hat)| ||."""
    a, b = _aligned(x, x_hat)
    if a.size == 0:
        return float("nan")
    fa = np.abs(np.fft.rfft(a))
    fb = np.abs(np.fft.rfft(b))
    return float(np.linalg.norm(fa - fb))


def spectral_distortion_normalized(x, x_hat):
    """Spectral distortion normalized by the original spectrum energy, so it is
    comparable across signal lengths: ||.|| / (|| |FFT(x)| || + eps)."""
    a, b = _aligned(x, x_hat)
    if a.size == 0:
        return float("nan")
    fa = np.abs(np.fft.rfft(a))
    fb = np.abs(np.fft.rfft(b))
    return float(np.linalg.norm(fa - fb) / (np.linalg.norm(fa) + _EPS))


def reconstruction_metrics(x, x_hat):
    """Fidelity metrics for a lossy (original, reconstruction) pair. Includes
    both raw and length-normalized forms so values are comparable across
    datasets and N."""
    return {
        "l2_error": l2_error(x, x_hat),
        "linf_error": linf_error(x, x_hat),
        "rmse": rmse(x, x_hat),
        "nrmse": nrmse(x, x_hat),
        "spectral_distortion": spectral_distortion(x, x_hat),
        "spectral_distortion_normalized": spectral_distortion_normalized(x, x_hat),
        "length_match": int(_as_float(x).size == _as_float(x_hat).size),
    }
