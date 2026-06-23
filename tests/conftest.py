import numpy as np
import pytest


@pytest.fixture
def seeded_rng():
    return np.random.default_rng(seed=1234)


@pytest.fixture
def random_bytes(seeded_rng):
    def _make(n):
        return seeded_rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
    return _make


@pytest.fixture
def periodic_bytes():
    """A smooth, low-frequency signal: the kind of input KSDZ's
    top-k spectral truncation can actually reconstruct well."""
    def _make(n, freq=2.0):
        t = np.linspace(0, 2 * np.pi, n)
        sig = np.sin(freq * t)
        return (((sig + 1.0) * 127.5).clip(0, 255)).astype(np.uint8).tobytes()
    return _make
