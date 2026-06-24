# Copyright (C) 2025 Senior AI Architect.
# Licensed under AGPLv3. For commercial licensing, contact the author.
#
# LOSSY TRANSFORM adapter for KSDZ.
#
# KSDZ is a lossy spectral projection (FFT -> keep top-k bins), NOT a lossless
# compressor. It is kept in its own class so it can never be ranked against
# zlib/zstd/lz4 on a single "compression ratio" leaderboard. For this class the
# meaningful metrics are reconstruction fidelity (RMSE/NRMSE, L2/L-inf) and
# spectral distortion; the size ratio is recorded only as a *pseudo* ratio,
# explicitly flagged non-equivalent to lossless CR.
#
# Interface is intentionally encode()/decode() (not compress/decompress) to
# signal at the type level that this is a transform codec, not a CompressionCodec.

import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

CLASS = "lossy_transform"


class LossyTransformCodec:
    """Base interface for lossy signal-reduction codecs. Distinct from the
    lossless CompressionCodec interface in baselines.py."""
    cls = CLASS
    lossless = False
    cr_comparable = False   # its size ratio is NOT comparable to lossless CR

    def encode(self, x):  # bytes -> bytes
        raise NotImplementedError

    def decode(self, x):  # bytes -> bytes (approximate)
        raise NotImplementedError


class KsdzTransform(LossyTransformCodec):
    def __init__(self, top_k=100):
        from ksdz_core import KSDZ_Quantum_Encoder
        self.name = f"ksdz_topk{top_k}"
        self.available = True
        self.params = {"top_k": top_k}
        self._enc = KSDZ_Quantum_Encoder()
        self._top_k = top_k

    def encode(self, x):
        return self._enc.compress(x, top_k=self._top_k)

    def decode(self, x):
        return self._enc.decompress(x)


class Unavailable(LossyTransformCodec):
    def __init__(self, name, reason, params):
        self.name = name
        self.available = False
        self.reason = reason
        self.params = params


def build_lossy_codecs(top_k=100):
    try:
        return [KsdzTransform(top_k=top_k)]
    except Exception as e:  # pragma: no cover
        return [Unavailable(f"ksdz_topk{top_k}", repr(e), {"top_k": top_k})]
