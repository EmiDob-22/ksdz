# Copyright (C) 2025 Senior AI Architect.
# Licensed under AGPLv3. For commercial licensing, contact the author.
#
# LOSSLESS baseline codecs for the KSDZ benchmark suite.
#
# This module deliberately contains ONLY lossless entropy/dictionary coders
# (zlib/zstd/lz4). They form the "lossless" metric class: the only meaningful
# metric is compression ratio, and exact round-trip is REQUIRED (the runner
# asserts it). The lossy KSDZ spectral transform lives in ksdz_adapter.py as a
# separate class and is never mixed into this list — comparing a lossy
# projection against lossless coders on one ratio leaderboard is semantically
# invalid.
#
# Uniform interface:
#     compress(x: bytes) -> bytes
#     decompress(x: bytes) -> bytes
# metadata: name, available, cls == "lossless", lossless == True, params.

import zlib

CLASS = "lossless"


class ZlibCodec:
    name = "zlib_l6"
    cls = CLASS
    available = True
    lossless = True
    params = {"level": 6}

    def compress(self, x):
        return zlib.compress(x, 6)

    def decompress(self, x):
        return zlib.decompress(x)


class _ZstdCodec:
    cls = CLASS
    lossless = True

    def __init__(self, level):
        import zstandard as zstd
        self.params = {"level": level}
        self._c = zstd.ZstdCompressor(level=level)
        self._d = zstd.ZstdDecompressor()

    def compress(self, x):
        return self._c.compress(x)

    def decompress(self, x):
        return self._d.decompress(x)


def _make_zstd(level):
    try:
        import zstandard  # noqa: F401
        codec = _ZstdCodec(level)
        codec.name = f"zstd_l{level}"
        codec.available = True
        return codec
    except Exception as e:  # pragma: no cover - environment dependent
        return Unavailable(f"zstd_l{level}", repr(e), params={"level": level})


class _Lz4Codec:
    name = "lz4"
    cls = CLASS
    available = True
    lossless = True
    params = {"format": "frame"}

    def __init__(self):
        import lz4.frame as lf
        self._lf = lf

    def compress(self, x):
        return self._lf.compress(x)

    def decompress(self, x):
        return self._lf.decompress(x)


def _make_lz4():
    try:
        return _Lz4Codec()
    except Exception as e:  # pragma: no cover
        return Unavailable("lz4", repr(e), params={"format": "frame"})


class Unavailable:
    """Placeholder for a lossless codec whose dependency is missing. Recorded in
    results with status 'unavailable' rather than silently dropped."""
    cls = CLASS
    lossless = True

    def __init__(self, name, reason, params):
        self.name = name
        self.available = False
        self.reason = reason
        self.params = params


def build_lossless_codecs():
    """Every lossless baseline. Unavailable ones come back as stubs."""
    return [
        ZlibCodec(),
        _make_zstd(3),
        _make_zstd(9),
        _make_lz4(),
    ]
