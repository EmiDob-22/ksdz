import struct

import numpy as np
import pytest

from ksdz_core import KSDZ_Quantum_Encoder


@pytest.fixture
def encoder():
    return KSDZ_Quantum_Encoder()


# --- P1: round-trip invariant -----------------------------------------

def test_round_trip_preserves_length_and_dtype(encoder, periodic_bytes):
    data = periodic_bytes(128)
    compressed = encoder.compress(data, top_k=10)
    decompressed = encoder.decompress(compressed)

    assert len(decompressed) == len(data)
    assert decompressed.dtype == np.uint8


def test_round_trip_reconstructs_smooth_signal_within_tolerance(encoder, periodic_bytes):
    data = periodic_bytes(128)
    compressed = encoder.compress(data, top_k=10)
    decompressed = encoder.decompress(compressed)

    original = np.frombuffer(data, dtype=np.uint8).astype(float)
    reconstructed = decompressed.astype(float)
    mean_abs_error = np.mean(np.abs(original - reconstructed))

    assert mean_abs_error < 10.0


def test_compress_is_deterministic(encoder, periodic_bytes):
    data = periodic_bytes(128)
    assert encoder.compress(data, top_k=10) == encoder.compress(data, top_k=10)


def test_imprint_is_deterministic(encoder, random_bytes):
    data = random_bytes(64)
    np.testing.assert_array_equal(encoder.imprint(data), encoder.imprint(data))


# --- P3: imprint stability ----------------------------------------------

def test_imprint_boosts_lotus_frequencies(encoder, random_bytes):
    data = random_bytes(64)
    imprinted = encoder.imprint(data)

    spectrum = np.fft.fft(encoder._to_signal(imprinted.tobytes()))
    for k in encoder.lotus_freqs:
        assert abs(spectrum[k]) > 5.0


def test_imprint_output_is_valid_byte_range(encoder, random_bytes):
    data = random_bytes(64)
    imprinted = encoder.imprint(data)

    assert imprinted.dtype == np.uint8
    assert imprinted.min() >= 0
    assert imprinted.max() <= 255
    assert len(imprinted) == len(data)


# --- _to_bytes edge cases -------------------------------------------------

def test_to_bytes_constant_signal_uses_offset_branch(encoder):
    constant_signal = np.zeros(16)
    result = encoder._to_bytes(constant_signal)

    assert np.all(result == 128)


def test_to_bytes_clips_out_of_range_values(encoder):
    signal = np.array([-10.0, 0.0, 10.0])
    result = encoder._to_bytes(signal)

    assert result.min() >= 0
    assert result.max() <= 255


# --- P2: malformed input resilience --------------------------------------
# These pin down the *current* (unvalidated) failure modes of decompress().
# They exist so any future input-validation hardening has a regression
# baseline to compare against, and so the failure surface is documented
# instead of implicit.

def test_decompress_truncated_header_raises(encoder):
    with pytest.raises(struct.error):
        encoder.decompress(b"\x00\x01\x02")


def test_decompress_truncated_payload_raises(encoder):
    header = struct.pack("<QH", 128, 5)
    with pytest.raises(struct.error):
        encoder.decompress(header)


def test_decompress_out_of_bounds_index_raises(encoder):
    header = struct.pack("<QH", 4, 1)
    gene = struct.pack(encoder.gene_format, 999, 1.0, 2.0)
    with pytest.raises(IndexError):
        encoder.decompress(header + gene)


def test_decompress_empty_input_raises(encoder):
    with pytest.raises(struct.error):
        encoder.decompress(b"")
