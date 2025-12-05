# Copyright (C) 2025 Senior AI Architect.
# Licensed under AGPLv3. For commercial licensing, contact the author.

import numpy as np
import struct

class KSDZ_Quantum_Encoder:
    def __init__(self):
        self.lotus_freqs = [1, 3, 8]
        self.gene_format = '<Iff' 
        self.gene_size = struct.calcsize(self.gene_format)
        
    def _to_signal(self, data_bytes):
        return (np.frombuffer(data_bytes, dtype=np.uint8) / 127.5) - 1.0

    def _to_bytes(self, signal):
        sig_min = np.min(signal)
        sig_max = np.max(signal)
        if sig_max > sig_min:
            normalized = 255 * (signal - sig_min) / (sig_max - sig_min)
        else:
            normalized = signal + 128
        return np.clip(normalized, 0, 255).astype(np.uint8)

    def imprint(self, data_bytes, strength_factor=2.0):
        sig = self._to_signal(data_bytes)
        spectrum = np.fft.fft(sig)
        N = len(spectrum)
        strength = N * strength_factor
        for k in self.lotus_freqs:
            if k < N:
                spectrum[k] = strength
                spectrum[-k] = strength
        return self._to_bytes(np.fft.ifft(spectrum).real)

    def compress(self, data_bytes, top_k=10):
        original_size = len(data_bytes)
        sig = self._to_signal(data_bytes)
        spectrum = np.fft.fft(sig)
        magnitudes = np.abs(spectrum)
        magnitudes[0] = 0 
        limit = min(top_k, len(spectrum) // 2)
        top_indices = np.argsort(magnitudes)[-limit:][::-1]
        
        header = struct.pack('<QH', original_size, len(top_indices))
        payload = bytearray(header)
        for idx in top_indices:
            val = spectrum[idx]
            gene = struct.pack(self.gene_format, int(idx), float(val.real), float(val.imag))
            payload.extend(gene)
        return bytes(payload)

    def decompress(self, ksdz_bytes):
        header_size = struct.calcsize('<QH')
        original_size, gene_count = struct.unpack('<QH', ksdz_bytes[:header_size])
        spectrum = np.zeros(original_size, dtype=np.complex64)
        offset = header_size
        for _ in range(gene_count):
            chunk = ksdz_bytes[offset : offset + self.gene_size]
            idx, real, imag = struct.unpack(self.gene_format, chunk)
            spectrum[idx] = complex(real, imag)
            if idx > 0: spectrum[-idx] = complex(real, -imag)
            offset += self.gene_size
        return self._to_bytes(np.fft.ifft(spectrum).real)
