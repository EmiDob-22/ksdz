# KSDZ v4.0 OMEGA - Quantum Spectral Compression Engine

![License: AGPLv3](https://img.shields.io/badge/License-AGPLv3-red.svg)
![Platform](https://img.shields.io/badge/Platform-ARM64%20%7C%20x86-blue)

**KSDZ v4.0** is a research "Spectral Imprinting" compression engine: it treats
a byte stream as a 1-D signal, takes its FFT, and keeps only the dominant
frequency components ("genes"). The transform is **lossy** — reconstruction is
approximate and fidelity depends heavily on the input.

> **All performance characteristics are unverified until benchmark suite execution.**
> Run `python benchmarks/benchmark_runner.py --all --output results.json` to
> produce measured compression ratio, reconstruction error (L2 / L∞), spectral
> distortion, and runtime against gzip/zstd/lz4 baselines on deterministic
> datasets. See `benchmarks/` for the methodology and `results_schema.json` for
> the output format.

## ⚠️ LICENSE WARNING: AGPLv3

This software is strictly licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.
- **Open Source:** Free to use if you open-source your code.
- **Commercial:** For proprietary/SaaS use, you **MUST** purchase a Commercial License.

[See COMMERCIAL_OFFER.md for pricing.](COMMERCIAL_OFFER.md)

## 🚀 Benchmark

**All performance characteristics are unverified until benchmark suite execution.**
The numbers below are placeholders until the suite writes `results.json`; they
are not slogans. Reproduce them yourself with the runner — do not cite a value
that is not present in a generated results file.

### Claims table

| Metric | Value | Status |
|--------|-------|--------|
| compression_ratio | undefined | unverified |
| throughput | undefined | unverified |
| reconstruction_error | undefined | unverified |
| spectral_distortion | undefined | unverified |

Baselines compared in the suite: `zlib` (level 6), `zstandard` (levels 3 and 9),
`lz4`. Datasets: sinusoidal mixtures, gaussian noise, Lorenz trajectory, random
walk — each deterministic (fixed seed + SHA256 input hash).

## 📞 Contact
For commercial licensing: **Senior AI Architect**
