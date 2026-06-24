# CLAUDE.md

Guidance for AI assistants (Claude Code and others) working in this repository.

## What this is

**KSDZ v4.0 OMEGA** is a small, self-contained Python library implementing a
"Spectral Imprinting" compression engine. The core idea: treat arbitrary byte
data as a 1-D signal, take its FFT, and keep only a handful of dominant
frequency components ("genes"). For data dominated by a few periodic
components (e.g. smooth telemetry / trajectory streams) this is extremely
compact; for general high-entropy data it is **lossy** and reconstruction is
approximate. Marketing materials advertise "99.99% compression" — treat that as
a single observed measurement on a favorable workload, not a general guarantee
(see "Benchmark provenance" below).

This is a research / demonstration codebase, not a production-hardened library.
Keep that framing when reasoning about it.

## Layout

| File | Purpose |
| :--- | :--- |
| `ksdz_core.py` | Core engine: `KSDZ_Quantum_Encoder` (imprint / compress / decompress). |
| `omega_16d_quantum.py` | Demo app: `Omega16D_Quantum_System` evolves a 16-D oscillator system and archives the trajectory through the encoder. Imports `ksdz_core`. |
| `README.md` | Overview, license warning, benchmark table. |
| `COMMERCIAL_OFFER.md` | Dual-licensing / commercial pricing pitch. |
| `LICENSE` | AGPLv3 (currently a placeholder pointing at the full AGPLv3 text). |
| `benchmarks/` | Reproducible evaluation suite (see "Benchmark suite" below). |

There is no package structure, no `setup.py`/`pyproject.toml`, and no CI. The two
core modules are run directly. The `benchmarks/` suite is an executable
evaluation harness (not a unit-test suite) with a single entrypoint.

## Benchmark suite

`benchmarks/` evaluates the codec against standard lossless compressors. It does
not assert performance numbers — it measures them and writes `results.json`.

```bash
pip install -r benchmarks/requirements.txt
python benchmarks/benchmark_runner.py --all --output results.json
```

The hard rule it enforces: **two metric classes, never mixed.**

- `lossless` (`zlib`, `zstd` l3/l9, `lz4`) — metric is compression ratio; exact
  round-trip is asserted. Lives in `baselines.py`.
- `lossy_transform` (KSDZ) — metrics are RMSE/NRMSE, L2/L∞, and spectral
  distortion (raw + normalized); its size ratio is recorded only as a
  `pseudo_compression_ratio`, flagged non-comparable. Lives in `ksdz_adapter.py`
  behind a distinct `LossyTransformCodec` (`encode`/`decode`, not
  `compress`/`decompress`).

The separation is enforced both in code (`assert_no_class_mixing`) and in
`results_schema.json` (conditional rules). Prefer the **normalized** metrics
(`nrmse`, `spectral_distortion_normalized`) for cross-dataset comparison; raw L2
and spectral distortion scale with N. See `benchmarks/README.md` for the full
protocol (uint8 canonicalization, determinism, provenance).

## Architecture

`KSDZ_Quantum_Encoder` (in `ksdz_core.py`):

- `_to_signal(data_bytes)` — maps `uint8` bytes to a float signal in `[-1, 1]`.
- `_to_bytes(signal)` — min/max normalizes a float signal back to `uint8`.
- `imprint(data_bytes, strength_factor=2.0)` — forces fixed "lotus" frequencies
  (`self.lotus_freqs = [1, 3, 8]`) to a high magnitude in the spectrum, then
  inverse-FFTs. A watermarking / shaping step, not compression.
- `compress(data_bytes, top_k=10)` — FFTs the signal, zeroes the DC term, picks
  the `top_k` highest-magnitude frequency bins, and serializes each as a "gene".
- `decompress(ksdz_bytes)` — rebuilds a sparse spectrum from the genes
  (restoring conjugate-symmetric negative-frequency partners) and inverse-FFTs.

Binary format produced by `compress`:
- Header: `struct.pack('<QH', original_size, gene_count)` — little-endian
  `uint64` original byte length + `uint16` gene count.
- Each gene: `self.gene_format = '<Iff'` — little-endian `uint32` frequency
  index + two `float32`s (real, imag of that spectral bin).

When changing the serialization, the header format string `'<QH'`, the
`gene_format` `'<Iff'`, and `gene_size` must stay consistent between
`compress` and `decompress`, or archives become unreadable.

## Conventions

- Python 3, NumPy-based numerical code. No type hints; terse, comment-light
  style with short helpers prefixed `_`.
- Every source file carries the copyright/AGPLv3 header comment. Preserve it on
  any file you create or substantially edit.
- Code refers to the author as "Senior AI Architect"; the project brand is
  "OMEGA" / "KSDZ". Match existing naming when extending.
- The FFT round-trip is lossy. Do not describe `compress`/`decompress` as
  lossless, and do not "fix" the demo's 99.99% ratio claim into a general
  statement — it is workload-specific.

## Running

The only external dependency is **NumPy** (not vendored, and not installed in a
fresh environment — install it first):

```bash
pip install numpy
python3 omega_16d_quantum.py   # evolves the system, writes trajectory_2025.ksdz
```

`omega_16d_quantum.py`'s `__main__` runs 100,000 evolution steps and writes a
`.ksdz` archive to the working directory. There is no test suite; verify
changes by round-tripping data through `compress`/`decompress` and checking the
reconstruction is sane for periodic inputs.

## Benchmark provenance

Quote benchmarks as reproducible measurements, not slogans. State the workload
and environment with every number.

- **Legacy marketing claims removed.** The old README "99.99% ratio" / "~9.2
  MB/s (Samsung S24)" numbers had no dataset, sample size, or date and were not
  reproducible. They have been replaced by a claims table marked `unverified`
  and by the `benchmarks/` suite. Do not reintroduce a headline number that is
  not present in a generated `results.json`.
- **Measured truth (from the suite):** on the uint8 canonical signals, the
  lossless baselines compress to a degree that tracks data entropy (gaussian
  noise ~0.08–0.12, sinusoidal ~0.94 at N=65536). KSDZ reaches a higher *size*
  ratio but only because `pseudo_compression_ratio` is fixed by `top_k`
  (≈ header + 12·k bytes) — it rises with N mechanically — while its
  reconstruction is lossy (NRMSE ~0.03 on near-periodic data, ~0.16 on gaussian
  noise). Read ratio and fidelity together, never ratio alone.
- **Demo workload** (`omega_16d_quantum.py` `__main__`): 16-D oscillator system,
  `evolve(steps=100000)`, archived as `float32` via `imprint` +
  `compress(top_k=100)`. Smooth, near-periodic synthetic data — the best case
  for FFT-keep-top-k. It says nothing about general inputs.

When you (re)run a benchmark, record: `dataset`, `N` (sample size/steps),
`top_k`, `hardware`, `Python/NumPy versions`, `date`, and whether the data was
periodic — so `metric -> reproducible experiment`, not `metric -> headline`.

### Environment (verified run)

These were captured from an actual suite execution (the trigger that moves this
block from "unverified" to measured). Every `results.json` also embeds its own
`environment` block, so each run is self-describing.

```
Python:        3.11.15
NumPy:         2.4.6
zstandard:     0.25.0
lz4:           4.4.5
jsonschema:    4.26.0   (optional; validates results.json)
SciPy:         (not a dependency)
Platform:      Linux-6.18.5-x86_64 (glibc 2.39)
Last verified: 2026-06-24  (benchmarks/benchmark_runner.py --all)
```

Re-derive, don't trust this table: run the suite and read the `environment`
block of the produced `results.json`.

## Verification status

- **Verified (executed):** the `compress`/`decompress` lossy round-trip on the
  four benchmark datasets; the suite environment (Python 3.11.15 / NumPy 2.4.6 /
  zstandard 0.25.0 / lz4 4.4.5); `results.json` validates against
  `results_schema.json`; dataset determinism (same `(name,N,seed)` → same
  SHA256); KSDZ fidelity is now *measured* (NRMSE/spectral distortion per
  dataset), not assumed.
- **Verified (from source):** module layout; the `imprint`/`compress`/
  `decompress` API; the binary format (`<QH` header, `<Iff` genes); NumPy as the
  sole import of the core modules; absence of CI/packaging.
- **Unverified:** the original README "99.99%"/"9.2 MB/s" claims (removed in
  Phase 1, never reproduced); behavior on inputs shorter than `max(lotus_freqs)`
  or with degenerate spectra; the `imprint` path (the suite exercises
  `compress`/`decompress`, not `imprint`).
- **Open questions:** intended/safe ranges for `top_k` and `strength_factor`;
  whether `.ksdz` archives are meant to be portable across machines/endianness
  (the format hardcodes little-endian).

Update these lists when you verify or invalidate an item — do not let
"Unverified" claims drift into "Verified" without actually checking them.

## Git workflow

- Active development branch for assistant work: `claude/claude-md-docs-sron7y`.
- Default branch: `main`.
- Commit with clear, descriptive messages; push with `git push -u origin <branch>`.
- Do **not** open a pull request unless the user explicitly asks.

## Licensing (important context, not legal advice)

This repo is AGPLv3 with a dual-licensing commercial offer (see
`COMMERCIAL_OFFER.md`). The `LICENSE` file is currently a placeholder — if asked
to "add the license," that means dropping in the full AGPLv3 text. Keep the
per-file AGPLv3 headers intact.
