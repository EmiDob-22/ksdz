# CLAUDE.md

Guidance for AI assistants (Claude Code and others) working in this repository.

## What this is

**KSDZ v4.0 OMEGA** is a small, self-contained Python library implementing a
"Spectral Imprinting" compression engine. The core idea: treat arbitrary byte
data as a 1-D signal, take its FFT, and keep only a handful of dominant
frequency components ("genes"). For data dominated by a few periodic
components (e.g. smooth telemetry / trajectory streams) this is extremely
compact; for general high-entropy data it is **lossy** and reconstruction is
approximate. Marketing materials advertise "99.99% compression" — this is the
ratio achieved on the favorable synthetic trajectory workload in
`omega_16d_quantum.py`, not a general-purpose guarantee.

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

There is no package structure, no `setup.py`/`pyproject.toml`, no tests, and no
CI. The two modules are run directly.

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
