# KSDZ benchmark suite

Reproducible, claims-free evaluation of the KSDZ spectral transform against
standard lossless compressors. Nothing here asserts a performance number; it
*measures* one and writes it to `results.json`.

## Run

```bash
pip install -r benchmarks/requirements.txt
python benchmarks/benchmark_runner.py --all --output results.json
```

Useful flags: `--datasets sinusoidal_mixture gaussian_noise`, `--sizes 4096 65536`,
`--seed 1234`, `--top-k 100`.

## Two metric classes (kept strictly separate)

Comparing a lossy spectral projection against lossless coders on a single
"compression ratio" leaderboard is semantically invalid. The suite enforces the
split in code (`assert_no_class_mixing`) and in `results_schema.json`
(`class` is required; conditional rules forbid cross-leakage).

| Class | Codecs | Metrics | Ratio field |
| :--- | :--- | :--- | :--- |
| `lossless` | `zlib_l6`, `zstd_l3`, `zstd_l9`, `lz4` | compression ratio; **exact round-trip asserted** | `compression_ratio` (comparable) |
| `lossy_transform` | `ksdz_topk{k}` | RMSE, NRMSE, L2, L∞, spectral distortion (raw + normalized) | `pseudo_compression_ratio` (flagged non-comparable) |

Lossless codecs that fail to reconstruct bit-for-bit are recorded as
`status: "error"`, never as a fast/small "win".

## Modules

| File | Role |
| :--- | :--- |
| `datasets.py` | Deterministic generators (sinusoidal mixture, gaussian noise, Lorenz RK4 trajectory, random walk). Float signal → uint8 canonical signal. Fixed seed + SHA256 input hash. |
| `baselines.py` | Lossless codecs only (`class = "lossless"`). |
| `ksdz_adapter.py` | `LossyTransformCodec` wrapping KSDZ (`class = "lossy_transform"`, `encode`/`decode` interface — deliberately not `compress`/`decompress`). |
| `metrics.py` | `compression_ratio`, `l2/linf`, `rmse`, `nrmse`, `spectral_distortion` (raw + normalized). |
| `benchmark_runner.py` | Single entrypoint; captures environment; writes structured JSON; prints two separate tables. |
| `results_schema.json` | JSON Schema (draft 2020-12) for the output; encodes the class separation. |

## Evaluation protocol (documented assumptions)

- **Canonical signal = uint8.** Each dataset's float signal is min/max quantized
  to uint8 [0,255]; that byte stream is what every codec sees. This matches
  KSDZ's native domain and keeps all error/spectral metrics finite for both
  classes. Trade-off: quantization discards float precision before compression —
  a deliberate, documented choice, not a hidden one.
- **Normalization.** Raw L2 and spectral distortion scale with N; `nrmse`
  (RMSE / dynamic range) and `spectral_distortion_normalized` (÷ source spectrum
  energy) are the cross-dataset-comparable forms. Prefer them when comparing.
- **Determinism.** A `(dataset, N, seed)` triple pins one byte-for-byte input;
  the SHA256 is recorded per result.
- **Provenance.** Every `results.json` embeds Python/NumPy/zstandard/lz4
  versions, platform, git commit, and a UTC timestamp.

## Known limitations

- KSDZ's `pseudo_compression_ratio` is dominated by `top_k` (≈ header + 12·k
  bytes), so it rises with N mechanically rather than reflecting content
  compressibility. Read it alongside the fidelity columns, never alone.
- `results.json` is a runtime artifact (gitignored). Commit a copy deliberately
  if you want to archive a specific run.
