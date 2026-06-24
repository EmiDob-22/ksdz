# Research Retrospective

*Epistemic history of the KSDZ investigation, in chronological order. Every entry
is traceable to a commit, a source file, or a measured value in this repository.
No code or benchmarks were modified to produce this document.*

Commit spine (this branch):

| commit | date | event |
|:--|:--|:--|
| `b9f5d0c`, `83276bd` | 2025-12-05 | original release + demo (pre-investigation) |
| `9d94b1e`–`c64949d` | 2026-06-23 | documentation / verification scaffolding |
| `279caf2` | 2026-06-24 | benchmark suite; claims normalized |
| `d18267f` | 2026-06-24 | operator model |
| `aee387f` | 2026-06-24 | rate–distortion study |
| `81f835e` | 2026-06-24 | compressibility-exponent study |
| `eafb55d` | 2026-06-24 | literature mapping |

---

## 1. Initial claims

As released (`README.md`, `COMMERCIAL_OFFER.md` at `b9f5d0c`/`83276bd`):

- "achieving **99.99% compression** where GZIP fails."
- Benchmark table: Ratio "99.99% (Success)" vs GZIP "−0.03% (Fail)"; Speed
  "~9.2 MB/s" on "Samsung S24 / Snapdragon 8 Gen 3".
- "transforms matter into frequency spectra"; "revolutionary."
- ROI: "Saves 90% on Cloud Storage (AWS S3) and Satellite Transfer costs."

These carried no dataset, sample size, date, or reconstruction-error figure, and
no reproducible procedure existed in the repository.

---

## 2. What was verified

**From source** (`ksdz_core.py`): module layout; the `imprint`/`compress`/
`decompress` API; the binary format (header `'<QH'` = 10 bytes, gene `'<Iff'` =
12 bytes); NumPy as the sole import; absence of tests/CI/packaging.

**By execution** (suite, `279caf2`; data in `results.json`,
`spectral_compressibility_data.json`):

- The `compress`/`decompress` round-trip runs on four deterministic datasets.
- Environment captured and reproducible: Python 3.11.15, NumPy 2.4.6,
  zstandard 0.25.0, lz4 4.4.5, jsonschema 4.26.0.
- `results.json` validates against `results_schema.json` (40 rows).
- Determinism: identical `(name, N, seed)` ⟹ identical SHA256 input hash.
- The pseudo-compression-ratio law $R(k)=1-(10+12k)/N$ matches measurement to
  all printed digits (e.g. $N{=}65536,k{=}100$: predicted 0.98154, measured
  0.9815).
- Random walk is a $1/f^2$ process; the predicted approximation exponent
  $\alpha=(\beta-1)/2=0.5$ matches the independently measured
  $\alpha_{\mathrm{Fourier}}=0.507$–$0.533$ across $N$.

---

## 3. What was falsified

- **"99.99% as a general compression result."** Lossless baselines on the uint8
  canonical signals compress only to the degree set by entropy (gaussian noise
  ratio ≈0.08–0.12; sinusoidal ≈0.94 at $N{=}65536$). KSDZ's high size ratio is
  fixed by `top_k` (payload $=10+12k$ bytes, content-independent) and rises with
  $N$ mechanically, while its reconstruction is lossy (NRMSE ≈0.03 on
  near-periodic data, ≈0.16 on gaussian noise). The headline conflated a
  workload-specific size ratio with general, lossless performance.
- **"~9.2 MB/s."** No dataset/date/procedure; never reproduced. Removed in
  Phase 1 (`279caf2`).
- **KSDZ as a general-purpose compressor.** It is a lossy transform; lossless
  coders reconstruct bit-for-bit on the same inputs, KSDZ does not.
- **α as a universal invariant.** Falsified for broadband signals:
  $\alpha_{\mathrm{Fourier}}\to0$ as $N{:}4096{\to}262144$ for gaussian
  (0.029→0.001) and lorenz (0.616→0.022).
- **The A/B/C α-threshold taxonomy.** Falsified: class B empty; assignments not
  invariant under estimator or $N$ (`81f835e`).

---

## 4. Operator reconstruction (`d18267f`)

The implemented map is

$$T_k = Q \circ \nu \circ \mathrm{Re} \circ F^{-1} \circ \Pi_{S(a(b))} \circ F \circ a,$$

**not** the clean $F^{-1}P_kF$. Established facts:

- The idealised core $\mathcal T_k=\mathcal F^{-1}\Pi_{S(x)}\mathcal F$ is best
  $k$-term Fourier truncation; by Parseval its error equals the discarded
  spectral energy, and top-$k$ selection is energy-optimal.
- Full $T_k$ is **nonlinear** (support depends on $x$; and $\nu$ is min/max
  normalisation, giving $T_k(cb)=T_k(b)$, degree-0 homogeneity).
- The linear core (fixed support) is an orthogonal projection ($P^2{=}P{=}P^\ast$);
  the full map is only quasi-idempotent (broken by $Q$); both are bounded
  (range $\subseteq[0,255]^N$).
- **Conjugate-doubling:** the decoder re-imposes the partner $\overline{\hat x_m}$,
  so each physical frequency costs two genes (≈2× storage redundancy; ≈$k/2$
  distinct frequencies for $k$ genes).

---

## 5. Benchmark results (`279caf2`)

Two metric classes, enforced in code and schema and never mixed:

- **lossless** (`zlib`, `zstd` l3/l9, `lz4`): metric = compression ratio; exact
  round-trip asserted. Measured ratios track entropy (gaussian ≈0.08–0.12,
  lorenz ≈0.38–0.61, random_walk ≈0.27–0.53, sinusoidal ≈0.82–0.94 across the
  codecs at $N{=}65536$).
- **lossy_transform** (KSDZ): fidelity metrics (RMSE/NRMSE, L2/L∞, spectral
  distortion raw+normalized) plus a flagged `pseudo_compression_ratio`. At
  $N{=}65536,k{=}100$: NRMSE 0.032 (random_walk), 0.034 (sinusoidal), 0.158
  (gaussian), 0.193 (lorenz); largest L∞ 227 (lorenz).

---

## 6. Rate–distortion results (`aee387f`)

Sweep $k\in\{1,\dots,256\}$, $N{=}65536$:

- $R(k)=1-(10+12k)/N$ exactly; content-independent.
- KSDZ NRMSE exponents $D\sim k^{-\alpha}$: random_walk 0.529, sinusoidal 0.466,
  gaussian 0.173, lorenz 0.163.
- $D(1)=D(2)$ for every dataset (conjugate-doubling, measured).
- Crossover $k^\star$ (per-doubling NRMSE gain $<10\%$ thereafter):
  gaussian 16 (high floor ≈0.16), sinusoidal 64 (low floor ≈0.030);
  random_walk and lorenz: none within the sweep.
- Non-monotonicity observed (e.g. sinusoidal $64{\to}128$ NRMSE rises).

---

## 7. Compressibility-exponent study (`81f835e`)

A second, KSDZ-free estimator was introduced — oracle best-$k$-term relative
error from sorted FFT energy — to test whether α is an instrument artifact.

- **Stability over $N$:** $\alpha_{\mathrm{Fourier}}$ is $N$-stable only for
  random_walk (≈0.52) and sinusoidal (≈0.627); gaussian and lorenz → 0.
- **Noise $\sigma\in\{0,\dots,1\}$:** $\alpha(\sigma)$ continuous and monotone;
  smooth crossover near $\sigma\approx0.15$; **no discontinuous phase transition.**
- **Entropy:** $\alpha$ anti-correlates with spectral entropy ($r=-0.951$ across
  datasets; $r=-0.883$ pooled, $n=28$); essentially uncorrelated with Shannon
  amplitude entropy ($r=+0.654$).
- **Estimator comparison ($N{=}65536$):** random_walk agrees within CI
  (0.565 KSDZ vs 0.507 Fourier); KSDZ **inflates** α for broadband (gaussian
  0.119 vs 0.003; lorenz 0.160 vs 0.060, non-overlapping) and **deflates** it for
  sinusoidal (0.416 vs 0.627).
- **Conclusion reached:** α is a useful empirical descriptor of spectral
  concentration; not a pure codec artifact (survives for concentrated spectra),
  not a universal invariant (N-dependent for broadband).

---

## 8. Literature placement (`eafb55d`)

The codec-free α is a known quantity:

- the **best $k$-term (nonlinear) approximation rate** in an ONB [DeVore 1998];
- the **Stechkin / compressibility exponent** $\alpha=1/p-1/2$ for weak-$\ell^p$
  coefficients [Cohen–Dahmen–DeVore 2009; Temlyakov 2011];
- $\alpha=(\beta-1)/2$ derived from a power spectrum $S(f)\sim f^{-\beta}$,
  consistent with random_walk ($\beta{=}2\Rightarrow\alpha{=}0.5$, measured
  ≈0.51–0.53); $\alpha\to0$ for $\beta\le1$ (consistent with gaussian/lorenz);
- equal to a Sobolev index for monotone spectra; analogous (not identical) to
  Besov index and to nonlinear $n$-widths; correlated-but-distinct from spectral
  entropy.
- $D(k)\sim k^{-\alpha}$ is a known theorem (Stechkin's lemma), not a new law.

Classification: **A — known quantity under another name**; KSDZ is a biased
estimator of it (B); novelty (C) rejected.

---

## 9. Hypotheses that survived

| hypothesis | surviving evidence |
|:--|:--|
| KSDZ is spectral truncation / a transform coder | operator reconstruction from source + measured behavior (§4, §8) |
| Size ratio rises with $N$ mechanically, not by content | exact law $R(k)=1-(10+12k)/N$ (§2, §6) |
| KSDZ is most faithful on near-periodic data | lowest NRMSE on random_walk/sinusoidal (§5, §6) |
| α is a real, codec-independent descriptor for concentrated spectra | Fourier estimator + theory match (random_walk α≈0.5) (§7, §8) |
| $\alpha=(\beta-1)/2$ | derivation + measured random_walk (§8) |
| Lossy and lossless ratios are not comparable | class-separation enforced; assertion gate (§5) |

---

## 10. Hypotheses that failed

| hypothesis | falsifying evidence |
|:--|:--|
| "99.99% general compression" | ratio fixed by `top_k`; reconstruction lossy (§3) |
| "~9.2 MB/s" throughput | no reproducible procedure; unverified (§3) |
| KSDZ competes with lossless coders on one ratio leaderboard | lossless exact vs KSDZ lossy; classes separated (§3, §5) |
| α is a universal spectral invariant | $\alpha_{\mathrm{Fourier}}\to0$ for broadband as $N\uparrow$ (§7) |
| A/B/C α-threshold taxonomy | class B empty; estimator/$N$-dependent (§3, §7) |
| α is a novel descriptor | it is classical nonlinear-approximation theory (§8) |
| KSDZ is an unbiased estimator of α | inflates broadband, deflates sinusoidal (§7) |

---

## 11. Lessons learned

Stated only as conclusions the evidence forced:

1. For a lossy transform, a size ratio without a fidelity figure is
   uninterpretable; the two-class separation was necessary to avoid comparing
   incomparable metrics (§5).
2. Raw error/energy quantities scale with $N$; normalized forms (NRMSE,
   normalized spectral distortion, the explicit $R(k)$ law) were required for any
   cross-$N$ or cross-dataset statement (§6).
3. An instrument can bias the quantity it measures. The KSDZ-free Fourier
   estimator was decisive: it separated a real signal property (α for
   concentrated spectra) from codec-induced bias (§7).
4. Claims need embedded provenance. The environment block, SHA256 input hashes,
   and schema validation made each number self-describing and reproducible
   (§2).
5. Separating Observed / Hypothesis / Verified categories (the documentation
   discipline from `9d94b1e`–`c64949d`) prevented unverified figures from being
   restated as facts.

---

## 12. Future directions

Open questions already posed and made well-defined by this work
(`operator_model.md` §8; `rate_distortion.md`; `spectral_compressibility.md`):

1. The measured **Lorenz > Gaussian** NRMSE/spectral inversion (lorenz worse than
   white noise despite structure) — prove or refute via a Gibbs-vs-noise model
   and locate the $(k,J,\mathrm{SNR})$ surface where the ordering flips.
2. A **renormalisation-corrected error law** quantifying how much $\nu$'s
   affine refit reduces measured NRMSE relative to the Parseval ideal
   $\sqrt{E_{\text{discarded}}}$ as a function of α.
3. A quantitative **idempotency-defect** bound $\|T_k^2-T_k\|$ in terms of the
   `uint8`/`float32` quantisation step.
4. Whether removing the **conjugate-doubling** redundancy changes only constants
   (predicted) and not the $\Theta(1/N)$ rate law.
5. **Rate–distortion curves per signal class** at a fixed distortion target,
   solving $k^\star(\varepsilon)$ from the measured α and the $R(k)$ law.
6. The **continuous $N\to\infty$ limit**: whether the discrete $\mathcal T_k$
   converges (operator/Γ-sense, on band-limited subspaces) to the continuous
   best-$k$-term Fourier projector.

These are open mathematical questions, not predictions of outcome.

---

*All figures above are reproducible from `benchmarks/` and the data archived in
`docs/spectral_compressibility_data.json`. This retrospective adds no claim not
already established in the cited phase documents and commits.*
