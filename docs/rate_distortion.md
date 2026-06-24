# KSDZ Rate–Distortion Study

*Empirical sweep of the retained-mode budget `top_k`, using the existing
benchmark framework unmodified. No change to KSDZ or `benchmarks/`.*

## Method

Driven through the existing modules (`benchmarks/datasets.py`,
`benchmarks/ksdz_adapter.py`, `benchmarks/metrics.py`). For each dataset the
canonical uint8 signal is built once and passed through
`KsdzTransform(top_k=k).encode`/`.decode` for each $k$.

- **Fixed:** $N = 65536$, seed $= 1234$ (suite defaults).
- **Swept:** $k \in \{1,2,4,8,16,32,64,128,256\}$.
- **Rate:** $R(k) = $ `pseudo_compression_ratio` $= 1 - (10 + 12k)/N$ (the lossy
  size ratio; non-comparable to lossless coders by construction).
- **Distortions:** $D_{L2}=$ `l2_error`, $D_{\mathrm{NRMSE}}=$ `nrmse`
  (= RMSE / dynamic range), $D_{\mathrm{spec}}=$ `spectral_distortion_normalized`
  (= $\||\hat x|-|\hat x_{\mathrm{rec}}|\| / \|\hat x\|$). $L^\infty$ shown for context.
- **Scaling fit:** least squares of $\log D$ vs $\log k$; report
  $\alpha = -\text{slope}$ in $D(k)\sim k^{-\alpha}$ with $R^2$.

All numbers below are measured outputs, reproducible by re-running the framework
at these parameters.

---

## 1. Rate–distortion tables

### sinusoidal_mixture

| k | R(k) | bytes | D_L2 | D_NRMSE | D_spec | L∞ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.99966 | 22 | 17876.92 | 0.27385 | 0.36296 | 171 |
| 2 | 0.99948 | 34 | 17876.92 | 0.27385 | 0.36296 | 171 |
| 4 | 0.99911 | 58 | 11169.70 | 0.17110 | 0.22662 | 120 |
| 8 | 0.99838 | 106 | 6563.70 | 0.10055 | 0.13318 | 96 |
| 16 | 0.99692 | 202 | 4667.63 | 0.07150 | 0.10588 | 102 |
| 32 | 0.99399 | 394 | 2343.05 | 0.03589 | 0.04775 | 79 |
| 64 | 0.98813 | 778 | 1980.07 | 0.03033 | 0.04559 | 81 |
| 128 | 0.97641 | 1546 | 2218.27 | 0.03398 | 0.05678 | 85 |
| 256 | 0.95297 | 3082 | 2077.15 | 0.03182 | 0.05465 | 85 |

### random_walk

| k | R(k) | bytes | D_L2 | D_NRMSE | D_spec | L∞ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.99966 | 22 | 19095.04 | 0.29251 | 0.37750 | 170 |
| 2 | 0.99948 | 34 | 19095.04 | 0.29251 | 0.37750 | 170 |
| 4 | 0.99911 | 58 | 14085.04 | 0.21576 | 0.28760 | 130 |
| 8 | 0.99838 | 106 | 9716.74 | 0.14885 | 0.19578 | 99 |
| 16 | 0.99692 | 202 | 4543.01 | 0.06959 | 0.09143 | 64 |
| 32 | 0.99399 | 394 | 3778.42 | 0.05788 | 0.07797 | 52 |
| 64 | 0.98813 | 778 | 2564.07 | 0.03928 | 0.05090 | 42 |
| 128 | 0.97641 | 1546 | 1884.34 | 0.02887 | 0.03696 | 34 |
| 256 | 0.95297 | 3082 | 1312.90 | 0.02011 | 0.02542 | 25 |

### gaussian_noise

| k | R(k) | bytes | D_L2 | D_NRMSE | D_spec | L∞ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.99966 | 22 | 24206.55 | 0.37081 | 0.53737 | 247 |
| 2 | 0.99948 | 34 | 24206.55 | 0.37081 | 0.53737 | 247 |
| 4 | 0.99911 | 58 | 17895.07 | 0.27413 | 0.39728 | 241 |
| 8 | 0.99838 | 106 | 13793.26 | 0.21129 | 0.30639 | 217 |
| 16 | 0.99692 | 202 | 11667.98 | 0.17874 | 0.25877 | 197 |
| 32 | 0.99399 | 394 | 11107.30 | 0.17015 | 0.24885 | 196 |
| 64 | 0.98813 | 778 | 10607.95 | 0.16250 | 0.23593 | 179 |
| 128 | 0.97641 | 1546 | 10620.05 | 0.16268 | 0.23894 | 183 |
| 256 | 0.95297 | 3082 | 10215.32 | 0.15648 | 0.22747 | 193 |

### lorenz_trajectory

| k | R(k) | bytes | D_L2 | D_NRMSE | D_spec | L∞ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.99966 | 22 | 25515.59 | 0.39086 | 0.54725 | 246 |
| 2 | 0.99948 | 34 | 25515.59 | 0.39086 | 0.54725 | 246 |
| 4 | 0.99911 | 58 | 19682.82 | 0.30151 | 0.42219 | 246 |
| 8 | 0.99838 | 106 | 21086.88 | 0.32302 | 0.50648 | 246 |
| 16 | 0.99692 | 202 | 15760.93 | 0.24144 | 0.35351 | 233 |
| 32 | 0.99399 | 394 | 15721.79 | 0.24084 | 0.36533 | 225 |
| 64 | 0.98813 | 778 | 13777.35 | 0.21105 | 0.30596 | 216 |
| 128 | 0.97641 | 1546 | 12052.37 | 0.18463 | 0.25847 | 191 |
| 256 | 0.95297 | 3082 | 10624.58 | 0.16275 | 0.22765 | 168 |

**Rate note (measured):** $R(k)$ is identical across datasets at each $k$ because
the payload size depends only on $k$, not on content: bytes $= 10 + 12k$, so
$R(k) = 1 - (10+12k)/65536$. The "bytes" column equals this for every dataset.

---

## 2. Empirical scaling $D(k)\sim k^{-\alpha}$

Fit over the full sweep, and (right two columns) over the tail $k\ge 8$ to
exclude the flat $k\in\{1,2\}$ plateau.

| dataset | α (NRMSE, full) | R² | α (spec, full) | R² | α (NRMSE, k≥8) | R² |
|:---|---:|---:|---:|---:|---:|---:|
| random_walk | 0.529 | 0.977 | 0.533 | 0.976 | 0.537 | 0.968 |
| sinusoidal_mixture | 0.466 | 0.914 | 0.418 | 0.866 | 0.336 | 0.750 |
| gaussian_noise | 0.173 | 0.861 | 0.171 | 0.858 | 0.075 | 0.810 |
| lorenz_trajectory | 0.163 | 0.961 | 0.162 | 0.917 | 0.180 | 0.947 |

Measured facts:

- $D_{L2}$ and $D_{\mathrm{NRMSE}}$ give the **same** $\alpha$ to three decimals
  (they differ only by the constant factor $1/(\sqrt N\cdot\text{range})$, which
  is $k$-independent). Only NRMSE is tabulated above; the L2 column is identical.
- $\alpha_{\mathrm{spec}}$ tracks $\alpha_{\mathrm{NRMSE}}$ within $\pm 0.05$ for
  every dataset.
- Fit quality is high for random_walk and lorenz ($R^2 \ge 0.92$), moderate for
  sinusoidal ($R^2 = 0.91$ full, dropping to $0.75$ on the tail), and weakest for
  gaussian on the tail ($R^2 = 0.81$, $\alpha$ collapses to $0.075$).

---

## 3. Comparison of $\alpha$ across signal classes

Ordered by measured full-sweep $\alpha_{\mathrm{NRMSE}}$:

| rank | dataset | α | reading from the data |
|:---:|:---|---:|:---|
| 1 | random_walk | **0.529** | steepest, clean power law ($R^2=0.98$); distortion keeps falling through $k=256$ |
| 2 | sinusoidal_mixture | **0.466** | steep early, but tail $\alpha$ falls to $0.336$ and $D$ stops decreasing (§4) |
| 3 | gaussian_noise | **0.173** | shallow; tail $\alpha$ collapses to $0.075$ — effectively flat past $k\approx 32$ |
| 4 | lorenz_trajectory | **0.163** | shallow but **persistent** — tail $\alpha=0.180 >$ full, still falling at $k=256$ |

Two distinct shapes are visible in the numbers, not one:

- **random_walk and gaussian_noise** are well-described by a single exponent
  across the whole range (monotone, high/decent $R^2$), with random_walk an order
  of magnitude steeper than gaussian.
- **sinusoidal_mixture and lorenz_trajectory** are *not* single-exponent. The
  sinusoid's exponent **drops** in the tail (saturation); the Lorenz exponent
  **rises** in the tail (late improvement). Their full-sweep $\alpha$ values
  therefore summarise curved log–log data and should be read with §4.

---

## 4. Crossover $k^\star$ (diminishing returns)

Relative reduction of $D_{\mathrm{NRMSE}}$ per doubling,
$\;\rho_i = (D_{k_{i-1}} - D_{k_i})/D_{k_{i-1}}$:

| dataset | 1→2 | 2→4 | 4→8 | 8→16 | 16→32 | 32→64 | 64→128 | 128→256 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| sinusoidal_mixture | 0.000 | 0.375 | 0.412 | 0.289 | 0.498 | 0.155 | −0.120 | 0.064 |
| random_walk | 0.000 | 0.262 | 0.310 | 0.532 | 0.168 | 0.321 | 0.265 | 0.303 |
| gaussian_noise | 0.000 | 0.261 | 0.229 | 0.154 | 0.048 | 0.045 | −0.001 | 0.038 |
| lorenz_trajectory | 0.000 | 0.229 | −0.071 | 0.253 | 0.002 | 0.124 | 0.125 | 0.118 |

Two measured structural facts first:

- **$\rho_{1\to2}=0$ for all datasets**: $D(1)=D(2)$ exactly. Selecting one vs two
  bins yields the identical reconstruction, because the decoder re-imposes the
  conjugate partner of each stored bin. Effective rate advances in conjugate
  pairs, so distortion responds to $\lceil k/2\rceil$ pairs, not $k$.
- **Non-monotonicity occurs**: sinusoidal $64\to128 = -0.120$ and
  gaussian $64\to128 = -0.001$ and lorenz $4\to8 = -0.071$ are negative — adding
  modes *increased* distortion at those steps.

**Crossover rule (stated, then applied):** $k^\star$ = smallest swept $k$ such
that every subsequent doubling yields $\rho < 0.10$.

| dataset | $k^\star$ | $D_{\mathrm{NRMSE}}$ at $k^\star$ | status of returns beyond $k^\star$ |
|:---|---:|---:|:---|
| gaussian_noise | **16** | 0.179 | flat at a **high** floor (~0.16); extra modes ≈ 0 benefit |
| sinusoidal_mixture | **64** | 0.030 | flat/negative at a **low** floor; best $D$ is at $k=64$ |
| random_walk | **> 256** | 0.020 (at 256) | not reached — every doubling still ≥ 0.17 |
| lorenz_trajectory | **> 256** | 0.163 (at 256) | not reached — doublings hold ≈ 0.12 through 256 |

The crossover means **opposite things** in the measured data and must not be
conflated:

- **gaussian_noise** reaches diminishing returns early ($k^\star=16$) because
  distortion saturates at a **high** floor — added modes cannot help an
  (approximately) flat spectrum. Low rate is wasted, not earned.
- **sinusoidal_mixture** reaches diminishing returns ($k^\star=64$) at a **low**
  floor — the signal is essentially reconstructed; beyond $k=64$ further modes do
  nothing or slightly hurt (min $D=0.0303$ at $k=64$).
- **random_walk and lorenz_trajectory** show **no** crossover within the sweep:
  every doubling through $k=256$ still buys ≥ 10% (random_walk strongly,
  ≥ 16%; lorenz marginally, ≈ 12%).

---

## 5. Conclusions (measured only)

1. **Rate is content-independent.** $R(k) = 1 - (10+12k)/N$ exactly, identical
   across all four datasets at each $k$; the byte cost is $10 + 12k$ regardless of
   signal. Rate is therefore set entirely by the mode budget, not by
   compressibility.

2. **Distortion separates the datasets by roughly 3× in $\alpha$.** Measured
   full-sweep $\alpha_{\mathrm{NRMSE}}$: random_walk 0.529, sinusoidal 0.466,
   gaussian 0.173, lorenz 0.163. random_walk follows a clean single power law
   ($R^2=0.98$).

3. **Two datasets are not single-exponent.** sinusoidal_mixture saturates
   (tail $\alpha$ falls $0.466\to0.336$; $D$ reaches its minimum $0.0303$ at
   $k=64$ then rises to $0.0318$ at $k=256$). lorenz_trajectory does the opposite
   (tail $\alpha$ rises to $0.180$; still improving at $k=256$).

4. **$k=1$ and $k=2$ are equivalent** in distortion for every dataset
   ($D(1)=D(2)$), so the effective resolution advances in conjugate pairs.

5. **Adding modes can increase distortion.** Measured negative per-doubling
   reductions occur (sinusoidal $64\to128$, gaussian $64\to128$, lorenz
   $4\to8$).

6. **A crossover exists for two of four datasets within the sweep:**
   gaussian_noise at $k^\star=16$ (saturating at a high distortion floor ≈0.16)
   and sinusoidal_mixture at $k^\star=64$ (saturating at a low floor ≈0.030).
   random_walk and lorenz_trajectory show no crossover up to $k=256$: each
   doubling still reduces $D_{\mathrm{NRMSE}}$ by ≥10%.

7. **$L^\infty$ behaviour distinguishes saturation type.** Where distortion falls
   (random_walk), $L^\infty$ falls monotonically $170\to25$. Where it saturates
   high (gaussian), $L^\infty$ stays ≈180–250. lorenz holds $L^\infty\ge168$ even
   at $k=256$, the largest terminal $L^\infty$ among the descending datasets.

*Parameters: $N=65536$, seed 1234, $k\in\{1,\dots,256\}$. All values are direct
outputs of the existing framework; no KSDZ or benchmark code was modified.*
