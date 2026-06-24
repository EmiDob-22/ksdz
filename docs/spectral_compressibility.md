# Spectral Compressibility: Is α Meaningful or a KSDZ Artifact?

*Phase 6. KSDZ and `benchmarks/` are used only as read-only instruments; no code
in either was modified. All numbers are measured outputs of
`datasets.py` signals analysed by two independent estimators. Raw values are
archived in `docs/spectral_compressibility_data.json`.*

## Question

The earlier rate–distortion study fitted $D(k)\sim k^{-\alpha}$ through KSDZ. This
phase asks whether $\alpha$ is (A) a codec artifact, (B) a useful empirical
descriptor, or (C) an invariant of spectral structure — decided only by measured
evidence.

## Two estimators

- **$\alpha_{\mathrm{KSDZ}}$** — fit of `nrmse` from
  `KsdzTransform(top_k=k).encode/decode` (the Phase-5 instrument), $k\in\{4,\dots,256\}$.
- **$\alpha_{\mathrm{Fourier}}$ (independent, Task 5)** — computed *directly from
  FFT coefficients*, with no KSDZ call, no reconstruction, no renormalisation, no
  quantisation. For signal $x$, with DC excluded (as KSDZ does), let
  $p_{(1)}\ge p_{(2)}\ge\cdots$ be the sorted bin powers $|\hat x_m|^2$. The
  oracle best-$k$-term relative $L^2$ error is
  $$
  D_{\mathrm{F}}(k) \;=\; \sqrt{\frac{\sum_{j>k} p_{(j)}}{\sum_j p_{(j)}}},
  \qquad \alpha_{\mathrm{Fourier}} = -\,\mathrm{slope}\big(\log D_{\mathrm{F}}\ \text{vs}\ \log k\big).
  $$

Both fits report a 95% CI from the regression slope standard error ($t$,
$\mathrm{dof}=n-2=5$, $t_{.975}=2.571$) and $R^2$. Fixed seed $1234$; noise seed
$7$.

```python
# the independent estimator, in full (no KSDZ):
X = np.fft.fft(x); p = np.abs(X)**2; p[0] = 0.0
p = np.sort(p)[::-1]; tot = p.sum(); cs = np.cumsum(p)
D_F = [np.sqrt(max(tot-cs[k-1],0)/tot) for k in ks]
```

---

## Task 1 — Stability of α over N (via KSDZ)

$\alpha_{\mathrm{KSDZ}}(N)$, reported as $\alpha \pm \text{CI}_{95}\ (R^2)$:

| dataset | N=4096 | N=16384 | N=65536 | N=262144 | trend |
|:---|:---|:---|:---|:---|:---|
| sinusoidal_mixture | 0.418 ± 0.193 (.84) | 0.416 ± 0.195 (.84) | 0.416 ± 0.196 (.83) | 0.416 ± 0.196 (.83) | **flat → 0.416** |
| random_walk | 0.681 ± 0.085 (.99) | 0.538 ± 0.027 (1.0) | 0.565 ± 0.090 (.98) | 0.580 ± 0.117 (.97) | **→ ≈0.57** |
| gaussian_noise | 0.132 ± 0.055 (.86) | 0.113 ± 0.060 (.80) | 0.119 ± 0.066 (.78) | 0.123 ± 0.075 (.75) | **flat ≈0.12** |
| lorenz_trajectory | 0.559 ± 0.243 (.86) | 0.241 ± 0.051 (.96) | 0.160 ± 0.045 (.93) | 0.100 ± 0.029 (.93) | **drifts ↓ toward 0** |

Measured: sinusoidal and random_walk give an $N$-stable $\alpha_{\mathrm{KSDZ}}$;
gaussian is stable at a small value; **lorenz is unstable** — $\alpha$ falls
monotonically $0.559\to0.100$ with no sign of a limit. So through KSDZ, $\alpha$
is $N$-stable for three of four datasets and clearly $N$-dependent for Lorenz.

---

## Task 5 (companion) — α_Fourier over N (independent)

Same fit on the independent estimator:

| dataset | N=4096 | N=16384 | N=65536 | N=262144 | trend |
|:---|:---|:---|:---|:---|:---|
| sinusoidal_mixture | 0.628 ± 0.055 | 0.627 ± 0.055 | 0.627 ± 0.055 | 0.627 ± 0.055 | **flat → 0.627** |
| random_walk | 0.527 ± 0.042 | 0.516 ± 0.028 | 0.507 ± 0.041 | 0.533 ± 0.023 | **→ ≈0.52** |
| lorenz_trajectory | 0.616 ± 0.307 | 0.173 ± 0.080 | 0.060 ± 0.026 | 0.022 ± 0.010 | **→ 0** |
| gaussian_noise | 0.029 ± 0.015 | 0.009 ± 0.005 | 0.003 ± 0.001 | 0.001 ± 0.000 | **→ 0** |

Independent of KSDZ, only **sinusoidal_mixture** and **random_walk** have an
$N$-stable spectral exponent. **gaussian_noise and lorenz_trajectory have
$\alpha_{\mathrm{Fourier}}\to 0$** as $N$ grows: their bin energy spreads over
$\Theta(N)$ modes, so a fixed top-$k$ budget captures a vanishing fraction and the
log–log slope flattens. The stable random_walk value $\approx 0.52$ matches the
$1/f^2$ prediction $\alpha=s-\tfrac12=\tfrac12$ for power law $|\hat x|\sim m^{-1}$.

**Conclusion of Tasks 1+5 (measured):** $\alpha$ is a genuine $N$-invariant only
for signals whose spectrum is line- or power-law-concentrated (sinusoidal,
random_walk). For broadband/noise signals it is $N$-dependent and tends to $0$ —
it is *not* invariant there.

---

## Task 2 — Noise perturbation α(σ)

$x_\sigma = \hat x + \sigma\eta$ on the standardised signal ($\eta\sim\mathcal N(0,1)$,
seed 7), $N=65536$, $\alpha_{\mathrm{Fourier}}$:

| dataset \\ σ | 0.00 | 0.01 | 0.05 | 0.10 | 0.20 | 0.50 | 1.00 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| sinusoidal_mixture | 0.628 | 0.623 | 0.543 | 0.426 | 0.273 | 0.102 | 0.036 |
| random_walk | 0.507 | 0.505 | 0.470 | 0.400 | 0.277 | 0.110 | 0.039 |
| lorenz_trajectory | 0.060 | 0.060 | 0.060 | 0.060 | 0.057 | 0.046 | 0.027 |
| gaussian_noise | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 |

Measured behaviour:

- **Continuous and monotone** in $\sigma$ for every dataset — no jump, no
  discontinuity. There is **no phase transition** in the strict (discontinuous)
  sense.
- The two concentrated signals (sinusoidal, random_walk) show a smooth
  **crossover** from signal-dominated ($\alpha\approx0.5$–$0.6$) to
  noise-dominated ($\alpha\to0$), steepest over $\sigma\in[0.1,0.2]$ (the per-step
  drop peaks there: random_walk $0.123,0.167$; sinusoidal $0.153,0.171$). By
  $\sigma=1$ both collapse to $\approx0.04$, i.e. the white-noise value.
- The two broadband signals start near $0$ and stay there (gaussian exactly flat;
  lorenz mildly decreasing) — adding noise to something already broadband barely
  moves $\alpha$.

So $\alpha(\sigma)$ is a continuous "melting" of the exponent, not a transition;
the crossover scale ($\sigma\approx0.15$) is where injected noise energy begins to
dominate the signal's spectral tail.

---

## Task 3 — Entropy comparison

$N=65536$. Shannon entropy of the uint8 value histogram (bits/symbol, max 8);
spectral entropy $=-\sum p_m\ln p_m/\ln M$ on normalised AC power (in $[0,1]$,
$1=$ flat):

| dataset | Shannon (bits) | Spectral entropy | α_Fourier | α_KSDZ |
|:---|---:|---:|---:|---:|
| gaussian_noise | 6.945 | 0.962 | 0.003 | 0.119 |
| lorenz_trajectory | 7.689 | 0.659 | 0.060 | 0.160 |
| random_walk | 7.568 | 0.256 | 0.507 | 0.565 |
| sinusoidal_mixture | 7.821 | 0.234 | 0.627 | 0.416 |

Measured Pearson correlations:

| pair | r | n |
|:---|---:|---:|
| α_Fourier vs Shannon entropy | **+0.654** | 4 |
| α_Fourier vs spectral entropy | **−0.951** | 4 |
| α_Fourier vs spectral entropy (pooled over the σ-sweep) | **−0.883** | 28 |

**Answer (measured):** $\alpha$ does **not** track amplitude (Shannon) entropy —
$r=+0.65$ over four points, weak and of the "wrong" sign for a compressibility
measure (the highest-Shannon signal, sinusoidal at 7.82 bits, has the *highest*
$\alpha$). $\alpha$ **strongly anti-correlates with spectral entropy**:
$r=-0.951$ across datasets and $r=-0.883$ over $28$ pooled (dataset, σ) points.
Low spectral entropy (concentrated spectrum) $\Rightarrow$ high $\alpha$. $\alpha$
is a measure of **spectral** concentration, not amplitude disorder.

---

## Task 4 — Universal classification attempt

Proposed: A $\alpha>0.45$, B $0.25<\alpha\le0.45$, C $\alpha\le0.25$. Assigning
with each estimator at $N=65536$:

| dataset | α_Fourier | class (F) | α_KSDZ | class (K) |
|:---|---:|:---:|---:|:---:|
| gaussian_noise | 0.003 | C | 0.119 | C |
| lorenz_trajectory | 0.060 | C | 0.160 | C |
| random_walk | 0.507 | A | 0.565 | A |
| sinusoidal_mixture | 0.627 | A | 0.416 | **B** |

**Boundaries are not supported by the data.**

1. **Class B is essentially empty.** Under the independent estimator the four
   values are $\{0.003,0.060\}$ and $\{0.507,0.627\}$ — two tight clusters with a
   **wide empty gap from $0.06$ to $0.51$**. No measured point lies in
   $(0.25,0.45]$; B is populated only as an artifact of the KSDZ estimator
   (sinusoidal $0.416$), which Task 5 shows is biased.
2. **Assignments are not estimator-invariant.** sinusoidal is A under Fourier but
   B under KSDZ.
3. **Assignments are not $N$-invariant.** From Tasks 1/5, lorenz would be class A
   at $N{=}4096$ ($\alpha\approx0.56$–$0.62$) and class C by $N{=}65536$
   ($\le0.16$); its true limit is $0$.

The specific thresholds $0.25$ and $0.45$ are therefore **rejected**. The only
separation the data supports is a **binary** one — *spectrally concentrated*
(sinusoidal, random_walk; stable $\alpha\gtrsim0.5$) vs *broadband*
(gaussian, lorenz; $\alpha\to0$) — and even that requires the stable Fourier
estimator and acknowledging the broadband limit is $0$, not a fixed class value.

---

## Task 5 — Independence from KSDZ (critical)

$\alpha_{\mathrm{KSDZ}}$ vs $\alpha_{\mathrm{Fourier}}$ at $N=65536$:

| dataset | α_KSDZ ± CI | α_Fourier ± CI | \|diff\| | CIs overlap? |
|:---|:---|:---|---:|:---:|
| random_walk | 0.565 ± 0.090 | 0.507 ± 0.041 | 0.059 | **yes** |
| sinusoidal_mixture | 0.416 ± 0.196 | 0.627 ± 0.055 | 0.211 | yes (wide KSDZ CI) |
| lorenz_trajectory | 0.160 ± 0.045 | 0.060 ± 0.026 | 0.099 | no |
| gaussian_noise | 0.119 ± 0.066 | 0.003 ± 0.001 | 0.116 | no |

Measured findings:

- **α survives removal of KSDZ for random_walk:** the two estimators agree within
  CI ($0.565$ vs $0.507$), and $\alpha_{\mathrm{Fourier}}\approx0.5$ is stable in
  $N$ and matches the $1/f^2$ theory. The exponent is a property of the signal, not
  the codec.
- **KSDZ systematically *inflates* α for broadband signals.** For gaussian and
  lorenz the true spectral exponent is $\approx0$ (and $\to0$ in $N$), yet KSDZ
  reports $0.12$ and $0.16$ — non-overlapping CIs. KSDZ's renormalisation and
  quantisation manufacture apparent decay where the spectrum has essentially none.
- **For sinusoidal, KSDZ *deflates* α** ($0.416$ vs $0.627$): its $k\!>\!64$
  saturation (Phase 5) bends the KSDZ slope down, while the oracle keeps
  improving. The wide KSDZ CI hides the disagreement.

So $\alpha$ is **not purely a codec artifact** (it survives for the concentrated
cases) but **KSDZ is a biased estimator of it** — upward for broadband signals,
downward for saturating ones. The trustworthy quantity is the codec-free
$\alpha_{\mathrm{Fourier}}$.

---

## Task 6 — Final assessment

> **B. α is a useful empirical descriptor** — specifically of *spectral
> concentration* — but it is a stable invariant only for line/power-law spectra,
> and KSDZ measures it with bias.

Exactly one option, supported only by measured evidence:

- **Not A (not a mere codec artifact):** the independent Fourier estimator,
  which never calls KSDZ, reproduces $\alpha$ for the concentrated signals
  (random_walk $\approx0.51$, sinusoidal $\approx0.63$), $N$-stably, with
  random_walk matching the $1/f^2$ value $0.5$. CIs overlap KSDZ for random_walk
  (Task 5). The exponent exists without the codec.
- **Not C (not a universal spectral invariant):** $\alpha_{\mathrm{Fourier}}(N)$
  is constant only for sinusoidal and random_walk; for gaussian and lorenz it
  drifts to $0$ (Tasks 1, 5). The proposed universal A/B/C thresholds are
  rejected (Task 4): class B is empty, and assignments are neither estimator- nor
  $N$-invariant. An invariant cannot depend on $N$ or on the measuring codec.
- **Therefore B:** $\alpha$ is a meaningful, codec-independent descriptor of how
  concentrated a signal's Fourier spectrum is. It correlates strongly and
  negatively with spectral entropy ($r=-0.95$ across datasets, $-0.88$ pooled,
  Task 3), is continuous under noise with a smooth crossover near $\sigma\approx
  0.15$ (Task 2), and cleanly separates concentrated from broadband signals — but
  it attains a stable, signal-intrinsic value only in the concentrated regime,
  and must be read from the Fourier estimator, not from KSDZ.

**One sentence:** $\alpha$ measures spectral concentration (a real signal
property, anti-correlated with spectral entropy), is reproducible without KSDZ
for line/power-law spectra, and is an $N$-stable invariant only there — so it is a
useful empirical descriptor, not a codec artifact and not a universal invariant.

---

*Parameters: seeds 1234 (signal) / 7 (noise); $k\in\{4,8,16,32,64,128,256\}$;
$N\in\{4096,16384,65536,262144\}$; CI = 95% regression-slope interval ($t$,
dof 5). Reproduce by re-running the two estimators on `datasets.py` signals; raw
values in `docs/spectral_compressibility_data.json`. No KSDZ or benchmark code
was modified.*
