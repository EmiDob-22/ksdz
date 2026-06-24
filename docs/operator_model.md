# A Mathematical Operator Model of KSDZ

*Reverse-engineering the mathematical object implemented by `ksdz_core.py`.*

This document analyses the map actually implemented by
`KSDZ_Quantum_Encoder.compress` followed by `decompress`. It introduces no code
changes; all empirical numbers are quoted from the benchmark suite
(`benchmarks/`, `results.json`, seed `1234`, `top_k = 100`). Where the
engineering implementation departs from the clean mathematical idealisation,
both are stated and the departure is named.

Notation conventions:

- Integers $b \in \{0,\dots,255\}^N$ are raw `uint8` inputs.
- $x \in \mathbb{R}^N$ is the working signal.
- The DFT follows NumPy's unnormalised convention
  $(Fx)_m = \hat{x}_m = \sum_{n=0}^{N-1} x_n e^{-2\pi i m n / N}$,
  with inverse $x_n = \tfrac1N \sum_m \hat{x}_m e^{+2\pi i m n/N}$.
- The **unitary** DFT is $\mathcal{F} = N^{-1/2} F$, so $\|\mathcal{F}x\|_2 = \|x\|_2$
  (Parseval). Clean energy statements use $\mathcal{F}$; the code uses $F$.

---

## 1. The signal space

**Discrete (what is implemented).** The native object is a finite real vector

$$
x \in \mathbb{R}^N, \qquad N = \texttt{len(data\_bytes)},
$$

obtained from bytes by the affine *de-quantisation* map
$a : \{0,\dots,255\}^N \to \mathbb{R}^N$,

$$
a(b) = \frac{b}{127.5} - \mathbf{1}, \qquad a(b)_n \in [-1,\, 1 - \tfrac1{127.5}].
$$

$a$ is affine and injective. The inner product is the Euclidean
$\langle x,y\rangle = \sum_n x_n y_n$, and $\ell^2(\mathbb{Z}_N) \cong \mathbb{R}^N$
carries the orthonormal Fourier basis
$\varphi_m = N^{-1/2}\big(e^{2\pi i m n/N}\big)_{n}$, $m \in \mathbb{Z}_N$.

**Continuous idealisation (for clean statements).** Let
$x \in L^2([0,1])$ with orthonormal Fourier basis
$e_m(t) = e^{2\pi i m t}$, $m \in \mathbb{Z}$, and coefficients
$c_m(x) = \langle x, e_m\rangle = \int_0^1 x(t) e^{-2\pi i m t}\,dt$. The discrete
suite is the sampled, band-limited shadow of this space; the truncation
operator below is defined identically in both, so $L^2([0,1])$ is used wherever
it makes a statement basis-independent.

The benchmark further pre-composes $a$ with a **uint8 quantiser** $Q$ (min/max
normalise then round), so the canonical *input* itself lives in
$Q(\mathbb{R}^N) \subset \{0,\dots,255\}^N$. That quantisation is a property of
the harness, not of KSDZ, and is treated as the ambient lattice of the domain.

---

## 2. KSDZ as an explicit operator composition

Read off `compress`∘`decompress` line by line. Write
$\mathbf{1}_S$ for the diagonal indicator (keep coordinates in $S$, zero the
rest), and for a set $S$ of positive frequencies let
$S^\ast = S \cup (-S) = S \cup \{N-m : m\in S\}$ be its conjugate completion.

1. **De-quantise** (`_to_signal`): $x = a(b)$. *Affine.*
2. **Analyse**: $\hat{x} = Fx$. *Linear, invertible.*
3. **Support selection** (`compress`, lines 40–43): set $\hat{x}_0 \mapsto 0$ for
   ranking, then choose
   $$
   S(x) = \operatorname*{arg\,top}_{m \neq 0}{}^{(k')}\,|\hat{x}_m|,
   \qquad k' = \min(k,\ \lfloor N/2\rfloor),
   $$
   the indices of the $k'$ largest-magnitude non-DC bins. **Data-dependent.**
4. **Coefficient quantise** (line 49): store $(m,\ \mathrm{fl}_{32}\,\Re\hat{x}_m,\ \mathrm{fl}_{32}\,\Im\hat{x}_m)$
   for $m \in S(x)$, where $\mathrm{fl}_{32}$ is rounding to `float32`. *Quantisation.*
5. **Synthesise sparse spectrum** (`decompress`, lines 61–62): build
   $\tilde{x}$ with $\tilde{x}_m = \hat{x}_m$ for $m \in S(x)$ and the **conjugate
   partner** $\tilde{x}_{-m} = \overline{\hat{x}_m}$ forced in; all other bins,
   including DC, are $0$.
6. **Reconstruct**: $y = \Re\big(F^{-1}\tilde{x}\big)$. *Linear + $\Re$.*
7. **Renormalise** (`_to_bytes`, lines 19–23): the min/max map
   $$
   \nu(y) = 255\,\frac{y - \min_n y_n}{\max_n y_n - \min_n y_n}
   \quad(\text{if } \max>\min), \qquad \nu(y)=y+128 \text{ otherwise}.
   $$
   **Data-dependent affine** (random offset and gain set by $y$).
8. **Output quantise**: $b' = Q(\nu(y)) = \mathrm{round}\,\mathrm{clip}_{[0,255]}\,\nu(y)$.

So the implemented operator $T_k : \{0,\dots,255\}^N \to \{0,\dots,255\}^N$ is

$$
\boxed{\;T_k \;=\; Q \,\circ\, \nu \,\circ\, \Re \,\circ\, F^{-1} \,\circ\, \Pi_{S(a(b))} \,\circ\, F \,\circ\, a\;}
$$

where $\Pi_S = \mathbf{1}_{S^\ast}$ is the (linear, once $S$ is fixed) restriction
to the conjugate-completed support $S^\ast$, with DC excluded.

**It is *not* simply $T = F^{-1}P_k F$.** That clean form is only the *core*

$$
\mathcal{T}_k \;=\; \mathcal{F}^{-1}\,\Pi_{S(x)}\,\mathcal{F},
$$

i.e. the best-$k'$-term Fourier truncation. KSDZ = this core wrapped in four
non-ideal layers that change its algebraic character:

| Layer | Symbol | Effect on the math |
| :--- | :--- | :--- |
| DC removal in selection | $\hat{x}_0 \mapsto 0$ | removes the mean; output is forced zero-mean **before** $\nu$ |
| best-$k$ selection | $S(\cdot)$ | makes the support, hence $\Pi$, depend on $x$ ⇒ **nonlinear** |
| coefficient rounding | $\mathrm{fl}_{32}$ | perturbs retained coefficients by $O(2^{-23}\lvert\hat x_m\rvert)$ |
| output renormalise | $\nu$ | data-dependent affine ⇒ **scale-invariant, nonlinear** |
| output quantise | $Q$ | rounds onto the `uint8` lattice ⇒ breaks exact idempotency |

Thus the *idealised* object is a projection-based **nonlinear $k$-term
approximation**; the *implemented* object is that, post-composed with a
scale-normalising quantiser.

### 2.1 The conjugate-doubling redundancy

For real $x$, $|\hat{x}_m| = |\hat{x}_{N-m}|$, so $m$ and $N-m$ are selected
together: $S(x)$ is (up to ties / the Nyquist self-conjugate bin $m=N/2$) a union
of conjugate pairs. Step 5 *also* re-imposes the partner. Hence each physical
frequency occupies **two genes** carrying $4$ floats for the $2$ real degrees of
freedom $(\Re\hat{x}_m,\Im\hat{x}_m)$ of one pair. Consequently:

- the number of distinct physical frequencies retained is $\approx k'/2$;
- the real dimension of $\operatorname{range}(\Pi_{S^\ast})$ is $\approx k'$;
- coefficient storage is $\approx 2\times$ redundant.

This factor of two matters for §6 (it does **not** change the $1/N$ scaling, only
the constant).

---

## 3. Error and discarded spectral energy

Work first with the clean core $\mathcal{T}_k$ (unitary basis, exact
coefficients, no $\nu,Q$). Let $\tilde{S} = S(x)^\ast$ be the retained set.
Because $\{\varphi_m\}$ is orthonormal and $\Pi_{\tilde S}$ keeps exactly those
basis coefficients,

$$
\mathcal{T}_k x = \sum_{m \in \tilde{S}} \langle x,\varphi_m\rangle \,\varphi_m,
\qquad
x - \mathcal{T}_k x = \sum_{m \notin \tilde{S}} \langle x,\varphi_m\rangle\,\varphi_m .
$$

By Parseval the reconstruction error is **exactly the discarded spectral
energy**:

$$
\boxed{\;\operatorname{error}(x,k)^2 \;=\; \|x - \mathcal{T}_k x\|_2^2
\;=\; \sum_{m \notin \tilde{S}} |\langle x,\varphi_m\rangle|^2
\;=\; \|x\|_2^2 - \sum_{m \in \tilde{S}} |\langle x,\varphi_m\rangle|^2.\;}
$$

In NumPy's unnormalised convention, with the DC term removed,
$\operatorname{error}(x,k)^2 = \tfrac1N \sum_{m \notin \tilde S,\, m\neq 0} |\hat{x}_m|^2$.

**Optimality.** Among all index sets of size $k'$, choosing the largest
$|\langle x,\varphi_m\rangle|$ *minimises* the discarded energy. Hence
$\mathcal{T}_k$ realises the **best $k'$-term approximation of $x$ in the Fourier
basis**:

$$
\|x - \mathcal{T}_k x\|_2 \;=\; \min_{|\Sigma| = k'} \Big\| x - \textstyle\sum_{m\in\Sigma}\langle x,\varphi_m\rangle\varphi_m \Big\|_2 .
$$

This is the orthonormal-basis special case of the Eckart–Young principle (greedy
coefficient thresholding is optimal). KSDZ’s selection rule is therefore not
heuristic — it is the energy-optimal hard threshold, *modulo* the DC exclusion
and the conjugate-pair budget accounting.

**Decay law.** Let $(|c|_{(j)})_{j\ge 1}$ be the coefficient magnitudes sorted
decreasingly. If $|c|_{(j)} \sim C j^{-s}$ then for $s > \tfrac12$

$$
\operatorname{error}(x,k)^2 \;=\; \sum_{j > k'} |c|_{(j)}^2 \;\sim\; \frac{C^2}{2s-1}\, (k')^{-(2s-1)},
\qquad
\operatorname{error}(x,k) \sim k'^{-(s - 1/2)} .
$$

Faster coefficient decay (larger $s$) ⇒ smaller truncation error at fixed $k$.
This single exponent organises §4.

**What the engineering layers do to this identity.** $\nu$ restores an affine
gain/offset, so the *measured* error is taken after the reconstruction is
re-stretched to $[0,255]$; this replaces $\|x-\mathcal T_k x\|$ by
$\|x - (\alpha\,\mathcal T_k x + \beta\mathbf 1)\|$ with $(\alpha,\beta)$ chosen by
$\nu$, which can only *reduce* the gap relative to a fixed scale (it fits one
optimal gain/offset) while *destroying absolute amplitude fidelity*. $Q$ adds
quantisation noise of order $1/2$ LSB. These shift constants, not the
qualitative $k^{-(s-1/2)}$ ordering — which is exactly what the benchmark shows.

---

## 4. Why each dataset behaves as measured

Benchmark fidelity at $N = 65536$, $k=100$ (`results.json`):

| Dataset | NRMSE | spec\_norm | L∞ | Coefficient picture |
| :--- | ---: | ---: | ---: | :--- |
| `random_walk` | 0.032 | 0.042 | 36 | $\sim 1/f^2$ power; $s \approx 1$ |
| `sinusoidal_mixture` | 0.034 | 0.055 | 84 | line spectrum; finite support |
| `gaussian_noise` | 0.158 | 0.231 | 182 | flat power; $s \approx 0$ |
| `lorenz_trajectory` | 0.193 | 0.270 | 227 | broadband + fast lobe switches |

**Periodic / quasi-periodic (`sinusoidal_mixture`) performs well.** A sum of $r$
sinusoids has a **line spectrum**: $c_m$ is supported on $2r$ conjugate bins, so
$\operatorname{error}(x,k) = 0$ as soon as $k' \ge 2r$. With $r=5 \ll k'=100$ the
discarded energy is (up to quantisation/leakage from the finite window) zero;
the residual NRMSE $0.034$ is dominated by spectral leakage of non-integer
periods and by $Q$, not by truncation. This is the regime the marketing
"99.99%" implicitly assumed.

**Gaussian white noise performs poorly.** White noise has, in expectation, a
**flat** power spectrum $\mathbb{E}|\hat{x}_m|^2 = \sigma^2 N$ (no decay, $s=0$).
Retaining $k'$ of the $\approx N/2$ AC pairs keeps an expected energy fraction
$\approx 2k'/N$, so

$$
\frac{\operatorname{error}(x,k)^2}{\|x_{\mathrm{AC}}\|_2^2} \;\approx\; 1 - \frac{2k'}{N}
\;=\; 1 - \frac{200}{65536} \;\approx\; 0.997 .
$$

Essentially **all** AC energy is discarded — the worst case for any fixed-basis
truncation. The reason the *measured* NRMSE is $0.158$ rather than $\approx 1$ is
the two engineering layers: (i) $\nu$ re-inflates the surviving $100$-mode
reconstruction to span $[0,255]$, and (ii) NRMSE normalises by dynamic range
while the discarded structure is incoherent with the original, capping the error
near the "two uncorrelated signals of comparable variance" ceiling
$\sqrt{2}\,\sigma/\text{range}$. The **spectral** diagnostic `spec_norm` $=0.231$,
which compares $|\hat{x}|$ directly, is $\sim 4\times$ worse than the periodic
case and is the honest indicator here.

**Lorenz trajectories degrade (worst NRMSE/spec, largest L∞).** The Lorenz
$x$-coordinate is **broadband**: low-frequency content from slow orbiting within
a lobe, plus near-discontinuous **lobe-switching transitions**. Locally the
signal is piecewise smooth with fast transitions, whose Fourier coefficients
decay only algebraically near each transition. Truncating to $100$ modes thus
incurs a **Gibbs phenomenon**: $O(1)$ overshoot localised at the switches that
does not diminish with $k$. This is visible as the **largest $L^\infty = 227$**
(near the full $255$ range) in the table — a signature of localised ringing
rather than uniform error. Hence Lorenz edges out even white noise on NRMSE and
spec\_norm: white-noise error is spread and partly absorbed by $\nu$, whereas
Lorenz concentrates error at transitions where renormalisation cannot help.

**Ordering, and an honest caveat.** The energy-decay model predicts
periodic $\approx$ random-walk $\ll$ broadband, and ranks white noise as the
energy-worst case. The data confirm the gross ordering but show
$\text{lorenz} > \text{gaussian}$ in *both* NRMSE and spec\_norm — i.e. Gibbs
localisation makes a *structured* broadband signal score worse than an
*unstructured* one. The pure $s$-exponent argument does not by itself predict
this inversion; the interaction of Gibbs overshoot with the $\nu$/range
normalisation does. This is flagged as open in §8.

---

## 5. Operator-theoretic classification

Let $T_k$ be the full implemented map and $\mathcal{T}_k = \mathcal F^{-1}\Pi_{S(x)}\mathcal F$
its idealised core.

### 5.1 Linearity — **nonlinear** (two independent reasons)

*Reason 1: data-dependent support.* Take $N$ large, $k=1$. Let $u = \varphi_a$ and
$v = 2\varphi_b$ with distinct non-DC frequencies $a \neq b$ (and partners). Then
$S(u)=\{a\}$, $S(v)=\{b\}$, but $S(u+v)=\{b\}$ since $|\widehat{u+v}_b|=2 > 1$.
Hence (even for the core)
$\mathcal{T}_1(u+v) = \Pi_{\{b\}}(u+v) \neq \mathcal{T}_1 u + \mathcal{T}_1 v$,
because the right side retains both $a$ and $b$. Additivity fails.

*Reason 2: scale invariance from $\nu$.* For any $c>0$, $\nu(cy)=\nu(y)$ exactly
(min/max cancels $c$), and $S(cx)=S(x)$. Therefore

$$
T_k(c\,b) = T_k(b) \quad\text{for all } c>0, \qquad\text{so}\qquad T_k(2b)=T_k(b) \neq 2\,T_k(b)
$$

in general. Positive homogeneity of degree **zero** is incompatible with
linearity (which requires degree one). KSDZ discards absolute amplitude entirely.

### 5.2 Idempotency — **the core yes, the full map no (only approximately)**

*Core:* with a **fixed** support $\Sigma$, $P_\Sigma = \mathcal F^{-1}\mathbf 1_{\Sigma^\ast}\mathcal F$
is an orthogonal projection:

$$
P_\Sigma^2 = \mathcal F^{-1}\mathbf 1_{\Sigma^\ast}\mathcal F\,\mathcal F^{-1}\mathbf 1_{\Sigma^\ast}\mathcal F
= \mathcal F^{-1}\mathbf 1_{\Sigma^\ast}^2\,\mathcal F = \mathcal F^{-1}\mathbf 1_{\Sigma^\ast}\mathcal F = P_\Sigma,
\qquad P_\Sigma^\ast = P_\Sigma,
$$

since $\mathbf 1_{\Sigma^\ast}$ is a diagonal $0/1$ idempotent and $\mathcal F$ is
unitary. So the linear core is a genuine self-adjoint projection ($P^2=P=P^\ast$).

*Full map:* $T_k(T_k(b))$ re-runs selection and renormalisation on an already
$k'$-band-limited, range-filling signal. If $k_{\mathrm{eff}}(T_k b) \le k'$ the
support is reselected identically and $\nu$ acts as near-identity (the signal
already spans $[0,255]$), so $T_k(T_k(b)) \approx T_k(b)$. Exact idempotency
fails because of $Q$: the `uint8` rounding on the second pass generically moves
$\ge 1$ coordinate by $1$ LSB, and `float32` re-rounding perturbs coefficients.
Formally, $T_k$ is **quasi-idempotent**: $\|T_k^2 b - T_k b\|_\infty \le$ a small
$O(1)$ quantisation bound, but $T_k^2 \neq T_k$ as exact maps. (Counterexample to
exactness: any $b$ for which $\nu(\cdot)$ lands a coordinate on a half-integer,
which $Q$ then rounds differently across passes.)

### 5.3 Boundedness — **yes (both senses)**

*Full map:* $\operatorname{range}(T_k) \subseteq \{0,\dots,255\}^N$, so for all $b$
$$
\|T_k b\|_\infty \le 255, \qquad \|T_k b\|_2 \le 255\sqrt{N}.
$$
$T_k$ is a bounded map (with bounded, in fact finite, range), notwithstanding its
nonlinearity.

*Core:* an orthogonal projection has operator norm
$\|P_\Sigma\|_{2\to2} = 1$ (it is a contraction; $=1$ whenever $\Sigma\neq\varnothing$).
The nonlinear selection $\mathcal T_k$ is $1$-Lipschitz on the *open* region where
the top-$k'$ support is locally constant, but **not globally Lipschitz**: across
the selection boundaries $\{|\hat x_a|=|\hat x_b|\}$ the support jumps and
$\mathcal T_k$ is discontinuous. So $\mathcal T_k$ is bounded and piecewise an
orthogonal projection, but not continuous.

**Summary.**

| Property | Idealised core $\mathcal{T}_k$ | Full implemented $T_k$ |
| :--- | :--- | :--- |
| Linear | No (support depends on $x$) | No (support **and** scale-invariant $\nu$) |
| Idempotent | Yes on fixed support; the nonlinear selector is not | Only quasi-idempotent (broken by $Q$) |
| Bounded | Yes; piecewise orthogonal projection, $\|\cdot\|=1$ | Yes; range $\subseteq[0,255]^N$ |
| Self-adjoint | Yes on fixed support | N/A (nonlinear) |
| Continuous | No (jumps at selection ties) | No |

---

## 6. The source of `pseudo_compression_ratio` growth

The compressed payload is, exactly (`compress`):

$$
\underbrace{|\langle\texttt{'<QH'}\rangle|}_{8+2\,=\,10\ \text{bytes (header)}}
\;+\;
k'\cdot\underbrace{|\langle\texttt{'<Iff'}\rangle|}_{4+4+4\,=\,12\ \text{bytes/gene}},
\qquad k' = \min(k,\lfloor N/2\rfloor).
$$

For $N \ge 2k$ (the benchmark regime, $k=100$), $k' = k$ is **constant in $N$**,
so the compressed size is the $N$-independent constant

$$
C(k) = 10 + 12k \quad\text{bytes} \qquad (C(100) = 1210).
$$

The original size is $N$ bytes (`uint8`). Therefore the reported ratio is the
**exact** rational function

$$
\boxed{\;\operatorname{pCR}(N,k) \;=\; 1 - \frac{C(k)}{N} \;=\; 1 - \frac{10 + 12k}{N}, \qquad N \ge 2k.\;}
$$

Check against measurement ($k=100$): $\operatorname{pCR}(4096)=1-1210/4096=0.7046$
and $\operatorname{pCR}(65536)=1-1210/65536=0.98154$ — matching the benchmark’s
$0.7046$ and $0.9815$ to all printed digits.

**Asymptotic statement.**

$$
\operatorname{pCR}(N,k) = 1 - \frac{10+12k}{N} = 1 - \Theta_k(N^{-1}) \xrightarrow{\;N\to\infty\;} 1,
$$

with $1 - \operatorname{pCR}(N,k) \sim (12k)\,N^{-1}$. For $N < 2k$ the regime
changes to $k'=\lfloor N/2\rfloor$, giving
$\operatorname{pCR} = 1 - (10 + 12\lfloor N/2\rfloor)/N \to 1 - 6 = -5$ as a lower
bound near $N\approx$ small (the payload can exceed the input).

**Interpretation.** The growth toward $1$ is a **fixed-budget artefact**: the
encoder emits a constant number of genes regardless of input length or content,
so the ratio rises mechanically as the denominator $N$ grows. It encodes **no**
statement about compressibility of the data. The conjugate-doubling of §2.1
means the constant $12k$ already pays $2\times$ for $\approx k/2$ physical
frequencies; halving it (storing one member per pair) would change $C(k)$ to
$10 + 6k$ but not the $\Theta(N^{-1})$ law. This is precisely why the suite
records it as a **pseudo** ratio, non-comparable to a lossless coder’s
content-adaptive ratio.

---

## 7. Placement within known mathematics

| Category | Verdict | Justification |
| :--- | :---: | :--- |
| **Spectral truncation** | ✅ core identity | $T_k$ keeps the largest-magnitude DFT bins and zeroes the rest: textbook spectral / Fourier truncation, in its *nonlinear best-$k$-term* form (greedy hard thresholding). §3 establishes the best-$k$-term optimality. |
| **Transform coding** | ✅ (incomplete) | The pipeline transform → select → quantise → store (FFT; top-$k$; `float32`; raw bytes) is exactly transform coding. It lacks the final **entropy-coding** stage and any rate–distortion bit allocation, so it is transform coding with a degenerate (fixed-length, non-entropy) back end. |
| **Projection operator** | ◐ core only | For a *fixed* support, the linear core $P_\Sigma=\mathcal F^{-1}\mathbf 1_{\Sigma^\ast}\mathcal F$ is an orthogonal projection (§5.2). The full operator is a nonlinear $k$-term *approximation*, not a projection, and $\nu,Q$ break idempotency. So: "projection" describes the core, not KSDZ. |
| **Low-rank approximation** | ◐ analogy | $\mathcal T_k$ maps onto a $\le k'$-dimensional subspace, so its *output* is "low rank" in the sense of few active modes. But the subspace is the **fixed Fourier basis**, chosen per-signal by thresholding — not a **data-adaptive** singular subspace. It is the diagonal (Fourier) analogue of truncated SVD, not SVD/PCA itself. Include as analogy, exclude as method. |
| **Reduced-order model** | ◐ structural kinship | ROM (POD/Galerkin) projects a **dynamical operator** onto dominant modes to cheapen *evolution*. KSDZ compresses a **static snapshot** and reduces no dynamics, no governing equation. It resembles the *projection step* of POD applied per snapshot, but provides no reduced dynamics. Exclude as a ROM proper. |
| **Compressed sensing** | ❌ | CS recovers a sparse signal from $m \ll N$ **incoherent/random linear measurements** via convex ($\ell_1$) or greedy recovery. KSDZ acquires the **full** signal, computes the **full** FFT, and keeps the largest coefficients by **oracle** thresholding — no undersampling, no random sensing matrix, no recovery optimisation. It is the *oracle best-$k$-term* that CS theory uses as the **benchmark to approximate**, hence categorically not CS. |

**One-line classification.** KSDZ is **nonlinear best-$k$-term Fourier spectral
truncation with fixed-length coefficient quantisation and amplitude
renormalisation** — i.e. an (entropy-coder-less) transform coder whose linear
core is an orthogonal Fourier projection.

---

## 8. Research questions

Open mathematical questions only; none are settled by the current suite.

1. **Renormalisation-corrected error law.** Derive a closed form for the
   *measured* error $\|x - (\alpha\mathcal T_k x + \beta\mathbf 1)\|$ where
   $(\alpha,\beta)$ are the min/max gains of $\nu$. How much does fitting one
   optimal affine gain/offset reduce NRMSE relative to the Parseval ideal
   $\sqrt{E_{\text{discarded}}}$, as a function of the coefficient-decay exponent
   $s$? Quantify the "$\nu$ absorbs error" effect that flattened white-noise NRMSE
   to $0.158$.

2. **The Lorenz > Gaussian inversion.** Prove or refute: under fixed-$k$ Fourier
   truncation followed by $\nu$, a piecewise-smooth signal with $J$ fast
   transitions (Gibbs, $L^\infty = \Theta(1)$) has larger NRMSE than i.i.d. noise
   of equal variance once $k \ll N$. Identify the critical $(k, J, \text{SNR})$
   surface where the ordering flips. This is the §4 caveat made precise.

3. **DC-removal bias.** KSDZ forces zero mean before $\nu$ re-introduces an
   offset. Characterise the bias this injects for signals whose information lives
   partly in the mean (e.g. asymmetric or heavy-tailed inputs), and whether an
   optimal *single* retained DC term would dominate any $k$ AC modes for some
   signal classes.

4. **Effective rank vs. gene budget.** Given conjugate-doubling (§2.1) and ties,
   what is the exact distribution of the number of *distinct* physical
   frequencies retained as a function of $k'$ and the spectral profile? When does
   the Nyquist self-conjugate bin or a magnitude tie make $|S^\ast| < k'$?

5. **Stability across selection boundaries.** $\mathcal T_k$ is discontinuous on
   the tie-set $\{|\hat x_a| = |\hat x_b|\}$. Bound the reconstruction jump
   $\|\mathcal T_k x^+ - \mathcal T_k x^-\|$ across a boundary crossing in terms of
   the swapped coefficients, and characterise the measure / codimension of the
   boundary set in $\mathbb{R}^N$.

6. **Optimal $k$ under a distortion target.** For each dataset’s coefficient
   decay $s$, solve for the minimal $k^\star(\varepsilon)$ achieving NRMSE
   $\le\varepsilon$, using the $k^{-(s-1/2)}$ law and the $\operatorname{pCR}$
   formula, to produce an honest rate–distortion curve
   $\operatorname{pCR}$ vs. NRMSE per signal class.

7. **Idempotency defect.** Make §5.2 quantitative: bound
   $\|T_k^2 - T_k\|$ (sup over inputs) in terms of the `uint8`/`float32`
   quantisation step, and determine whether iterating $T_k^n$ converges to a fixed
   point (a quantised $k$-band-limited signal) and at what rate.

8. **Continuous limit.** As $N\to\infty$ with the sampling of a fixed
   $x \in L^2([0,1])$, does the discrete $\mathcal T_k$ Γ-converge (or converge in
   operator sense on band-limited subspaces) to the continuous best-$k$-term
   Fourier projector, and is the convergence uniform on Sobolev balls $H^s$?

---

*This document is descriptive mathematics about existing code. It prescribes no
implementation change. Every empirical value is reproducible via
`python benchmarks/benchmark_runner.py --all --output results.json`.*
