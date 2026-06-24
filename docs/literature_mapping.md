# Literature Mapping of the Exponent α

*Phase 7. Pure literature identification — no code, benchmark, or KSDZ change.
Goal: determine whether the Fourier-measured exponent $\alpha$ in
$D(k)\sim k^{-\alpha}$ (Phase 6) is already a named quantity, and cite the
sources. Claims here are bibliographic; the measured numbers they are checked
against come from `docs/spectral_compressibility*.{md,json}`.*

## What α is, precisely

In Phase 6, $D_{\mathrm{F}}(k)=\big(\sum_{j>k}p_{(j)}/\sum_j p_{(j)}\big)^{1/2}$ is
the **relative $L^2$ error of the best $k$-term approximation of the signal in the
Fourier basis** ($p_{(j)}$ = sorted bin powers). Hence

$$
D_{\mathrm F}(k) = \frac{\sigma_k(x)_2}{\|x\|_2},\qquad
\sigma_k(x)_2 := \min_{|\Lambda|=k}\Big\|x-\sum_{m\in\Lambda}\langle x,\varphi_m\rangle\varphi_m\Big\|_2 ,
$$

and $\alpha$ is the polynomial decay rate of $\sigma_k$. This object —
$\sigma_k$, the **best $k$-term (nonlinear) approximation error**, and its rate —
is the central quantity of *nonlinear approximation theory*
[DeVore 1998; DeVore–Lorentz 1993; Temlyakov 2011]. The remainder of this
document identifies the equivalences.

---

## Tasks 1 & 2 — candidate concepts: definition, relation, (non-)equivalence, refs

### 1. Nonlinear approximation / best $m$-term approximation
- **Definition.** $\sigma_m(f)_X=\inf_{|\Lambda|=m}\inf_{c}\|f-\sum_{k\in\Lambda}c_k\psi_k\|_X$; the *approximation order* is the largest $\alpha$ with $\sigma_m=O(m^{-\alpha})$ [DeVore 1998, §1, §7].
- **Relation to α.** Identical: our $\alpha$ **is** the best $k$-term approximation order in the Fourier ONB, $X=L^2$.
- **Equivalence.** **Equivalent (definitional).**
- **Refs.** R. DeVore, "Nonlinear approximation," *Acta Numerica* 7 (1998) 51–150; DeVore & Lorentz, *Constructive Approximation*, Springer GMW 303 (1993); V. Temlyakov, *Greedy Approximation*, Cambridge (2011).

### 2. Sparsity / compressibility exponents (weak-$\ell^p$, Stechkin)
- **Definition.** A sequence $c$ is in weak-$\ell^p$ ($\ell^{p,\infty}$) if its decreasing rearrangement obeys $c_{(j)}\le C j^{-1/p}$; "compressible" signals are those with $1/p>1/2$ [Cohen–Dahmen–DeVore 2009; Candès–Tao 2006].
- **Relation to α.** **Stechkin's lemma:** for an ONB, $c\in\ell^{p,\infty}$, $0<p<2$ ⟹ $\sigma_m(f)_2\le C\,\|c\|_{\ell^{p,\infty}}\,m^{-(1/p-1/2)}$, i.e. $\boxed{\alpha = \tfrac1p-\tfrac12}$ [DeVore 1998, Thm 7.x; Temlyakov 2011].
- **Equivalence.** **Equivalent.** $\alpha$ is exactly the *compressibility exponent*; $p=2/(2\alpha+1)$.
- **Refs.** A. Cohen, W. Dahmen, R. DeVore, "Compressed sensing and best $k$-term approximation," *J. Amer. Math. Soc.* 22 (2009) 211–231; E. Candès & T. Tao, "Near-optimal signal recovery from random projections," *IEEE Trans. IT* 52 (2006) 5406–5425; D. Donoho, "Compressed sensing," *IEEE Trans. IT* 52(4) (2006) 1289–1306; S. B. Stechkin (1955), on absolute convergence of orthogonal series.

### 3. Compressibility exponents in transform coding
- **Definition.** Decay rate of sorted transform coefficients governing rate–distortion of a transform coder [Mallat 2009, Ch. 9–10; Donoho 1993].
- **Relation to α.** The high-rate transform-coding distortion satisfies $D(R)$ tied to the same coefficient-decay rate; $\alpha$ controls the operational $D(R)$ slope.
- **Equivalence.** **Equivalent in substance** (it is the same coefficient-decay exponent under another community's name).
- **Refs.** S. Mallat, *A Wavelet Tour of Signal Processing: The Sparse Way*, 3rd ed., Academic Press (2009); D. Donoho, "Unconditional bases are optimal bases for data compression and for statistical estimation," *Appl. Comput. Harmon. Anal.* 1(1) (1993) 100–115.

### 4. Power-law spectra
- **Definition.** Power spectral density $S(f)\sim f^{-\beta}$ (e.g. $1/f^\beta$ processes) [Percival & Walden 1993; Mandelbrot & Van Ness 1968].
- **Relation to α.** Derived in Task 4: $\alpha=(\beta-1)/2$ for $\beta>1$ (else $\alpha\to0$).
- **Equivalence.** **Equivalent for monotone power-law spectra** (the regime where top-$k$-by-magnitude = lowest-$k$ frequencies).
- **Refs.** D. Percival & A. Walden, *Spectral Analysis for Physical Applications*, Cambridge (1993); B. Mandelbrot & J. Van Ness, "Fractional Brownian motions, fractional noises and applications," *SIAM Review* 10 (1968) 422–437.

### 5. Sobolev regularity
- **Definition.** $f\in H^s$ iff $\sum_m |\hat f_m|^2(1+|m|^2)^s<\infty$ [standard].
- **Relation to α.** Sobolev smoothness governs **linear** Fourier approximation: $f\in H^s\Rightarrow \|f-S_k f\|_2=O(k^{-s})$ ($S_k$ = first $k$ modes). For a monotone power-law spectrum, linear = nonlinear, so $\alpha=s=(\beta-1)/2$.
- **Equivalence.** **Partial** — equal to $\alpha$ only when best-$k$ = lowest-$k$ (monotone spectrum); for general spectra the nonlinear rate $\alpha$ can exceed the linear (Sobolev) rate.
- **Refs.** A. Pinkus, *n-Widths in Approximation Theory*, Springer (1985), Ch. on Sobolev classes; DeVore 1998 §3 (linear vs nonlinear).

### 6. Besov regularity
- **Definition.** $B^s_{q}(L^p)$ smoothness; in a wavelet ONB, membership ⟺ a weighted $\ell^{p,q}$ condition on coefficients [DeVore–Jawerth–Popov 1992].
- **Relation to α.** The **nonlinear** approximation rate characterizes Besov smoothness: in wavelet bases $\sigma_m(f)_2=O(m^{-\alpha})$ ⟺ $f\in B^{\alpha\cdot d}_{\tau}(L^\tau)$ with $1/\tau=\alpha+1/2$ (the "DeVore diagonal") [DeVore 1998, §7].
- **Equivalence.** **Equivalent in spirit** (rate ↔ Besov index) for unconditional/wavelet bases; for the **Fourier** basis the Besov characterization is *not* exact (Fourier is not an unconditional basis for $L^p$, $p\ne2$), so this is an analogy, not an identity, in our setting.
- **Refs.** R. DeVore, B. Jawerth, V. Popov, "Compression of wavelet decompositions," *Amer. J. Math.* 114(4) (1992) 737–785; DeVore 1998 §7.

### 7. Kolmogorov $n$-widths
- **Definition.** $d_n(K,X)=\inf_{\dim V=n}\sup_{f\in K}\inf_{g\in V}\|f-g\|_X$ — best **linear** $n$-dim approximation of a class $K$ [Pinkus 1985; Kolmogorov 1936].
- **Relation to α.** For Sobolev balls $d_n(H^s)\asymp n^{-s}$. $\alpha$ is the **nonlinear** analogue (best $k$-*term*, basis fixed), not the linear width; the two coincide only for monotone spectra.
- **Equivalence.** **Non-equivalent in general** (linear vs nonlinear width); parallel role.
- **Refs.** A. Pinkus, *n-Widths in Approximation Theory*, Springer (1985); A. Kolmogorov, "Über die beste Annäherung...," *Ann. of Math.* 37 (1936) 107–110.

### 8. Spectral entropy / spectral flatness (Wiener entropy)
- **Definition.** Normalised power $p_m=|\hat x_m|^2/\sum|\hat x|^2$; spectral entropy $H=-\sum p_m\ln p_m/\ln M\in[0,1]$; spectral flatness = geometric/arithmetic mean ratio [Gray & Markel 1974].
- **Relation to α.** A **bounded scalar functional** of the spectrum, not an asymptotic rate. Phase 6 measured a strong empirical anti-correlation ($r=-0.95$ across datasets, $-0.88$ pooled, $n=28$).
- **Equivalence.** **Non-equivalent** (different mathematical type: scalar vs decay exponent); **empirically correlated** via shared dependence on spectral concentration.
- **Refs.** A. Gray & J. Markel, "A spectral-flatness measure for studying the autocorrelation method of linear prediction of speech," *IEEE Trans. ASSP* 22(3) (1974) 207–217.

### 9. Rate–distortion theory
- **Definition.** $D(R)=\min_{p(\hat x|x): I\le R}\mathbb E\,d(x,\hat x)$ [Cover–Thomas 2006; Berger 1971].
- **Relation to α.** $\alpha$ is an **operational, single-realisation transform-coding** exponent, not Shannon's information-theoretic $D(R)$; high-rate transform-coding theory links coefficient decay to $D(R)$ but the two are distinct objects.
- **Equivalence.** **Non-equivalent** (operational best-$k$-term vs Shannon limit); related through high-rate transform coding.
- **Refs.** T. Cover & J. Thomas, *Elements of Information Theory*, 2nd ed., Wiley (2006); T. Berger, *Rate Distortion Theory*, Prentice-Hall (1971); A. Gersho & R. Gray, *Vector Quantization and Signal Compression*, Kluwer (1992).

**Summary of types:** $\alpha$ is *definitionally* the nonlinear best-$k$-term
approximation order (1); *equivalently* the Stechkin/weak-$\ell^p$ compressibility
exponent (2,3) with $\alpha=1/p-1/2$; *equivalent for monotone power-law spectra*
to a Sobolev index via $\alpha=(\beta-1)/2$ (4,5); *analogous* to Besov index and
to nonlinear $n$-widths (6,7); and merely *correlated* with the scalar spectral
entropy (8) and *related to but distinct from* Shannon rate–distortion (9).

---

## Task 3 — Is $D(k)\sim k^{-\alpha}$ a known theorem?

**Yes.** For an orthonormal basis it is **Stechkin's lemma** and its sharp
companions:

> If the coefficient sequence $c=(\langle f,\varphi_m\rangle)$ lies in
> $\ell^{p,\infty}$, $0<p<2$, then the best $m$-term approximation in $L^2$
> satisfies $\sigma_m(f)_2 \le C_p\,\|c\|_{\ell^{p,\infty}}\,m^{-(1/p-1/2)}$.
> Conversely, $\sigma_m(f)_2=O(m^{-\alpha})$ for all $f$ in the class iff the
> coefficients lie in the corresponding Lorentz space, with $\alpha=1/p-1/2$.
> [DeVore 1998, §7; Temlyakov 2011, Ch. on best $m$-term approximation.]

For **Sobolev/power-law classes** the matching two-sided estimate
$\sigma_k\asymp k^{-\alpha}$ is classical (upper bound from Stechkin, lower bound
from the $n$-width of the class) [Pinkus 1985; DeVore 1998 §3]. So $D(k)\sim
k^{-\alpha}$ is **not an empirical fit of a new law** — it is the expected form of a
known theorem, and the Phase-6 measurement of $\alpha$ estimates the exponent in
that theorem.

---

## Task 4 — Deriving α from $S(f)\sim f^{-\beta}$

Let the power spectrum decay as $S(f)\sim f^{-\beta}$, i.e. bin magnitudes
$|\hat x_m|\sim m^{-\beta/2}$. For a **monotone** spectrum the decreasing
rearrangement coincides with the frequency ordering, $p_{(j)}\sim j^{-\beta}$.
Then, with DC excluded,

$$
\sigma_k(x)_2^2 = \sum_{j>k} p_{(j)} \sim \int_k^\infty j^{-\beta}\,dj
= \frac{k^{-(\beta-1)}}{\beta-1}\quad(\beta>1),
$$

so

$$
\boxed{\;\alpha = \frac{\beta-1}{2}\quad(\beta>1);\qquad \alpha\to 0\ \ (\beta\le 1).\;}
$$

For $\beta\le1$ the tail sum diverges with $N$ (energy spreads over $\Theta(N)$
modes) — no scale-invariant decay — which is exactly why Phase 6 found
$\alpha_{\mathrm{Fourier}}\to0$ for gaussian ($\beta\approx0$) and lorenz
(broadband, $\beta$ small). The three classical relations agree:

| route | statement | consistency |
| :--- | :--- | :--- |
| power spectrum | $\alpha=(\beta-1)/2$ | — |
| Stechkin / weak-$\ell^p$ | $\alpha=1/p-1/2$, $\;p=2/\beta$ | $1/p-1/2=\beta/2-1/2=(\beta-1)/2$ ✓ |
| Sobolev (monotone) | $\alpha=s$, $\;f\in H^{s}$, $s<(\beta-1)/2$ | ✓ |

**Measured check.** Random walk is a $1/f^2$ process [Mandelbrot–Van Ness 1968],
$\beta=2\Rightarrow\alpha=(2-1)/2=0.5$. Phase 6 measured
$\alpha_{\mathrm{Fourier}}=0.507$–$0.533$ across $N$ — agreement with the theorem.
The sinusoidal case is a degenerate **line** spectrum (finite support); its
measured $\alpha\approx0.627$ reflects spectral **leakage** tails of non-integer
periods (Dirichlet-kernel decay), not a clean power law, and the power-law model
applies only approximately there.

---

## Task 5 — Final classification

> **A. α is a known quantity under another name.**

The codec-independent quantity $\alpha_{\mathrm{Fourier}}$ is, *by definition*, the
**best $k$-term (nonlinear) approximation rate** in the Fourier basis
[DeVore 1998], equal to the **compressibility / Stechkin exponent**
$\alpha=1/p-1/2$ for weak-$\ell^p$ coefficients [Cohen–Dahmen–DeVore 2009;
Temlyakov 2011], and to $(\beta-1)/2$ for power-law spectra (Task 4), coinciding
with a Sobolev smoothness index for monotone spectra [Pinkus 1985]. Its
governing law $D(k)\sim k^{-\alpha}$ is a known theorem (Task 3), and the
single empirical value that could be checked — random walk, $\beta=2\Rightarrow
\alpha=0.5$ — matches the prediction.

The **KSDZ** estimator of the same quantity is best described by **B (known
theory, different — and biased — estimator):** Phase 6 showed
$\alpha_{\mathrm{KSDZ}}$ over/under-estimates $\alpha_{\mathrm{Fourier}}$ because of
KSDZ's renormalisation, quantisation, and $k$-saturation. It measures the
established exponent through a non-standard instrument.

**C (potentially novel) is rejected:** the quantity is textbook nonlinear
approximation theory; no new descriptor is introduced. What Phases 5–6 added is an
*empirical measurement* of a classical exponent on specific synthetic signals
through a specific codec — engineering, not new mathematics.

**One sentence.** $\alpha$ is the best $k$-term approximation rate /
compressibility exponent of classical nonlinear approximation theory
($\alpha=1/p-1/2=(\beta-1)/2$), a known quantity governed by a known theorem;
KSDZ is merely a biased estimator of it.

---

## References

1. R. A. DeVore. *Nonlinear approximation.* Acta Numerica 7 (1998) 51–150.
2. R. A. DeVore, G. G. Lorentz. *Constructive Approximation.* Springer GMW 303 (1993).
3. V. N. Temlyakov. *Greedy Approximation.* Cambridge University Press (2011).
4. A. Cohen, W. Dahmen, R. DeVore. *Compressed sensing and best $k$-term approximation.* J. Amer. Math. Soc. 22 (2009) 211–231.
5. E. J. Candès, T. Tao. *Near-optimal signal recovery from random projections: universal encoding strategies?* IEEE Trans. Inf. Theory 52 (2006) 5406–5425.
6. D. L. Donoho. *Compressed sensing.* IEEE Trans. Inf. Theory 52(4) (2006) 1289–1306.
7. D. L. Donoho. *Unconditional bases are optimal bases for data compression and for statistical estimation.* Appl. Comput. Harmon. Anal. 1(1) (1993) 100–115.
8. R. A. DeVore, B. Jawerth, V. Popov. *Compression of wavelet decompositions.* Amer. J. Math. 114(4) (1992) 737–785.
9. S. Mallat. *A Wavelet Tour of Signal Processing: The Sparse Way*, 3rd ed. Academic Press (2009).
10. A. Pinkus. *n-Widths in Approximation Theory.* Springer (1985).
11. A. N. Kolmogorov. *Über die beste Annäherung von Funktionen einer gegebenen Funktionenklasse.* Ann. of Math. 37 (1936) 107–110.
12. D. B. Percival, A. T. Walden. *Spectral Analysis for Physical Applications.* Cambridge University Press (1993).
13. B. B. Mandelbrot, J. W. Van Ness. *Fractional Brownian motions, fractional noises and applications.* SIAM Review 10 (1968) 422–437.
14. A. H. Gray, J. D. Markel. *A spectral-flatness measure for studying the autocorrelation method of linear prediction of speech signals.* IEEE Trans. Acoust. Speech Signal Process. 22(3) (1974) 207–217.
15. T. M. Cover, J. A. Thomas. *Elements of Information Theory*, 2nd ed. Wiley (2006).
16. T. Berger. *Rate Distortion Theory.* Prentice-Hall (1971).
17. A. Gersho, R. M. Gray. *Vector Quantization and Signal Compression.* Kluwer (1992).
18. S. B. Stechkin. *On absolute convergence of orthogonal series.* Dokl. Akad. Nauk SSSR 102 (1955) 37–40 (in Russian).

*Bibliographic note: citations give author/title/venue/year for traceability;
where a specific theorem is attributed (Stechkin's lemma), it is located via the
survey references [1, 3] that reproduce it. No claim above relies on the KSDZ
implementation; all are standard approximation-theory results.*
