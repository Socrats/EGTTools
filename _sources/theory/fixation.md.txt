# Fixation and absorption under the pairwise comparison process

This page specifies the fixation probabilities, absorption probabilities and absorption
times of the pairwise comparison process without mutation, and how EGTtools computes
them. It builds on [the pairwise comparison process](pairwise_comparison.md), whose
definitions are cited as PC 2.1, PC 4.1, etc. Every function listed in
[Implementations](#implementations) computes one of the numbered definitions below, and
`tests/test_fixation_equations.py` checks each one against a direct transcription of
these equations.

Notation as in the pairwise comparison page: $Z$ is the population size, $k$ the number of
strategies, $\beta \ge 0$ the intensity of selection, and $\phi$ the Fermi imitation
probability (PC 3.1). Throughout this page $\mu = 0$.

## 1. Absorbing states and time

**Definition 1.1 (Absorbing states).** Without mutation, a strategy that is absent never
reappears, and a homogeneous population never changes. The $k$ homogeneous states
$Z \mathbf e_l$ are absorbing; every other state is transient, and the process reaches a
homogeneous state with probability 1.

**Definition 1.2 (Time).** Time is counted in elementary updates (one step of PC 3.2). A
generation is $Z$ elementary updates, so that each individual is updated once per
generation on average.

## 2. Fixation of a single invader

**Definition 2.1 (Fixation probability).** The fixation probability $\rho_{a \to b}$ of
strategy $a$ in a population of strategy $b$ is the probability that the process started in
the state with one $a$-strategist and $Z - 1$ $b$-strategists,
$\mathbf e_a + (Z - 1)\,\mathbf e_b$, is absorbed in $Z \mathbf e_a$.

Only $a$ and $b$ can be present along the way, so the process is a birth-death chain on the
number $i \in \{0, \dots, Z\}$ of $a$-strategists, with state
$\mathbf n_i = i\,\mathbf e_a + (Z - i)\,\mathbf e_b$ and, by PC 4.1 with $\mu = 0$,

$$
T^+(i) = T_{b \to a}(\mathbf n_i) = \frac{i}{Z}\,\frac{Z - i}{Z - 1}\,
\phi\big(f_b(\mathbf n_i), f_a(\mathbf n_i)\big), \qquad
T^-(i) = T_{a \to b}(\mathbf n_i) = \frac{i}{Z}\,\frac{Z - i}{Z - 1}\,
\phi\big(f_a(\mathbf n_i), f_b(\mathbf n_i)\big).
$$

With the Fermi rule their ratio depends only on the fitness difference
$\Delta f(i) = f_a(\mathbf n_i) - f_b(\mathbf n_i)$:

$$
\frac{T^-(i)}{T^+(i)} = e^{-\beta\,\Delta f(i)} .
$$

**Proposition 2.2 (Closed form).** For a birth-death chain absorbing at $0$ and $Z$, the
probability of absorption at $Z$ from $i$ is (Karlin & Taylor 1975; Nowak 2006)

$$
\rho(i) = \frac{1 + \sum_{m=1}^{i-1} \prod_{j=1}^{m} \frac{T^-(j)}{T^+(j)}}
{1 + \sum_{m=1}^{Z-1} \prod_{j=1}^{m} \frac{T^-(j)}{T^+(j)}} ,
$$

and for a single invader, with the Fermi rule (Traulsen, Nowak & Pacheco 2006),

$$
\rho_{a \to b} = \rho(1) = \frac{1}{1 + \sum_{m=1}^{Z-1} e^{L_m}}, \qquad
L_m = -\beta \sum_{j=1}^{m} \Delta f(j) .
$$

Under neutral selection ($\beta = 0$ or $\Delta f \equiv 0$), $\rho_{a \to b} = 1/Z$.

**Definition 2.3 (Computation in log space).** EGTtools computes
$\Lambda = \log \sum_{m=1}^{Z-1} e^{L_m}$ with log-sum-exp and returns
$\rho_{a \to b} = 1 / (1 + e^{\Lambda})$ and $\log \rho_{a \to b} = -\log(1 + e^{\Lambda})$.
The high-precision variant evaluates the same expression in 50-digit decimal arithmetic.

## 3. Absorption with several strategies

**Definition 3.1 (Absorption probabilities).** For a state $\mathbf n$, $B_l(\mathbf n)$ is the
probability that the process started at $\mathbf n$ is absorbed in $Z \mathbf e_l$. For a single
invader, $B_a\big(\mathbf e_a + (Z - 1)\,\mathbf e_b\big) = \rho_{a \to b}$.

**Definition 3.2 (Absorption time).** $\tau(\mathbf n)$ is the number of elementary updates
(Definition 1.2) until the process started at $\mathbf n$ first reaches a homogeneous state;
$t(\mathbf n) = \mathbb E[\tau(\mathbf n)]$ is the mean absorption time.

**Proposition 3.3 (Absorbing chain).** Order the transient states first and write the
transition matrix of PC 4.1 (with $\mu = 0$) as

$$
P = \begin{pmatrix} Q & R \\ 0 & I \end{pmatrix} ,
$$

with $Q$ the transitions among transient states and $R$ those from transient to absorbing
states. With the fundamental matrix $N = (I - Q)^{-1}$ (Kemeny & Snell 1960),

$$
B = N R, \qquad t = N \mathbf 1, \qquad
\mathrm{Var}[\tau] = (2N - I)\,t - t \circ t ,
$$

where $\circ$ is the entrywise product. For $k = 2$, $B$ agrees with Proposition 2.2.

## 4. Monte Carlo estimators

The estimators simulate PC 3.2 with $\mu = 0$ from a given state until absorption, in
independent runs.

**Definition 4.1 (Fixation probability estimate).** Each run starts from one invader and
$Z - 1$ residents and simulates at most a given number of elementary updates. Runs that are
not absorbed within that limit are discarded, and the estimate is the fraction of the
remaining runs absorbed in $Z \mathbf e_a$ (0 if every run was discarded). It estimates
$\rho_{a \to b}$ when the limit is long enough for every run to be absorbed; otherwise it is
conditional on absorption within the limit.

**Definition 4.2 (Absorption probability estimate).** Each run continues until absorption,
without a limit; the estimate of $B_l(\mathbf n)$ is the fraction of runs absorbed in
$Z \mathbf e_l$.

**Definition 4.3 (Absorption time estimate).** Each run continues until absorption, without
a limit; the estimate of $t(\mathbf n)$ is the mean of $\tau(\mathbf n)$ over the runs, with
its standard error (sample standard deviation over $\sqrt{\text{runs}}$).

Runs of Definitions 4.2 and 4.3 have no step limit, and the absorption time can grow
exponentially with $Z$ under strong selection in coexistence games (Antal & Scheuring 2006).

## Implementations

| Function | Definition | Status |
|----------|------------|--------|
| `egttools.analytical.PairwiseComparison.calculate_fixation_probability` | 2.2, 2.3 | verified |
| `egttools.analytical.PairwiseComparison.calculate_fixation_probability(..., high_precision=True)` | 2.2, 2.3 | verified |
| `egttools.analytical.PairwiseComparison.calculate_log_fixation_probability` | 2.3 | verified |
| `egttools.numerical.PairwiseComparisonNumerical.estimate_fixation_probability` | 4.1 | verified |
| `egttools.numerical.PairwiseComparisonNumerical.estimate_absorption_probabilities` | 4.2 | verified |
| `egttools.numerical.PairwiseComparisonNumerical.estimate_mean_absorption_time` | 4.3 | verified |

"Verified" means agreement with a direct transcription for two-player and three-player games,
every pair of strategies and $\beta$ from 0 to strong selection; the estimators agree with
Propositions 2.2 and 3.3 within four standard errors, with fixed seeds.

The fixation probabilities on networks (`egttools.numerical.structure`) follow a different
process and are specified on the networks page.

## References

- Antal, T. & Scheuring, I. (2006). Fixation of strategies for an evolutionary game in
  finite populations. *Bulletin of Mathematical Biology* 68(8), 1923–1944.
- Karlin, S. & Taylor, H. M. (1975). *A First Course in Stochastic Processes*, 2nd ed.
  Academic Press.
- Kemeny, J. G. & Snell, J. L. (1960). *Finite Markov Chains*. Van Nostrand.
- Nowak, M. A. (2006). *Evolutionary Dynamics: Exploring the Equations of Life*. Harvard
  University Press.
- Traulsen, A., Nowak, M. A. & Pacheco, J. M. (2006). Stochastic dynamics of invasion and
  fixation. *Physical Review E* 74(1), 011909.
