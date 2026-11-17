# Pairwise comparison process in finite well-mixed populations

This page specifies the stochastic process that EGTtools implements for evolution
by social learning (imitation) in a finite, well-mixed population, and the
quantities derived from it. Every function listed in
[Implementations](#implementations) must compute exactly one of the numbered
definitions below, and the test suite checks each one against a direct
transcription of these equations (`tests/test_pairwise_comparison_equations.py`).

Notation: $Z$ is the population size, $k$ the number of strategies, $\beta \ge 0$
the intensity of selection, and $\mu \in [0, 1]$ the mutation (exploration)
probability.

## 1. State space

**Definition 1.1 (Population state).** The state is the vector of strategy counts

$$
\mathbf n = (n_1, \dots, n_k) \in \mathbb Z_{\ge 0}^k, \qquad \sum_{l=1}^k n_l = Z .
$$

The state space $S$ has $|S| = \binom{Z + k - 1}{k - 1}$ elements. EGTtools indexes
states with `egttools.calculate_state` and its inverse `egttools.sample_simplex`.
$\mathbf e_l$ denotes the $l$-th unit vector.

## 2. Fitness

The fitness of an individual is its expected payoff when interacting with the
other $Z - 1$ members of the population, so the focal individual is excluded.

**Definition 2.1 (Two-player games).** For a payoff matrix $A \in \mathbb R^{k \times k}$,
where $A_{lm}$ is the payoff of strategy $l$ against strategy $m$,

$$
f_l(\mathbf n) = \sum_{m=1}^k A_{lm} \, \frac{n_m - \delta_{lm}}{Z - 1} .
$$

**Definition 2.2 (N-player games).** For groups of size $N$ formed by sampling
without replacement, with $a_l(\mathbf g)$ the payoff of a focal $l$-strategist in a
group whose other $N - 1$ members have composition $\mathbf g$
($\sum_m g_m = N - 1$),

$$
f_l(\mathbf n) = \sum_{\mathbf g} a_l(\mathbf g) \,
\frac{\prod_{m} \binom{n_m - \delta_{lm}}{g_m}}{\binom{Z - 1}{N - 1}} ,
$$

i.e. the expected payoff under the multivariate hypergeometric distribution of
co-players (Pacheco et al. 2009; Gokhale & Traulsen 2010). For $N = 2$ this
reduces to Definition 2.1.

## 3. The update rule

**Definition 3.1 (Fermi imitation probability).** A focal individual with fitness
$f_{\text{focal}}$ adopts the strategy of a role model with fitness
$f_{\text{model}}$ with probability

$$
\phi(f_{\text{focal}}, f_{\text{model}}) =
\frac{1}{1 + e^{-\beta (f_{\text{model}} - f_{\text{focal}})}}
$$

(Szabó & Tőke 1998; Traulsen, Nowak & Pacheco 2006). Note
$\phi(a, b) - \phi(b, a) = \tanh\!\big(\tfrac{\beta}{2}(b - a)\big)$.

**Definition 3.2 (Pairwise comparison process with mutation).** One time step
(one elementary update) consists of:

1. Select a focal individual uniformly at random, and a role model uniformly at
   random among the remaining $Z - 1$ individuals.
2. With probability $\mu$, the focal individual mutates: it switches to one of
   the other $k - 1$ strategies, chosen uniformly at random.
3. Otherwise, it adopts the role model's strategy with probability
   $\phi(f_{\text{focal}}, f_{\text{model}})$ (Definition 3.1).

The mutation probability $\mu$ is the total probability of a mutation per update,
split uniformly over the other $k-1$ strategies (Fudenberg & Imhof 2006; Traulsen,
Nowak & Pacheco 2006). With $\mu > 0$ the chain is irreducible and has a unique
stationary distribution.

## 4. Transition probabilities

**Definition 4.1 (One-step transitions).** For $j \ne l$ with $n_j > 0$, the
probability that one $j$-strategist becomes an $l$-strategist is

$$
T_{j \to l}(\mathbf n) \equiv P(\mathbf n \to \mathbf n - \mathbf e_j + \mathbf e_l)
= \frac{n_j}{Z} \left[ (1 - \mu) \, \frac{n_l}{Z - 1} \,
\phi\big(f_j(\mathbf n), f_l(\mathbf n)\big) + \frac{\mu}{k - 1} \right] ,
$$

and $P(\mathbf n \to \mathbf n) = 1 - \sum_{j \ne l} T_{j \to l}(\mathbf n)$. All other
transitions have probability zero. This follows from Definition 3.2: the focal
individual is a $j$-strategist with probability $n_j / Z$; it then either mutates
to $l$ (probability $\mu / (k-1)$) or, without mutation, picks an $l$-strategist
as role model (probability $n_l / (Z - 1)$) and imitates it.

## 5. Gradient of selection

**Definition 5.1 (Gradient of selection).** The gradient of selection is the
expected change of the state in one time step,

$$
G_l(\mathbf n) = \mathbb E\big[n_l(t+1) - n_l(t) \,\big|\, \mathbf n(t) = \mathbf n\big]
= \sum_{j \ne l} \Big( T_{j \to l}(\mathbf n) - T_{l \to j}(\mathbf n) \Big) ,
$$

i.e. the probability of gaining minus the probability of losing an
$l$-strategist, $T_l^+ - T_l^-$ (Traulsen, Nowak & Pacheco 2006; Pacheco et al. 2009;
Santos & Pacheco 2011). It is the drift term of the process and needs only the
fitnesses at $\mathbf n$; no solution of the master equation is required.
Substituting Definition 4.1:

$$
G_l(\mathbf n) = (1 - \mu) \sum_{j \ne l} \frac{n_l \, n_j}{Z (Z - 1)}
\tanh\!\Big(\frac{\beta}{2}\big(f_l(\mathbf n) - f_j(\mathbf n)\big)\Big)
\;+\; \frac{\mu}{k - 1} \, \frac{Z - k \, n_l}{Z} .
$$

The first term is the selection part and the second the mutation part. Both
parts sum to zero over $l$, so $\sum_l G_l(\mathbf n) = 0$. For $k = 2$ and
$\mu = 0$ this is the classical $g(n) = T^+(n) - T^-(n)$ for the number $n$ of
individuals of one strategy.

## Implementations

| Function | Definition | Status |
|----------|------------|--------|
| `egttools.games.*.calculate_fitness` (2-player and N-player holders) | 2.1, 2.2 | verified |
| `egttools.analytical.PairwiseComparison.calculate_transition_matrix` | 4.1 | verified |
| `egttools.analytical.PairwiseComparison.calculate_gradient_of_selection` | 5.1, $\mu = 0$ | verified |
| `egttools.analytical.PairwiseComparison.calculate_gradient_of_selection_with_mutation` | 5.1 | verified |
| `egttools.analytical.StochDynamics.full_gradient_selection` | 5.1 | verified |
| `egttools.analytical.StochDynamics.full_gradient_selection_without_mutation` | 5.1, $\mu = 0$ | verified |
| `egttools.numerical.PairwiseComparisonNumerical.run_with_mutation` | 3.2 | verified |
| `egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution{,_sparse}` | 3.2 | verified |
| `egttools.numerical.PairwiseComparisonNumerical.estimate_strategy_distribution` | 3.2 | verified |
| `egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_indicators` | 3.2 | verified |

"Verified" means the function agrees with a direct transcription of its
definition to floating-point precision for $k \in \{2, 3, 4\}$, $\mu \in \{0, 0.05, 0.3\}$,
and every state of small populations. The simulation and the Monte Carlo
estimators are checked statistically: the one-step drift of a simulated trajectory
against Definition 5.1, and the estimates against the exact stationary distribution
of Definition 4.1, each within five standard errors.

**Simulating homogeneous populations.** In a homogeneous population only a
mutation can change the state, so the simulation draws the number of updates
until the next mutation at once instead of simulating them one by one. Under
Definition 3.2 the population stays unchanged for $K \sim \mathrm{Geometric}(\mu)$
updates (the number of failures before the first success) and mutates on update
$K + 1$; if the observation window ends first, the population stays homogeneous
until its end.

## References

- Fudenberg, D. & Imhof, L. A. (2006). Imitation processes with small mutations.
  *Journal of Economic Theory* 131(1), 251–262.
- Gokhale, C. S. & Traulsen, A. (2010). Evolutionary games in the multiverse.
  *PNAS* 107(12), 5500–5504.
- Pacheco, J. M., Santos, F. C., Souza, M. O. & Skyrms, B. (2009). Evolutionary
  dynamics of collective action in N-person stag hunt dilemmas.
  *Proceedings of the Royal Society B* 276(1655), 315–321.
- Santos, F. C. & Pacheco, J. M. (2011). Risk of collective failure provides an
  escape from the tragedy of the commons. *PNAS* 108(26), 10421–10425.
- Szabó, G. & Tőke, C. (1998). Evolutionary prisoner's dilemma game on a square
  lattice. *Physical Review E* 58(1), 69–73.
- Traulsen, A., Nowak, M. A. & Pacheco, J. M. (2006). Stochastic dynamics of
  invasion and fixation. *Physical Review E* 74(1), 011909.
- Traulsen, A., Pacheco, J. M. & Nowak, M. A. (2007). Pairwise comparison and
  selection temperature in evolutionary game dynamics. *Journal of Theoretical
  Biology* 246(3), 522–529.
