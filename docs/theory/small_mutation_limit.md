# Small-mutation limit

This page specifies the small-mutation limit (SML) of the pairwise comparison process: the
reduced Markov chain on the homogeneous states, its stationary distribution, and the invasion
graph drawn from it. It builds on [the pairwise comparison process](pairwise_comparison.md)
(cited as PC 3.2, PC 4.1, ...) and on [fixation and absorption](fixation.md) (cited as F 2.2,
...). Every function listed in [Implementations](#implementations) computes one of the numbered
definitions below, and `tests/test_sml_equations.py` checks each one against a direct
transcription of these equations.

Notation as on those pages; $\rho_{a \to b}$ is the fixation probability of one $a$-mutant in a
population of $b$ (F 2.1).

## 1. The limit

With mutation probability $\mu > 0$ the process PC 3.2 has a unique stationary distribution
$\pi^{(\mu)}$ over all states. When mutations are rare, a mutant either fixes or goes extinct
long before the next mutation, so the population is homogeneous almost all the time, and it
moves from one homogeneous state to another through the fixation of a single mutant.

**Theorem 1.1 (Fudenberg & Imhof 2006).** If every fixation probability $\rho_{a \to b}$ is
positive (true for the Fermi rule with finite $\beta$), then as $\mu \to 0$, $\pi^{(\mu)}$
concentrates on the homogeneous states, and its restriction to them converges to the stationary
distribution of the reduced chain of Definition 2.2. The error is of order $\mu$.

## 2. The reduced chain

**Definition 2.1 (Fixation matrix).** $\rho$ is the $k \times k$ matrix with
$\rho_{ij} = \rho_{j \to i}$ for $i \ne j$: the probability that one $j$-mutant fixes in a
population of $i$ (F 2.2). Its diagonal is not used.

**Definition 2.2 (SML transition matrix).** One step of the reduced chain is one mutation: in a
population of $i$, a mutant of one of the other $k - 1$ strategies, chosen uniformly (PC 3.2),
appears and either fixes or goes extinct. The transition matrix is

$$
T_{ij} = \frac{\rho_{ij}}{k - 1} \quad (i \ne j), \qquad T_{ii} = 1 - \sum_{j \ne i} T_{ij}
$$

(Fudenberg & Imhof 2006; Imhof, Fudenberg & Nowak 2005). $T$ is row-stochastic: row $i$ holds
the transitions out of state $i$.

**Definition 2.3 (Computation in log space).** The log-space variant computes $\log \rho_{ij}$
with log-sum-exp (F 2.3) and returns $T^{(c)} = I + c\,(T - I)$ with
$c = e^{-\max_{i \ne j} \log \rho_{ij}}$, which has the same stationary distribution as $T$.

## 3. Stationary distribution

**Definition 3.1 (SML stationary distribution).** $\pi$ is the probability vector with
$\pi T = \pi$; $\pi_i$ is the long-run fraction of time the population spends homogeneous in
strategy $i$. It is unique when every $\rho_{ij} > 0$.

The functions in `egttools.utils` take the column-stochastic matrix $T^\top$.

## 4. Invasion graph

**Definition 4.1 (Invasion graph).** With the neutral fixation probability $\rho_N = 1/Z$ and the
ratios $r_{ij} = \rho_{ij} / \rho_N$, the invasion graph has one node per strategy and an edge
$i \to j$ when $j$ invades $i$ by selection, $r_{ij} > 1$ (Imhof, Fudenberg & Nowak 2005; Hauert
et al. 2007). EGTtools classifies each pair once, with a tolerance $\varepsilon$ (default
$0.005$, so that ratios shown as $1.00$ count as neutral):

- an edge $i \to j$ for every direction with $r_{ij} > 1 + \varepsilon$ (both if each invades
  the other);
- one undirected drift edge if neither direction exceeds $1 + \varepsilon$ and at least one is
  within $\varepsilon$ of 1;
- no edge otherwise.

Node sizes or labels show $\pi$ (Definition 3.1).

## Implementations

| Function | Definition | Status |
|----------|------------|--------|
| `egttools.analytical.PairwiseComparison.calculate_transition_and_fixation_matrix_sml` | 2.1, 2.2 | verified |
| `egttools.analytical.PairwiseComparison.calculate_transition_and_log_fixation_matrix_sml` | 2.3 | verified |
| `egttools.analytical.StochDynamics.transition_and_fixation_matrix` | 2.1, 2.2 | verified; returns $T^\top$ (API-06) |
| `egttools.analytical.StochDynamics.transition_and_log_fixation_matrix` | 2.3 | verified; returns $(T^{(c)})^\top$ (API-06) |
| `egttools.analytical.StochDynamics.calculate_stationary_distribution` (with `mu = 0`) | 3.1 | verified |
| `egttools.utils.calculate_stationary_distribution` (input $T^\top$) | 3.1 | verified |
| `egttools.utils.calculate_stationary_distribution_non_hermitian` (input $T^\top$) | 3.1 | verified |
| `egttools.plotting.invasion_graph` | 4.1 | verified (`tests/test_invasion_diagram.py`) |

`StochDynamics` returns the transposed (column-stochastic) matrices, while `PairwiseComparison`
returns row-stochastic ones; API-06 makes them consistent. With `mu > 0`,
`StochDynamics.calculate_stationary_distribution` solves the full chain of PC 4.1 instead.

"Verified" means agreement with a direct transcription for two-player and three-player games
with three and four strategies and $\beta$ from 0 to strong selection; Theorem 1.1 is checked on
the full chain.

## References

- Fudenberg, D. & Imhof, L. A. (2006). Imitation processes with small mutations.
  *Journal of Economic Theory* 131(1), 251–262.
- Hauert, C., Traulsen, A., Brandt, H., Nowak, M. A. & Sigmund, K. (2007). Via freedom to
  coercion: the emergence of costly punishment. *Science* 316(5833), 1905–1907.
- Imhof, L. A., Fudenberg, D. & Nowak, M. A. (2005). Evolutionary cycles of cooperation and
  defection. *PNAS* 102(31), 10797–10800.
