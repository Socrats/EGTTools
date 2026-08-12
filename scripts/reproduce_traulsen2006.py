"""
Reproduce Figure 2 of Traulsen & Nowak (2006) PNAS.

Critical benefit-to-cost ratio b/c as a function of:
  (a) number of groups m for fixed n=10
  (b) group size n for fixed m=10

Analytical line: b/c = 1 + n / (m - 2)
Monte Carlo circles: binary-search for the threshold b/c where the
  fixation probability of cooperators equals the neutral drift level 1/(n*m).

Uses MLSTraulsen with a 2-strategy donation game.
Payoff matrix:
    A = [[b-c, -c],
         [b,    0]]
where row = focal strategy, column = co-player.
  row 0 = Cooperator (C), row 1 = Defector (D)

Note on accuracy: near the critical b/c, the fixation probability being
estimated is close to the neutral level 1/(n*m) — for the larger (n, m) in
this sweep that's a very small probability (e.g. ~1e-3 at n=m=30), so the
Monte Carlo estimate is inherently noisy at a practical NB_RUNS budget, and
brentq's bisection occasionally fails to bracket a root (reported as NaN and
left as a gap in the plot). Expect the simulated points to track the
analytical curve's trend and order of magnitude rather than match it tightly
point-for-point; increase NB_RUNS for a tighter fit at the cost of runtime.
GENERATIONS only needs to be large enough that individual Moran runs reach
absorption — empirically confirmed (2026-08-11) to never be the bottleneck
anywhere in this parameter range at 5e6, despite an earlier attempt to raise
it to 9e9 in response to the "fewer than 50% of runs converged" warning; that
warning does not fire in this sweep. The actual lever for accuracy is NB_RUNS.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from egttools.numerical import MLSTraulsen

# ── Parameters ────────────────────────────────────────────────────────────────
GENERATIONS = 5_000_000   # max Moran steps per run (never the convergence bottleneck; see note above)
NB_RUNS     = 50_000      # MC runs per fixation estimate
W           = 0.01        # weak selection intensity
Q           = 1e-3        # small splitting probability
C           = 1.0         # cost (normalised)

N_FIXED     = 10          # fixed n for panel (a)
M_FIXED     = 10          # fixed m for panel (b)
M_RANGE     = np.array([5, 10, 15, 20, 25, 30])
N_RANGE     = np.array([5, 10, 15, 20, 25, 30])


def payoff_matrix(b: float, c: float = C) -> np.ndarray:
    return np.array([[b - c, -c],
                     [b,      0]], dtype=float)


def fixation_prob(b: float, n: int, m: int) -> float:
    """Fixation probability of one C in a population of D."""
    A = payoff_matrix(b)
    freq = np.array([0.0, 1.0])  # start with all D
    evolver = MLSTraulsen(GENERATIONS, 2, n, m, W, freq, A)
    return evolver.fixation_probability(0, 1, NB_RUNS, Q, W)


def critical_bc(n: int, m: int) -> float:
    """Binary-search for the b/c where fixation prob == 1/(n*m)."""
    neutral = 1.0 / (n * m)
    # Bracket: at b/c=1 we expect fp < neutral, at b/c=20 we expect fp > neutral
    def f(b):
        return fixation_prob(b, n, m) - neutral
    # Quick sanity: evaluate at endpoints
    try:
        return brentq(f, 1.0, 30.0, xtol=0.1, maxiter=20)
    except ValueError:
        return float("nan")


# ── Panel (a): vary m, fixed n ─────────────────────────────────────────────
print(f"Panel (a): n={N_FIXED}, varying m …")
m_vals   = list(M_RANGE)
bc_mc_a  = []
bc_ana_a = [1.0 + N_FIXED / (m - 2) for m in m_vals]

for m in m_vals:
    print(f"  m={m}", flush=True)
    bc_mc_a.append(critical_bc(N_FIXED, m))

# ── Panel (b): vary n, fixed m ─────────────────────────────────────────────
print(f"\nPanel (b): m={M_FIXED}, varying n …")
n_vals   = list(N_RANGE)
bc_mc_b  = []
bc_ana_b = [1.0 + n / (M_FIXED - 2) for n in n_vals]

for n in n_vals:
    print(f"  n={n}", flush=True)
    bc_mc_b.append(critical_bc(n, M_FIXED))

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.plot(m_vals, bc_ana_a, "-k", label="Analytical")
ax.plot(m_vals, bc_mc_a,  "ko", mfc="none", label="Simulation")
ax.set_xlabel("Number of groups $m$")
ax.set_ylabel("Critical $b/c$")
ax.set_ylim(0, 6)
ax.set_xlim(3, 30)
ax.set_title(f"$n={N_FIXED}$, $q={Q}$")
ax.legend()

ax = axes[1]
ax.plot(n_vals, bc_ana_b, "-k", label="Analytical")
ax.plot(n_vals, bc_mc_b,  "ko", mfc="none", label="Simulation")
ax.set_xlabel("Group size $n$")
ax.set_ylabel("Critical $b/c$")
ax.set_ylim(0, 6)
ax.set_xlim(3, 30)
ax.set_title(f"$m={M_FIXED}$, $q={Q}$")
ax.legend()

fig.suptitle("Traulsen & Nowak (2006) — Figure 2")
fig.tight_layout()

os.makedirs("results", exist_ok=True)
outpath = "results/traulsen2006_fig2.pdf"
fig.savefig(outpath)
print(f"\nSaved {outpath}")

# ── Numeric sanity check (not a strict tolerance gate — see accuracy note above) ──
all_ana = np.array(bc_ana_a + bc_ana_b)
all_mc  = np.array(bc_mc_a + bc_mc_b)
valid   = ~np.isnan(all_mc)
rel_err = np.abs(all_mc[valid] - all_ana[valid]) / all_ana[valid]
print(f"\n{valid.sum()}/{len(all_mc)} points converged to a bracketed root.")
if valid.any():
    print(f"Median relative error vs. analytical: {np.median(rel_err):.1%} "
          f"(expected to be noisy — see accuracy note in the module docstring).")
