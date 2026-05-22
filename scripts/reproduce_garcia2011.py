"""
Reproduce Figures 2 and 4 of García & van den Bergh (2011).

4-strategy game: Altruist (A), Parochialist (P), Traitor (T), Egoist (E).
  A = (C in-group, C out-group)
  P = (C in-group, D out-group)
  T = (D in-group, C out-group)
  E = (D in-group, D out-group)

In-group payoffs (Prisoner's Dilemma):
    A_in[i,j] = b*delta(j is C-in) - c*delta(i is C-in)
Out-group payoffs (Prisoner's Dilemma):
    A_out[i,j] = b*delta(j is C-out) - c*delta(i is C-out)

Default parameters (Table 1 of García & van den Bergh 2011):
  n=10, m=10, q=0.01, alpha=0.8, b/c=2, w=0.1, kappa=0, z=0, lambda=0
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from egttools.numerical import MLSGarcia

# ── Strategy indices ──────────────────────────────────────────────────────────
ALTRUIST     = 0   # (C, C)
PAROCHIALIST = 1   # (C, D)
TRAITOR      = 2   # (D, C)
EGOIST       = 3   # (D, D)
NB_STRAT     = 4

# in-group cooperation flags: A, P = cooperate; T, E = defect
C_IN  = np.array([1, 1, 0, 0], dtype=float)
# out-group cooperation flags: A, T = cooperate; P, E = defect
C_OUT = np.array([1, 0, 1, 0], dtype=float)


def build_payoff_matrices(b: float, c: float):
    """
    Return (A_in, A_out) payoff matrices of shape (4,4).
    A_in[i,j]  = b * C_IN[j]  - c * C_IN[i]
    A_out[i,j] = b * C_OUT[j] - c * C_OUT[i]
    """
    A_in  = np.outer(np.ones(NB_STRAT), b * C_IN)  - np.outer(c * C_IN,  np.ones(NB_STRAT))
    A_out = np.outer(np.ones(NB_STRAT), b * C_OUT) - np.outer(c * C_OUT, np.ones(NB_STRAT))
    return A_in.copy(), A_out.copy()


def fixation_AB(invader: int, resident: int, n: int, m: int,
                b: float, c: float = 1.0,
                q: float = 0.01, lam: float = 0.0, w: float = 0.1,
                alpha: float = 0.8, kappa: float = 0.0, z: float = 0.0,
                nb_runs: int = 3_000, generations: int = 5_000_000) -> float:
    A_in, A_out = build_payoff_matrices(b, c)
    evolver = MLSGarcia(generations, NB_STRAT, n, m)
    return evolver.fixation_probability(
        invader, resident, nb_runs,
        q, lam, w, alpha, kappa, z,
        A_in, A_out)


# ── Figure 2: fixation probability vs b/c ─────────────────────────────────
print("Figure 2: fixation probability vs b/c …")
N, M = 10, 10
BC_RANGE = np.linspace(1.0, 5.0, 10)
neutral  = 1.0 / (N * M)

fp_A_over_P = []
fp_A_over_E = []

for bc in BC_RANGE:
    print(f"  b/c={bc:.2f}", flush=True)
    fp_A_over_P.append(fixation_AB(ALTRUIST, PAROCHIALIST, N, M, b=bc))
    fp_A_over_E.append(fixation_AB(ALTRUIST, EGOIST,       N, M, b=bc))

# ── Figure 4a: fixation prob vs group size n ──────────────────────────────
print("\nFigure 4a: vs group size n …")
BC_DEFAULT = 2.0
N_RANGE = range(4, 21, 2)
fp_n_AP = []
fp_n_AE = []

for n in N_RANGE:
    print(f"  n={n}", flush=True)
    fp_n_AP.append(fixation_AB(ALTRUIST, PAROCHIALIST, n, M, b=BC_DEFAULT))
    fp_n_AE.append(fixation_AB(ALTRUIST, EGOIST,       n, M, b=BC_DEFAULT))

# ── Figure 4b: fixation prob vs number of groups m ───────────────────────
print("\nFigure 4b: vs number of groups m …")
M_RANGE = range(3, 21, 2)
fp_m_AP = []
fp_m_AE = []

for m in M_RANGE:
    print(f"  m={m}", flush=True)
    fp_m_AP.append(fixation_AB(ALTRUIST, PAROCHIALIST, N, m, b=BC_DEFAULT))
    fp_m_AE.append(fixation_AB(ALTRUIST, EGOIST,       N, m, b=BC_DEFAULT))

# ── Plot Figure 2 ─────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(6, 4))
ax.axhline(neutral, color="gray", linestyle="--", label="Neutral drift $1/mn$")
ax.plot(BC_RANGE, fp_A_over_P, "b-o", label="Altruist vs Parochialist")
ax.plot(BC_RANGE, fp_A_over_E, "r-s", label="Altruist vs Egoist")
ax.set_xlabel("Benefit-to-cost ratio $b/c$")
ax.set_ylabel("Fixation probability")
ax.set_title("García & van den Bergh (2011) — Figure 2")
ax.legend()
fig2.tight_layout()

# ── Plot Figure 4 ─────────────────────────────────────────────────────────
fig4, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
neutral_n = [1.0 / (n * M) for n in N_RANGE]
ax.plot(list(N_RANGE), neutral_n,  "k--", label="Neutral drift")
ax.plot(list(N_RANGE), fp_n_AP, "b-o", label="vs Parochialist")
ax.plot(list(N_RANGE), fp_n_AE, "r-s", label="vs Egoist")
ax.set_xlabel("Group size $n$")
ax.set_ylabel("Fixation probability")
ax.set_title(f"$m={M}$, $b/c={BC_DEFAULT}$")
ax.legend()

ax = axes[1]
neutral_m = [1.0 / (N * m) for m in M_RANGE]
ax.plot(list(M_RANGE), neutral_m,  "k--", label="Neutral drift")
ax.plot(list(M_RANGE), fp_m_AP, "b-o", label="vs Parochialist")
ax.plot(list(M_RANGE), fp_m_AE, "r-s", label="vs Egoist")
ax.set_xlabel("Number of groups $m$")
ax.set_ylabel("Fixation probability")
ax.set_title(f"$n={N}$, $b/c={BC_DEFAULT}$")
ax.legend()

fig4.suptitle("García & van den Bergh (2011) — Figure 4")
fig4.tight_layout()

os.makedirs("results", exist_ok=True)
fig2.savefig("results/garcia2011_fig2.pdf")
fig4.savefig("results/garcia2011_fig4.pdf")
print("\nSaved results/garcia2011_fig2.pdf")
print("Saved results/garcia2011_fig4.pdf")
