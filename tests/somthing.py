import matplotlib.pyplot as plt
import numpy as np
import egttools as egt

import warnings

# Parameters and evolver
beta = 5
Z = 50
N = 6
M = 3
b = 1
c = 0.1
pop_states = np.arange(0, Z + 1, 1)

colors = ['black', 'blue', 'cadetblue', 'orange', 'red']
risks = [1.0, 0.75, 0.5, 0.25, 0.0]

with warnings.catch_warnings():
    fig, ax = plt.subplots(figsize=(6, 5))

    for i, risk in enumerate(risks):
        game = egt.games.OneShotCRD(b, c, risk, N, M)
        evolver = egt.analytical.StochDynamics(game.nb_strategies(), game.payoffs(), Z, group_size=N, mu=0)
        gradients = np.array([evolver.full_gradient_selection(egt.sample_simplex(index, Z, 2), beta)
                              for index in range(egt.calculate_nb_states(Z, 2))])
        egt.plotting.indicators.plot_gradients(gradients[:, 1],
                                               marker_facecolor='white',
                                               xlabel="frequency of cooperators (k/Z)", marker="o",
                                               marker_size=30, marker_plot_freq=2,
                                               linelabel="$r={:.2f}$".format(risk),
                                               marker_edgecolor=colors[i], color=colors[i], ax=ax)

    ax.legend(frameon=False, fontsize=14)

    plt.show()

with warnings.catch_warnings():
    fig, ax = plt.subplots(figsize=(6, 5))

    for i, risk in enumerate(risks):
        game = egt.games.OneShotCRD(b, c, risk, N, M)
        evolver = egt.analytical.PairwiseComparison(Z, game)
        gradients = np.array([evolver.calculate_gradient_of_selection(beta, egt.sample_simplex(index, Z, 2))
                              for index in range(egt.calculate_nb_states(Z, 2))])
        egt.plotting.indicators.plot_gradients(gradients[:, 1],
                                               marker_facecolor='white',
                                               xlabel="frequency of cooperators (k/Z)", marker="o",
                                               marker_size=30, marker_plot_freq=2,
                                               linelabel="$r={:.2f}$".format(risk),
                                               marker_edgecolor=colors[i], color=colors[i], ax=ax)

    ax.legend(frameon=False, fontsize=14)

    plt.show()
