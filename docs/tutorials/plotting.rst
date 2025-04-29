.. _visualizing-results:

Visualizing Results
===================

EGTtools provides a set of flexible plotting functions to help you visualize evolutionary dynamics, fixation probabilities, strategy distributions, and gradients of selection.  
The plotting utilities are organized into:
- 2-strategy plots,
- Simplex plots for 3 strategies,
- General-purpose plots for higher-dimensional cases.

---

Populations with 2 Strategies
------------------------------

The Gradient of Selection and Stability in Infinite Populations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To visualize the gradient of selection under replicator dynamics for 2-strategy games, use:

- `egttools.plotting.indicators.plot_gradients`

Example:

.. code-block:: python

    import numpy as np
    import egttools as egt

    payoffs = np.array([[-0.5, 2.], [0., 0]])
    x = np.linspace(0, 1, 101)
    gradient_function = lambda x: egt.analytical.replicator_equation(x, payoffs)
    gradients = np.array([gradient_function([xi, 1-xi]) for xi in x])

    egt.plotting.indicators.plot_gradients(
        gradients[:, 0],
        xlabel="frequency of hawks",
        roots=[0.0, 0.5, 1.0],
        stability=[-1, 1, -1]
    )

This function can also indicate the stability of roots using different markers.

---

The Gradient of Selection in Finite Populations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To plot gradients obtained under stochastic evolutionary dynamics in finite populations, use the same plotting function:

- `egttools.plotting.indicators.plot_gradients`

Example:

.. code-block:: python

    from egttools.analytical import PairwiseComparison

    Z = 100  # Population size
    beta = 1
    A = np.array([[-0.5, 2.], [0., 0]])

    game = egt.games.Matrix2PlayerGameHolder(2, A)
    evolver = PairwiseComparison(Z, game)

    gradients = np.array([
        evolver.calculate_gradient_of_selection(beta, np.array([k, Z-k]))
        for k in range(Z + 1)
    ])

    egt.plotting.indicators.plot_gradients(
        gradients[:, 0],
        xlabel="frequency of hawks (k/Z)"
    )

---

Plotting the Stationary Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To visualize stationary distributions over states or strategies:

- `egttools.plotting.helpers.plot_stationary_distribution`
- `egttools.plotting.helpers.plot_strategy_distribution`

Example for a simple distribution:

.. code-block:: python

    stationary_distribution = np.random.rand(101)
    stationary_distribution /= stationary_distribution.sum()

    egt.plotting.helpers.plot_stationary_distribution(
        stationary_distribution,
        xlabel="number of cooperators",
        ylabel="probability"
    )

For strategy distributions (especially when the state space is too large to track full states):

.. code-block:: python

    strategy_distribution = np.random.rand(2)
    strategy_distribution /= strategy_distribution.sum()

    egt.plotting.helpers.plot_strategy_distribution(
        strategy_distribution,
        strategy_labels=["Hawk", "Dove"]
    )

---

Populations with 3 Strategies
------------------------------

The Simplex2D Class
^^^^^^^^^^^^^^^^^^^

For visualizing three-strategy dynamics on a 2D simplex, EGTtools provides:

- `egttools.plotting.simplex2d.Simplex2D`

The `Simplex2D` class allows plotting gradients, stationary distributions, and evolutionary trajectories on the 2-simplex (triangle).

Initialization:

.. code-block:: python

    from egttools.plotting.simplex2d import Simplex2D

    simplex = Simplex2D(nb_points_per_axis=50)

You can plot vector fields, color maps, or trajectories using its methods.

---

The Gradient of Selection and Stability in Infinite Populations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can plot the deterministic flow of replicator dynamics in a three-strategy game:

.. code-block:: python

    # Assuming you have a function that computes dx/dt given x
    def gradient_function(x):
        # Example dummy function
        return np.array([-x[0] * (1 - x[0]), x[1] * (1 - x[1])])

    simplex.plot_quiver(gradient_function)

---

The Gradient of Selection and Stationary Distribution in Finite Populations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also plot heatmaps of stationary distributions over the simplex:

.. code-block:: python

    stationary_distribution = np.random.rand(simplex.nb_points)
    simplex.plot_stationary_distribution(stationary_distribution)

---

Populations with More Than 3 Strategies
---------------------------------------

Currently, for more than 3 strategies, EGTtools provides basic plotting utilities such as bar plots for strategy frequencies:

- `egttools.plotting.helpers.plot_strategy_distribution`

Example:

.. code-block:: python

    strategy_distribution = np.random.rand(5)
    strategy_distribution /= strategy_distribution.sum()

    egt.plotting.helpers.plot_strategy_distribution(
        strategy_distribution,
        strategy_labels=["A", "B", "C", "D", "E"]
    )

Advanced visualization methods (e.g., 3D simplexes for 4 strategies) are under development.

.. note::
    Support for plotting 3D simplexes (tetrahedrons) for 4-strategy populations is planned for future versions.

