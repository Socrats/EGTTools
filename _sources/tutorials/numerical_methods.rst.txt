.. _numerical-methods:

Numerical Methods in EGTtools
=============================

In many cases, the complexity of the problem—whether due to a large number of strategies or a large population size—makes **numerical simulations** necessary.
`egttools` implements several methods for estimating key indicators that characterize a population’s stochastic evolutionary dynamics. These methods are based on the analytical approaches described :doc:`here <../tutorials/analytical_methods>` and are available in the `egttools.numerical.PairwiseMoranNumerical` class.

To use these methods, you need to provide:
- A `game` object, which must inherit from the `egttools.games.AbstractGame` class,
- The **population size**,
- A **cache size** to control memory usage.

.. note::
    The `cache` parameter determines the size of the fitness cache in memory (RAM). A larger cache speeds up simulations by avoiding redundant calculations but also increases memory consumption. Choose a value appropriate for the available system memory.

---

Methods Overview
----------------

- **`estimate_fixation_probability(index_invading_strategy, index_resident_strategy, nb_runs, nb_generations, beta)`**
  Estimates the probability that a mutant adopting strategy `index_invading_strategy` fixates in a resident population adopting strategy `index_resident_strategy`.
  - `nb_runs`: Number of independent simulations (higher values yield better estimations).
  - `nb_generations`: Maximum number of generations per simulation.
  - `beta`: Intensity of selection.

- **`estimate_stationary_distribution(nb_runs, nb_generations, transitory, beta, mu)`**
  Estimates the stationary distribution over all possible population states.
  - `transitory`: Number of generations discarded at the beginning (burn-in).
  - `mu`: Mutation rate.

- **`estimate_stationary_distribution_sparse(nb_runs, nb_generations, transitory, beta, mu)`**
  Same as `estimate_stationary_distribution`, but returns a **sparse matrix**, suitable for very large state spaces.

- **`estimate_strategy_distribution(nb_runs, nb_generations, transitory, beta, mu)`**
  For very large systems where the number of states exceeds what can be stored in a 64-bit integer, this method estimates the **distribution of strategies** instead of full population states.

- **`evolve(nb_generations, beta, mu, init_state)`**
  Runs a single simulation for `nb_generations` generations and returns only the final population state.
  - `init_state`: A `numpy.ndarray` representing the initial counts of each strategy.

- **`run_with_mutation(nb_generations, transient, beta, mu, init_state)`**
  Runs a simulation **with mutation**.
  After an initial `transient` phase (burn-in), returns the sequence of visited states.
  - `beta`: Intensity of selection.
  - `mu`: Mutation rate.
  - `init_state`: Initial population configuration (`numpy.ndarray`).

- **`run_without_mutation(nb_generations, transient, beta, init_state)`**
  Runs a simulation **without mutation** (`mu = 0`).
  After an initial `transient` phase, returns the sequence of visited states.
  - `beta`: Intensity of selection.
  - `init_state`: Initial population configuration (`numpy.ndarray`).

---

Tutorials
---------

Estimate Fixation Probabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Estimate the probability that a rare mutant strategy successfully invades and takes over a resident population.
Useful for studying **invasion dynamics** and **evolutionary stability**.

Estimate Stationary Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. warning::
    Only use this method if the number of possible states fits within a 64-bit integer (`int64_t`).
    For very large systems, use `estimate_strategy_distribution` instead.

Estimate the long-term probability distribution over all possible population states.
This provides insight into **stable evolutionary states** under mutation-selection balance.

Estimate Strategy Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For extremely large systems, where tracking full states is infeasible, estimate the distribution of strategies across the population.

Run a Single Simulation
^^^^^^^^^^^^^^^^^^^^^^^
Use `evolve` when you are only interested in the **final state** of the population after a fixed number of generations.

Run Full Trajectories
^^^^^^^^^^^^^^^^^^^^^
Use `run_with_mutation` or `run_without_mutation` to obtain the **full trajectory** of population states over time.
Useful for studying dynamic evolutionary paths.

---

.. note::
    Currently, `egttools.numerical` supports simulations only in **well-mixed populations**.
    Planned future extensions:
    - **Version 0.14.0**: Support for **structured populations** (e.g., networks),
    - **Version 0.15.0**: Support for **multi-level selection** models.
