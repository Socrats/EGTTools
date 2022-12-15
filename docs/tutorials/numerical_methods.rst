Apply numerical methods
=======================

Often the complexity of the problem (either due to a high number of strategies or a big population size) makes
numerical simulations a requirement. `egttools` implements several method to estimate the main indicators required
to characterize a population through the stochastic evolutionary dynamics methods described in
:doc:`here <../tutorials/analytical_methods>` in the `egttools.numerical.PairwiseMoranNumerical` class.

All this class requires is a `game` which must inherit from the `egttools.games.Abstract` game class and population
size as parameters. It also requires that you specify a `cache` size. You should
take into account the memory (RAM) available in the machine you will run the simulations,
as this parameter is used to determine the size of the cache where the computed fitness values
will be stored, so that the simulation can run faster. It implements the following methods:

- `estimate_fixation_probability(index_invading_strategy, index_resident_strategy, nb_runs, nb_generations, beta))`:
    Estimates the probability that one mutant of the strategy with index `index_invading_strategy` will
    fixate in a population where all members adopt strategy of index `index_resident_strategy`. `nb_runs` specifies
    the number of (parallel) runs that shall be executed. The higher this number the better the estimation will be.
    `nb_generations` indicates the total number of generations that a simulation will run. The simulation will
    be stopped even if the population did not converge to a monomorphic state. `beta` represents the intensity of
    selection.

- `estimate_stationary_distribution(nb_runs, nb_generations, transitory, beta, mu)`:
    Estimates the stationary distribution. This method will run `nb_runs` (parallel) simulations for `nb_generations`
    generations. After an initial `transitory` number of generations, this methods will count the number of times
    each population state is visited. The final estimation will be an average of all the independent simulations.
    `beta` represents the intensity of selection and `mu` the mutation rate.

- `estimate_stationary_distribution_sparse(nb_runs, nb_generations, transitory, beta, mu)`:
    Same as above, but returns a Sparse Matrix, which is useful for very large state-spaces.

- `estimate_strategy_distribution(nb_runs, nb_generations, transitory, beta, mu)`:
    When the state space is too large to be represented in a 64 bit integer, then we can no longer estimate
    the stationary distribution with `PairwiseComparisonNumerical`. Instead, we can estimate directly the strategy
    distribution using this method.

- `evolve(nb_generations, beta, mu, init_state)`:
    This method will run a single simulation for `nb_generations` generations and return the
    final state of the population. `init_state` is a `numpy.array` containing the initial
    counts of each strategy in the population.

- `run(nb_generations, beta, mu, init_state)`:
    Same as evolve, but instead of just the last state, it will return all states the population went through.

- `run(nb_generations, transient, beta, mu, init_state)`:
    Same as above, but will not return the first `transient` generations. This is useful, as long simulations can
    occupy a lot of memory.

- `run(nb_generations, transient, beta, init_state)`:
    This version of run assumes that mutation is 0.


Estimate fixation probabilities
-------------------------------


Estimate stationary distributions
---------------------------------

.. warning::
    This method should not use for states spaces larger than the number which can be stored in
    a 64 bit - `int64_t` - integer!

Estimate strategy distributions
-------------------------------


Run a single simulation
-----------------------


Evolve a population for a given number of rounds
------------------------------------------------

.. note::
    Although at the moment `egttools.numerical` only contain methods to
    study evolutionary dynamics in well-mixed populations, we have planned
    to add support for simulations in complex networks in version 0.14.0,
    and for multi-level selection in version 0.15.0.