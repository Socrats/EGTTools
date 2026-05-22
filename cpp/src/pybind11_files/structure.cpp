/** Copyright (c) 2022-2026  Elias Fernandez
*
* This file is part of EGTtools.
*
* EGTtools is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* EGTtools is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with EGTtools.  If not, see <http://www.gnu.org/licenses/>
*/
#include "structure.hpp"

// =========================================================================
// Helper: bind all common NetworkMCEstimator methods to a py::class_<T>.
// This is a free template function so it compiles before init_structure.
// =========================================================================
template<typename T>
static void bind_network_mc_estimator_methods(py::class_<T> &cls) {
    using GameT = egttools::FinitePopulations::games::AbstractSpatialGame;
    using NDT = egttools::FinitePopulations::structure::NodeDictionary;

    cls.def(
        py::init([](GameT &game, const NDT &topo,
                    int nb_s, double beta, double mu, int cache) {
            return new T(game, topo, nb_s, beta, mu, cache);
        }),
        py::arg("game"),
        py::arg("topology"),
        py::arg("nb_strategies"),
        py::arg("beta"),
        py::arg("mu"),
        py::arg("cache_size") = 100000,
        py::keep_alive<1, 2>(),
        R"pbdoc(
Construct a NetworkMCEstimator.

Parameters
----------
game : egttools.games.AbstractSpatialGame
    Spatial game used to evaluate node fitness.
topology : dict[int, list[int]]
    Network adjacency dictionary (e.g. from NetworkX ``G.adjacency()``).
nb_strategies : int
    Number of distinct strategies.
beta : float
    Selection intensity.
mu : float
    Mutation probability per time step.
cache_size : int, optional
    Per-thread LRU fitness cache size (default 100000).
)pbdoc"
    );

    cls.def(
        "calculate_gradient_of_selection",
        [](T &self, const std::vector<int> &population) {
            return self.calculate_gradient_of_selection(population);
        },
        py::arg("population"),
        R"pbdoc(
Numerically exact gradient of selection for the given per-node strategy assignment.

Parameters
----------
population : list[int]
    Strategy index for each node (length = population_size).

Returns
-------
numpy.ndarray
    Gradient vector of length nb_strategies.
)pbdoc"
    );

    cls.def(
        "estimate_fixation_probability",
        [](T &self, int invader, int resident, int64_t nb_runs, int64_t nb_generations) {
            py::gil_scoped_release release;
            return self.estimate_fixation_probability(invader, resident, nb_runs, nb_generations);
        },
        py::arg("invader"),
        py::arg("resident"),
        py::arg("nb_runs"),
        py::arg("nb_generations"),
        R"pbdoc(
Estimate the fixation probability of a single invader in an otherwise resident population.

Parameters
----------
invader : int
resident : int
nb_runs : int
nb_generations : int
    Maximum time-steps per trial.

Returns
-------
float
    Estimated fixation probability in [0, 1].
)pbdoc"
    );

    cls.def(
        "estimate_strategy_distribution",
        [](T &self, int64_t nb_runs, int64_t nb_generations, int64_t transitory,
           double tolerance, int64_t check_every) {
            py::gil_scoped_release release;
            return self.estimate_strategy_distribution(nb_runs, nb_generations, transitory,
                                                       tolerance, check_every);
        },
        py::arg("nb_runs"),
        py::arg("nb_generations"),
        py::arg("transitory"),
        py::arg("tolerance") = 0.0,
        py::arg("check_every") = 0,
        R"pbdoc(
Estimate time-averaged strategy frequencies after the transitory period.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray]
    (mean_frequencies, standard_errors), each of length nb_strategies.
)pbdoc"
    );

    cls.def(
        "run",
        [](T &self, int64_t nb_generations, int64_t transitory,
           const egttools::VectorXui &init_state) {
            py::gil_scoped_release release;
            return self.run(nb_generations, transitory, init_state);
        },
        py::arg("nb_generations"),
        py::arg("transitory"),
        py::arg("init_state"),
        R"pbdoc(
Run a single trajectory and return aggregate strategy counts per generation.

Returns
-------
numpy.ndarray
    Matrix of shape (nb_generations - transitory, nb_strategies).
)pbdoc"
    );

    cls.def(
        "run_snapshots",
        [](T &self, int64_t nb_generations, int64_t transitory,
           int64_t snapshot_interval, const egttools::VectorXui &init_state,
           const py::object &callback) {
            self.run_snapshots(nb_generations, transitory, snapshot_interval,
                               init_state,
                               [&callback](int64_t t, const std::vector<int> &pop) {
                                   py::gil_scoped_acquire acquire;
                                   callback(t, pop);
                               });
        },
        py::arg("nb_generations"),
        py::arg("transitory"),
        py::arg("snapshot_interval"),
        py::arg("init_state"),
        py::arg("callback"),
        R"pbdoc(
Run a trajectory and call ``callback(generation, population)`` at each snapshot.

Parameters
----------
snapshot_interval : int
    Call callback every this many generations after transitory.
callback : callable
    Called as ``callback(generation: int, population: list[int])``.
)pbdoc"
    );

    cls.def(
        "estimate_agos",
        [](T &self, int64_t nb_runs, int64_t nb_generations, int64_t transitory,
           int64_t runs_per_j) {
            py::gil_scoped_release release;
            return self.estimate_agos(nb_runs, nb_generations, transitory, runs_per_j);
        },
        py::arg("nb_runs"),
        py::arg("nb_generations"),
        py::arg("transitory") = 0,
        py::arg("runs_per_j") = 0,
        R"pbdoc(
Estimate the time-independent Average Gradient of Selection G^A(j).

Runs independent trajectories; at each post-transitory generation computes
the numerically exact gradient and bins the result by cooperator count j.
OpenMP-parallelised over runs; per-thread caches prevent contention.

Parameters
----------
nb_runs : int
    Total trajectories when ``runs_per_j == 0`` (random starts).
nb_generations : int
    Generations (each = N elementary steps) per trajectory.
transitory : int, optional
    Burn-in generations not counted (default 0).
runs_per_j : int, optional
    When > 0, uses the paper's sampling scheme: for each initial cooperator
    count j0 ∈ {1, …, N-1} exactly ``runs_per_j`` trajectories are started
    with j0 cooperators on random nodes.  Total runs = runs_per_j × (N-1);
    ``nb_runs`` is ignored.  This gives uniform coverage of all j values.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray]
    (mean_G, se_G) each of shape (N+1, nb_strategies).
    Row j holds the average/SE gradient at cooperator count j.
    Rows 0 and N are zero (absorbing states).
)pbdoc"
    );

    cls.def(
        "estimate_agos_time_dependent",
        [](T &self, int64_t nb_runs, int64_t nb_generations) {
            // Release the GIL only for the pure C++ computation.
            egttools::Matrix3D mean_t, se_t;
            {
                py::gil_scoped_release release;
                auto result = self.estimate_agos_time_dependent(nb_runs, nb_generations);
                mean_t = std::move(result.first);
                se_t   = std::move(result.second);
            }
            // GIL is now held again; safe to create Python objects.
            const int T_  = static_cast<int>(mean_t.size());
            const int NP1 = T_ > 0 ? static_cast<int>(mean_t[0].rows()) : 0;
            const int K_  = T_ > 0 ? static_cast<int>(mean_t[0].cols()) : 0;

            py::array_t<double> arr_mean(std::vector<ssize_t>{T_, NP1, K_});
            py::array_t<double> arr_se(std::vector<ssize_t>{T_, NP1, K_});
            auto m = arr_mean.mutable_unchecked<3>();
            auto s = arr_se.mutable_unchecked<3>();
            for (int t = 0; t < T_; ++t)
                for (int j = 0; j < NP1; ++j)
                    for (int k = 0; k < K_; ++k) {
                        m(t, j, k) = mean_t[t](j, k);
                        s(t, j, k) = se_t[t](j, k);
                    }
            return py::make_tuple(arr_mean, arr_se);
        },
        py::arg("nb_runs"),
        py::arg("nb_generations"),
        R"pbdoc(
Estimate the time-dependent Average Gradient of Selection G^A(j, t).

Like estimate_agos but preserves the generation index, letting you study how
the gradient landscape evolves from the initial transient to the stationary
regime (e.g. Fig. 2 of Pinheiro et al. 2012).

Returns
-------
tuple[numpy.ndarray, numpy.ndarray]
    (mean_G_t, se_G_t) each of shape (nb_generations, N+1, nb_strategies).
    ``mean_G_t[t, j, k]`` is the mean gradient of strategy k at generation t
    and cooperator count j.
)pbdoc"
    );

    cls.def("population_size", [](const T &self) { return self.population_size(); });
    cls.def("nb_strategies", [](const T &self) { return self.nb_strategies(); });
    cls.def("beta", [](const T &self) { return self.beta(); });
    cls.def("mu", [](const T &self) { return self.mu(); });
    cls.def("set_beta", [](T &self, double b) { self.set_beta(b); }, py::arg("beta"));
    cls.def("set_mu", [](T &self, double m) { self.set_mu(m); }, py::arg("mu"));
    cls.def("topology", [](const T &self) -> const egttools::FinitePopulations::AdjacencyList & {
        return self.topology();
    }, py::return_value_policy::reference_internal);

    // Step-by-step simulation API
    cls.def(
        "initialize",
        [](T &self, const egttools::VectorXui &init_state) { self.initialize(init_state); },
        py::arg("init_state"),
        R"pbdoc(
Initialise a manual-stepping session with the given strategy counts.

Parameters
----------
init_state : numpy.ndarray
    Strategy count vector of length nb_strategies; sum must equal population_size.
)pbdoc"
    );
    cls.def(
        "initialize",
        [](T &self) { self.initialize(); },
        R"pbdoc(
Initialise a manual-stepping session with a uniformly random strategy assignment.
)pbdoc"
    );
    cls.def(
        "step",
        [](T &self) { self.step(); },
        R"pbdoc(
Advance the session by one generation.

For asynchronous rules: N individual update steps (one per node on average).
For synchronous rules (e.g. LinearProportional): one full simultaneous sweep.
Raises RuntimeError if initialize() has not been called first.
)pbdoc"
    );
    cls.def(
        "population_strategies",
        [](const T &self) -> std::vector<int> { return self.population_strategies(); },
        R"pbdoc(
Return per-node strategy assignments as a list of integers (length = population_size).
)pbdoc"
    );
    cls.def(
        "mean_population_state",
        [](const T &self) -> egttools::VectorXui { return self.mean_population_state(); },
        R"pbdoc(
Return strategy count vector (length = nb_strategies).
)pbdoc"
    );
}

// =========================================================================
// Helper: bind NetworkCoEvolutionary methods to a py::class_<T>.
// =========================================================================
template<typename T>
static void bind_network_coevo_methods(py::class_<T> &cls) {
    using GameT = egttools::FinitePopulations::games::AbstractSpatialGame;
    using NDT = egttools::FinitePopulations::structure::NodeDictionary;

    cls.def(
        py::init([](GameT &game, const NDT &topo, int nb_s,
                    double beta, double mu, double rewiring_prob, int cache) {
            return new T(game, topo, nb_s, beta, mu, rewiring_prob, cache);
        }),
        py::arg("game"),
        py::arg("topology"),
        py::arg("nb_strategies"),
        py::arg("beta"),
        py::arg("mu"),
        py::arg("rewiring_probability"),
        py::arg("cache_size") = 100000,
        py::keep_alive<1, 2>(),
        R"pbdoc(
Construct a NetworkCoEvolutionary estimator.

Parameters
----------
game : egttools.games.AbstractSpatialGame
topology : dict[int, list[int]]
nb_strategies : int
beta : float
mu : float
    Mutation probability.
rewiring_probability : float
    Probability that a time step is a rewiring event rather than a strategy update.
cache_size : int, optional
)pbdoc"
    );

    cls.def(
        "estimate_strategy_distribution",
        [](T &self, int64_t nb_runs, int64_t nb_generations, int64_t transitory,
           double tolerance, int64_t check_every) {
            py::gil_scoped_release release;
            return self.estimate_strategy_distribution(nb_runs, nb_generations, transitory,
                                                       tolerance, check_every);
        },
        py::arg("nb_runs"),
        py::arg("nb_generations"),
        py::arg("transitory"),
        py::arg("tolerance") = 0.0,
        py::arg("check_every") = 0,
        R"pbdoc(
Estimate time-averaged strategy frequencies and edge homophily after the transitory period.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray, float, float]
    (mean_frequencies, se_frequencies, mean_homophily, se_homophily)
)pbdoc"
    );

    cls.def(
        "estimate_fixation_probability",
        [](T &self, int invader, int resident, int64_t nb_runs, int64_t nb_generations) {
            py::gil_scoped_release release;
            return self.estimate_fixation_probability(invader, resident, nb_runs, nb_generations);
        },
        py::arg("invader"),
        py::arg("resident"),
        py::arg("nb_runs"),
        py::arg("nb_generations"),
        R"pbdoc(
Estimate the fixation probability of a single invader in an otherwise resident population.
)pbdoc"
    );

    cls.def(
        "run",
        [](T &self, int64_t nb_generations, int64_t transitory,
           const egttools::VectorXui &init_state) {
            py::gil_scoped_release release;
            return self.run(nb_generations, transitory, init_state);
        },
        py::arg("nb_generations"),
        py::arg("transitory"),
        py::arg("init_state"),
        R"pbdoc(
Run a single co-evolutionary trajectory and return aggregate strategy counts per generation.
)pbdoc"
    );

    cls.def(
        "run_snapshots",
        [](T &self, int64_t nb_generations, int64_t transitory,
           int64_t snapshot_interval, const egttools::VectorXui &init_state,
           const py::object &strategy_callback,
           int64_t topology_interval, const py::object &topology_callback) {
            typename T::AdjacencyList dummy_net;
            std::function<void(int64_t, const typename T::AdjacencyList &)> topo_cb = nullptr;
            if (!topology_callback.is_none()) {
                topo_cb = [&topology_callback](int64_t t, const typename T::AdjacencyList &net) {
                    py::gil_scoped_acquire acquire;
                    topology_callback(t, net);
                };
            }
            self.run_snapshots(nb_generations, transitory, snapshot_interval, init_state,
                               [&strategy_callback](int64_t t, const std::vector<int> &pop) {
                                   py::gil_scoped_acquire acquire;
                                   strategy_callback(t, pop);
                               },
                               topology_interval, topo_cb);
        },
        py::arg("nb_generations"),
        py::arg("transitory"),
        py::arg("snapshot_interval"),
        py::arg("init_state"),
        py::arg("strategy_callback"),
        py::arg("topology_interval") = 0,
        py::arg("topology_callback") = py::none(),
        R"pbdoc(
Run a co-evolutionary trajectory, calling callbacks at each snapshot.

Parameters
----------
strategy_callback : callable
    Called as ``strategy_callback(generation, population)`` every ``snapshot_interval`` gens.
topology_interval : int, optional
    Call topology_callback every this many generations (0 = never).
topology_callback : callable, optional
    Called as ``topology_callback(generation, adjacency_list)``.
)pbdoc"
    );

    cls.def("population_size", [](const T &self) { return self.population_size(); });
    cls.def("nb_strategies", [](const T &self) { return self.nb_strategies(); });
    cls.def("beta", [](const T &self) { return self.beta(); });
    cls.def("mu", [](const T &self) { return self.mu(); });
    cls.def("rewiring_probability", [](const T &self) { return self.rewiring_probability(); });
    cls.def("set_beta", [](T &self, double b) { self.set_beta(b); }, py::arg("beta"));
    cls.def("set_mu", [](T &self, double m) { self.set_mu(m); }, py::arg("mu"));
    cls.def("set_rewiring_probability", [](T &self, double p) {
        self.set_rewiring_probability(p); }, py::arg("p"));
    cls.def("initial_topology", [](const T &self) -> const egttools::FinitePopulations::AdjacencyList & {
        return self.initial_topology();
    }, py::return_value_policy::reference_internal);
}

void init_structure(py::module_ &m) {
    py::options options;
    options.disable_function_signatures();

    py::class_<egttools::FinitePopulations::structure::AbstractStructure,
                stubs::PyAbstractStructure>(
                m,
                "AbstractStructure",
                R"pbdoc(
Abstract base class for population structures.

This class defines the common interface for structures that contain a population
and update the behavior of individuals over time.

Subclasses must implement at least the following methods:

- `initialize()`
- `update_population()`
- `mean_population_state()`
- `nb_strategies()`
)pbdoc"
            )
            .def(py::init<>())

            .def(
                "initialize",
                &egttools::FinitePopulations::structure::AbstractStructure::initialize,
                R"pbdoc(
Initialize the population.

In evolutionary games, this usually means assigning an initial strategy to each
individual.
)pbdoc"
            )

            .def(
                "update_population",
                &egttools::FinitePopulations::structure::AbstractStructure::update_population,
                R"pbdoc(
Update the population by one generation.
)pbdoc"
            )

            .def(
                "mean_population_state",
                &egttools::FinitePopulations::structure::AbstractStructure::mean_population_state,
                py::return_value_policy::reference_internal,
                R"pbdoc(
Return the current aggregate population state.

Returns
-------
numpy.ndarray
    Total counts of each strategy in the population.
)pbdoc"
            )

            .def(
                "nb_strategies",
                &egttools::FinitePopulations::structure::AbstractStructure::nb_strategies,
                R"pbdoc(
Return the maximum number of strategies that can be present in the population.

Returns
-------
int
    Number of strategies.
)pbdoc"
            );

    py::class_<egttools::FinitePopulations::structure::AbstractNetworkStructure,
                stubs::PyAbstractNetworkStructure,
                egttools::FinitePopulations::structure::AbstractStructure>(
                m,
                "AbstractNetworkStructure",
                R"pbdoc(
Abstract base class for network-structured populations.

This class extends `AbstractStructure` for populations represented as nodes in a
network, where edges define who interacts with whom.

Subclasses must implement at least the following methods:

- `initialize()`
- `initialize_state(state)`
- `update_population()`
- `calculate_average_gradient_of_selection()`
- `mean_population_state()`
- `nb_strategies()`
- `population_size()`
)pbdoc"
            )
            .def(py::init<>())

            .def(
                "initialize_state",
                &egttools::FinitePopulations::structure::AbstractNetworkStructure::initialize_state,
                py::arg("state"),
                R"pbdoc(
Initialize the population at a specified aggregate state.

Parameters
----------
state : numpy.ndarray
    One-dimensional array containing the counts of each strategy in the population.
)pbdoc"
            )

            .def(
                "update_node",
                &egttools::FinitePopulations::structure::AbstractNetworkStructure::update_node,
                py::arg("node"),
                R"pbdoc(
Update the strategy of a given node.

Parameters
----------
node : int
    Index of the node to update.
)pbdoc"
            )

            .def(
                "calculate_average_gradient_of_selection",
                &egttools::FinitePopulations::structure::AbstractNetworkStructure::calculate_average_gradient_of_selection,
                py::return_value_policy::reference_internal,
                R"pbdoc(
Calculate the average gradient of selection at the current network state.

Returns
-------
numpy.ndarray
    Averaged gradient of selection for each strategy.
)pbdoc"
            )

            .def(
                "calculate_average_gradient_of_selection_and_update_population",
                &egttools::FinitePopulations::structure::AbstractNetworkStructure::calculate_average_gradient_of_selection_and_update_population,
                py::return_value_policy::reference_internal,
                R"pbdoc(
Calculate the average gradient of selection and update the population.

Returns
-------
numpy.ndarray
    Averaged gradient of selection for each strategy.
)pbdoc"
            )

            .def(
                "population_size",
                &egttools::FinitePopulations::structure::AbstractNetworkStructure::population_size,
                R"pbdoc(
Return the population size.

Returns
-------
int
    Number of nodes in the network.
)pbdoc"
            )

            .def(
                "network",
                &egttools::FinitePopulations::structure::AbstractNetworkStructure::network,
                py::return_value_policy::reference_internal,
                R"pbdoc(
Return the network adjacency structure.

Returns
-------
list[list[int]]
    Adjacency list: entry i contains the neighbor node indices of node i.
)pbdoc"
            );


    options.enable_function_signatures();

    {
        auto cls = py::class_<NetworkMCEstimatorPC>(
            m, "NetworkMCEstimatorPC",
            R"pbdoc(
Monte Carlo estimator for evolutionary games on networks using the Pairwise Comparison update rule.

Provides gradient of selection, fixation probability, strategy distribution estimation,
and trajectory generation. Uses OpenMP for parallel runs and an LRU cache for fitness
memoization.

The network topology is stored as a contiguous adjacency list for O(1) neighbour lookup.
)pbdoc");
        bind_network_mc_estimator_methods(cls);
    }

    {
        auto cls = py::class_<NetworkMCEstimatorBD>(
            m, "NetworkMCEstimatorBD",
            R"pbdoc(
Monte Carlo estimator for evolutionary games on networks using the Birth-Death update rule.

In each step a node is selected to reproduce proportional to fitness, then replaces a
random neighbour.
)pbdoc");
        bind_network_mc_estimator_methods(cls);
    }

    {
        auto cls = py::class_<NetworkMCEstimatorDB>(
            m, "NetworkMCEstimatorDB",
            R"pbdoc(
Monte Carlo estimator for evolutionary games on networks using the Death-Birth update rule.

In each step a random node dies, and a neighbour is selected to reproduce proportional to
fitness.
)pbdoc");
        bind_network_mc_estimator_methods(cls);
    }

    {
        auto cls = py::class_<NetworkMCEstimatorTDPC>(
            m, "NetworkMCEstimatorTDPC",
            R"pbdoc(
Monte Carlo estimator for evolutionary games on networks with time-dependent beta/mu schedules.

Uses the Pairwise Comparison update rule with piecewise-constant C++ schedules for the
selection intensity beta(t) and mutation rate mu(t).
)pbdoc");
        bind_network_mc_estimator_methods(cls);
    }

    {
        auto cls = py::class_<NetworkMCEstimatorLP>(
            m, "NetworkMCEstimatorLP",
            R"pbdoc(
Monte Carlo estimator for evolutionary games on networks using the Linear-Proportional synchronous update rule.

Implements the update rule from Santos, Pacheco & Lenaerts (2006) PNAS.
Each generation all nodes simultaneously accumulate payoffs against all neighbours,
then simultaneously copy a neighbour's strategy with probability proportional to
the payoff advantage, normalised by max(degree) * D_>, where D_> = max(T,1) - min(S,0).

Pass ``beta = max(T, 1.0) - min(S, 0.0)`` as the selection intensity.
)pbdoc");
        bind_network_mc_estimator_methods(cls);
    }

    {
        auto cls = py::class_<NetworkCoEvoPCRandom>(
            m, "NetworkCoEvolutionaryPC",
            R"pbdoc(
Co-evolutionary network estimator: Pairwise Comparison updates + Random rewiring.

Each time step either rewires an edge (with probability ``rewiring_probability``) or
performs a pairwise-comparison strategy update. This is the Santos & Pacheco (2006) model.
)pbdoc");
        bind_network_coevo_methods(cls);
    }

    {
        auto cls = py::class_<NetworkCoEvoPCHomophilic>(
            m, "NetworkCoEvolutionaryPCHomophilic",
            R"pbdoc(
Co-evolutionary network estimator: Pairwise Comparison updates + Homophilic rewiring.

When a rewiring event occurs, the focal node severs a link to a different-strategy neighbour
and preferentially reconnects to a same-strategy non-neighbour. Models social polarisation
(Borges et al. 2023).
)pbdoc");
        bind_network_coevo_methods(cls);
    }

    // -------------------------------------------------------------------------
    // run_network_sweep — OpenMP parallel parameter sweep
    // -------------------------------------------------------------------------
    m.def(
        "run_network_sweep",
        [](std::vector<py::object>                               &py_games,
           py::array_t<double, py::array::c_style>               py_betas,
           std::vector<egttools::FinitePopulations::structure::NodeDictionary> &py_topos,
           int nb_strategies, double mu,
           int64_t nb_runs, int64_t avg_gens, int64_t transitory,
           py::array_t<uint64_t, py::array::c_style>             py_init,
           const std::string                                     &update_rule,
           int cache_size) -> py::array_t<double> {

            using namespace egttools::FinitePopulations;
            using GameT = games::AbstractSpatialGame;

            // --- Validate game types and collect raw pointers (GIL held) ---
            std::vector<GameT *> cpp_games;
            cpp_games.reserve(py_games.size());
            for (auto &obj : py_games) {
                auto *g = obj.cast<GameT *>();
                if (py::get_override(g, "calculate_fitness")) {
                    throw py::type_error(
                        "run_network_sweep requires C++ game classes "
                        "(e.g. NormalFormNetworkGame). Python subclasses of "
                        "AbstractSpatialGame are not supported in the parallel path "
                        "because calculate_fitness would be called without the GIL.");
                }
                cpp_games.push_back(g);
            }

            // --- Convert NodeDictionary -> AdjacencyList (GIL held) ---
            std::vector<structure::AdjacencyList> topos;
            topos.reserve(py_topos.size());
            for (auto &nd : py_topos)
                topos.push_back(structure::dict_to_adjacency_list(nd));

            // --- Copy numeric inputs ---
            std::vector<double> betas(py_betas.data(),
                                      py_betas.data() + py_betas.size());

            egttools::VectorXui init_state(py_init.size());
            for (py::ssize_t i = 0; i < py_init.size(); ++i)
                init_state(static_cast<Eigen::Index>(i)) =
                    static_cast<unsigned int>(py_init.data()[i]);

            egttools::Matrix2D result;
            {
                py::gil_scoped_release release;
                if (update_rule == "LP" || update_rule == "LinearProportional") {
                    result = run_network_sweep<update_rules::LinearProportional>(
                        cpp_games, betas, topos, nb_strategies, mu,
                        nb_runs, avg_gens, transitory, init_state, cache_size);
                } else if (update_rule == "PC" || update_rule == "PairwiseComparison") {
                    result = run_network_sweep<update_rules::PairwiseComparison>(
                        cpp_games, betas, topos, nb_strategies, mu,
                        nb_runs, avg_gens, transitory, init_state, cache_size);
                } else if (update_rule == "BD" || update_rule == "BirthDeath") {
                    result = run_network_sweep<update_rules::BirthDeath>(
                        cpp_games, betas, topos, nb_strategies, mu,
                        nb_runs, avg_gens, transitory, init_state, cache_size);
                } else if (update_rule == "DB" || update_rule == "DeathBirth") {
                    result = run_network_sweep<update_rules::DeathBirth>(
                        cpp_games, betas, topos, nb_strategies, mu,
                        nb_runs, avg_gens, transitory, init_state, cache_size);
                } else {
                    throw std::invalid_argument(
                        "Unknown update_rule '" + update_rule + "'. "
                        "Choose from: 'LP'/'LinearProportional', 'PC'/'PairwiseComparison', "
                        "'BD'/'BirthDeath', 'DB'/'DeathBirth'.");
                }
            }

            // Return as a 2-D numpy array (row-major copy of Eigen column-major data)
            const Eigen::Index rows = result.rows();
            const Eigen::Index cols = result.cols();
            py::array_t<double> out({static_cast<py::ssize_t>(rows),
                                     static_cast<py::ssize_t>(cols)});
            auto buf = out.mutable_unchecked<2>();
            for (Eigen::Index r = 0; r < rows; ++r)
                for (Eigen::Index c = 0; c < cols; ++c)
                    buf(r, c) = result(r, c);
            return out;
        },
        py::arg("games"),
        py::arg("betas"),
        py::arg("topologies"),
        py::arg("nb_strategies"),
        py::arg("mu"),
        py::arg("nb_runs"),
        py::arg("avg_gens"),
        py::arg("transitory"),
        py::arg("init_state"),
        py::arg("update_rule") = "PC",
        py::arg("cache_size")  = 100000,
        R"pbdoc(
Run a parallel parameter sweep over multiple games and topologies using OpenMP.

Each (game, topology) combination is simulated independently with ``nb_runs`` runs.
Topologies are shared read-only across threads — the adjacency list is never copied —
so even a complete graph at large N uses only one copy of memory.

Parameters
----------
games : list[AbstractSpatialGame]
    One C++ game object per parameter combination (e.g. one per (T, S) grid point).
    **Must be C++ classes** such as ``NormalFormNetworkGame`` or
    ``OneShotCRDNetworkGame``.  Python subclasses of ``AbstractSpatialGame`` raise
    ``TypeError`` because ``calculate_fitness`` would be called without the GIL.
betas : np.ndarray, shape (n_games,)
    Selection-intensity / payoff-normalisation constant, one per game.
    For the LP rule pass ``max(T, 1.0) - min(S, 0.0)`` per (T, S) point.
topologies : list[dict[int, list[int]]]
    Network adjacency dicts (NetworkX format).  Shared read-only across threads.
nb_strategies : int
    Number of strategies.
mu : float
    Mutation probability per time step.
nb_runs : int
    Independent simulation runs per (game, topology) combination.
avg_gens : int
    Generations to record after the transitory.
transitory : int
    Burn-in generations (discarded before recording).
init_state : np.ndarray[uint64], shape (nb_strategies,)
    Initial strategy counts; must sum to population_size.
update_rule : str, optional
    ``"LP"`` / ``"LinearProportional"`` (default for Santos 2006),
    ``"PC"`` / ``"PairwiseComparison"``, ``"BD"`` / ``"BirthDeath"``,
    ``"DB"`` / ``"DeathBirth"``.
cache_size : int, optional
    Per-thread LRU fitness cache capacity (default 100 000).

Returns
-------
np.ndarray, shape (n_games, n_topologies)
    Mean cooperation fraction (strategy-0 count / N) for each (game, topology)
    pair, averaged over ``nb_runs`` runs and ``avg_gens`` recording generations.
    Average over the topology axis to get a single cooperation frequency per
    parameter combination.
)pbdoc");
}
