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

namespace egttools {
    std::unique_ptr<NetworkStructure> init_network_structure(
        int nb_strategies,
        double beta,
        double mu,
        egttools::FinitePopulations::structure::NodeDictionary &network,
        egttools::FinitePopulations::games::AbstractSpatialGame &game,
        int cache_size
    ) {
        egttools::FinitePopulations::structure::NodeDictionary network_copy(network.begin(), network.end());
        return std::make_unique<NetworkStructure>(nb_strategies, beta, mu, network_copy, game, cache_size);
    }

    std::unique_ptr<NetworkStructureSync> init_network_structure_sync(
        int nb_strategies,
        double beta,
        double mu,
        egttools::FinitePopulations::structure::NodeDictionary &network,
        egttools::FinitePopulations::games::AbstractSpatialGame &game,
        int cache_size
    ) {
        egttools::FinitePopulations::structure::NodeDictionary network_copy(network.begin(), network.end());
        return std::make_unique<NetworkStructureSync>(nb_strategies, beta, mu, network_copy, game, cache_size);
    }

    std::unique_ptr<NetworkGroupStructure> init_network_group_structure(
        int nb_strategies,
        double beta,
        double mu,
        egttools::FinitePopulations::structure::NodeDictionary &network,
        egttools::FinitePopulations::games::AbstractSpatialGame &game,
        int cache_size
    ) {
        egttools::FinitePopulations::structure::NodeDictionary network_copy(network.begin(), network.end());
        return std::make_unique<NetworkGroupStructure>(nb_strategies, beta, mu, network_copy, game, cache_size);
    }

    std::unique_ptr<NetworkGroupStructureSync> init_network_group_structure_sync(
        int nb_strategies,
        double beta,
        double mu,
        egttools::FinitePopulations::structure::NodeDictionary &network,
        egttools::FinitePopulations::games::AbstractSpatialGame &game,
        int cache_size
    ) {
        egttools::FinitePopulations::structure::NodeDictionary network_copy(network.begin(), network.end());
        return std::make_unique<NetworkGroupStructureSync>(nb_strategies, beta, mu, network_copy, game, cache_size);
    }
} // namespace egttools

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
dict[int, list[int]]
    Mapping from each node to its neighbors.
)pbdoc"
            );

    py::class_<NetworkStructure,
                egttools::FinitePopulations::structure::AbstractNetworkStructure>(
                m,
                "Network",
                R"pbdoc(
Asynchronous network population structure with pairwise imitation updates.
)pbdoc"
            )
            .def(
                py::init(&egttools::init_network_structure),
                py::arg("nb_strategies"),
                py::arg("beta"),
                py::arg("mu"),
                py::arg("network"),
                py::arg("game"),
                py::arg("cache_size") = 1000,
                py::keep_alive<1, 6>(),
                R"pbdoc(
Construct a network structure.

Parameters
----------
nb_strategies : int
    Maximum number of strategies in the population.
beta : float
    Intensity of selection.
mu : float
    Mutation probability.
network : dict[int, list[int]]
    Network adjacency dictionary.
game : egttools.games.AbstractSpatialGame
    Spatial game played by the population.
cache_size : int, optional
    Cache size used for fitness evaluations.
)pbdoc"
            )

            .def(
                "initialize",
                &NetworkStructure::initialize,
                R"pbdoc(
Initialize the population.

Each individual adopts one of the available strategies with approximately equal
probability.
)pbdoc"
            )

            .def(
                "initialize_state",
                &NetworkStructure::initialize_state,
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
                "update_population",
                &NetworkStructure::update_population,
                R"pbdoc(
Update the population by one generation.
)pbdoc"
            )

            .def(
                "update_node",
                &NetworkStructure::update_node,
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
                &NetworkStructure::calculate_average_gradient_of_selection,
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
                &NetworkStructure::calculate_average_gradient_of_selection_and_update_population,
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
                "calculate_fitness",
                &NetworkStructure::calculate_fitness,
                py::arg("index"),
                R"pbdoc(
Calculate the fitness of the individual at a given node.

Parameters
----------
index : int
    Index of the node whose fitness is calculated.

Returns
-------
float
    Fitness of the individual at the given node.
)pbdoc"
            )

            .def("population_size", &NetworkStructure::population_size)
            .def("nb_strategies", &NetworkStructure::nb_strategies)

            .def(
                "network",
                &NetworkStructure::network,
                py::return_value_policy::reference_internal,
                R"pbdoc(
Return the network adjacency structure.

Returns
-------
dict[int, list[int]]
    Mapping from each node to its neighbors.
)pbdoc"
            )

            .def(
                "population_strategies",
                &NetworkStructure::population_strategies,
                py::return_value_policy::reference_internal,
                R"pbdoc(
Return the strategy currently adopted by each node.

Returns
-------
list[int]
    Strategy index for each node.
)pbdoc"
            )

            .def(
                "mean_population_state",
                &NetworkStructure::mean_population_state,
                py::return_value_policy::reference_internal,
                R"pbdoc(
Return the aggregate population state.

Returns
-------
numpy.ndarray
    Total counts of each strategy in the population.
)pbdoc"
            )

            .def(
                "game",
                &NetworkStructure::game,
                py::return_value_policy::reference_internal,
                R"pbdoc(
Return the game played by the population.

Returns
-------
egttools.games.AbstractSpatialGame
    Game used to evaluate fitness.
)pbdoc"
            );

    py::class_<NetworkGroupStructure,
                egttools::FinitePopulations::structure::AbstractNetworkStructure>(
                m,
                "NetworkGroup",
                R"pbdoc(
Asynchronous network-group population structure with group interactions.
)pbdoc"
            )
            .def(
                py::init(&egttools::init_network_group_structure),
                py::arg("nb_strategies"),
                py::arg("beta"),
                py::arg("mu"),
                py::arg("network"),
                py::arg("game"),
                py::arg("cache_size") = 1000,
                py::keep_alive<1, 6>(),
                R"pbdoc(
Construct a network-group structure.

Parameters
----------
nb_strategies : int
    Maximum number of strategies in the population.
beta : float
    Intensity of selection.
mu : float
    Mutation probability.
network : dict[int, list[int]]
    Network adjacency dictionary.
game : egttools.games.AbstractSpatialGame
    Spatial game played by the population.
cache_size : int, optional
    Cache size used for fitness evaluations.
)pbdoc"
            )

            .def("initialize", &NetworkGroupStructure::initialize)
            .def("initialize_state", &NetworkGroupStructure::initialize_state, py::arg("state"))
            .def("update_population", &NetworkGroupStructure::update_population)
            .def("update_node", &NetworkGroupStructure::update_node, py::arg("node"))

            .def(
                "calculate_average_gradient_of_selection",
                &NetworkGroupStructure::calculate_average_gradient_of_selection,
                py::return_value_policy::reference_internal
            )

            .def(
                "calculate_average_gradient_of_selection_and_update_population",
                &NetworkGroupStructure::calculate_average_gradient_of_selection_and_update_population,
                py::return_value_policy::reference_internal
            )

            .def(
                "calculate_fitness",
                &NetworkGroupStructure::calculate_fitness,
                py::arg("index"),
                R"pbdoc(
Calculate the fitness of the individual at a given node.

The fitness is the accumulated payoff over the focal interaction and the
interactions centered on neighboring nodes.

Parameters
----------
index : int
    Index of the node whose fitness is calculated.

Returns
-------
float
    Fitness of the individual at the given node.
)pbdoc"
            )

            .def(
                "calculate_game_payoff",
                &NetworkGroupStructure::calculate_game_payoff,
                py::arg("index"),
                R"pbdoc(
Calculate the game payoff of the individual at a given node.

Parameters
----------
index : int
    Index of the node whose payoff is calculated.

Returns
-------
float
    Payoff of the individual at the given node.
)pbdoc"
            )

            .def("population_size", &NetworkGroupStructure::population_size)
            .def("nb_strategies", &NetworkGroupStructure::nb_strategies)

            .def("network", &NetworkGroupStructure::network,
                 py::return_value_policy::reference_internal)
            .def("population_strategies", &NetworkGroupStructure::population_strategies,
                 py::return_value_policy::reference_internal)
            .def("mean_population_state", &NetworkGroupStructure::mean_population_state,
                 py::return_value_policy::reference_internal)
            .def("game", &NetworkGroupStructure::game,
                 py::return_value_policy::reference_internal);

    py::class_<NetworkStructureSync,
                egttools::FinitePopulations::structure::AbstractNetworkStructure>(
                m,
                "NetworkSync",
                R"pbdoc(
Synchronous network population structure with pairwise imitation updates.
)pbdoc"
            )
            .def(
                py::init(&egttools::init_network_structure_sync),
                py::arg("nb_strategies"),
                py::arg("beta"),
                py::arg("mu"),
                py::arg("network"),
                py::arg("game"),
                py::arg("cache_size") = 1000,
                py::keep_alive<1, 6>()
            )

            .def("initialize", &NetworkStructureSync::initialize)
            .def("initialize_state", &NetworkStructureSync::initialize_state, py::arg("state"))
            .def("update_population", &NetworkStructureSync::update_population)
            .def("update_node", &NetworkStructureSync::update_node, py::arg("node"))

            .def(
                "calculate_average_gradient_of_selection",
                &NetworkStructureSync::calculate_average_gradient_of_selection,
                py::return_value_policy::reference_internal
            )

            .def(
                "calculate_average_gradient_of_selection_and_update_population",
                &NetworkStructureSync::calculate_average_gradient_of_selection_and_update_population,
                py::return_value_policy::reference_internal
            )

            .def(
                "calculate_fitness",
                &NetworkStructureSync::calculate_fitness,
                py::arg("index"),
                R"pbdoc(
Calculate the fitness of the individual at a given node.

Parameters
----------
index : int
    Index of the node whose fitness is calculated.

Returns
-------
float
    Fitness of the individual at the given node.
)pbdoc"
            )

            .def("population_size", &NetworkStructureSync::population_size)
            .def("nb_strategies", &NetworkStructureSync::nb_strategies)
            .def("network", &NetworkStructureSync::network,
                 py::return_value_policy::reference_internal)
            .def("population_strategies", &NetworkStructureSync::population_strategies,
                 py::return_value_policy::reference_internal)
            .def("mean_population_state", &NetworkStructureSync::mean_population_state,
                 py::return_value_policy::reference_internal)
            .def("game", &NetworkStructureSync::game,
                 py::return_value_policy::reference_internal);

    py::class_<NetworkGroupStructureSync,
                egttools::FinitePopulations::structure::AbstractNetworkStructure>(
                m,
                "NetworkGroupSync",
                R"pbdoc(
Synchronous network-group population structure with group interactions.
)pbdoc"
            )
            .def(
                py::init(&egttools::init_network_group_structure_sync),
                py::arg("nb_strategies"),
                py::arg("beta"),
                py::arg("mu"),
                py::arg("network"),
                py::arg("game"),
                py::arg("cache_size") = 1000,
                py::keep_alive<1, 6>()
            )

            .def("initialize", &NetworkGroupStructureSync::initialize)
            .def("initialize_state", &NetworkGroupStructureSync::initialize_state, py::arg("state"))
            .def("update_population", &NetworkGroupStructureSync::update_population)
            .def("update_node", &NetworkGroupStructureSync::update_node, py::arg("node"))

            .def(
                "calculate_average_gradient_of_selection",
                &NetworkGroupStructureSync::calculate_average_gradient_of_selection,
                py::return_value_policy::reference_internal
            )

            .def(
                "calculate_average_gradient_of_selection_and_update_population",
                &NetworkGroupStructureSync::calculate_average_gradient_of_selection_and_update_population,
                py::return_value_policy::reference_internal
            )

            .def(
                "calculate_fitness",
                &NetworkGroupStructureSync::calculate_fitness,
                py::arg("index"),
                R"pbdoc(
Calculate the fitness of the individual at a given node.

The fitness is the accumulated payoff over the focal interaction and the
interactions centered on neighboring nodes.

Parameters
----------
index : int
    Index of the node whose fitness is calculated.

Returns
-------
float
    Fitness of the individual at the given node.
)pbdoc"
            )

            .def(
                "calculate_game_payoff",
                &NetworkGroupStructureSync::calculate_game_payoff,
                py::arg("index"),
                R"pbdoc(
Calculate the game payoff of the individual at a given node.

Parameters
----------
index : int
    Index of the node whose payoff is calculated.

Returns
-------
float
    Payoff of the individual at the given node.
)pbdoc"
            )

            .def("population_size", &NetworkGroupStructureSync::population_size)
            .def("nb_strategies", &NetworkGroupStructureSync::nb_strategies)
            .def("network", &NetworkGroupStructureSync::network,
                 py::return_value_policy::reference_internal)
            .def("population_strategies", &NetworkGroupStructureSync::population_strategies,
                 py::return_value_policy::reference_internal)
            .def("mean_population_state", &NetworkGroupStructureSync::mean_population_state,
                 py::return_value_policy::reference_internal)
            .def("game", &NetworkGroupStructureSync::game,
                 py::return_value_policy::reference_internal);

    options.enable_function_signatures();
}
