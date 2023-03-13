//
// Created by Elias Fernandez on 08/01/2023.
//
#include "structure.hpp"

namespace egttools {
    std::unique_ptr<NetworkStructure> init_network_structure(int nb_strategies, double beta,
                                                             double mu, egttools::FinitePopulations::structure::NodeDictionary &network,
                                                             egttools::FinitePopulations::games::AbstractSpatialGame &game,
                                                             int cache_size) {
        egttools::FinitePopulations::structure::NodeDictionary network_copy(network.begin(), network.end());

        return std::make_unique<NetworkStructure>(nb_strategies, beta, mu, network_copy, game, cache_size);
    }

}// namespace egttools

void init_structure(py::module_ &m) {
    py::class_<egttools::FinitePopulations::structure::AbstractStructure, stubs::PyAbstractStructure>(m, "AbstractStructure")
            .def(py::init<>(), R"pbdoc(
                Abstract class which must be implemented by any new population structure.

                This class provides a common interface for classes that implement general
                behavioral updates for a population of a given structure.

                A population structure in egttools is meant to contain a population
                (which may be structure in any way in function of the implementation)
                and whose state (the behaviors of the individuals) are updated according
                to a given function.

                For this reason, a Structure class must expose a method to initialize the population
                (specific implementations may add other initialization methods), a method
                to update the population given its current state, a method that returns the number
                of possible strategies in the population, and a method that returns the mean
                state of the population, i.e., how many individuals, in total, in the population
                adopt each strategy.

                Moreover, we encourage specific implementations to also expose the structure of the
                population and the changes in the behavior of each individual to the user.

                You must implement the following methods:
                - initialize()
                - update_population()
                - mean_population_state()
                - nb_strategies()

                See Also
                --------
                egttools.numerical.structures.Network

                Note
                ----
                This is still a first implementation.
                This class might be renamed ini the future and the API might change.
                )pbdoc")
            .def("initialize", &egttools::FinitePopulations::structure::AbstractStructure::initialize, R"pbdoc(
                    Initializes each element of the structure.

                    In Evolutionary games, this means that each individual in the structure is
                    assigned a strategy according to some algorithm (generally
                    it will be a random assignment). It is recommended
                    that subclasses which wish to implement other assignment
                    types, create different methods with more concrete name,
                    e.g., initialize_all_black, would initialize each individual
                    with the black strategy.
                    )pbdoc")
            .def("update_population", &egttools::FinitePopulations::structure::AbstractStructure::initialize, R"pbdoc(
                    Updates the strategy of an individual inside of the population.

                    The developer has freedom to implement this method. All that is required
                    is that it applies some change to the population every time it is called
                    given the current population state.
                    )pbdoc")
            .def("mean_population_state", &egttools::FinitePopulations::structure::AbstractStructure::initialize, R"pbdoc(
                    Returns the mean population state.

                    Returns
                    -------
                    numpy.ndarray
                        The total counts of each strategy in the population.
                    )pbdoc")
            .def("nb_strategies", &egttools::FinitePopulations::structure::AbstractStructure::initialize, R"pbdoc(
                    Returns the maximum number of strategies that can be present in the population.

                    Returns
                    -------
                    int
                        The maximum number of strategies that can be present in the population.
                    )pbdoc");

    py::class_<NetworkStructure, egttools::FinitePopulations::structure::AbstractStructure>(m, "Network")
            .def(py::init(&egttools::init_network_structure),
                 py::arg("nb_strategies"),
                 py::arg("beta"),
                 py::arg("mu"),
                 py::arg("network"),
                 py::arg("game"),
                 py::arg("cache_size") = 1000, py::keep_alive<1, 6>(),
                 R"pbdoc(
                    Network structure.

                    This population structure assumes that players are connected in a network.
                    The `network` input parameter, is a dictionary which indicates how individuals
                    are connected. The keys of the dictionary are the nodes in the network and the
                    value associated with each key is a list of neighbors to which the node is connected.

                    We assume that the network is directed. To produce an undirected network, it is enough
                    to consider that 2 connected nodes are in the neighborhood of each other.

                    This structure only implements asynchronous updates at the moment. This means
                    that at each update step, only one member of the population can be updated.
                    Moreover, the network is static (no links are changed between the nodes).

                    The update mechanism is pairwise imitation. This is, at a given time step
                    an individual i is selected randomly to die. Another individual j, is selected randomly from
                    i's neighborhood. i will imitate the strategy of j with probability proportional
                    to the difference of payoffs between them (using the fermi distribution with temperature
                    beta).

                    The fitness of an individual is given by the calculate_fitness function of the
                    Game object.

                    Parameters
                    ----------
                    nb_strategies: int
                        The maximum number of strategies that can exist in the population.
                    beta: float
                        The intensity of selection.
                    mu: float
                        Mutation probability
                    network: Dict[int, List[int]]
                        Dictionary containing the list of neighbors for each node in the network.
                    game: egttools.games.AbstractSpatialGame
                        The game that the population will play.
                    cache_size: int
                        The size of the cache memory used to save the fitness of each strategy given
                        a neighborhood state.
                )pbdoc")
            .def("initialize", &NetworkStructure::initialize,
                 R"pbdoc(
                    Initializes each element of the structure.

                    Each individual will adopt any of the available strategies with equal probability.
                    If the population is big enough and divisible by the number of strategies,
                    This should result in a roughly equal distribution of each strategy.
                    )pbdoc")
            .def("update_population", &NetworkStructure::update_population,
                 R"pbdoc(
                    Initializes each element of the structure.

                    Each individual will adopt any of the available strategies with equal probability.
                    If the population is big enough and divisible by the number of strategies,
                    This should result in a roughly equal distribution of each strategy.
                    )pbdoc")

            .def("calculate_fitness", &NetworkStructure::calculate_fitness,
                 R"pbdoc(
                    Calculates the fitness of a strategy given a neighborhood state.

                    The neighborhood state is the counts of each strategy in the neighborhood.

                    Parameters
                    ----------
                    index: int
                        The index of the node whose fitness shall be calculated.

                    Returns
                    -------
                    float
                        The fitness of the individual at the node `index`.
                    )pbdoc")

            .def("population_size", &NetworkStructure::population_size,
                 R"pbdoc(
                    Returns the number of individuals in the population.

                    Returns
                    -------
                    int
                        The number of individuals in the population.
                    )pbdoc")

            .def("nb_strategies", &NetworkStructure::nb_strategies,
                 R"pbdoc(
                    Returns the maximum number of strategies in the population

                    Returns
                    -------
                    int
                        The maximum number of strategies in the population.
                    )pbdoc")
            .def("network", &NetworkStructure::network,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
                    Returns a dictionary with list of neighbors for each node in the network.

                    Returns
                    -------
                    Dict[int, List[int]]
                        A dictionary with list of neighbors for each node in the network.
                    )pbdoc")
            .def("population_strategies", &NetworkStructure::population_strategies,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
                    Returns a list containing the strategy adopted by each node in the network.

                    Returns
                    -------
                    List[int]
                        A list containing the strategy adopted by each node in the network.
                    )pbdoc")
            .def("mean_population_state", &NetworkStructure::mean_population_state,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
                    Returns a numpy array contain the total counts of strategies in the population.

                    Returns
                    -------
                    numpy.ndarray
                        A numpy array contain the total counts of strategies in the population.
                    )pbdoc")
            .def("game", &NetworkStructure::game,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
                    Returns the game played by the population

                    Returns
                    -------
                    egttools.games.AbstractSpatialGame
                        The game played by the population.
                    )pbdoc");
}