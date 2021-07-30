/** Copyright (c) 2019-2021  Elias Fernandez
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

#include "egttools.h"

#include "version.h"

#define XSTR(s) STR(s)
#define STR(s) #s

namespace py = pybind11;
using namespace std::string_literals;
using namespace egttools;
using PairwiseComparison = egttools::FinitePopulations::PairwiseMoran<egttools::Utils::LRUCache<std::string, double>>;

namespace egttools {

    std::string call_get_action(const py::list &strategies, size_t time_step, size_t action) {
        std::stringstream result;
        result << "(";
        for (py::handle strategy : strategies) {
            result << py::cast<egttools::FinitePopulations::behaviors::AbstractNFGStrategy *>(strategy)->get_action(time_step, action) << ", ";
        }
        result << ")";
        return result.str();
    }

    std::unique_ptr<egttools::FinitePopulations::NormalFormGame> init_normal_form_game_from_python_list(size_t nb_rounds,
                                                                                                        const Eigen::Ref<const Matrix2D> &payoff_matrix, const py::list &strategies) {
        egttools::FinitePopulations::NFGStrategyVector strategies_cpp;
        for (py::handle strategy : strategies) {
            strategies_cpp.push_back(py::cast<egttools::FinitePopulations::behaviors::AbstractNFGStrategy *>(strategy));
        }
        return std::make_unique<egttools::FinitePopulations::NormalFormGame>(nb_rounds, payoff_matrix, strategies_cpp);
    }

    std::unique_ptr<egttools::FinitePopulations::CRDGame> init_crd_game_from_python_list(int endowment,
                                                                                         int threshold,
                                                                                         int nb_rounds,
                                                                                         int group_size,
                                                                                         double risk,
                                                                                         const py::list &strategies) {
        egttools::FinitePopulations::CRDStrategyVector strategies_cpp;
        for (py::handle strategy : strategies) {
            strategies_cpp.push_back(py::cast<egttools::FinitePopulations::behaviors::AbstractCRDStrategy *>(strategy));
        }
        return std::make_unique<egttools::FinitePopulations::CRDGame>(endowment, threshold, nb_rounds,
                                                                      group_size, risk, strategies_cpp);
    }

    egttools::VectorXli sample_simplex_directly(int64_t nb_strategies, int64_t pop_size) {
        std::mt19937_64 generator(egttools::Random::SeedGenerator::getInstance().getSeed());
        egttools::VectorXli state = egttools::VectorXli::Zero(nb_strategies);

        egttools::FinitePopulations::sample_simplex_direct_method<long int, egttools::VectorXli, std::mt19937_64>(nb_strategies, pop_size, state, generator);

        return state;
    }
    egttools::Vector sample_unit_simplex(int64_t nb_strategies) {
        std::mt19937_64 generator(egttools::Random::SeedGenerator::getInstance().getSeed());
        auto real_rand = std::uniform_real_distribution<double>(0, 1);
        egttools::Vector state = egttools::Vector::Zero(nb_strategies);
        egttools::FinitePopulations::sample_unit_simplex<int64_t, std::mt19937_64>(nb_strategies, state, real_rand, generator);

        return state;
    }
}// namespace egttools

namespace {

    inline std::string attr_doc(const py::module_ &m, const char *name, const char *doc) {
        auto attr = m.attr(name);
        return ".. data:: "s + name + "\n    :annotation: = "s + py::cast<std::string>(py::repr(attr)) + "\n\n    "s + doc + "\n\n"s;
    }

}// namespace

PYBIND11_MODULE(numerical, m) {
    m.attr("__version__") = py::str(XSTR(EGTTOOLS_VERSION));
    m.attr("VERSION") = py::str(XSTR(EGTTOOLS_VERSION));
    m.attr("__init__") = py::str(
            "The `numerical` module contains optimized "
            "functions and classes to simulate evolutionary dynamics in large populations.");

    m.doc() =
            "The `numerical` module contains optimized functions and classes to simulate "
            "evolutionary dynamics in large populations. This module is written in C++.";


    // Use this function to get access to the singleton
    py::class_<Random::SeedGenerator, std::unique_ptr<Random::SeedGenerator, py::nodelete>>(m, "Random", "Random seed generator.")
            .def_static(
                    "init", []() {
                        return std::unique_ptr<Random::SeedGenerator, py::nodelete>(&Random::SeedGenerator::getInstance());
                    },
                    R"pbdoc(
            This static method initializes the random seed generator from random_device
            and returns an instance of egttools::Random::SeedGenerator which is used
            to seed the random generators used across egttools.

            Parameters
            ----------

            Returns
            -------
            egttools.Random
                An instance of the random seed generator.
           )pbdoc")
            .def_static(
                    "init", [](unsigned long int seed) {
                        auto instance = std::unique_ptr<Random::SeedGenerator, py::nodelete>(&Random::SeedGenerator::getInstance());
                        instance->setMainSeed(seed);
                        return instance;
                    },
                    R"pbdoc(
            This static method initializes the random seed generator from seed
            and returns an instance of egttools.Random which is used
            to seed the random generators used across egttools.

            Parameters
            ----------
            seed : int
                Integer value used to seed the random generator.

            Returns
            -------
            egttools.Random
                An instance of the random seed generator.
           )pbdoc",
                    py::arg("seed"))
            .def_property_readonly_static(
                    "_seed", [](const py::object &) {
                        return egttools::Random::SeedGenerator::getInstance().getMainSeed();
                    },
                    "The initial seed of egttools.Random.")
            .def_static(
                    "generate", []() {
                        return egttools::Random::SeedGenerator::getInstance().getSeed();
                    },
                    R"pbdoc(
                    Generates a random seed.

                    Parameters
                    ----------

                    Returns
                    -------
                    int
                        A seed.
                    )pbdoc")
            .def_static(
                    "seed", [](unsigned long int seed) {
                        egttools::Random::SeedGenerator::getInstance().setMainSeed(seed);
                    },
                    R"pbdoc(
                    This static methods changes the seed of egttools.Random.

                    Parameters
                    ----------
                    int
                        A seed.

                    Returns
                    -------

                    )pbdoc",
                    py::arg("seed"));

    // Now we define a submodule
    auto mGames = m.def_submodule("games");
    auto mBehaviors = m.def_submodule("behaviors");
    auto mCRD = mBehaviors.def_submodule("CRD");
    auto mNF = mBehaviors.def_submodule("NormalForm");
    auto mNFTwoActions = mNF.def_submodule("TwoActions");
    auto mData = m.def_submodule("DataStructures");
    auto mDistributions = m.def_submodule("distributions");

    mGames.attr("__init__") = py::str("The `game` submodule contains the available games.");
    mBehaviors.attr("__init__") = py::str("The `behaviors` submodule contains the available strategies to evolve.");
    mNF.attr("__init__") = py::str("The `NormalForm` submodule contains the strategies for normal form games.");
    mNFTwoActions.attr("__init__") = py::str(
            "The `TwoActions` submodule contains the strategies for "
            "normal form games with 2 actions.");
    mData.attr("__init__") = py::str("The `DataStructures` submodule contains helpful data structures to store.");
    mDistributions.attr("__init__") = py::str(
            "The `distributions` submodule contains "
            "functions and classes that produce stochastic distributions.");

    py::class_<egttools::FinitePopulations::AbstractGame, stubs::PyAbstractGame>(mGames, "AbstractGame")
            .def(py::init<>())
            .def("play", &egttools::FinitePopulations::AbstractGame::play, R"pbdoc(
                    Updates the vector of payoffs with the payoffs of each player after playing the game.

                    This method will run the game using the players and player types defined in :param group_composition,
                    and will update the vector :param game_payoffs with the resulting payoff of each player.

                    Parameters
                    ----------
                    group_composition : List[int]
                        A list with counts of the number of players of each strategy in the group.
                    game_payoffs : List[float]
                        A list used as container for the payoffs of each player
                    )pbdoc",
                 py::arg("group_composition"), py::arg("game_payoffs"))
            .def("calculate_payoffs", &egttools::FinitePopulations::AbstractGame::calculate_payoffs,
                 R"pbdoc(
                    Estimates the payoffs for each strategy and returns the values in a matrix.
                    Each row of the matrix represents a strategy and each column a game state.
                    E.g., in case of a 2 player game, each entry a_ij gives the payoff for strategy
                    i against strategy j. In case of a group game, each entry a_ij gives the payoff
                    of strategy i for game state j, which represents the group composition.

                    Returns
                    -------
                    numpy.ndarray[np.float64]
                        A matrix with the expected payoffs for each strategy given each possible game
                        state.
                    )pbdoc")
            .def("calculate_fitness", &egttools::FinitePopulations::AbstractGame::calculate_fitness,
                 R"pbdoc(
                    Estimates the fitness for a player_type in the population with state :param strategies.

                    This function assumes that the player with strategy player_type is not included in
                    the vector of strategy counts strategies.

                    Parameters
                    ----------
                    player_type : int
                        The index of the strategy used by the player.
                    pop_size : int
                        The size of the population.
                    strategies : numpy.ndarray[numpy.uint64]
                        A vector of counts of which strategy. The current state of the population

                    Returns
                    -------
                    float
                        The fitness of the strategy in the population state given by strategies.
                    )pbdoc",
                 py::arg("player_type"), py::arg("pop_size"), py::arg("strategies"))
            .def("__str__", &egttools::FinitePopulations::AbstractGame::toString)
            .def("type", &egttools::FinitePopulations::AbstractGame::type, "returns the type of game.")
            .def("payoffs", &egttools::FinitePopulations::AbstractGame::payoffs, "returns the payoff matrix of the game.")
            .def("payoff", &egttools::FinitePopulations::AbstractGame::payoff,
                 R"pbdoc(
                    Returns the payoff of a strategy given a group composition.

                    If the group composition does not include the strategy, the payoff should be zero.

                    Parameters
                    ----------
                    strategy : int
                        The index of the strategy used by the player.
                    group_composition : List[int]
                        List with the group composition. The structure of this list
                        depends on the particular implementation of this abstract method.

                    Returns
                    -------
                    float
                        The payoff value.
                    )pbdoc",
                 py::arg("strategy"), py::arg("group_composition"))
            .def_property_readonly("nb_strategies", &egttools::FinitePopulations::AbstractGame::nb_strategies,
                                   "Number of different strategies playing the game.")
            .def("save_payoffs", &egttools::FinitePopulations::AbstractGame::save_payoffs,
                 R"pbdoc(
                    Stores the payoff matrix in a txt file.

                    Parameters
                    ----------
                    file_name : str
                        Name of the file in which the data will be stored.
                    )pbdoc",
                 py::arg("file_name"));

    m.def("calculate_state",
          static_cast<size_t (*)(const size_t &, const egttools::Factors &)>(&egttools::FinitePopulations::calculate_state),
          R"pbdoc(
                    This function converts a vector containing counts into an index.

                    This method was copied from @Svalorzen.

                    Parameters
                    ----------
                    group_size : int
                        Maximum bin size (it can also be the population size).
                    group_composition : List[int]
                        The vector to convert from simplex coordinates to index.

                    Returns
                    -------
                    int
                        The unique index in [0, egttools.calculate_nb_states(group_size, len(group_composition))
                        representing the n-dimensional simplex.

                    See Also
                    --------
                    egttools.sample_simplex, egttools.calculate_nb_states
                    )pbdoc",
          py::arg("group_size"), py::arg("group_composition"));
    m.def("calculate_state",
          static_cast<size_t (*)(const size_t &,
                                 const Eigen::Ref<const egttools::VectorXui> &)>(&egttools::FinitePopulations::calculate_state),
          R"pbdoc(
                    This function converts a vector containing counts into an index.

                    This method was copied from @Svalorzen.

                    Parameters
                    ----------
                    group_size : int
                        Maximum bin size (it can also be the population size).
                    group_composition : numpy.ndarray[int]
                        The vector to convert from simplex coordinates to index.

                    Returns
                    -------
                    int
                        The unique index in [0, egttools.calculate_nb_states(group_size, len(group_composition))
                        representing the n-dimensional simplex.

                    See Also
                    --------
                    egttools.sample_simplex, egttools.calculate_nb_states
                    )pbdoc",
          py::arg("group_size"), py::arg("group_composition"));
    m.def("sample_simplex",
          static_cast<egttools::VectorXui (*)(size_t, const size_t &, const size_t &)>(&egttools::FinitePopulations::sample_simplex),
          R"pbdoc(
                    Transforms a state index into a vector.

                    Parameters
                    ----------
                    index : int
                        State index.
                    pop_size : int
                        Size of the population.
                    nb_strategies : int
                        Number of strategies.

                    Returns
                    -------
                    numpy.ndarray[int]
                        Vector with the sampled state.

                    See Also
                    --------
                    egttools.numerical.calculate_state, egttools.numerical.calculate_nb_states
                    )pbdoc",
          py::arg("index"), py::arg("pop_size"),
          py::arg("nb_strategies"));
    m.def("sample_simplex_directly",
          &sample_simplex_directly,
          R"pbdoc(
                    Samples an N-dimensional point directly from the simplex.
                    N is the number of strategies.

                    Parameters
                    ----------
                    nb_strategies : int
                        Number of strategies.
                    pop_size : int
                        Size of the population.

                    Returns
                    -------
                    numpy.ndarray[numpy.int64]
                        Vector with the sampled state.

                    See Also
                    --------
                    egttools.numerical.calculate_state, egttools.numerical.calculate_nb_states, egttools.numerical.sample_simplex
                    )pbdoc",
          py::arg("nb_strategies"),
          py::arg("pop_size"));
    m.def("sample_unit_simplex",
          &sample_unit_simplex,
          R"pbdoc(
                    Samples uniformly at random the unit simplex with nb_strategies dimensionse.

                    Parameters
                    ----------
                    nb_strategies : int
                        Number of strategies.

                    Returns
                    -------
                    numpy.ndarray[numpy.int64]
                        Vector with the sampled state.

                    See Also
                    --------
                    egttools.numerical.calculate_state, egttools.numerical.calculate_nb_states, egttools.numerical.sample_simplex
                    )pbdoc",
          py::arg("nb_strategies"));

    m.def("calculate_nb_states",
          &egttools::starsBars<size_t>,
          R"pbdoc(
                    Calculates the number of states (combinations) of the members of a group in a subgroup.
                    It can be used to calculate the maximum number of states in a discrete simplex.

                    The implementation of this method follows the stars and bars algorithm (see Wikipedia).

                    Parameters
                    ----------
                    group_size : int
                        Size of the group (maximum number of players/elements that can adopt each possible strategy).
                    nb_strategies : int
                        number of strategies that can be assigned to players.

                    Returns
                    -------
                    int
                        Number of states (possible combinations of strategies and players).

                    See Also
                    --------
                    egttools.numerical.calculate_state, egttools.numerical.sample_simplex
                    )pbdoc",
          py::arg("group_size"), py::arg("nb_strategies"));

    m.def("calculate_strategies_distribution",
          static_cast<egttools::Vector (*)(size_t, size_t, egttools::SparseMatrix2D &)>(&egttools::utils::calculate_strategies_distribution),
          R"pbdoc(
                        Calculates the average frequency of each strategy available in
                        the population given the stationary distribution.

                        Parameters
                        ----------
                        pop_size : int
                            Size of the population.
                        nb_strategies : int
                            Number of strategies that can be assigned to players.
                        stationary_distribution : scipy.sparse.csr_matrix
                            A sparse matrix which contains the stationary distribution (the frequency with which the evolutionary system visits each
                            stationary state).

                        Returns
                        -------
                        numpy.ndarray[numpy.float64[m, 1]]
                            Average frequency of each strategy in the stationary evolutionary system.

                        See Also
                        --------
                        egttools.numerical.calculate_state, egttools.numerical.sample_simplex,
                        egttools.numerical.calculate_nb_states, egttools.numerical.PairwiseMoran.stationary_distribution_sparse
                        )pbdoc",
          py::arg("pop_size"), py::arg("nb_strategies"), py::arg("stationary_distribution"));

    py::class_<egttools::FinitePopulations::NormalFormGame, egttools::FinitePopulations::AbstractGame>(mGames, "NormalFormGame")
            .def(py::init<size_t, const Eigen::Ref<const Matrix2D> &>(),
                 R"pbdoc(
                    Normal Form Game. This constructor assumes that there are only two possible strategies and two possible actions.

                    This method will run the game using the players and player types defined in :param group_composition,
                    and will update the vector :param game_payoffs with the resulting payoff of each player.

                    Parameters
                    ----------
                    nb_rounds : int
                        Number of rounds of the game.
                    payoff_matrix : numpy.ndarray[float]
                        A payoff matrix of shape (nb_actions, nb_actions).

                    See Also
                    --------
                    egttools.games.AbstractGame
                    )pbdoc",
                 py::arg("nb_rounds"),
                 py::arg("payoff_matrix"))
            .def(py::init(&egttools::init_normal_form_game_from_python_list),
                 R"pbdoc(
                    Normal Form Game. This constructor allows you to define any number of strategies
                    by passing a list of pointers to them. All strategies must by of type AbstractNFGStrategy *.

                    Parameters
                    ----------
                    nb_rounds : int
                        Number of rounds of the game.
                    payoff_matrix : numpy.ndarray[float]
                        A payoff matrix of shape (nb_actions, nb_actions).
                    strategies : List[egttools.behaviors.AbstractNFGStrategy]
                        A list containing references of AbstractNFGStrategy strategies (or child classes).

                    See Also
                    --------
                    egttools.games.AbstractGame
                    )pbdoc",
                 py::arg("nb_rounds"),
                 py::arg("payoff_matrix"), py::arg("strategies"), py::return_value_policy::reference_internal)
            .def("play", &egttools::FinitePopulations::NormalFormGame::play)
            .def("calculate_payoffs", &egttools::FinitePopulations::NormalFormGame::calculate_payoffs,
                 "updates the internal payoff and coop_level matrices by calculating the payoff of each strategy "
                 "given any possible strategy pair")
            .def("calculate_fitness", &egttools::FinitePopulations::NormalFormGame::calculate_fitness,
                 "calculates the fitness of an individual of a given strategy given a population state."
                 "It always assumes that the population state does not contain the current individual",
                 py::arg("player_strategy"),
                 py::arg("pop_size"), py::arg("population_state"))
            .def("calculate_cooperation_rate", &egttools::FinitePopulations::NormalFormGame::calculate_cooperation_level,
                 "calculates the rate/level of cooperation in the population at a given state.",
                 py::arg("population_size"), py::arg("population_state"))
            .def("__str__", &egttools::FinitePopulations::NormalFormGame::toString)
            .def("type", &egttools::FinitePopulations::NormalFormGame::type)
            .def("payoffs", &egttools::FinitePopulations::NormalFormGame::payoffs)
            .def("payoff", &egttools::FinitePopulations::NormalFormGame::payoff,
                 "returns the payoff of a strategy given a strategy pair.", py::arg("strategy"),
                 py::arg("strategy pair"))
            .def("expected_payoffs", &egttools::FinitePopulations::NormalFormGame::expected_payoffs, "returns the expected payoffs of each strategy vs another")
            .def_property_readonly("nb_strategies", &egttools::FinitePopulations::NormalFormGame::nb_strategies,
                                   "Number of different strategies which are playing the game.")
            .def_property_readonly("nb_rounds", &egttools::FinitePopulations::NormalFormGame::nb_rounds,
                                   "Number of rounds of the game.")
            .def_property_readonly("nb_states", &egttools::FinitePopulations::NormalFormGame::nb_states,
                                   "Number of combinations of 2 strategies that can be matched in the game.")
            .def_property_readonly("strategies", &egttools::FinitePopulations::NormalFormGame::strategies,
                                   "A list with pointers to the strategies that are playing the game.")
            .def("save_payoffs", &egttools::FinitePopulations::NormalFormGame::save_payoffs,
                 "Saves the payoff matrix in a txt file.");

    py::class_<egttools::FinitePopulations::CRDGame, egttools::FinitePopulations::AbstractGame>(mGames, "CRDGame")
            .def(py::init(&egttools::init_crd_game_from_python_list),
                 R"pbdoc(
                    Collective Risk Dilemma. This allows you to define any number of strategies by passing them
                    as a list. All strategies must be of type AbstractCRDStrategy *.

                    The CRD dilemma implemented here follows the description of:
                    Milinski, M., Sommerfeld, R. D., Krambeck, H.-J., Reed, F. A.,
                    & Marotzke, J. (2008). The collective-risk social dilemma and the prevention of simulated
                    dangerous climate change. Proceedings of the National Academy of Sciences of the United States of America, 105(7),
                    2291–2294. https://doi.org/10.1073/pnas.0709546105

                    Parameters
                    ----------
                    endowment : int
                        Initial endowment for all players.
                    threshold : int
                        Collective target that the group must reach.
                    nb_rounds : int
                        Number of rounds of the game.
                    group_size : int
                        Size of the group that will play the CRD.
                    risk : float
                        The probability that all members will lose their remaining endowment if the threshold is not achieved.
                    strategies : List[egttools.behaviors.AbstractCRDStrategy]
                        A list containing references of AbstractCRDStrategy strategies (or child classes).

                    See Also
                    --------
                    egttools.games.AbstractGame
                    egttools.games.NormalFormGame
                    )pbdoc",
                 py::arg("endowment"),
                 py::arg("threshold"),
                 py::arg("nb_rounds"),
                 py::arg("group_size"),
                 py::arg("risk"),
                 py::arg("strategies"), py::return_value_policy::reference_internal)
            .def("play", &egttools::FinitePopulations::CRDGame::play)
            .def("calculate_payoffs", &egttools::FinitePopulations::CRDGame::calculate_payoffs,
                 "updates the internal payoff and coop_level matrices by calculating the payoff of each strategy "
                 "given any possible strategy pair")
            .def("calculate_fitness", &egttools::FinitePopulations::CRDGame::calculate_fitness,
                 "calculates the fitness of an individual of a given strategy given a population state."
                 "It always assumes that the population state does not contain the current individual",
                 py::arg("player_strategy"),
                 py::arg("pop_size"), py::arg("population_state"))
            .def("calculate_population_group_achievement", &egttools::FinitePopulations::CRDGame::calculate_population_group_achievement,
                 "calculates the group achievement in the population at a given state.",
                 py::arg("population_size"), py::arg("population_state"))
            .def("calculate_group_achievement", &egttools::FinitePopulations::CRDGame::calculate_group_achievement,
                 "calculates the group achievement for a given stationary distribution.",
                 py::arg("population_size"), py::arg("stationary_distribution"))
            .def("calculate_polarization", &egttools::FinitePopulations::CRDGame::calculate_polarization,
                 "calculates the fraction of players that contribute above, below or equal to the fair contribution (E/2)"
                 "in a give population state.",
                 py::arg("population_size"), py::arg("population_state"))
            .def("calculate_polarization_success", &egttools::FinitePopulations::CRDGame::calculate_polarization_success,
                 "calculates the fraction of players (from successful groups)) that contribute above, below or equal to the fair contribution (E/2)"
                 "in a give population state.",
                 py::arg("population_size"), py::arg("population_state"))
            .def("__str__", &egttools::FinitePopulations::CRDGame::toString)
            .def("type", &egttools::FinitePopulations::CRDGame::type)
            .def("payoffs", &egttools::FinitePopulations::CRDGame::payoffs, "returns the expected payoffs of each strategy vs each possible game state")
            .def("payoff", &egttools::FinitePopulations::CRDGame::payoff,
                 "returns the payoff of a strategy given a group composition.", py::arg("strategy"),
                 py::arg("strategy pair"))
            .def_property_readonly("nb_strategies", &egttools::FinitePopulations::CRDGame::nb_strategies,
                                   "Number of different strategies which are playing the game.")
            .def_property_readonly("endowment", &egttools::FinitePopulations::CRDGame::endowment,
                                   "Initial endowment for all players.")
            .def_property_readonly("target", &egttools::FinitePopulations::CRDGame::target,
                                   "Collective target which needs to be achieved by the group.")
            .def_property_readonly("group_size", &egttools::FinitePopulations::CRDGame::group_size,
                                   "Size of the group which will play the game.")
            .def_property_readonly("risk", &egttools::FinitePopulations::CRDGame::risk,
                                   "Probability that all players will lose their remaining endowment if the target si not achieved.")
            .def_property_readonly("nb_rounds", &egttools::FinitePopulations::CRDGame::nb_rounds,
                                   "Number of rounds of the game.")
            .def_property_readonly("nb_states", &egttools::FinitePopulations::CRDGame::nb_states,
                                   "Number of combinations of 2 strategies that can be matched in the game.")
            .def_property_readonly("strategies", &egttools::FinitePopulations::CRDGame::strategies,
                                   "A list with pointers to the strategies that are playing the game.")
            .def("save_payoffs", &egttools::FinitePopulations::CRDGame::save_payoffs,
                 "Saves the payoff matrix in a txt file.");

    py::class_<egttools::FinitePopulations::behaviors::AbstractNFGStrategy, stubs::PyAbstractNFGStrategy>(mNF, "AbstractNFGStrategy")
            .def(py::init<>())
            .def("get_action", &egttools::FinitePopulations::behaviors::AbstractNFGStrategy::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.behaviors.NormalForm.TwoActions.Cooperator, egttools.behaviors.NormalForm.TwoActions.Defector,
                egttools.behaviors.NormalForm.TwoActions.Random, egttools.behaviors.NormalForm.TwoActions.TFT,
                egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT, egttools.behaviors.NormalForm.TwoActions.GenerousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::AbstractNFGStrategy::type, "Returns a string indicating the Strategy Type.")
            .def_property_readonly("is_stochastic", &egttools::FinitePopulations::behaviors::AbstractNFGStrategy::isStochastic,
                                   "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::AbstractCRDStrategy, stubs::PyAbstractCRDStrategy>(mCRD, "AbstractCRDStrategy")
            .def(py::init<>())
            .def("get_action", &egttools::FinitePopulations::behaviors::AbstractCRDStrategy::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                group_contributions_prev : int
                    Sum of contributions of the other members of the group (excluding the focal player) in the previous round.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.behaviors.CRD.CRDMemoryOnePlayer,
                egttools.behaviors.NormalForm.TwoActions.Cooperator, egttools.behaviors.NormalForm.TwoActions.Defector,
                egttools.behaviors.NormalForm.TwoActions.Random, egttools.behaviors.NormalForm.TwoActions.TFT,
                egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT, egttools.behaviors.NormalForm.TwoActions.GenerousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("group_contributions_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::AbstractCRDStrategy::type, "Returns a string indicating the Strategy Type.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::Cooperator,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Cooperator")
            .def(py::init<>(), "This strategy always cooperates.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::Cooperator::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Defector,
                egttools.behaviors.NormalForm.TwoActions.Random, egttools.behaviors.NormalForm.TwoActions.TFT,
                egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT, egttools.behaviors.NormalForm.TwoActions.GenerousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::Cooperator::type, "Returns a string indicating the Strategy Type.")
            .def_property_readonly("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::Cooperator::isStochastic,
                                   "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::Defector,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Defector")
            .def(py::init<>(), "This strategy always defects.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::Defector::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Random, egttools.behaviors.NormalForm.TwoActions.TFT,
                egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT, egttools.behaviors.NormalForm.TwoActions.GenerousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::Defector::type, "Returns a string indicating the Strategy Type.")
            .def_property_readonly("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::Defector::isStochastic,
                                   "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::RandomPlayer,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Random")
            .def(py::init<>(), "This players chooses cooperation with uniform random probability.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.TFT,
                egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT, egttools.behaviors.NormalForm.TwoActions.GenerousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::type, "Returns a string indicating the Strategy Type.")
            .def_property_readonly("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::isStochastic,
                                   "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::TitForTat,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "TFT")
            .def(py::init<>(), "Tit for Tat: Cooperates in the first round and imitates the opponent's move thereafter.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::TitForTat::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT, egttools.behaviors.NormalForm.TwoActions.GenerousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::TitForTat::type, "Returns a string indicating the Strategy Type.")
            .def_property_readonly("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::TitForTat::isStochastic,
                                   "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "SuspiciousTFT")
            .def(py::init<>(), "Defects on the first round and imitates its opponent's previous move thereafter.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.GenerousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::type, "Returns a string indicating the Strategy Type.")
            .def_property_readonly("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::isStochastic,
                                   "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::GenerousTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "GenerousTFT")
            .def(py::init<double, double, double, double>(),
                 "Cooperates on the first round and after its opponent "
                 "cooperates. Following a defection,it cooperates with probability\n"
                 "@f[ p(R,P,T,S) = min{1 - \\frac{T-R}{R-S}, \\frac{R-P}{T-P}} @f]\n"
                 "where R, P, T and S are the reward, punishment, temptation and "
                 "suckers payoffs.",
                 py::arg("R"), py::arg("P"),
                 py::arg("T"), py::arg("S"))
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::type, "Returns a string indicating the Strategy Type.")
            .def_property_readonly("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::isStochastic,
                                   "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::GradualTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "GradualTFT")
            .def(py::init<>(),
                 "TFT with two differences:\n"
                 "(1) it increases the string of punishing defection responses "
                 "with each additional defection by its opponent\n"
                 "(2) it apologizes for each string of defections"
                 "by cooperating in the subsequent two rounds.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::GradualTFT::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::GradualTFT::type, "Returns a string indicating the Strategy Type.")
            .def_property_readonly("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::GradualTFT::isStochastic,
                                   "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "ImperfectTFT")
            .def(py::init<double>(),
                 "Imitates opponent as in TFT, but makes mistakes "
                 "with :param error_probability.",
                 py::arg("error_probability"))
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::type, "Returns a string indicating the Strategy Type.")
            .def_property_readonly("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::isStochastic,
                                   "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::TFTT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "TFTT")
            .def(py::init<>(), "Tit for 2 tats: Defects if defected twice.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::TFTT::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::TFTT::type, "Returns a string indicating the Strategy Type.")
            .def_property_readonly("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::TFTT::isStochastic,
                                   "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::TTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "TTFT")
            .def(py::init<>(), "2 Tits for tat: Defects twice if defected.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::TTFT::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT, egttools.behaviors.NormalForm.TwoActions.TFTT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::TTFT::type, "Returns a string indicating the Strategy Type.")
            .def_property_readonly("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::TTFT::isStochastic,
                                   "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::GRIM,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "GRIM")
            .def(py::init<>(),
                 "Grim (Trigger): Cooperates until its opponent has defected once, "
                 "and then defects for the rest of the game.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::GRIM::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT, egttools.behaviors.NormalForm.TwoActions.TFTT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::GRIM::type, "Returns a string indicating the Strategy Type.")
            .def_property_readonly("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::GRIM::isStochastic,
                                   "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::Pavlov,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Pavlov")
            .def(py::init<>(),
                 "Win-stay loose-shift: Cooperates if it and its opponent moved alike in"
                 "previous move and defects if they moved differently.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::Pavlov::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT, egttools.behaviors.NormalForm.TwoActions.TFTT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.GRIM
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::Pavlov::type, "Returns a string indicating the Strategy Type.")
            .def_property_readonly("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::Pavlov::isStochastic,
                                   "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::ActionInertia,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "ActionInertia")
            .def(py::init<double, double>(),
                 R"pbdoc(
                Always repeats the same action, but explores a different action with
                probability :param epsilon. In the first round it will cooperate with
                probability :param p.

                Parameters
                ----------
                epsilon : double
                    Probability of changing action.
                p : double
                    Probability of cooperation in the first round

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT, egttools.behaviors.NormalForm.TwoActions.TFTT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.GRIM
                )pbdoc",
                 py::arg("epsilon"), py::arg("p"))
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::ActionInertia::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT, egttools.behaviors.NormalForm.TwoActions.TFTT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.GRIM
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::ActionInertia::type, "Returns a string indicating the Strategy Type.")
            .def_property_readonly("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::ActionInertia::isStochastic,
                                   "Indicates whether the strategy is stochastic.");


    py::class_<egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer,
               egttools::FinitePopulations::behaviors::AbstractCRDStrategy>(mCRD, "CRDMemoryOnePlayer")
            .def(py::init<int, int, int, int, int>(),
                 R"pbdoc(
                This strategy contributes in function of the contributions of the rest
                of the group in the previous round.

                This strategy contributes @param initial_action in the first round of the game,
                afterwards compares the sum of contributions of the other members of the group
                in the previous round (a_{-i}(t-1)) to a :param personal_threshold. If the
                a_{-i}(t-1)) > personal_threshold the agent contributions :param action_above,
                if a_{-i}(t-1)) = personal_threshold it contributes :param action_equal
                or if a_{-i}(t-1)) < personal_threshold it contributes :param action_below.

                Parameters
                ----------
                personal_threshold : int
                    threshold value compared to the contributions of the other members of the group
                initial_action : int
                    Contribution in the first round
                action_above : int
                    contribution if a_{-i}(t-1)) > personal_threshold
                action_equal : int
                    contribution if a_{-i}(t-1)) = personal_threshold
                action_below : int
                    contribution if a_{-i}(t-1)) < personal_threshold

                See Also
                --------
                egttools.behaviors.AbstractGame,
                )pbdoc",
                 py::arg("personal_threshold"), py::arg("initial_action"),
                 py::arg("action_above"), py::arg("action_equal"), py::arg("action_below"))
            .def("get_action", &egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer::get_action,
                 R"pbdoc(
                Returns an action in function of time_step
                round and the previous action action_prev
                of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                group_contributions_prev : int
                    Sum of contributions of the other members of the group (without
                    the focal player) in the previous round.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.behaviors.AbstractGame,
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT, egttools.behaviors.NormalForm.TwoActions.TFTT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.GRIM
                )pbdoc",
                 py::arg("time_step"), py::arg("group_contributions_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer::type, "Returns a string indicating the Strategy Type.")
            .def("__str__", &egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer::toString);

    py::class_<PairwiseComparison>(m, "PairwiseMoran")
            .def(py::init<size_t, egttools::FinitePopulations::AbstractGame &, size_t>(),
                 "Runs a moran process with pairwise comparison and calculates fitness according to game",
                 py::arg("pop_size"), py::arg("game"), py::arg("cache_size"), py::keep_alive<1, 3>())
            .def("evolve",
                 static_cast<egttools::VectorXui (PairwiseComparison::*)(size_t, double, double,
                                                                         const Eigen::Ref<const egttools::VectorXui> &)>(&PairwiseComparison::evolve),
                 py::keep_alive<1, 5>(),
                 "evolves the strategies for a maximum of nb_generations", py::arg("nb_generations"), py::arg("beta"),
                 py::arg("mu"), py::arg("init_state"))
            .def("run",
                 static_cast<egttools::MatrixXui2D (PairwiseComparison::*)(size_t, double,
                                                                           const Eigen::Ref<const egttools::VectorXui> &)>(&PairwiseComparison::run),
                 "runs the moran process with social imitation and returns a matrix with all the states the system went through",
                 py::arg("nb_generations"),
                 py::arg("beta"),
                 py::arg("init_state"))
            .def("run",
                 static_cast<egttools::MatrixXui2D (PairwiseComparison::*)(size_t, double, double,
                                                                           const Eigen::Ref<const egttools::VectorXui> &)>(&PairwiseComparison::run),
                 "runs the moran process with social imitation and returns a matrix with all the states the system went through",
                 py::arg("nb_generations"),
                 py::arg("beta"),
                 py::arg("mu"),
                 py::arg("init_state"))
            .def("fixation_probability", &PairwiseComparison::fixationProbability,
                 "Estimates the fixation probability of an strategy in the population.",
                 py::arg("mutant"), py::arg("resident"), py::arg("nb_runs"), py::arg("nb_generations"), py::arg("beta"))
            .def("stationary_distribution", &PairwiseComparison::stationaryDistribution,
                 py::call_guard<py::gil_scoped_release>(),
                 "Estimates the stationary distribution of the population of strategies given the game.",
                 py::arg("nb_runs"), py::arg("nb_generations"), py::arg("transitory"), py::arg("beta"), py::arg("mu"))
            .def("stationary_distribution_sparse", &PairwiseComparison::estimate_stationary_distribution_sparse,
                 py::call_guard<py::gil_scoped_release>(),
                 "Estimates the stationary distribution of the population of strategies given the game.",
                 py::arg("nb_runs"), py::arg("nb_generations"), py::arg("transitory"), py::arg("beta"), py::arg("mu"))
            .def_property_readonly("nb_strategies", &PairwiseComparison::nb_strategies, "Number of strategies in the population.")
            .def_property_readonly("payoffs", &PairwiseComparison::payoffs,
                                   "Payoff matrix containing the payoff of each strategy (row) for each game state (column)")
            .def_property("pop_size", &PairwiseComparison::population_size, &PairwiseComparison::set_population_size,
                          "Size of the population.")
            .def_property("cache_size", &PairwiseComparison::cache_size, &PairwiseComparison::set_cache_size,
                          "Maximum memory which can be used to cache the fitness calculations.");

    py::class_<egttools::DataStructures::DataTable>(mData, "DataTable")
            .def(py::init<size_t,
                          size_t,
                          std::vector<std::string> &,
                          std::vector<std::string> &>(),
                 "Data structure that allows to store information in table format. Headers give the ",
                 py::arg("nb_rows"),
                 py::arg("nb_columns"),
                 py::arg("headers"),
                 py::arg("column_types"))
            .def_readonly("rows", &egttools::DataStructures::DataTable::nb_rows, "returns the number of rows")
            .def_readonly("cols", &egttools::DataStructures::DataTable::nb_columns, "returns the number of columns")
            .def_readwrite("data", &egttools::DataStructures::DataTable::data, py::return_value_policy::reference_internal)
            .def_readwrite("headers", &egttools::DataStructures::DataTable::header,
                           py::return_value_policy::reference_internal)
            .def_readwrite("column_types", &egttools::DataStructures::DataTable::column_types,
                           py::return_value_policy::reference_internal);

    m.def("call_get_action", &egttools::call_get_action,
          "Returns a string with a tuple of actions",
          py::arg("strategies"),
          py::arg("time_step"),
          py::arg("prev_action"));

    mDistributions.def("multivariate_hypergeometric_pdf",
                       static_cast<double (*)(size_t, size_t, size_t, const std::vector<size_t> &,
                                              const Eigen::Ref<const VectorXui> &)>(&egttools::multivariateHypergeometricPDF),
                       "Calculates the probability density function of a multivariate hypergeometric distribution.",
                       py::arg("m"),
                       py::arg("k"),
                       py::arg("n"),
                       py::arg("sample_counts"),
                       py::arg("population_counts"));
}