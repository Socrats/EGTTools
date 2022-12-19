/** Copyright (c) 2019-2023  Elias Fernandez
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

#define XSTR(s) STR(s)
#define STR(s) #s

namespace py = pybind11;
using namespace std::string_literals;
using namespace egttools;
using PairwiseComparison = egttools::FinitePopulations::PairwiseMoran<egttools::Utils::LRUCache<std::string, double>>;

namespace egttools {

    std::string call_get_action(const py::list &strategies, size_t time_step, int action) {
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
                                                                                         double enhancement_factor,
                                                                                         const py::list &strategies) {
        egttools::FinitePopulations::CRDStrategyVector strategies_cpp;
        for (py::handle strategy : strategies) {
            strategies_cpp.push_back(py::cast<egttools::FinitePopulations::behaviors::AbstractCRDStrategy *>(strategy));
        }
        return std::make_unique<egttools::FinitePopulations::CRDGame>(endowment, threshold, nb_rounds,
                                                                      group_size, risk, enhancement_factor, strategies_cpp);
    }

    std::unique_ptr<egttools::FinitePopulations::games::CRDGameTU> init_crd_tu_game_from_python_list(int endowment,
                                                                                                     int threshold,
                                                                                                     int nb_rounds,
                                                                                                     int group_size,
                                                                                                     double risk,
                                                                                                     egttools::utils::TimingUncertainty<> tu,
                                                                                                     const py::list &strategies) {
        egttools::FinitePopulations::games::CRDStrategyVector strategies_cpp;
        for (py::handle strategy : strategies) {
            strategies_cpp.push_back(py::cast<egttools::FinitePopulations::behaviors::AbstractCRDStrategy *>(strategy));
        }
        return std::make_unique<egttools::FinitePopulations::games::CRDGameTU>(endowment, threshold, nb_rounds,
                                                                               group_size, risk, tu, strategies_cpp);
    }

    egttools::VectorXli sample_simplex_directly(int64_t nb_strategies, int64_t pop_size) {
        std::mt19937_64 generator(egttools::Random::SeedGenerator::getInstance().getSeed());
        egttools::VectorXli state = egttools::VectorXli::Zero(nb_strategies);

        egttools::FinitePopulations::sample_simplex_direct_method<long int, long int, egttools::VectorXli, std::mt19937_64>(nb_strategies, pop_size, state, generator);

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

//namespace {
//
////    inline std::string attr_doc(const py::module_ &m, const char *name, const char *doc) {
////        auto attr = m.attr(name);
////        return ".. data:: "s + name + "\n    :annotation: = "s + py::cast<std::string>(py::repr(attr)) + "\n\n    "s + doc + "\n\n"s;
////    }
//
//}// namespace

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
            and returns an instance of egttools.Random which is used
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
            and returns an instance of `egttools.Random` which is used
            to seed the random generators used across `egttools`.

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
                    "The initial seed of `egttools.Random`.")
            .def_static(
                    "generate", []() {
                        return egttools::Random::SeedGenerator::getInstance().getSeed();
                    },
                    R"pbdoc(
                    Generates a random seed.

                    The generated seed can be used to seed other pseudo-random generators,
                    so that the initial state of the simulation can always be tracked and
                    the simulation can be reproduced. This is very important both for debugging
                    purposes as well as for scientific research. However, this approach should
                    NOT be used in any cryptographic applications, it is NOT safe.

                    Returns
                    -------
                    int
                        A random seed which can be used to seed new random generators.
                    )pbdoc")
            .def_static(
                    "seed", [](unsigned long int seed) {
                        egttools::Random::SeedGenerator::getInstance().setMainSeed(seed);
                    },
                    R"pbdoc(
                    This static methods changes the seed of `egttools.Random`.

                    Parameters
                    ----------
                    int
                        The new seed for the `egttools.Random` module which is used to seed
                        every other pseudo-random generation in the `egttools` package.
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

    mGames.attr("__init__") = py::str("The `egttools.numerical.games` submodule contains the available games.");
    mBehaviors.attr("__init__") = py::str("The `egttools.numerical.behaviors` submodule contains the available strategies to evolve.");
    mNF.attr("__init__") = py::str("The `egttools.numerical.behaviors.NormalForm` submodule contains the strategies for normal form games.");
    mCRD.attr("__init__") = py::str("The `egttools.numerical.behaviors.CRD` submodule contains the strategies for the CRD.");
    mNFTwoActions.attr("__init__") = py::str(
            "The `TwoActions` submodule contains the strategies for "
            "normal form games with 2 actions.");
    mData.attr("__init__") = py::str("The `egttools.numerical.DataStructures` submodule contains helpful data structures.");
    mDistributions.attr("__init__") = py::str(
            "The `egttools.numerical.distributions` submodule contains "
            "functions and classes that produce stochastic distributions.");

    py::class_<egttools::utils::TimingUncertainty<>>(mDistributions, "TimingUncertainty")
            .def(py::init<double, int>(),
                 R"pbdoc(
                    Timing uncertainty distribution container.

                    This class provides methods to calculate the final round of the game according to some predifined distribution, which is geometric by default.

                    Parameters
                    ----------
                    p : float
                        Probability that the game will end after the minimum number of rounds.
                    max_rounds : int
                        maximum number of rounds that the game can take (if 0, there is no maximum).
                    )pbdoc",
                 py::arg("p"), py::arg("max_rounds") = 0)
            .def("calculate_end", &egttools::utils::TimingUncertainty<>::calculate_end,
                 "Calculates the final round limiting by max_rounds, i.e., outputs a value between"
                 "[min_rounds, max_rounds].",
                 py::arg("min_rounds"), py::arg("random_generator"))
            .def("calculate_full_end", &egttools::utils::TimingUncertainty<>::calculate_full_end,
                 "Calculates the final round, i.e., outputs a value between"
                 "[min_rounds, Inf].",
                 py::arg("min_rounds"), py::arg("random_generator"))
            .def_property_readonly("p", &egttools::utils::TimingUncertainty<>::probability)
            .def_property("max_rounds", &egttools::utils::TimingUncertainty<>::max_rounds,
                          &egttools::utils::TimingUncertainty<>::set_max_rounds);

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
                    numpy.ndarray[numpy.float64[m, n]]
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
                    strategy_index : int
                        The index of the strategy used by the player.
                    pop_size : int
                        The size of the population.
                    strategies : numpy.ndarray[numpy.uint64[m, 1]]
                        A vector of counts of each strategy. The current state of the population.

                    Returns
                    -------
                    float
                        The fitness of the strategy in the population state given by strategies.
                    )pbdoc",
                 py::arg("strategy_index"), py::arg("pop_size"), py::arg("strategies"))
            .def("__str__", &egttools::FinitePopulations::AbstractGame::toString)
            .def("type", &egttools::FinitePopulations::AbstractGame::type, "returns the type of game.")
            .def("payoffs", &egttools::FinitePopulations::AbstractGame::payoffs,
                 R"pbdoc(
                    Returns the payoff matrix of the game.

                    Returns
                    -------
                    numpy.ndarray
                        The payoff matrix.
                    )pbdoc")
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
            .def("nb_strategies", &egttools::FinitePopulations::AbstractGame::nb_strategies,
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

    py::class_<egttools::FinitePopulations::AbstractNPlayerGame, stubs::PyAbstractNPlayerGame, egttools::FinitePopulations::AbstractGame>(mGames, "AbstractNPlayerGame")
            .def(py::init_alias<int, int>(),
                 R"pbdoc(
                    Abstract N-Player Game.

                    This abstract Game class can be used in most scenarios where the fitness of a strategy is calculated as its
                    expected payoff given the population state.

                    It assumes that the game is N player, since the fitness of a strategy given a population state is calculated
                    as the expected payoff of that strategy over all possible group combinations in the given state.

                    Notes
                    -----
                    It might be a good idea to overwrite the methods `__str__`, `type`, and `save_payoffs` to adapt to your
                    given game implementation

                    It assumes that you have at least the following attributes:
                    1. And an attribute `self.nb_strategies_` which contains the number of strategies
                    that you are going to analyse for the given game.
                    2. `self.payoffs_` which must be a numpy.ndarray and contain the payoff matrix of the game. This array
                    must be of shape (self.nb_strategies_, nb_group_configurations), where nb_group_configurations is the number
                    of possible combinations of strategies in the group. Thus, each row should give the (expected) payoff of the row
                    strategy when playing in a group with the column configuration. The `payoff` method provides an easy way to access
                    the payoffs for any group composition, by taking as arguments the index of the row strategy
                    and a List with the count of each possible strategy in the group.

                    You must still implement the methods `play` and `calculate_payoffs` which should define how the game assigns
                    payoffs to each strategy for each possible game context. In particular, `calculate_payoffs` should fill the
                    array `self.payoffs_` with the correct values as explained above. We recommend that you run this method in
                    the `__init__` (initialization of the object) since, these values must be set before passing the game object
                    to the numerical simulator (e.g., egttools.numerical.PairwiseComparisonNumerical).

                    Parameters
                    ----------
                    nb_strategies: int
                        total number of possible strategies.
                    group_size: int
                        size of the group in which the game will take place.
                    )pbdoc",
                 py::arg("nb_strategies"), py::arg("group_size"), py::return_value_policy::reference_internal)
            .def("play", &egttools::FinitePopulations::AbstractNPlayerGame::play, R"pbdoc(
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
            .def("calculate_payoffs", &egttools::FinitePopulations::AbstractNPlayerGame::calculate_payoffs,
                 R"pbdoc(
                    Estimates the payoffs for each strategy and returns the values in a matrix.
                    Each row of the matrix represents a strategy and each column a game state.
                    E.g., in case of a 2 player game, each entry a_ij gives the payoff for strategy
                    i against strategy j. In case of a group game, each entry a_ij gives the payoff
                    of strategy i for game state j, which represents the group composition.

                    Returns
                    -------
                    numpy.ndarray[numpy.float64[m, n]]
                        A matrix with the expected payoffs for each strategy given each possible game
                        state.
                    )pbdoc")
            .def("calculate_fitness", &egttools::FinitePopulations::AbstractNPlayerGame::calculate_fitness,
                 R"pbdoc(
                    Estimates the fitness for a player_type in the population with state :param strategies.

                    This function assumes that the player with strategy player_type is not included in
                    the vector of strategy counts strategies.

                    Parameters
                    ----------
                    strategy_index : int
                        The index of the strategy used by the player.
                    pop_size : int
                        The size of the population.
                    strategies : numpy.ndarray[numpy.uint64[m, 1]]
                        A vector of counts of each strategy. The current state of the population.

                    Returns
                    -------
                    float
                        The fitness of the strategy in the population state given by strategies.
                    )pbdoc",
                 py::arg("strategy_index"), py::arg("pop_size"), py::arg("strategies"))
            .def("__str__", &egttools::FinitePopulations::AbstractNPlayerGame::toString)
            .def("type", &egttools::FinitePopulations::AbstractNPlayerGame::type, "returns the type of game.")
            .def("payoffs", &egttools::FinitePopulations::AbstractNPlayerGame::payoffs,
                 R"pbdoc(
                    Returns the payoff matrix of the game.

                    Returns
                    -------
                    numpy.ndarray
                        The payoff matrix.
                    )pbdoc")
            .def("payoff", &egttools::FinitePopulations::AbstractNPlayerGame::payoff,
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
            .def("update_payoff", &egttools::FinitePopulations::AbstractNPlayerGame::update_payoff, "update an entry of the payoff matrix",
                 py::arg("strategy_index"), py::arg("group_configuration_index"), py::arg("value"))
            .def("nb_strategies", &egttools::FinitePopulations::AbstractNPlayerGame::nb_strategies,
                 "Number of different strategies playing the game.")
            .def("group_size", &egttools::FinitePopulations::AbstractNPlayerGame::group_size,
                 "Size of the group.")
            .def("nb_group_configurations", &egttools::FinitePopulations::AbstractNPlayerGame::nb_group_configurations,
                 "Number of different group configurations.")
            .def("save_payoffs", &egttools::FinitePopulations::AbstractNPlayerGame::save_payoffs,
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
                    group_composition : numpy.ndarray[numpy.int64[m, 1]]
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
                    numpy.ndarray[numpy.int64[m, 1]]
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
                    numpy.ndarray[numpy.int64[m, 1]]
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
                    numpy.ndarray[numpy.int64[m, 1]]
                        Vector with the sampled state.

                    See Also
                    --------
                    egttools.numerical.calculate_state, egttools.numerical.calculate_nb_states, egttools.numerical.sample_simplex
                    )pbdoc",
          py::arg("nb_strategies"));

#if (HAS_BOOST)
    m.def(
            "calculate_nb_states", [](size_t group_size, size_t nb_strategies) {
                auto result = starsBars<size_t, mp::uint128_t>(group_size, nb_strategies);
                return result.convert_to<size_t>();
            },
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
#else
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
#endif

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
                        egttools.numerical.calculate_nb_states, egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution
                        egttools.numerical.calculate_nb_states, egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse
                        )pbdoc",
          py::arg("pop_size"), py::arg("nb_strategies"), py::arg("stationary_distribution"));

    py::class_<egttools::FinitePopulations::NormalFormGame, egttools::FinitePopulations::AbstractGame>(mGames, "NormalFormGame")
            .def(py::init<size_t, const Eigen::Ref<const Matrix2D> &>(),
                 R"pbdoc(
                    Normal Form Game. This constructor assumes that there are only two possible strategies and two possible actions.

                    This class will run the game using the players and player types defined in :param group_composition,
                    and will update the vector :param game_payoffs with the resulting payoff of each player.

                    Parameters
                    ----------
                    nb_rounds : int
                        Number of rounds of the game.
                    payoff_matrix : numpy.ndarray[numpy.float64[m, m]]
                        A payoff matrix of shape (nb_actions, nb_actions).

                    See Also
                    --------
                    egttools.games.AbstractGame,
                    egttools.games.CRDGame,
                    egttools.games.CRDGameTU,
                    egttools.behaviors.NormalForm.TwoActions
                    )pbdoc",
                 py::arg("nb_rounds"),
                 py::arg("payoff_matrix"), py::return_value_policy::reference_internal)
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
            .def("play", &egttools::FinitePopulations::NormalFormGame::play,
                 R"pbdoc(
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
            .def("calculate_payoffs", &egttools::FinitePopulations::NormalFormGame::calculate_payoffs,
                 R"pbdoc(
                    Estimates the payoffs for each strategy and returns the values in a matrix.
                    Each row of the matrix represents a strategy and each column a game state.
                    E.g., in case of a 2 player game, each entry a_ij gives the payoff for strategy
                    i against strategy j. In case of a group game, each entry a_ij gives the payoff
                    of strategy i for game state j, which represents the group composition.

                    This method also updates a matrix that stores the cooperation level of each strategy
                    against any other.

                    Returns
                    -------
                    numpy.ndarray[numpy.float64[m, n]]
                        A matrix with the expected payoffs for each strategy given each possible game
                        state.
                    )pbdoc")
            .def("calculate_fitness", &egttools::FinitePopulations::NormalFormGame::calculate_fitness,
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
                    strategies : numpy.ndarray[numpy.uint64[m, 1]]
                        A vector of counts of each strategy. The current state of the population.

                    Returns
                    -------
                    float
                        The fitness of the strategy in the population state given by strategies.
                    )pbdoc",
                 py::arg("player_strategy"),
                 py::arg("population_size"), py::arg("population_state"))
            .def("calculate_cooperation_rate", &egttools::FinitePopulations::NormalFormGame::calculate_cooperation_level,
                 R"pbdoc(
                    Calculates the rate/level of cooperation in the population at a given population state.

                    Parameters
                    ----------
                    population_size : int
                        The size of the population.
                    population_state : numpy.ndarray[numpy.uint64[m, 1]]
                        A vector of counts of each strategy in the population.
                        The current state of the population.

                    Returns
                    -------
                    float
                        The level of cooperation at the population_state.
                    )pbdoc",
                 py::arg("population_size"), py::arg("population_state"))
            .def("__str__", &egttools::FinitePopulations::NormalFormGame::toString)
            .def("type", &egttools::FinitePopulations::NormalFormGame::type)
            .def("payoffs", &egttools::FinitePopulations::NormalFormGame::payoffs,
                 R"pbdoc(
                    Returns the payoff matrix of the game.

                    Returns
                    -------
                    numpy.ndarray
                        The payoff matrix.
                    )pbdoc",
                 py::return_value_policy::reference_internal)
            .def("payoff", &egttools::FinitePopulations::NormalFormGame::payoff,
                 R"pbdoc(
                    Returns the payoff of a strategy given a strategy pair.

                    If the group composition does not include the strategy, the payoff should be zero.

                    Parameters
                    ----------
                    strategy : int
                        The index of the strategy used by the player.
                    strategy_pair : List[int]
                        List with the group composition. The structure of this list
                        depends on the particular implementation of this abstract method.

                    Returns
                    -------
                    float
                        The payoff value.
                    )pbdoc",
                 py::arg("strategy"),
                 py::arg("strategy_pair"))
            .def("expected_payoffs", &egttools::FinitePopulations::NormalFormGame::expected_payoffs, "returns the expected payoffs of each strategy vs another", py::return_value_policy::reference_internal)
            .def("nb_strategies", &egttools::FinitePopulations::NormalFormGame::nb_strategies,
                 "Number of different strategies which are playing the game.")
            .def_property_readonly("nb_rounds", &egttools::FinitePopulations::NormalFormGame::nb_rounds,
                                   "Number of rounds of the game.")
            .def_property_readonly("nb_states", &egttools::FinitePopulations::NormalFormGame::nb_states,
                                   "Number of combinations of 2 strategies that can be matched in the game.")
            .def_property_readonly("strategies", &egttools::FinitePopulations::NormalFormGame::strategies,
                                   "A list with pointers to the strategies that are playing the game.")
            .def("save_payoffs", &egttools::FinitePopulations::NormalFormGame::save_payoffs,
                 R"pbdoc(
                    Stores the payoff matrix in a txt file.

                    Parameters
                    ----------
                    file_name : str
                        Name of the file in which the data will be stored.
                    )pbdoc");

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
                    enhancement_factor: float
                        The payoffs of each strategy are multiplied by this factor if the target is reached
                        (this may enables the inclusion of a surplus for achieving the goal).
                    strategies : List[egttools.behaviors.CRD.AbstractCRDStrategy]
                        A list containing references of AbstractCRDStrategy strategies (or child classes).

                    See Also
                    --------
                    egttools.games.AbstractGame,
                    egttools.games.NormalFormGame,
                    egttools.behaviors.CRD.AbstractCRDStrategy
                    )pbdoc",
                 py::arg("endowment"),
                 py::arg("threshold"),
                 py::arg("nb_rounds"),
                 py::arg("group_size"),
                 py::arg("risk"),
                 py::arg("enhancement_factor"),
                 py::arg("strategies"), py::return_value_policy::reference_internal)
            .def("play", &egttools::FinitePopulations::CRDGame::play,
                 R"pbdoc(
                    Updates the vector of payoffs with the payoffs of each player after playing the game.

                    This method will run the game using the players and player types defined in :param group_composition,
                    and will update the vector :param game_payoffs with the resulting payoff of each player.

                    Parameters
                    ----------
                    group_composition : List[int]
                        A list with counts of the number of players of each strategy in the group.
                    game_payoffs : List[float]
                        A list used as container for the payoffs of each player
                    )pbdoc")
            .def("calculate_payoffs", &egttools::FinitePopulations::CRDGame::calculate_payoffs,
                 R"pbdoc(
                    Estimates the payoffs for each strategy and returns the values in a matrix.
                    Each row of the matrix represents a strategy and each column a game state.
                    Therefore, each entry a_ij gives the payoff
                    of strategy i for game state j, which represents the group composition.

                    It also updates the coop_level matrices by calculating level of cooperation
                    at any given population state

                    Returns
                    -------
                    numpy.ndarray[numpy.float64[m, n]]
                        A matrix with the expected payoffs for each strategy given each possible game
                        state.
                    )pbdoc")
            .def("calculate_fitness", &egttools::FinitePopulations::CRDGame::calculate_fitness,
                 R"pbdoc(
                    Estimates the fitness for a player_type in the population with state :param strategies.

                    This function assumes that the player with strategy player_type is not included in
                    the vector of strategy counts strategies.

                    Parameters
                    ----------
                    player_strategy : int
                        The index of the strategy used by the player.
                    pop_size : int
                        The size of the population.
                    population_state : numpy.ndarray[numpy.uint64[m, 1]]
                        A vector of counts of each strategy. The current state of the population.

                    Returns
                    -------
                    float
                        The fitness of the strategy in the population state given by strategies.
                    )pbdoc",
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
            .def("payoffs", &egttools::FinitePopulations::CRDGame::payoffs,
                 R"pbdoc(
                    Returns the expected payoffs of each strategy vs each possible game state.

                    Returns
                    -------
                    numpy.ndarray[np.float64[m,n]]
                        The payoff matrix.
                    )pbdoc")
            .def("payoff", &egttools::FinitePopulations::CRDGame::payoff,
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
                 py::arg("strategy"),
                 py::arg("group_composition"))
            .def_property_readonly("group_achievement_per_group", &egttools::FinitePopulations::CRDGame::group_achievements)
            .def("nb_strategies", &egttools::FinitePopulations::CRDGame::nb_strategies,
                 "Number of different strategies which are playing the game.")
            .def_property_readonly("endowment", &egttools::FinitePopulations::CRDGame::endowment,
                                   "Initial endowment for all players.")
            .def_property_readonly("target", &egttools::FinitePopulations::CRDGame::target,
                                   "Collective target which needs to be achieved by the group.")
            .def_property_readonly("group_size", &egttools::FinitePopulations::CRDGame::group_size,
                                   "Size of the group which will play the game.")
            .def_property_readonly("risk", &egttools::FinitePopulations::CRDGame::risk,
                                   "Probability that all players will lose their remaining endowment if the target si not achieved.")
            .def_property_readonly("enhancement_factor", &egttools::FinitePopulations::CRDGame::enhancement_factor,
                                   "The payoffs of each strategy are multiplied by this factor if the target is reached (this may enables the inclusion of a surplus for achieving the goal).")
            .def_property_readonly("nb_rounds", &egttools::FinitePopulations::CRDGame::nb_rounds,
                                   "Number of rounds of the game.")
            .def_property_readonly("nb_states", &egttools::FinitePopulations::CRDGame::nb_states,
                                   "Number of combinations of 2 strategies that can be matched in the game.")
            .def_property_readonly("strategies", &egttools::FinitePopulations::CRDGame::strategies,
                                   "A list with pointers to the strategies that are playing the game.")
            .def("save_payoffs", &egttools::FinitePopulations::CRDGame::save_payoffs,
                 "Saves the payoff matrix in a txt file.");

    py::class_<egttools::FinitePopulations::games::CRDGameTU, egttools::FinitePopulations::AbstractGame>(mGames, "CRDGameTU")
            .def(py::init(&egttools::init_crd_tu_game_from_python_list),
                 R"pbdoc(
                    This class implements a One-Shot Collective Risk Dilemma.

                    This N-player game was first introduced in "Santos, F. C., & Pacheco, J. M. (2011).
                    Risk of collective failure provides an escape from the tragedy of the commons.
                    Proceedings of the National Academy of Sciences of the United States of America, 108(26), 10421–10425.".

                    The game consists of a group of size ``group_size`` (N) which can be composed of
                    Cooperators (Cs) who will contribute a fraction ``cost`` (c) of their
                    ``endowment`` (b) to the public good. And of Defectors (Ds) who contribute 0.

                    If the total contribution of the group is equal or surpasses the collective target Mcb,
                    with M being the ``min_nb_cooperators``, then all participants will receive as payoff
                    their remaining endowment. Which is, Cs receive b - cb and Ds receive b. Otherwise, all
                    participants receive 0 endowment with a probability equal to ``risk`` (r), and will
                    keep their endowment with probability 1-r. This means that each group must have at least
                    M Cs for the collective target to be achieved.

                    Parameters
                    ----------
                    endowment : float
                        The initial endowment (b) received by all participants
                    cost : float
                        The fraction of the endowment that Cooperators contribute to the public good.
                        This value must be in the interval [0, 1]
                    risk : float
                        The risk that all members of the group will lose their remaining endowment if the
                        collective target is not achieved.
                    group_size : int
                        The size of the group (N)
                    min_nb_cooperators : int
                        The minimum number of cooperators (M) required to reach the collective target.
                        In other words, the collective target is reached if the collective effort is
                        at least Mcb. This value must be in the discrete interval [[0, N]].

                    See Also
                    --------
                    egttools.games.CRDGame, egttools.games.CRDGameTU
                    )pbdoc",
                 py::arg("endowment"),
                 py::arg("threshold"),
                 py::arg("min_rounds"),
                 py::arg("group_size"),
                 py::arg("risk"),
                 py::arg("tu"),
                 py::arg("strategies"), py::return_value_policy::reference_internal)
            .def("play", &egttools::FinitePopulations::games::CRDGameTU::play,
                 R"pbdoc(
                    Plays the One-shop CRD and update the game_payoffs given the group_composition.

                    We always assume that strategy 0 is D and strategy 1 is C.

                    The payoffs of Defectors and Cooperators are described by the following equations:

                    .. math::
                        \Pi_{D}(k) = b\{\theta(k-M)+ (1-r)[1 - \theta(k-M)]\}

                        \Pi_{C}(k) = \Pi_{D}(k) - cb

                        \text{where } \theta(x) = 0 \text{if } x < 0 \text{ and 1 otherwise.}

                    Parameters
                    ----------
                    group_composition : Union[List[int], numpy.ndarray]
                        A list or array containing the counts of how many members of each strategy are
                        present in the group.
                    game_payoffs: numpy.ndarray
                        A vector in which the payoffs of the game will be stored.
                    )pbdoc")
            .def("calculate_payoffs", &egttools::FinitePopulations::games::CRDGameTU::calculate_payoffs,
                 R"pbdoc(
                    Calculates the payoffs of every strategy in each possible group composition.

                    Returns
                    -------
                    numpy.ndarray
                        A matrix containing the payoff of each strategy in every possible group composition.
                    )pbdoc")
            .def("calculate_fitness", &egttools::FinitePopulations::games::CRDGameTU::calculate_fitness,
                 R"pbdoc(
                    Calculates the fitness of a strategy given a population state.

                    Parameters
                    ----------
                    player_type : int
                        The index of the strategy whose fitness will be calculated.
                    pop_size : int
                        The size of the population (Z).
                    population_state : numpy.ndarray
                        A vector containing the counts of each strategy in the population.

                    Returns
                    -------
                    float
                        The fitness of the strategy in the current population state.
                    )pbdoc",
                 py::arg("player_strategy"),
                 py::arg("pop_size"), py::arg("population_state"))
            .def("calculate_population_group_achievement", &egttools::FinitePopulations::games::CRDGameTU::calculate_population_group_achievement,
                 "calculates the group achievement in the population at a given state.",
                 py::arg("population_size"), py::arg("population_state"))
            .def("calculate_group_achievement", &egttools::FinitePopulations::games::CRDGameTU::calculate_group_achievement,
                 "calculates the group achievement for a given stationary distribution.",
                 py::arg("population_size"), py::arg("stationary_distribution"))
            .def("calculate_polarization", &egttools::FinitePopulations::games::CRDGameTU::calculate_polarization,
                 py::call_guard<py::gil_scoped_release>(),
                 "calculates the fraction of players that contribute above, below or equal to the fair contribution (E/2)"
                 "in a give population state.",
                 py::arg("population_size"), py::arg("population_state"))
            .def("calculate_polarization_success", &egttools::FinitePopulations::games::CRDGameTU::calculate_polarization_success,
                 py::call_guard<py::gil_scoped_release>(),
                 "calculates the fraction of players (from successful groups)) that contribute above, below or equal to the fair contribution (E/2)"
                 "in a give population state.",
                 py::arg("population_size"), py::arg("population_state"))
            .def("__str__", &egttools::FinitePopulations::games::CRDGameTU::toString)
            .def("type", &egttools::FinitePopulations::games::CRDGameTU::type)
            .def("payoffs", &egttools::FinitePopulations::games::CRDGameTU::payoffs, "returns the expected payoffs of each strategy vs each possible game state")
            .def("payoff", &egttools::FinitePopulations::games::CRDGameTU::payoff,
                 "returns the payoff of a strategy given a group composition.", py::arg("strategy"),
                 py::arg("strategy pair"))
            .def("nb_strategies", &egttools::FinitePopulations::games::CRDGameTU::nb_strategies,
                 "Number of different strategies which are playing the game.")
            .def_property_readonly("endowment", &egttools::FinitePopulations::games::CRDGameTU::endowment,
                                   "Initial endowment for all players.")
            .def_property_readonly("target", &egttools::FinitePopulations::games::CRDGameTU::target,
                                   "Collective target which needs to be achieved by the group.")
            .def_property_readonly("group_size", &egttools::FinitePopulations::games::CRDGameTU::group_size,
                                   "Size of the group which will play the game.")
            .def_property_readonly("risk", &egttools::FinitePopulations::games::CRDGameTU::risk,
                                   "Probability that all players will lose their remaining endowment if the target si not achieved.")
            .def_property_readonly("min_rounds", &egttools::FinitePopulations::games::CRDGameTU::min_rounds,
                                   "Minimum number of rounds of the game.")
            .def_property_readonly("nb_states", &egttools::FinitePopulations::games::CRDGameTU::nb_states,
                                   "Number of combinations of 2 strategies that can be matched in the game.")
            .def_property_readonly("strategies", &egttools::FinitePopulations::games::CRDGameTU::strategies,
                                   "A list with pointers to the strategies that are playing the game.")
            .def("save_payoffs", &egttools::FinitePopulations::games::CRDGameTU::save_payoffs,
                 "Saves the payoff matrix in a txt file.");

    py::class_<egttools::FinitePopulations::OneShotCRD, egttools::FinitePopulations::AbstractGame>(mGames, "OneShotCRD")
            .def(py::init<double, double, double, int, int>(),
                 R"pbdoc(
                    One-Shot Collective Risk Dilemma (CRD).

                    The full description of the One-shot CRD can be found in:
                    Santos and Pacheco, ‘Risk of Collective Failure Provides an Escape from the Tragedy of the Commons’.

                    Parameters
                    ----------
                    endowment : float
                        Initial endowment for all players. This is parameter `b` in the mentioned article.
                    cost : float
                        Cost of cooperation.
                    risk : float
                        The probability that all members will lose their remaining endowment if the threshold is not achieved.
                    group_size : int
                        Size of the group that will play the CRD.
                    min_nb_cooperators: int
                        Minimum number of cooperators required to avoid the risk of collective loss.
                    See Also
                    --------
                    egttools.games.AbstractGame
                    egttools.games.NormalFormGame
                    )pbdoc",
                 py::arg("endowment"),
                 py::arg("cost"),
                 py::arg("risk"),
                 py::arg("group_size"),
                 py::arg("min_nb_cooperators"), py::return_value_policy::reference_internal)
            .def("play", &egttools::FinitePopulations::OneShotCRD::play)
            .def("calculate_payoffs", &egttools::FinitePopulations::OneShotCRD::calculate_payoffs,
                 "updates the internal payoff and coop_level matrices by calculating the payoff of each strategy "
                 "given any possible strategy pair")
            .def("calculate_fitness", &egttools::FinitePopulations::OneShotCRD::calculate_fitness,
                 "calculates the fitness of an individual of a given strategy given a population state."
                 "It always assumes that the population state does not contain the current individual",
                 py::arg("player_strategy"),
                 py::arg("pop_size"), py::arg("population_state"))
            .def("calculate_population_group_achievement", &egttools::FinitePopulations::OneShotCRD::calculate_population_group_achievement,
                 "calculates the group achievement in the population at a given state.",
                 py::arg("population_size"), py::arg("population_state"))
            .def("calculate_group_achievement", &egttools::FinitePopulations::OneShotCRD::calculate_group_achievement,
                 "calculates the group achievement for a given stationary distribution.",
                 py::arg("population_size"), py::arg("stationary_distribution"))
            .def("__str__", &egttools::FinitePopulations::OneShotCRD::toString)
            .def("type", &egttools::FinitePopulations::OneShotCRD::type)
            .def("payoffs", &egttools::FinitePopulations::OneShotCRD::payoffs, "returns the expected payoffs of each strategy vs each possible game state")
            .def("payoff", &egttools::FinitePopulations::OneShotCRD::payoff,
                 "returns the payoff of a strategy given a group composition.", py::arg("strategy"),
                 py::arg("strategy pair"))
            .def_property_readonly("group_achievement_per_group", &egttools::FinitePopulations::OneShotCRD::group_achievements)
            .def("nb_strategies", &egttools::FinitePopulations::OneShotCRD::nb_strategies,
                 "Number of different strategies which are playing the game.")
            .def_property_readonly("endowment", &egttools::FinitePopulations::OneShotCRD::endowment,
                                   "Initial endowment for all players.")
            .def_property_readonly("min_nb_cooperators", &egttools::FinitePopulations::OneShotCRD::min_nb_cooperators,
                                   "Minimum number of cooperators to reach the target.")
            .def_property_readonly("group_size", &egttools::FinitePopulations::OneShotCRD::group_size,
                                   "Size of the group which will play the game.")
            .def_property_readonly("risk", &egttools::FinitePopulations::OneShotCRD::risk,
                                   "Probability that all players will lose their remaining endowment if the target si not achieved.")
            .def_property_readonly("cost", &egttools::FinitePopulations::OneShotCRD::cost,
                                   "Cost of cooperation.")
            .def_property_readonly("nb_states", &egttools::FinitePopulations::OneShotCRD::nb_group_compositions,
                                   "Number of combinations of Cs and Ds that can be matched in the game.")
            .def("save_payoffs", &egttools::FinitePopulations::OneShotCRD::save_payoffs,
                 "Saves the payoff matrix in a txt file.");

    py::class_<egttools::FinitePopulations::NPlayerStagHunt, egttools::FinitePopulations::AbstractGame>(mGames, "NPlayerStagHunt")
            .def(py::init<int, int, double, double>(),
                 R"pbdoc(
                    N-Player Stag Hunt (NPSH).

                    NPSH follows the description of:
                    Pacheco et al., ‘Evolutionary Dynamics of Collective Action in N -Person Stag Hunt Dilemmas’.

                    Parameters
                    ----------
                    group_size : int
                        Number of players in the group. Parameter `N` in the article.
                    cooperation_threshold : int
                        Minimum number of cooperators required to provide the collective good. Parameter `M` in the article.
                    enhancement_factor : float
                        Number of rounds of the game. Parameter `F` in the article.
                    cost : float
                        Size of the group that will play the CRD. Parameter `c` in the article.

                    See Also
                    --------
                    egttools.games.AbstractGame
                    egttools.games.NormalFormGame
                    )pbdoc",
                 py::arg("group_size"),
                 py::arg("cooperation_threshold"),
                 py::arg("enhancement_factor"),
                 py::arg("cost"), py::return_value_policy::reference_internal)
            .def("play", &egttools::FinitePopulations::NPlayerStagHunt::play)
            .def("calculate_payoffs", &egttools::FinitePopulations::NPlayerStagHunt::calculate_payoffs,
                 "updates the internal payoff and coop_level matrices by calculating the payoff of each strategy "
                 "given any possible strategy pair")
            .def("calculate_fitness", &egttools::FinitePopulations::NPlayerStagHunt::calculate_fitness,
                 "calculates the fitness of an individual of a given strategy given a population state."
                 "It always assumes that the population state does not contain the current individual",
                 py::arg("player_strategy"),
                 py::arg("pop_size"), py::arg("population_state"))
            .def("calculate_population_group_achievement", &egttools::FinitePopulations::NPlayerStagHunt::calculate_population_group_achievement,
                 "calculates the group achievement in the population at a given state.",
                 py::arg("population_size"), py::arg("population_state"))
            .def("calculate_group_achievement", &egttools::FinitePopulations::NPlayerStagHunt::calculate_group_achievement,
                 "calculates the group achievement for a given stationary distribution.",
                 py::arg("population_size"), py::arg("stationary_distribution"))
            .def("__str__", &egttools::FinitePopulations::NPlayerStagHunt::toString)
            .def("type", &egttools::FinitePopulations::NPlayerStagHunt::type)
            .def("payoffs", &egttools::FinitePopulations::NPlayerStagHunt::payoffs, "returns the expected payoffs of each strategy vs each possible game state")
            .def("payoff", &egttools::FinitePopulations::NPlayerStagHunt::payoff,
                 "returns the payoff of a strategy given a group composition.", py::arg("strategy"),
                 py::arg("strategy pair"))
            .def_property_readonly("group_achievement_per_group", &egttools::FinitePopulations::NPlayerStagHunt::group_achievements)
            .def("nb_strategies", &egttools::FinitePopulations::NPlayerStagHunt::nb_strategies,
                 "Number of different strategies which are playing the game.")
            .def("strategies", &egttools::FinitePopulations::NPlayerStagHunt::strategies,
                 "List containing the names of the strategies used in the game, in the correct order.")
            .def("nb_group_configurations", &egttools::FinitePopulations::NPlayerStagHunt::nb_group_configurations,
                 "Number of combinations of Cs and Ds that can be matched in the game.")
            .def_property_readonly("group_size", &egttools::FinitePopulations::NPlayerStagHunt::group_size,
                                   "Size of the group which will play the game.")
            .def_property_readonly("cooperation_threshold", &egttools::FinitePopulations::NPlayerStagHunt::cooperation_threshold,
                                   "Minimum number of cooperators to provide the public good.")
            .def_property_readonly("enhancement_factor", &egttools::FinitePopulations::NPlayerStagHunt::enhancement_factor,
                                   "Enhancement factor F.")
            .def_property_readonly("cost", &egttools::FinitePopulations::NPlayerStagHunt::cost,
                                   "Cost of cooperation.")
            .def("save_payoffs", &egttools::FinitePopulations::NPlayerStagHunt::save_payoffs,
                 "Saves the payoff matrix in a txt file.");

    py::class_<egttools::FinitePopulations::Matrix2PlayerGameHolder, egttools::FinitePopulations::AbstractGame>(mGames, "Matrix2PlayerGameHolder")
            .def(py::init<int, egttools::Matrix2D>(),
                 R"pbdoc(Holder class for 2-player games for which the expected payoff between strategies has already been calculated.

                    This class is useful to store the matrix of expected payoffs between strategies
                    in an 2-player game and keep the methods to calculate the fitness between these strategies.

                    Parameters
                    ----------
                    nb_strategies : int
                        number of strategies in the game
                    payoff_matrix : numpy.ndarray
                        matrix of shape (nb_strategies, nb_strategies) containing the payoffs
                        of each strategy against any other strategy.

                    See Also
                    --------
                    egttools.games.Matrix2NlayerGameHolder,
                    egttools.games.AbstractGame
                    )pbdoc",
                 py::arg("nb_strategies"),
                 py::arg("payoff_matrix"), py::return_value_policy::reference_internal)
            .def("play", &egttools::FinitePopulations::Matrix2PlayerGameHolder::play,
                 R"pbdoc(
                    Plays the One-shop CRD and update the game_payoffs given the group_composition.

                    We always assume that strategy 0 is D and strategy 1 is C.

                    The payoffs of Defectors and Cooperators are described by the following equations:

                    .. math::
                        \Pi_{D}(k) = b\{\theta(k-M)+ (1-r)[1 - \theta(k-M)]\}

                        \Pi_{C}(k) = \Pi_{D}(k) - cb

                        \text{where } \theta(x) = 0 \text{if } x < 0 \text{ and 1 otherwise.}

                    Parameters
                    ----------
                    group_composition : Union[List[int], numpy.ndarray]
                        A list or array containing the counts of how many members of each strategy are
                        present in the group.
                    game_payoffs: numpy.ndarray
                        A vector in which the payoffs of the game will be stored.
                    )pbdoc")
            .def("calculate_payoffs", &egttools::FinitePopulations::Matrix2PlayerGameHolder::calculate_payoffs,
                 R"pbdoc(
                    Calculates the payoffs of every strategy in each possible group composition.

                    Returns
                    -------
                    numpy.ndarray
                        A matrix containing the payoff of each strategy in every possible group composition.
                    )pbdoc")
            .def("calculate_fitness", &egttools::FinitePopulations::Matrix2PlayerGameHolder::calculate_fitness,
                 R"pbdoc(
                    Calculates the fitness of a strategy given a population state.

                    Parameters
                    ----------
                    player_type : int
                        The index of the strategy whose fitness will be calculated.
                    pop_size : int
                        The size of the population (Z).
                    population_state : numpy.ndarray
                        A vector containing the counts of each strategy in the population.

                    Returns
                    -------
                    float
                        The fitness of the strategy in the current population state.
                    )pbdoc",
                 py::arg("player_strategy"),
                 py::arg("pop_size"), py::arg("population_state"))
            .def("__str__", &egttools::FinitePopulations::Matrix2PlayerGameHolder::toString)
            .def("type", &egttools::FinitePopulations::Matrix2PlayerGameHolder::type)
            .def("payoffs", &egttools::FinitePopulations::Matrix2PlayerGameHolder::payoffs, "returns the expected payoffs of each strategy vs each possible game state")
            .def("payoff", &egttools::FinitePopulations::Matrix2PlayerGameHolder::payoff,
                 "returns the payoff of a strategy given a group composition.", py::arg("strategy"),
                 py::arg("strategy pair"))
            .def("nb_strategies", &egttools::FinitePopulations::Matrix2PlayerGameHolder::nb_strategies,
                 "Number of different strategies which are playing the game.")
            .def("update_payoff_matrix", &egttools::FinitePopulations::Matrix2PlayerGameHolder::update_payoff_matrix,
                 "updates the values of the payoff matrix.", py::arg("payoff_matrix"))
            .def("save_payoffs", &egttools::FinitePopulations::Matrix2PlayerGameHolder::save_payoffs,
                 "Saves the payoff matrix in a txt file.");

    py::class_<egttools::FinitePopulations::MatrixNPlayerGameHolder, egttools::FinitePopulations::AbstractGame>(mGames, "MatrixNPlayerGameHolder")
            .def(py::init<int, int, egttools::Matrix2D>(),
                 R"pbdoc(Holder class for N-player games for which the expected payoff between strategies has already been calculated.

                    This class is useful to store the matrix of expected payoffs between strategies
                    in an N-player game and keep the methods to calculate the fitness between these strategies.

                    Parameters
                    ----------
                    nb_strategies : int
                        number of strategies in the game
                    group_size : int
                        size of the group
                    payoff_matrix : numpy.ndarray
                        matrix of shape (nb_strategies, nb_group_configurations) containing the payoffs
                        of each strategy against any other strategy.

                    See Also
                    --------
                    egttools.games.Matrix2PlayerGameHolder,
                    egttools.games.AbstractGame
                    )pbdoc",
                 py::arg("nb_strategies"), py::arg("group_size"),
                 py::arg("payoff_matrix"), py::return_value_policy::reference_internal)
            .def("play", &egttools::FinitePopulations::MatrixNPlayerGameHolder::play,
                 R"pbdoc(
                    Plays the One-shop CRD and update the game_payoffs given the group_composition.

                    We always assume that strategy 0 is D and strategy 1 is C.

                    The payoffs of Defectors and Cooperators are described by the following equations:

                    .. math::
                        \Pi_{D}(k) = b\{\theta(k-M)+ (1-r)[1 - \theta(k-M)]\}

                        \Pi_{C}(k) = \Pi_{D}(k) - cb

                        \text{where } \theta(x) = 0 \text{if } x < 0 \text{ and 1 otherwise.}

                    Parameters
                    ----------
                    group_composition : Union[List[int], numpy.ndarray]
                        A list or array containing the counts of how many members of each strategy are
                        present in the group.
                    game_payoffs: numpy.ndarray
                        A vector in which the payoffs of the game will be stored.
                    )pbdoc")
            .def("calculate_payoffs", &egttools::FinitePopulations::MatrixNPlayerGameHolder::calculate_payoffs,
                 R"pbdoc(
                    Calculates the payoffs of every strategy in each possible group composition.

                    Returns
                    -------
                    numpy.ndarray
                        A matrix containing the payoff of each strategy in every possible group composition.
                    )pbdoc")
            .def("calculate_fitness", &egttools::FinitePopulations::MatrixNPlayerGameHolder::calculate_fitness,
                 R"pbdoc(
                    Calculates the fitness of a strategy given a population state.

                    Parameters
                    ----------
                    player_type : int
                        The index of the strategy whose fitness will be calculated.
                    pop_size : int
                        The size of the population (Z).
                    population_state : numpy.ndarray
                        A vector containing the counts of each strategy in the population.

                    Returns
                    -------
                    float
                        The fitness of the strategy in the current population state.
                    )pbdoc",
                 py::arg("player_strategy"),
                 py::arg("pop_size"), py::arg("population_state"))
            .def("__str__", &egttools::FinitePopulations::MatrixNPlayerGameHolder::toString)
            .def("type", &egttools::FinitePopulations::MatrixNPlayerGameHolder::type)
            .def("payoffs", &egttools::FinitePopulations::MatrixNPlayerGameHolder::payoffs, "returns the expected payoffs of each strategy vs each possible game state")
            .def("payoff", &egttools::FinitePopulations::MatrixNPlayerGameHolder::payoff,
                 "returns the payoff of a strategy given a group composition.", py::arg("strategy"),
                 py::arg("strategy pair"))
            .def("nb_strategies", &egttools::FinitePopulations::MatrixNPlayerGameHolder::nb_strategies,
                 "Number of different strategies which are playing the game.")
            .def("group_size", &egttools::FinitePopulations::MatrixNPlayerGameHolder::group_size,
                 "Size of the group.")
            .def("nb_group_configurations", &egttools::FinitePopulations::MatrixNPlayerGameHolder::nb_group_configurations,
                 "Number of different group configurations.")
            .def("update_payoff_matrix", &egttools::FinitePopulations::MatrixNPlayerGameHolder::update_payoff_matrix,
                 "updates the values of the payoff matrix.", py::arg("payoff_matrix"))
            .def("save_payoffs", &egttools::FinitePopulations::MatrixNPlayerGameHolder::save_payoffs,
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
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::AbstractNFGStrategy::is_stochastic,
                 "Property indicating if the strategy is stochastic.");

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
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::Cooperator::is_stochastic,
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
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::Defector::is_stochastic,
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
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::is_stochastic,
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
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::TitForTat::is_stochastic,
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
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::is_stochastic,
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
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::is_stochastic,
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
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::GradualTFT::is_stochastic,
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
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::is_stochastic,
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
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::TFTT::is_stochastic,
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
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::TTFT::is_stochastic,
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
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::GRIM::is_stochastic,
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
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::Pavlov::is_stochastic,
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
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::ActionInertia::is_stochastic,
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
                egttools.behaviors.CRD.AbstractCRDStrategy
                egttools.games.AbstractGame,
                egttools.games.CRDGame,
                egttools.games.CRDGameTU
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
                egttools.behaviors.CRD.AbstractCRDStrategy
                egttools.games.AbstractGame,
                egttools.games.CRDGame,
                egttools.games.CRDGameTU,
                egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT, egttools.behaviors.NormalForm.TwoActions.TFTT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.GRIM
                )pbdoc",
                 py::arg("time_step"), py::arg("group_contributions_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer::type, "Returns a string indicating the Strategy Type.")
            .def("__str__", &egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer::toString);

    py::class_<egttools::FinitePopulations::analytical::PairwiseComparison>(m, "PairwiseComparison")
            .def(py::init<int, egttools::FinitePopulations::AbstractGame &>(),
                 R"pbdoc(
                A class containing methods to study analytically the evolutionary dynamics using the Pairwise comparison rule.

                This class defines methods to compute fixation probabilities, transition matrices in the Small Mutation
                Limit (SML), gradients of selection, and the full transition matrices of the system when considering
                mutation > 0.

                Parameters
                ----------
                population_size : int
                    Size of the population.
                game : egttools.games.AbstractGame
                    A game object which must implement the abstract class `egttools.games.AbstractGame`.
                    This game will contain the expected payoffs for each strategy in the game, or at least
                    a method to compute it, and a method to calculate the fitness of each strategy for a given
                    population state.

                See Also
                --------
                egttools.numerical.PairwiseComparisonNumerical
                egttools.analytical.StochDynamics
                egttools.games.AbstractGame

                Note
                -----
                Analytical computations should be avoided for problems with very large state spaces.
                This means very big populations with many strategies. The bigger the state space, the
                more memory and time these methods will require!

                Also, for now it is not possible to update the game without having to instantiate PairwiseComparison
                again. Hopefully, this will be fixed in the future.
                )pbdoc",
                 py::arg("population_size"), py::arg("game"), py::keep_alive<1, 2>())
            .def("calculate_transition_matrix",
                 &egttools::FinitePopulations::analytical::PairwiseComparison::calculate_transition_matrix,
                 "Calculates the transition matrix of the Markov Chain that defines the dynamics of the system.",
                 py::arg("beta"), py::arg("mu"), py::return_value_policy::move)
            .def("calculate_gradient_of_selection", &egttools::FinitePopulations::analytical::PairwiseComparison::calculate_gradient_of_selection,
                 "Calculates the gradient of selection at the given state.",
                 py::arg("beta"), py::arg("state"))
            .def("calculate_fixation_probability", &egttools::FinitePopulations::analytical::PairwiseComparison::calculate_fixation_probability,
                 "Calculates the fixation probability of the invading strategy in a population of the resident strategy.",
                 py::arg("invading_strategy_index"), py::arg("resident_strategy_index"), py::arg("beta"))
            .def("calculate_transition_and_fixation_matrix_sml", &egttools::FinitePopulations::analytical::PairwiseComparison::calculate_transition_and_fixation_matrix_sml,
                 "calculates the transition and fixation probabilities matrices assuming the samll mutation limit",
                 py::arg("beta"), py::return_value_policy::move)
            .def("update_population_size", &egttools::FinitePopulations::analytical::PairwiseComparison::update_population_size)
            .def("nb_strategies", &egttools::FinitePopulations::analytical::PairwiseComparison::nb_strategies)
            .def("nb_states", &egttools::FinitePopulations::analytical::PairwiseComparison::nb_states)
            .def("population_size", &egttools::FinitePopulations::analytical::PairwiseComparison::population_size)
            .def("game", &egttools::FinitePopulations::analytical::PairwiseComparison::game);

    py::class_<PairwiseComparison>(m, "PairwiseComparisonNumerical")
            .def(py::init<size_t, egttools::FinitePopulations::AbstractGame &, size_t>(),
                 R"pbdoc(
                A class containing methods to study numerically the evolutionary dynamics using the Pairwise comparison rule.

                This class defines methods to estimate numerically fixation probabilities, stationary distributions with or without
                mutation, and strategy distributions.

                Parameters
                ----------
                population_size : int
                    Size of the population.
                game : egttools.games.AbstractGame
                    A game object which must implement the abstract class `egttools.games.AbstractGame`.
                    This game will contain the expected payoffs for each strategy in the game, or at least
                    a method to compute it, and a method to calculate the fitness of each strategy for a given
                    population state.
                cache_size : int
                    The maximum size of the cache.

                See Also
                --------
                egttools.analytical.PairwiseComparison
                egttools.analytical.StochDynamics
                egttools.games.AbstractGame

                Note
                -----
                Numerical computations are not exact. Moreover, for now we still did not implement a method to automatically
                detect if the precision of the estimation of the stationary and strategy distributions are good enough and,
                thus, stop the simulation. You are advised to test different nb_generations and transitory periods for your
                specific problem (game).

                If you want to have exact calculations, you can use egttools.analytical.PairwiseComparison. However, this
                is only advisable for systems with a smaller number of states (i.e., not too big population size or number of strategies).
                Otherwise, the calculations might require too much memory.
                )pbdoc",
                 py::arg("pop_size"), py::arg("game"), py::arg("cache_size"), py::keep_alive<1, 3>())
            .def("evolve",
                 static_cast<egttools::VectorXui (PairwiseComparison::*)(size_t, double, double,
                                                                         const Eigen::Ref<const egttools::VectorXui> &)>(&PairwiseComparison::evolve),
                 py::keep_alive<1, 5>(),
                 "evolves the strategies for a maximum of nb_generations", py::arg("nb_generations"), py::arg("beta"),
                 py::arg("mu"), py::arg("init_state"))
            .def("run",
                 static_cast<egttools::MatrixXui2D (PairwiseComparison::*)(int, double,
                                                                           const Eigen::Ref<const egttools::VectorXui> &)>(&PairwiseComparison::run),
                 "runs the moran process with social imitation and returns a matrix with all the states the system went through",
                 py::arg("nb_generations"),
                 py::arg("beta"),
                 py::arg("init_state"), py::return_value_policy::move)
            .def("run",
                 static_cast<egttools::MatrixXui2D (PairwiseComparison::*)(int, int, double, double,
                                                                           const Eigen::Ref<const egttools::VectorXui> &)>(&PairwiseComparison::run),
                 "runs the moran process with social imitation and returns a matrix with all the states the system went through excluding the transient period",
                 py::arg("nb_generations"),
                 py::arg("transient"),
                 py::arg("beta"),
                 py::arg("mu"),
                 py::arg("init_state"), py::return_value_policy::move)
            .def("run",
                 static_cast<egttools::MatrixXui2D (PairwiseComparison::*)(int, double, double,
                                                                           const Eigen::Ref<const egttools::VectorXui> &)>(&PairwiseComparison::run),
                 "runs the moran process with social imitation and returns a matrix with all the states the system went through",
                 py::arg("nb_generations"),
                 py::arg("beta"),
                 py::arg("mu"),
                 py::arg("init_state"), py::return_value_policy::move)
            .def("estimate_fixation_probability", &PairwiseComparison::estimate_fixation_probability,
                 py::call_guard<py::gil_scoped_release>(),
                 "Estimates the fixation probability of an strategy in the population.",
                 py::arg("mutant"), py::arg("resident"), py::arg("nb_runs"), py::arg("nb_generations"), py::arg("beta"))
            .def("estimate_stationary_distribution", &PairwiseComparison::estimate_stationary_distribution,
                 py::call_guard<py::gil_scoped_release>(),
                 "Estimates the stationary distribution of the population of strategies given the game.",
                 py::arg("nb_runs"), py::arg("nb_generations"), py::arg("transitory"), py::arg("beta"), py::arg("mu"))
            .def("estimate_stationary_distribution_sparse", &PairwiseComparison::estimate_stationary_distribution_sparse,
                 py::call_guard<py::gil_scoped_release>(),
                 "Estimates the stationary distribution of the population of strategies given the game.",
                 py::arg("nb_runs"), py::arg("nb_generations"), py::arg("transitory"), py::arg("beta"), py::arg("mu"))
            .def("estimate_strategy_distribution", &PairwiseComparison::estimate_strategy_distribution,
                 py::call_guard<py::gil_scoped_release>(),
                 R"pbdoc(
                Estimates the distribution of strategies in the population given the current game.

                This method directly estimates how frequent each strategy is in the population, without calculating
                the stationary distribution as an intermediary step. You should use this method when the number
                of states of the system is bigger than MAX_LONG_INT, since it would not be possible to index the states
                in this case, and estimate_stationary_distribution and estimate_stationary_distribution_sparse would run into an
                overflow error.

                Parameters
                ----------
                nb_runs : int
                    Number of independent simulations to perform. The final result will be an average over all the runs.
                nb_generations : int
                    Total number of generations.
                transitory: int
                    Transitory period. These generations will be excluded from the final average. Thus, only the last
                    nb_generations - transitory generations will be taken into account. This is important, since in
                    order to obtain a correct average at the steady state, we need to skip the transitory period.
                beta: float
                    Intensity of selection. This parameter determines how important the difference in payoff between players
                    is for the probability of imitation. If beta is small, the system will mostly undergo random drift
                    between strategies. If beta is high, a slight difference in payoff will make a strategy disapear.
                mu: float
                    Probability of mutation. This parameter defines how likely it is for a mutation event to occur at a given generation

                Returns
                -------
                numpy.ndarray[numpy.float64[m, 1]]
                    The average frequency of each strategy in the population.

                See Also
                --------
                egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution,
                egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse
                )pbdoc",
                 py::arg("nb_runs"), py::arg("nb_generations"), py::arg("transitory"), py::arg("beta"), py::arg("mu"))
            .def_property_readonly("nb_strategies", &PairwiseComparison::nb_strategies, "Number of strategies in the population.")
            .def_property_readonly("payoffs", &PairwiseComparison::payoffs,
                                   "Payoff matrix containing the payoff of each strategy (row) for each game state (column)")
            .def_property_readonly("nb_states", &PairwiseComparison::nb_states, "number of possible population states")
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

    mDistributions.def("binom",
                       &egttools::binomialCoeff<double, int64_t>,
                       "Calculates the probability density function of a multivariate hypergeometric distribution.",
                       py::arg("n"),
                       py::arg("k"));

#if (HAS_BOOST)
    mDistributions.def(
            "comb", [](size_t n, size_t k) {
                auto result = egttools::binomial_precision(n, k);
                return result.convert_to<size_t>();
            },
            "Calculates the probability density function of a multivariate hypergeometric distribution.", py::arg("n"), py::arg("k"));
#else
    mDistributions.def("comb",
                       &egttools::binomialCoeff<size_t, size_t>,
                       "Calculates the probability density function of a multivariate hypergeometric distribution.",
                       py::arg("n"),
                       py::arg("k"));
#endif
}