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
        egttools::FinitePopulations::StrategyVector strategies_cpp;
        for (py::handle strategy : strategies) {
            strategies_cpp.push_back(py::cast<egttools::FinitePopulations::behaviors::AbstractNFGStrategy *>(strategy));
        }
        return std::make_unique<egttools::FinitePopulations::NormalFormGame>(nb_rounds, payoff_matrix, strategies_cpp);
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

    m.doc() = R"pbdoc(
        EGTtools: Toolbox of efficient Evolutionary Game Theory (EGT) models and methods.
        This library is written in C++ (with python bindings) and pure Python.

        Note:
        This module is part a larger library named Dyrwin which also includes other methods to model
        social dynamics which use learning theory (e.g., Reinforcement Learning), and methods to analyze, process
        and model experimental data.
        -----------------------
        .. currentmodule:: egttools
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc" + attr_doc(m, "VERSION", "This version of EGTtools.");


    // Use this function to get access to the singleton
    py::class_<Random::SeedGenerator, std::unique_ptr<Random::SeedGenerator, py::nodelete>>(m, "Random")
            .def(
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
            .def(
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
           )pbdoc")
            .def_property_readonly_static(
                    "seed_", [](const py::object &) {
                        return egttools::Random::SeedGenerator::getInstance().getMainSeed();
                    },
                    R"pbdoc(
                    Property which returns the integer used to see egttools.Random.

                    Parameters
                    ----------

                    Returns
                    -------
                    int
                        The value used to see egttools.Random.
                    )pbdoc")
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

                    )pbdoc");

    // Now we define a submodule
    auto mGames = m.def_submodule("games");
    auto mBehaviors = m.def_submodule("behaviors");
    auto mNF = mBehaviors.def_submodule("NormalForm");
    auto mNFTwoActions = mNF.def_submodule("TwoActions");
    auto mData = m.def_submodule("DataStructures");
    auto mDistributions = m.def_submodule("distributions");

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

                    Returns
                    -------
                    )pbdoc")
            .def("calculate_payoffs", &egttools::FinitePopulations::AbstractGame::calculate_payoffs,
                 R"pbdoc(
                    Estimates the payoff matrix for each strategy.

                    Parameters
                    ----------
                    group_composition : List[int]
                        A list with counts of the number of players of each strategy in the group.
                    game_payoffs : List[float]
                        A list used as container for the payoffs of each player

                    Returns
                    -------
                    )pbdoc")
            .def("calculate_fitness", &egttools::FinitePopulations::AbstractGame::calculate_fitness,
                 R"pbdoc(
                    Estimates the fitness for a :param player_type in the population with state :param strategies.

                    This function assumes that the player with strategy @param player_type is not included in
                    the vector of strategy counts :param strategies.

                    Parameters
                    ----------
                    player_type : int
                        The index of the strategy used by the player.
                    pop_size : int
                        The size of the population.
                    strategies : numpy.ndarray[np.uint64]
                        A vector of counts of which strategy. The current state of the population

                    Returns
                    -------
                    float
                        The fitness of the strategy give the population state.
                    )pbdoc")
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
                    )pbdoc")
            .def_property_readonly("nb_strategies", &egttools::FinitePopulations::AbstractGame::nb_strategies)
            .def("save_payoffs", &egttools::FinitePopulations::AbstractGame::save_payoffs,
                 R"pbdoc(
                    Stores the payoff matrix in a txt file

                    Parameters
                    ----------
                    file_name : str
                        Name of the file in which the data will be stored.

                    Returns
                    -------
                    )pbdoc");

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
                    group_composition : np.ndarray[int]
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
                    egttools.calculate_state, egttools.calculate_nb_states
                    )pbdoc",
          py::arg("index"), py::arg("pop_size"),
          py::arg("nb_strategies"));
    m.def("calculate_nb_states",
          &egttools::starsBars,
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
                    egttools.calculate_state, egttools.sample_simplex
                    )pbdoc",
          py::arg("group_size"), py::arg("nb_strategies"));

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

                    Returns
                    -------

                    See Also
                    --------
                    egttools.games.AbstractGame
                    )pbdoc",
                 py::arg("nb_rounds"),
                 py::arg("payoff_matrix"))
            .def(py::init(&egttools::init_normal_form_game_from_python_list),
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
                    strategies : List[egttools.behaviors.AbstractNFGStrategy]
                        A list containing references of AbstractNFGStrategy strategies (or child classes).

                    Returns
                    -------

                    See Also
                    --------
                    egttools.games.AbstractGame
                    )pbdoc", py::arg("nb_rounds"),
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
            .def_property_readonly("nb_strategies", &egttools::FinitePopulations::NormalFormGame::nb_strategies)
            .def_property_readonly("nb_rounds", &egttools::FinitePopulations::NormalFormGame::nb_rounds)
            .def_property_readonly("nb_states", &egttools::FinitePopulations::NormalFormGame::nb_states)
            .def_property_readonly("strategies", &egttools::FinitePopulations::NormalFormGame::strategies)
            .def("save_payoffs", &egttools::FinitePopulations::NormalFormGame::save_payoffs);

    py::class_<egttools::FinitePopulations::behaviors::AbstractNFGStrategy, stubs::PyAbstractNFGStrategy>(mNF, "AbstractNFGStrategy")
            .def(py::init<>())
            .def("get_action", &egttools::FinitePopulations::behaviors::AbstractNFGStrategy::get_action,
                 R"pbdoc(Returns an action in function of :param time_step round and the previous action :param action_prev of the opponent.)pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::AbstractNFGStrategy::type, "Returns a string indicating the Strategy Type.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::Cooperator,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Cooperator")
            .def(py::init<>(), "This strategy always cooperates.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::Cooperator::get_action,
                 R"pbdoc(Returns an action in function of :param time_step round and the previous action :param action_prev of the opponent.)pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::Cooperator::type, "Returns a string indicating the Strategy Type.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::Defector,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Defector")
            .def(py::init<>(), "This strategy always defects.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::Defector::get_action,
                 R"pbdoc(Returns an action in function of :param time_step round and the previous action :param action_prev of the opponent.)pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::Defector::type, "Returns a string indicating the Strategy Type.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::RandomPlayer,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Random")
            .def(py::init<>(), "This players chooses cooperation with uniform random probability.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::get_action,
                 R"pbdoc(Returns an action in function of :param time_step round and the previous action :param action_prev of the opponent.)pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::type, "Returns a string indicating the Strategy Type.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::TitForTat,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "TFT")
            .def(py::init<>(), "Tit for Tat: Cooperates in the first round and imitates the opponent's move thereafter.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::TitForTat::get_action,
                 R"pbdoc(Returns an action in function of :param time_step round and the previous action :param action_prev of the opponent.)pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::TitForTat::type, "Returns a string indicating the Strategy Type.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "SuspiciousTFT")
            .def(py::init<>(), "Defects on the first round and imitates its opponent's previous move thereafter.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::get_action,
                 R"pbdoc(Returns an action in function of :param time_step round and the previous action :param action_prev of the opponent.)pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::type, "Returns a string indicating the Strategy Type.");

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
                 R"pbdoc(Returns an action in function of :param time_step round and the previous action :param action_prev of the opponent.)pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::type, "Returns a string indicating the Strategy Type.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::GradualTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "GradualTFT")
            .def(py::init<>(),
                 "TFT with two differences:\n"
                 "(1) it increases the string of punishing defection responses "
                 "with each additional defection by its opponent\n"
                 "(2) it apologizes for each string of defections"
                 "by cooperating in the subsequent two rounds.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::GradualTFT::get_action,
                 R"pbdoc(Returns an action in function of :param time_step round and the previous action :param action_prev of the opponent.)pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::GradualTFT::type, "Returns a string indicating the Strategy Type.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "ImperfectTFT")
            .def(py::init<double>(),
                 "Imitates opponent as in TFT, but makes mistakes "
                 "with :param error_probability.",
                 py::arg("error_probability"))
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::get_action,
                 R"pbdoc(Returns an action in function of :param time_step round and the previous action :param action_prev of the opponent.)pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::type, "Returns a string indicating the Strategy Type.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::TFTT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "TFTT")
            .def(py::init<>(), "Tit for 2 tats: Defects if defected twice.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::TFTT::get_action,
                 R"pbdoc(Returns an action in function of :param time_step round and the previous action :param action_prev of the opponent.)pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::TFTT::type, "Returns a string indicating the Strategy Type.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::TTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "TTFT")
            .def(py::init<>(), "2 Tits for tat: Defects twice if defected.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::TTFT::get_action,
                 R"pbdoc(Returns an action in function of :param time_step round and the previous action :param action_prev of the opponent.)pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::TTFT::type, "Returns a string indicating the Strategy Type.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::GRIM,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "GRIM")
            .def(py::init<>(),
                 "Grim (Trigger): Cooperates until its opponent has defected once, "
                 "and then defects for the rest of the game.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::GRIM::get_action,
                 R"pbdoc(Returns an action in function of :param time_step round and the previous action :param action_prev of the opponent.)pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::GRIM::type, "Returns a string indicating the Strategy Type.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::Pavlov,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Pavlov")
            .def(py::init<>(),
                 "Win-stay loose-shift: Cooperates if it and its opponent moved alike in"
                 "previous move and defects if they moved differently.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::Pavlov::get_action,
                 R"pbdoc(Returns an action in function of :param time_step round and the previous action :param action_prev of the opponent.)pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::Pavlov::type, "Returns a string indicating the Strategy Type.");

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
                 static_cast<egttools::MatrixXui2D (PairwiseComparison::*)(size_t, double, double,
                                                                           const Eigen::Ref<const egttools::VectorXui> &)>(&PairwiseComparison::run),
                 "runs the moran process with social imitation and returns a matrix with all the states the system went through",
                 py::arg("nb_generations"),
                 py::arg("beta"),
                 py::arg("mu"),
                 py::arg("init_state"))
            .def("run",
                 static_cast<egttools::MatrixXui2D (PairwiseComparison::*)(size_t, double,
                                                                           const Eigen::Ref<const egttools::VectorXui> &)>(&PairwiseComparison::run),
                 "runs the moran process with social imitation and returns a matrix with all the states the system went through",
                 py::arg("nb_generations"),
                 py::arg("beta"),
                 py::arg("init_state"))
            .def("fixation_probability", &PairwiseComparison::fixationProbability,
                 "Estimates the fixation probability of an strategy in the population.",
                 py::arg("mutant"), py::arg("resident"), py::arg("nb_runs"), py::arg("nb_generations"), py::arg("beta"))
            .def("stationary_distribution", &PairwiseComparison::stationaryDistribution,
                 py::call_guard<py::gil_scoped_release>(),
                 "Estimates the stationary distribution of the population of strategies given the game.",
                 py::arg("nb_runs"), py::arg("nb_generations"), py::arg("transitory"), py::arg("beta"), py::arg("mu"))
            .def_property_readonly("nb_strategies", &PairwiseComparison::nb_strategies)
            .def_property_readonly("payoffs", &PairwiseComparison::payoffs)
            .def_property("pop_size", &PairwiseComparison::population_size, &PairwiseComparison::set_population_size)
            .def_property("cache_size", &PairwiseComparison::cache_size, &PairwiseComparison::set_cache_size);

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