/** Copyright (c) 2022-2023  Elias Fernandez
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

#include "games.hpp"

namespace egttools {
    std::unique_ptr<egttools::FinitePopulations::NormalFormGame> init_normal_form_game_from_python_list(size_t nb_rounds,
                                                                                                        const Eigen::Ref<const Matrix2D> &payoff_matrix, const py::list &strategies) {
        egttools::FinitePopulations::NFGStrategyVector strategies_cpp;
        for (py::handle strategy : strategies) {
            strategies_cpp.push_back(py::cast<egttools::FinitePopulations::behaviors::AbstractNFGStrategy *>(strategy));
        }
        return std::make_unique<egttools::FinitePopulations::NormalFormGame>(nb_rounds, payoff_matrix, strategies_cpp);
    }

    std::unique_ptr<egttools::FinitePopulations::games::NormalFormNetworkGame> init_normal_form_network_game_from_python_list(int nb_rounds,
                                                                                                                              const Eigen::Ref<const Matrix2D> &payoff_matrix, const py::list &strategies) {
        egttools::FinitePopulations::games::NFGStrategyVector strategies_cpp;
        for (py::handle strategy : strategies) {
            strategies_cpp.push_back(py::cast<egttools::FinitePopulations::games::AbstractNFGStrategy_ptr>(strategy));
        }
        return std::make_unique<egttools::FinitePopulations::games::NormalFormNetworkGame>(nb_rounds, payoff_matrix, strategies_cpp);
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

}// namespace egttools

void init_games(py::module_ &mGames) {
    mGames.attr("__init__") = py::str("The `egttools.numerical.games` submodule contains the available games.");

    py::class_<egttools::FinitePopulations::AbstractGame, stubs::PyAbstractGame>(mGames, "AbstractGame")
            .def(py::init<>(), R"pbdoc(
                    Abstract class which must be implemented by any new game.

                    This class provides a common interface for Games, so that they can be passed to the methods
                    (both analytical and numerical) implemented in `egttools`.

                    You must implement the following methods:
                    - play(group_composition: List[int], game_payoffs: List[float]) -> None
                    - calculate_payoffs() -> numpy.ndarray[numpy.float64[m, n]]
                    - calculate_fitness(strategy_index: int, pop_size: int, strategies: numpy.ndarray[numpy.uint64[m, 1]]) -> float
                    - __str__
                    - type() -> str
                    - payoffs() -> numpy.ndarray[numpy.float64[m, n]]
                    - payoff(strategy: int, group_composition: List[int]) -> float
                    - nb_strategies() -> int
                    - save_payoffs(file_name: str) -> None

                    See Also
                    --------
                    egttools.games.AbstractNPlayerGame
                    )pbdoc")
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

                    See Also
                    --------
                    egttools.games.AbstractGame
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

    py::class_<egttools::FinitePopulations::NormalFormGame, egttools::FinitePopulations::AbstractGame>(mGames, "NormalFormGame")
            .def(py::init<size_t, const Eigen::Ref<const egttools::Matrix2D> &>(),
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
                    egttools.games.AbstractNPlayerGame,
                    egttools.games.CRDGame,
                    egttools.games.CRDGameTU,
                    egttools.behaviors.NormalForm.TwoActions
                    )pbdoc",
                 py::arg("nb_rounds"),
                 py::arg("payoff_matrix"), py::return_value_policy::reference_internal)
            .def(py::init(&egttools::init_normal_form_game_from_python_list),
                 R"pbdoc(
                    Normal Form Game.

                    This constructor allows you to define any number of strategies
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
                    Collective Risk Dilemma.

                    This allows you to define any number of strategies by passing them
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
                 R"pbdoc(
                    Holder class for 2-player games for which the expected payoff between strategies has already been calculated.

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
                    egttools.games.Matrix2NlayerGameHolder
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
                 R"pbdoc(
                    Holder class for N-player games for which the expected payoff between strategies has already been calculated.

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
                    egttools.games.Matrix2PlayerGameHolder
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


    py::class_<egttools::FinitePopulations::games::AbstractSpatialGame, stubs::PyAbstractSpatialGame>(mGames, "AbstractSpatialGame")
            .def(py::init<>(),
                 R"pbdoc(
                    Common interface for spatial games.

                    All that is required to be able to run a Spatial game in egttools is
                    for the game to be able to compute the fitness of a strategy given a state vector.
                    This state vector may mean anything. We leave that implementation dependant. This
                    Way developers may define any kind of dynamics.

                    Nevertheless, as an example, in a Network game, the state vector can be
                    the counts of each strategy in the neighborhood of the focal player.

                    Note
                    ----
                    This API is still not stable, there might be changes in the future.
                    )pbdoc")
            .def("calculate_fitness", &egttools::FinitePopulations::games::AbstractSpatialGame::calculate_fitness,
                 py::arg("strategy_index"),
                 py::arg("state"),
                 R"pbdoc(
                    Calculates the fitness of the `strategy_index` at a given state.

                    Parameters
                    ----------
                    strategy_index: int
                        The index of the strategy adopted by the individual's whose payoff must be calculated.
                    state: numpy.ndarray
                        An array of integers containing information necessary to calculate the fitness

                    Returns
                    -------
                    double
                        The fitness of `strategy_index` at `state`.
                    )pbdoc")
            .def("nb_strategies", &egttools::FinitePopulations::games::AbstractSpatialGame::nb_strategies,
                 "Returns the number of strategies in the population.")
            .def("__str__", &egttools::FinitePopulations::games::AbstractSpatialGame::toString,
                 "A string representation of the game.")
            .def("type", &egttools::FinitePopulations::games::AbstractSpatialGame::type,
                 "the type of game.");

    py::class_<egttools::FinitePopulations::games::NormalFormNetworkGame, egttools::FinitePopulations::games::AbstractSpatialGame>(mGames, "NormalFormNetworkGame")
            .def(py::init<size_t, const Eigen::Ref<const egttools::Matrix2D> &>(),
                 R"pbdoc(
                    Normal Form Network Game. This constructor assumes that there are only two possible strategies and two possible actions.

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
                    egttools.games.AbstractNPlayerGame,
                    egttools.games.CRDGame,
                    egttools.games.CRDGameTU,
                    egttools.behaviors.NormalForm.TwoActions
                    )pbdoc",
                 py::arg("nb_rounds"),
                 py::arg("payoff_matrix"), py::return_value_policy::reference_internal)
            .def(py::init(&egttools::init_normal_form_network_game_from_python_list),
                 R"pbdoc(
                    Normal Form Network Game.

                    This constructor allows you to define any number of strategies
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
            .def("calculate_payoffs", &egttools::FinitePopulations::games::NormalFormNetworkGame::calculate_payoffs,
                 "calculates the expected payoff matrix.")
            .def("calculate_fitness", &egttools::FinitePopulations::games::NormalFormNetworkGame::calculate_fitness,
                 py::arg("strategy_index"),
                 py::arg("state"),
                 R"pbdoc(
                    Calculates the fitness of the `strategy_index` at a given neighborhood state.

                    Parameters
                    ----------
                    strategy_index: int
                        The index of the strategy adopted by the individual's whose payoff must be calculated.
                    state: numpy.ndarray
                        An array of integers containing the counts of strategies in the neighborhood.

                    Returns
                    -------
                    double
                        The fitness of `strategy_index` at the neighborhood `state`.
                    )pbdoc")
            .def("calculate_cooperation_level_neighborhood", &egttools::FinitePopulations::games::NormalFormNetworkGame::calculate_cooperation_level_neighborhood,
                 py::arg("strategy_index"),
                 py::arg("state"),
                 R"pbdoc(
                    Calculates the level of cooperation at the neighbourhood.

                    This method is only relevant when the nb_rounds > 1 and there are conditional
                    or stochastic strategies in the population

                    Parameters
                    ----------
                    index_strategy_focal: int
                        The index of the strategy adopted by the individual's whose payoff must be calculated.
                    neighborhood_state: numpy.ndarray
                        An array of integers containing the counts of strategies in the neighborhood.

                    Returns
                    -------
                    double
                        The fitness of `strategy_index` at the neighborhood `state`.
                    )pbdoc")
            .def("nb_strategies", &egttools::FinitePopulations::games::NormalFormNetworkGame::nb_strategies,
                 "Returns the number of strategies in the population.")
            .def("nb_rounds", &egttools::FinitePopulations::games::NormalFormNetworkGame::nb_rounds,
                 "The number of rounds of the game.")
            .def("__str__", &egttools::FinitePopulations::games::NormalFormNetworkGame::toString,
                 "A string representation of the game.")
            .def("type", &egttools::FinitePopulations::games::NormalFormNetworkGame::type,
                 "the type of game.")
            .def("expected_payoffs", &egttools::FinitePopulations::games::NormalFormNetworkGame::expected_payoffs,
                 "Expected payoffs.")
            .def("strategies", &egttools::FinitePopulations::games::NormalFormNetworkGame::strategies,
                 "The strategies that play the game.");

    py::class_<egttools::FinitePopulations::games::OneShotCRDNetworkGame, egttools::FinitePopulations::games::AbstractSpatialGame>(mGames, "OneShotCRDNetworkGame")
            .def(py::init<double, double, double, int>(),
                 R"pbdoc(
                    One-Shot Collective Risk Dilemma (CRD) in Networks.

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
                    min_nb_cooperators: int
                        Minimum number of cooperators required to avoid the risk of collective loss.

                    See Also
                    --------
                    egttools.games.AbstractGame
                    egttools.games.OneShotCRD
                    egttools.games.NormalFormGame
                    egttools.games.NormalFormNetworkGame
                    )pbdoc",
                 py::arg("endowment"),
                 py::arg("cost"),
                 py::arg("risk"),
                 py::arg("min_nb_cooperators"), py::return_value_policy::reference_internal)
            .def("calculate_fitness", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::calculate_fitness,
                 py::arg("strategy_index"),
                 py::arg("state"),
                 R"pbdoc(
                    Calculates the fitness of the `strategy_index` at a given neighborhood state.

                    Parameters
                    ----------
                    strategy_index: int
                        The index of the strategy adopted by the individual's whose payoff must be calculated.
                    state: numpy.ndarray
                        An array of integers containing the counts of strategies in the neighborhood.

                    Returns
                    -------
                    double
                        The fitness of `strategy_index` at the neighborhood `state`.
                    )pbdoc")
            .def("nb_strategies", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::nb_strategies,
                 "Returns the number of strategies in the population.")
            .def("endowment", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::endowment,
                 "Returns the endowment.")
            .def("cost", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::cost,
                 "Returns the cost of cooperation.")
            .def("risk", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::risk,
                 "Returns the risk.")
            .def("min_nb_cooperators", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::min_nb_cooperators,
                 "Returns the minimum number of cooperators.")
            .def("__str__", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::toString,
                 "A string representation of the game.")
            .def("type", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::type,
                 "the type of game.");
}
