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

#include "games.hpp"

namespace egttools {
    std::unique_ptr<FinitePopulations::NormalFormGame>
    init_normal_form_game_from_python_list(
        size_t nb_rounds,
        const Eigen::Ref<const Matrix2D> &payoff_matrix,
        const py::list &strategies
    ) {
        FinitePopulations::NFGStrategyVector strategies_cpp;
        for (const py::handle &strategy: strategies) {
            strategies_cpp.push_back(py::cast<FinitePopulations::behaviors::AbstractNFGStrategy *>(strategy));
        }
        return std::make_unique<FinitePopulations::NormalFormGame>(nb_rounds, payoff_matrix, strategies_cpp);
    }

    std::unique_ptr<FinitePopulations::games::NormalFormNetworkGame>
    init_normal_form_network_game_from_python_list(
        int nb_rounds,
        const Eigen::Ref<const Matrix2D> &payoff_matrix,
        const py::list &strategies
    ) {
        FinitePopulations::games::NFGStrategyVector strategies_cpp;
        for (const py::handle &strategy: strategies) {
            strategies_cpp.push_back(py::cast<FinitePopulations::games::AbstractNFGStrategy_ptr>(strategy));
        }
        return std::make_unique<FinitePopulations::games::NormalFormNetworkGame>(
            nb_rounds, payoff_matrix, strategies_cpp);
    }

    std::unique_ptr<FinitePopulations::CRDGame>
    init_crd_game_from_python_list(
        int endowment,
        int threshold,
        int nb_rounds,
        int group_size,
        double risk,
        double enhancement_factor,
        const py::list &strategies
    ) {
        FinitePopulations::CRDStrategyVector strategies_cpp;
        for (const py::handle &strategy: strategies) {
            strategies_cpp.push_back(py::cast<FinitePopulations::behaviors::AbstractCRDStrategy *>(strategy));
        }
        return std::make_unique<FinitePopulations::CRDGame>(
            endowment, threshold, nb_rounds, group_size, risk, enhancement_factor, strategies_cpp);
    }

    std::unique_ptr<FinitePopulations::games::CRDGameTU>
    init_crd_tu_game_from_python_list(
        int endowment,
        int threshold,
        int nb_rounds,
        int group_size,
        double risk,
        utils::TimingUncertainty<> tu,
        const py::list &strategies
    ) {
        FinitePopulations::games::CRDStrategyVector strategies_cpp;
        for (const py::handle &strategy: strategies) {
            strategies_cpp.push_back(py::cast<FinitePopulations::behaviors::AbstractCRDStrategy *>(strategy));
        }
        return std::make_unique<FinitePopulations::games::CRDGameTU>(
            endowment, threshold, nb_rounds, group_size, risk, tu, strategies_cpp);
    }
} // namespace egttools

void init_games(const py::module_ &mGames) {
    mGames.attr("__doc__") = py::str(
        R"pbdoc(
The `egttools.numerical.games` submodule provides access to all game implementations available in EGTtools.

It includes abstract base classes for defining new games, as well as concrete implementations such as
`NormalFormGame`, `CRDGame`, `NPlayerStagHunt`, and others.

These classes support the modeling of evolutionary dynamics in finite populations and can be used
with various numerical tools available in EGTtools to simulate and analyze game-theoretic behavior.

See Also
--------
egttools.numerical.PairwiseComparisonNumerical
egttools.analytical.PairwiseComparison
egttools.plotting
)pbdoc"
    );

    py::class_<egttools::infinite_populations::AbstractReplicatorGame, stubs::PyAbstractReplicatorGame>(
                mGames,
                "AbstractReplicatorGame",
                R"pbdoc(
Base class for games that define fitness in the infinite-population limit.

This abstract class defines the interface required for a game to be used with
replicator dynamics in EGTtools. Concrete implementations must provide the
number of strategies, the group size, a method to compute the fitness vector at
a given population state, and access to the corresponding payoff table when
available.
)pbdoc")
            .def(py::init<>())

            .def("calculate_payoffs",
                 &egttools::infinite_populations::AbstractReplicatorGame::calculate_payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
Computes or refreshes the payoff table of the game.

Implementations may use this method to lazily compute and cache the payoff
structure associated with the game.

Returns
-------
numpy.ndarray
    The payoff table of the game.
)pbdoc")

            .def("calculate_fitness",
                 &egttools::infinite_populations::AbstractReplicatorGame::calculate_fitness,
                 py::arg("frequencies"),
                 R"pbdoc(
Returns the expected fitness of all strategies at a given population state.

Parameters
----------
frequencies : numpy.ndarray
    One-dimensional array containing the frequency of each strategy in the population.

Returns
-------
numpy.ndarray
    One-dimensional array containing the expected fitness of each strategy.
)pbdoc")

            .def("__str__",
                 &egttools::infinite_populations::AbstractReplicatorGame::toString,
                 R"pbdoc(
Returns a string representation of the game object.

Returns
-------
str
    A short description of the game.
)pbdoc")

            .def("type",
                 &egttools::infinite_populations::AbstractReplicatorGame::type,
                 R"pbdoc(
Returns the type of the game as a string.

Returns
-------
str
    A label identifying the game type.
)pbdoc")

            .def("payoffs",
                 &egttools::infinite_populations::AbstractReplicatorGame::payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
Returns the currently stored payoff table of the game.

If the payoff table is computed lazily, `calculate_payoffs()` should be called
first to ensure that the returned table is initialized and up to date.

Returns
-------
numpy.ndarray
    The current payoff table of the game.
)pbdoc")

            .def("nb_strategies",
                 &egttools::infinite_populations::AbstractReplicatorGame::nb_strategies,
                 R"pbdoc(
Returns the number of strategies available in the game.

Returns
-------
int
    The total number of strategies.
)pbdoc")

            .def("group_size",
                 &egttools::infinite_populations::AbstractReplicatorGame::group_size,
                 R"pbdoc(
Returns the group size of the game.

Returns
-------
int
    The number of individuals in each interacting group.
)pbdoc");

    py::class_<egttools::FinitePopulations::AbstractGame, stubs::PyAbstractGame>(
                mGames,
                "AbstractGame",
                R"pbdoc(
Base class for all game-theoretic models in EGTtools.

This abstract class defines the required interface for any game to be used in
evolutionary dynamics models. All concrete games must inherit from this class
and implement its methods.
)pbdoc")
            .def(py::init<>())

            .def("play",
                 &egttools::FinitePopulations::AbstractGame::play,
                 py::arg("group_composition"),
                 py::arg("game_payoffs"),
                 R"pbdoc(
Computes the payoff of each strategy for a given group composition.

This method modifies `game_payoffs` in-place to store the payoff of each strategy,
given a group composed of the specified number of individuals per strategy.

Parameters
----------
group_composition : numpy.ndarray
    One-dimensional array indicating the number of individuals playing each strategy in the group.
game_payoffs : numpy.ndarray
    Pre-allocated one-dimensional array that will be updated with the payoffs of each strategy.

Returns
-------
None
    This function modifies `game_payoffs` directly.
)pbdoc")

            .def("calculate_payoffs",
                 &egttools::FinitePopulations::AbstractGame::calculate_payoffs,
                 R"pbdoc(
Calculates and stores all payoffs internally for all possible group compositions.

This method must be called before computing fitness values or using the game in simulations.
)pbdoc")

            .def("calculate_fitness",
                 &egttools::FinitePopulations::AbstractGame::calculate_fitness,
                 py::arg("strategy_index"),
                 py::arg("pop_size"),
                 py::arg("strategies"),
                 R"pbdoc(
Computes the fitness of a given strategy in a population.

Parameters
----------
strategy_index : int
    The index of the strategy whose fitness is being computed.
pop_size : int
    Total population size.
strategies : numpy.ndarray
    One-dimensional array representing the number of individuals using each strategy.

Returns
-------
float
    The computed fitness of the specified strategy.
)pbdoc")

            .def("__str__",
                 &egttools::FinitePopulations::AbstractGame::toString,
                 R"pbdoc(
Returns a string representation of the game object.

Returns
-------
str
    A string describing the game instance.
)pbdoc")

            .def("type",
                 &egttools::FinitePopulations::AbstractGame::type,
                 R"pbdoc(
Returns the type of the game as a string.

Returns
-------
str
    A label identifying the game type.
)pbdoc")

            .def("payoffs",
                 &egttools::FinitePopulations::AbstractGame::payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
Returns the current payoff matrix of the game.

Returns
-------
numpy.ndarray
    The stored payoff matrix used in the game.
)pbdoc")

            .def("payoff",
                 &egttools::FinitePopulations::AbstractGame::payoff,
                 py::arg("strategy"),
                 py::arg("group_composition"),
                 R"pbdoc(
Returns the expected payoff of a specific strategy in a group.

Parameters
----------
strategy : int
    Index of the focal strategy.
group_composition : numpy.ndarray
    One-dimensional array specifying the number of individuals using each strategy.

Returns
-------
float
    Expected payoff of the strategy in the given group context.
)pbdoc")

            .def("nb_strategies",
                 &egttools::FinitePopulations::AbstractGame::nb_strategies,
                 R"pbdoc(
Returns the number of strategies available in the game.

Returns
-------
int
    The total number of strategies.
)pbdoc")

            .def("save_payoffs",
                 &egttools::FinitePopulations::AbstractGame::save_payoffs,
                 py::arg("file_name"),
                 R"pbdoc(
Saves the current payoff matrix to a file.

Parameters
----------
file_name : str
    Name of the file to which the matrix should be saved.
)pbdoc");

    py::class_<egttools::FinitePopulations::AbstractNPlayerGame,
                stubs::PyAbstractNPlayerGame,
                egttools::FinitePopulations::AbstractGame>(mGames, "AbstractNPlayerGame")
            .def(py::init_alias<int, int>(),
                 R"pbdoc(
Abstract N-player game.

This abstract base class represents a symmetric N-player game in which each strategy's
fitness is computed as the expected payoff over all group compositions in a population.

Parameters
----------
nb_strategies : int
    Total number of strategies in the game.
group_size : int
    Size of the interacting group.
)pbdoc",
                 py::arg("nb_strategies"),
                 py::arg("group_size"))

            .def("play",
                 &egttools::FinitePopulations::AbstractNPlayerGame::play,
                 R"pbdoc(
Executes the game for a given group composition and fills the payoff vector.

Parameters
----------
group_composition : numpy.ndarray
    One-dimensional array containing the number of players of each strategy in the group.
game_payoffs : numpy.ndarray
    Output container where the payoff of each strategy will be written.
)pbdoc",
                 py::arg("group_composition"),
                 py::arg("game_payoffs"))

            .def("calculate_payoffs",
                 &egttools::FinitePopulations::AbstractNPlayerGame::calculate_payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
Computes and returns the full payoff matrix.

Returns
-------
numpy.ndarray
    A matrix with expected payoffs. Each row represents a strategy,
    and each column a group configuration.
)pbdoc")

            .def("calculate_fitness",
                 &egttools::FinitePopulations::AbstractNPlayerGame::calculate_fitness,
                 R"pbdoc(
Computes the fitness of a given strategy in a population state.

Parameters
----------
strategy_index : int
    The strategy of the focal player.
pop_size : int
    Total population size.
strategies : numpy.ndarray
    Population state as a strategy count vector.

Returns
-------
float
    Fitness of the focal strategy in the given state.
)pbdoc",
                 py::arg("strategy_index"),
                 py::arg("pop_size"),
                 py::arg("strategies"))

            .def("__str__", &egttools::FinitePopulations::AbstractNPlayerGame::toString)
            .def("type", &egttools::FinitePopulations::AbstractNPlayerGame::type)

            .def("payoffs",
                 &egttools::FinitePopulations::AbstractNPlayerGame::payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
Returns the payoff matrix.

Returns
-------
numpy.ndarray
    Matrix of shape (nb_strategies, nb_group_configurations).
)pbdoc")

            .def("payoff",
                 &egttools::FinitePopulations::AbstractNPlayerGame::payoff,
                 R"pbdoc(
Returns the payoff of a strategy in a given group context.

Parameters
----------
strategy : int
    The strategy index.
group_composition : numpy.ndarray
    The group configuration.

Returns
-------
float
    The corresponding payoff.
)pbdoc",
                 py::arg("strategy"),
                 py::arg("group_composition"))

            .def("update_payoff",
                 &egttools::FinitePopulations::AbstractNPlayerGame::update_payoff,
                 R"pbdoc(
Updates an entry in the payoff matrix.

Parameters
----------
strategy_index : int
    Index of the strategy.
group_configuration_index : int
    Index of the group composition.
value : float
    The new payoff value.
)pbdoc",
                 py::arg("strategy_index"),
                 py::arg("group_configuration_index"),
                 py::arg("value"))

            .def("nb_strategies", &egttools::FinitePopulations::AbstractNPlayerGame::nb_strategies)
            .def("group_size", &egttools::FinitePopulations::AbstractNPlayerGame::group_size)
            .def("nb_group_configurations", &egttools::FinitePopulations::AbstractNPlayerGame::nb_group_configurations)

            .def("save_payoffs",
                 &egttools::FinitePopulations::AbstractNPlayerGame::save_payoffs,
                 R"pbdoc(
Saves the payoff matrix to a text file.

Parameters
----------
file_name : str
    Destination file path.
)pbdoc",
                 py::arg("file_name"));

    py::class_<egttools::FinitePopulations::AbstractNPlayerStateGame,
                stubs::PyAbstractNPlayerStateGame,
                egttools::FinitePopulations::AbstractNPlayerGame>(mGames, "AbstractNPlayerStateGame",
        R"pbdoc(
Abstract base class for N-player games with state-dependent payoffs.

Use this class when payoffs cannot be precomputed at initialization because they
depend on the current population state (e.g., games with variable risk functions
whose value changes with population composition).

Subclasses must implement `get_payoffs_for_player`, which is called *once* per
`calculate_fitness` invocation. The C++ base class then runs the full
hypergeometric sampling loop in C++, reducing the number of Python call-throughs
from O(nb_group_configurations) to exactly 1 per fitness evaluation.

Parameters
----------
nb_strategies : int
    Number of strategies in the game.
group_size : int
    Number of players per interacting group.

Abstract Methods
----------------
get_payoffs_for_player(player_type, state_index, state) -> np.ndarray
    Returns a 1-D array of length `nb_group_configurations` with the expected
    payoff for `player_type` under each possible group configuration, given that
    the full population state (including the focal player) has linear index
    `state_index`.

play(group_composition, game_payoffs)
    Fills `game_payoffs` in-place for a concrete group sample.

calculate_payoffs() -> np.ndarray
    Optionally pre-computes and stores the payoff matrix for inspection.
    Not used by `calculate_fitness`.

Notes
-----
The `state_index` passed to `get_payoffs_for_player` is computed by
`egttools.calculate_state(group_size, full_state)`, where `full_state` is
`state` with `state[player_type] + 1`.

Example
-------
>>> import numpy as np
>>> import egttools as egt
>>>
>>> class MyGame(egt.games.AbstractNPlayerStateGame):
...     def __init__(self, nb_strategies, group_size, risk_func, payoff_func):
...         super().__init__(nb_strategies, group_size)
...         self._configs = [egt.sample_simplex(i, group_size, nb_strategies)
...                          for i in range(self.nb_group_configurations)]
...         self.risk_func = risk_func
...         self.payoff_func = payoff_func
...
...     def get_payoffs_for_player(self, player_type, state_index, state):
...         risk = self.risk_func(state_index)
...         return np.array([self.payoff_func(risk, gc)[player_type]
...                          for gc in self._configs])
...
...     def play(self, group_composition, game_payoffs): ...
...     def calculate_payoffs(self): return self.payoffs()
...     def payoffs(self): return np.zeros((self.nb_strategies, self.nb_group_configurations))
...     def payoff(self, strategy, group_composition): return 0.0
...     def save_payoffs(self, file_name): pass
...     def __str__(self): return "MyGame"
...     def type(self): return "MyGame"
)pbdoc")
            .def(py::init_alias<int, int>(),
                 py::arg("nb_strategies"),
                 py::arg("group_size"))

            .def("get_payoffs_for_player",
                 &egttools::FinitePopulations::AbstractNPlayerStateGame::get_payoffs_for_player,
                 R"pbdoc(
Returns the payoff row for `player_type` across all group configurations.

This method is called once per `calculate_fitness` invocation. Implement it
in your Python subclass to return the payoffs that depend on the current
population state.

Parameters
----------
player_type : int
    Index of the focal player's strategy.
state_index : int
    Linear index of the full population state (including the focal player),
    as returned by `egttools.calculate_state(group_size, full_state)`.
state : np.ndarray
    Population state vector *excluding* the focal player (same as the
    `strategies` argument passed to `calculate_fitness`).

Returns
-------
np.ndarray
    1-D array of length `nb_group_configurations` with the payoff for
    `player_type` in each possible group composition drawn from `state`.
)pbdoc",
                 py::arg("player_type"),
                 py::arg("state_index"),
                 py::arg("state"))

            .def("calculate_fitness",
                 &egttools::FinitePopulations::AbstractNPlayerStateGame::calculate_fitness,
                 R"pbdoc(
Computes the fitness of `player_type` in a population with state `strategies`.

Calls `get_payoffs_for_player` once to obtain the full payoff row, then
evaluates the hypergeometric expectation in C++.

Parameters
----------
player_type : int
    Index of the focal player's strategy.
pop_size : int
    Total population size (excluding the focal player).
strategies : np.ndarray
    Strategy counts in the population, excluding the focal player.

Returns
-------
float
    Expected fitness of `player_type`.
)pbdoc",
                 py::arg("player_type"),
                 py::arg("pop_size"),
                 py::arg("strategies"))

            .def("play",
                 &egttools::FinitePopulations::AbstractNPlayerStateGame::play,
                 py::arg("group_composition"),
                 py::arg("game_payoffs"))
            .def("calculate_payoffs",
                 &egttools::FinitePopulations::AbstractNPlayerStateGame::calculate_payoffs,
                 py::return_value_policy::reference_internal)
            .def("__str__", &egttools::FinitePopulations::AbstractNPlayerStateGame::toString)
            .def("type", &egttools::FinitePopulations::AbstractNPlayerStateGame::type)
            .def("payoffs",
                 &egttools::FinitePopulations::AbstractNPlayerStateGame::payoffs,
                 py::return_value_policy::reference_internal)
            .def("payoff",
                 &egttools::FinitePopulations::AbstractNPlayerStateGame::payoff,
                 py::arg("strategy"),
                 py::arg("group_composition"))
            .def("nb_strategies", &egttools::FinitePopulations::AbstractNPlayerStateGame::nb_strategies)
            .def("group_size", &egttools::FinitePopulations::AbstractNPlayerStateGame::group_size)
            .def("nb_group_configurations", &egttools::FinitePopulations::AbstractNPlayerStateGame::nb_group_configurations)
            .def("save_payoffs",
                 &egttools::FinitePopulations::AbstractNPlayerStateGame::save_payoffs,
                 py::arg("file_name"));

    py::class_<egttools::FinitePopulations::NormalFormGame,
                egttools::FinitePopulations::AbstractGame>(mGames, "NormalFormGame")
            .def(py::init<size_t, const Eigen::Ref<const egttools::Matrix2D> &>(),
                 R"pbdoc(
Normal-form game with repeated pairwise interactions.

Parameters
----------
nb_rounds : int
    Number of rounds played by each strategy pair.
payoff_matrix : numpy.ndarray
    Payoff matrix where entry (i, j) gives the payoff of strategy i against j.
)pbdoc",
                 py::arg("nb_rounds"),
                 py::arg("payoff_matrix"),
                 py::return_value_policy::reference_internal,
                 py::keep_alive<0, 2>())

            .def(py::init(&egttools::init_normal_form_game_from_python_list),
                 R"pbdoc(
Normal-form game initialized with custom strategy classes.

Parameters
----------
nb_rounds : int
    Number of rounds in the repeated game.
payoff_matrix : numpy.ndarray
    Payoff matrix.
strategies : list[AbstractNFGStrategy]
    List of strategy instances.
)pbdoc",
                 py::arg("nb_rounds"),
                 py::arg("payoff_matrix"),
                 py::arg("strategies"),
                 py::return_value_policy::reference_internal)

            .def("play",
                 &egttools::FinitePopulations::NormalFormGame::play,
                 R"pbdoc(
Executes a game round and stores the resulting payoffs.

Parameters
----------
group_composition : numpy.ndarray
    Composition of the pairwise game.
game_payoffs : numpy.ndarray
    Output array to store individual payoffs.
)pbdoc",
                 py::arg("group_composition"),
                 py::arg("game_payoffs"))

            .def("calculate_payoffs",
                 &egttools::FinitePopulations::NormalFormGame::calculate_payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
Calculates the expected payoff matrix for all strategy pairs.

Returns
-------
numpy.ndarray
    Matrix of expected payoffs between strategies.
)pbdoc")

            .def("calculate_fitness",
                 &egttools::FinitePopulations::NormalFormGame::calculate_fitness,
                 R"pbdoc(
Computes the fitness of a strategy in a given population state.

Parameters
----------
player_strategy : int
    Index of the focal strategy.
population_size : int
    Total number of individuals.
population_state : numpy.ndarray
    Strategy counts in the population.

Returns
-------
float
    Fitness of the focal strategy.
)pbdoc",
                 py::arg("player_strategy"),
                 py::arg("population_size"),
                 py::arg("population_state"))

            .def("__str__", &egttools::FinitePopulations::NormalFormGame::toString)
            .def("type", &egttools::FinitePopulations::NormalFormGame::type)

            .def("payoffs",
                 &egttools::FinitePopulations::NormalFormGame::payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
Returns the payoff matrix.

Returns
-------
numpy.ndarray
    Matrix of expected payoffs between strategies.
)pbdoc")

            .def("payoff",
                 &egttools::FinitePopulations::NormalFormGame::payoff,
                 R"pbdoc(
Returns the payoff for a given strategy in a specific match-up.

Parameters
----------
strategy : int
    Index of the strategy used by the player.
strategy_pair : list[int]
    Pair of strategy indices in the match-up.

Returns
-------
float
)pbdoc",
                 py::arg("strategy"),
                 py::arg("strategy_pair"))

            .def("expected_payoffs",
                 &egttools::FinitePopulations::NormalFormGame::expected_payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(Returns the matrix of expected payoffs between strategies.)pbdoc")

            .def("nb_strategies", &egttools::FinitePopulations::NormalFormGame::nb_strategies)
            .def_property_readonly("nb_rounds", &egttools::FinitePopulations::NormalFormGame::nb_rounds)
            .def_property_readonly("nb_states", &egttools::FinitePopulations::NormalFormGame::nb_states)

            .def_property_readonly(
                "strategies",
                [](egttools::FinitePopulations::NormalFormGame &self) -> py::list {
                    py::list out;
                    for (auto *s: self.strategies()) out.append(py::cast(s));
                    return out;
                },
                R"pbdoc(List of strategies participating in the game.)pbdoc")

            .def("save_payoffs",
                 &egttools::FinitePopulations::NormalFormGame::save_payoffs,
                 R"pbdoc(
Saves the payoff matrix to a text file.

Parameters
----------
file_name : str
    File path where the matrix will be saved.
)pbdoc");

    py::class_<egttools::FinitePopulations::CRDGame,
                egttools::FinitePopulations::AbstractGame>(mGames, "CRDGame")
            .def(py::init(&egttools::init_crd_game_from_python_list),
                 R"pbdoc(
Collective risk dilemma game.

Parameters
----------
endowment : int
    Initial endowment of each player.
threshold : int
    Collective target the group must achieve to avoid risk.
nb_rounds : int
    Number of rounds in the game.
group_size : int
    Number of players in each group.
risk : float
    Probability of losing remaining endowment if the target is not met.
enhancement_factor : float
    Multiplier for successful cooperation.
strategies : list[AbstractCRDStrategy]
    List of strategy instances.
)pbdoc",
                 py::arg("endowment"),
                 py::arg("threshold"),
                 py::arg("nb_rounds"),
                 py::arg("group_size"),
                 py::arg("risk"),
                 py::arg("enhancement_factor"),
                 py::arg("strategies"),
                 py::return_value_policy::reference_internal,
                 py::keep_alive<0, 7>())

            .def("play",
                 &egttools::FinitePopulations::CRDGame::play,
                 R"pbdoc(
Plays a single round of the CRD game for the specified group composition.

Parameters
----------
group_composition : numpy.ndarray
    Number of players using each strategy.
game_payoffs : numpy.ndarray
    Output vector to store player payoffs.
)pbdoc")

            .def("calculate_payoffs",
                 &egttools::FinitePopulations::CRDGame::calculate_payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
Computes the expected payoffs for each strategy under all group configurations.

Returns
-------
numpy.ndarray
)pbdoc")

            .def("calculate_fitness",
                 &egttools::FinitePopulations::CRDGame::calculate_fitness,
                 R"pbdoc(
Calculates the fitness of a strategy in a given population state.

Parameters
----------
player_strategy : int
    Index of the focal strategy.
pop_size : int
    Total population size.
population_state : numpy.ndarray
    Vector of strategy counts.

Returns
-------
float
)pbdoc",
                 py::arg("player_strategy"),
                 py::arg("pop_size"),
                 py::arg("population_state"))

            .def("calculate_population_group_achievement",
                 &egttools::FinitePopulations::CRDGame::calculate_population_group_achievement,
                 R"pbdoc(Calculates group achievement for the population at a given state.)pbdoc",
                 py::arg("population_size"),
                 py::arg("population_state"))

            .def("calculate_group_achievement",
                 &egttools::FinitePopulations::CRDGame::calculate_group_achievement,
                 R"pbdoc(Calculates group achievement given a stationary distribution.)pbdoc",
                 py::arg("population_size"),
                 py::arg("stationary_distribution"))

            .def("calculate_polarization",
                 &egttools::FinitePopulations::CRDGame::calculate_polarization,
                 R"pbdoc(Computes contribution polarization relative to the fair contribution.)pbdoc",
                 py::arg("population_size"),
                 py::arg("population_state"))

            .def("calculate_polarization_success",
                 &egttools::FinitePopulations::CRDGame::calculate_polarization_success,
                 R"pbdoc(Computes contribution polarization among successful groups.)pbdoc",
                 py::arg("population_size"),
                 py::arg("population_state"))

            .def("__str__", &egttools::FinitePopulations::CRDGame::toString)
            .def("type", &egttools::FinitePopulations::CRDGame::type)

            .def("payoffs",
                 &egttools::FinitePopulations::CRDGame::payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(Returns the payoff matrix for all strategies and group configurations.)pbdoc")

            .def("payoff",
                 &egttools::FinitePopulations::CRDGame::payoff,
                 R"pbdoc(
Returns the payoff of a strategy in a given group composition.

Parameters
----------
strategy : int
    Index of the strategy.
group_composition : list[int]
    Group composition vector.

Returns
-------
float
)pbdoc",
                 py::arg("strategy"),
                 py::arg("group_composition"))

            .def("nb_strategies", &egttools::FinitePopulations::CRDGame::nb_strategies)
            .def_property_readonly("endowment", &egttools::FinitePopulations::CRDGame::endowment)
            .def_property_readonly("target", &egttools::FinitePopulations::CRDGame::target)
            .def_property_readonly("group_size", &egttools::FinitePopulations::CRDGame::group_size)
            .def_property_readonly("risk", &egttools::FinitePopulations::CRDGame::risk)
            .def_property_readonly("enhancement_factor", &egttools::FinitePopulations::CRDGame::enhancement_factor)
            .def_property_readonly("nb_rounds", &egttools::FinitePopulations::CRDGame::nb_rounds)
            .def_property_readonly("nb_states", &egttools::FinitePopulations::CRDGame::nb_states)

            .def_property_readonly(
                "strategies",
                [](egttools::FinitePopulations::CRDGame &self) -> py::list {
                    py::list out;
                    for (auto *s: self.strategies()) out.append(py::cast(s));
                    return out;
                },
                R"pbdoc(List of strategy instances in the game.)pbdoc")

            .def("save_payoffs",
                 &egttools::FinitePopulations::CRDGame::save_payoffs,
                 R"pbdoc(
Saves the payoff matrix to a file.

Parameters
----------
file_name : str
    Output file path.
)pbdoc");

    py::class_<egttools::FinitePopulations::games::CRDGameTU,
                egttools::FinitePopulations::AbstractGame>(mGames, "CRDGameTU")
            .def(py::init(&egttools::init_crd_tu_game_from_python_list),
                 R"pbdoc(
Collective risk dilemma with timing uncertainty.

Parameters
----------
endowment : int
    Initial endowment of each player.
threshold : int
    Collective target required to avoid risk.
nb_rounds : int
    Maximum number of rounds.
group_size : int
    Number of players per group.
risk : float
    Probability of failure if the target is not met.
tu : TimingUncertainty
    Object modeling timing uncertainty.
strategies : list[AbstractCRDStrategy]
    List of strategy instances.
)pbdoc",
                 py::arg("endowment"),
                 py::arg("threshold"),
                 py::arg("nb_rounds"),
                 py::arg("group_size"),
                 py::arg("risk"),
                 py::arg("tu"),
                 py::arg("strategies"),
                 py::return_value_policy::reference_internal,
                 py::keep_alive<0, 7>())

            .def("play",
                 &egttools::FinitePopulations::games::CRDGameTU::play,
                 R"pbdoc(
Executes one iteration of the CRD game using a specific group composition.

Parameters
----------
group_composition : numpy.ndarray
    Number of players per strategy in the group.
game_payoffs : numpy.ndarray
    Output vector for player payoffs.
)pbdoc")

            .def("calculate_payoffs",
                 &egttools::FinitePopulations::games::CRDGameTU::calculate_payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
Computes the expected payoffs for each strategy across all group configurations.

Returns
-------
numpy.ndarray
    Matrix of expected payoffs.
)pbdoc")

            .def("calculate_fitness",
                 &egttools::FinitePopulations::games::CRDGameTU::calculate_fitness,
                 R"pbdoc(
Computes the fitness of a strategy in a given population state.

Parameters
----------
player_strategy : int
    Index of the focal strategy.
pop_size : int
    Total population size.
population_state : numpy.ndarray
    Vector of strategy counts.

Returns
-------
float
)pbdoc",
                 py::arg("player_strategy"),
                 py::arg("pop_size"),
                 py::arg("population_state"))

            .def("calculate_population_group_achievement",
                 &egttools::FinitePopulations::games::CRDGameTU::calculate_population_group_achievement,
                 R"pbdoc(Calculates group achievement for a given population state.)pbdoc",
                 py::arg("population_size"),
                 py::arg("population_state"))

            .def("calculate_group_achievement",
                 &egttools::FinitePopulations::games::CRDGameTU::calculate_group_achievement,
                 R"pbdoc(Calculates group achievement based on a stationary distribution.)pbdoc",
                 py::arg("population_size"),
                 py::arg("stationary_distribution"))

            .def("calculate_polarization",
                 &egttools::FinitePopulations::games::CRDGameTU::calculate_polarization,
                 py::call_guard<py::gil_scoped_release>(),
                 R"pbdoc(Computes contribution polarization in a given population state.)pbdoc",
                 py::arg("population_size"),
                 py::arg("population_state"))

            .def("calculate_polarization_success",
                 &egttools::FinitePopulations::games::CRDGameTU::calculate_polarization_success,
                 py::call_guard<py::gil_scoped_release>(),
                 R"pbdoc(Computes contribution polarization among successful groups.)pbdoc",
                 py::arg("population_size"),
                 py::arg("population_state"))

            .def("__str__", &egttools::FinitePopulations::games::CRDGameTU::toString)
            .def("type", &egttools::FinitePopulations::games::CRDGameTU::type)

            .def("payoffs",
                 &egttools::FinitePopulations::games::CRDGameTU::payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(Returns the matrix of expected payoffs.)pbdoc")

            .def("payoff",
                 &egttools::FinitePopulations::games::CRDGameTU::payoff,
                 R"pbdoc(
Returns the payoff of a strategy given a group composition.

Parameters
----------
strategy : int
    Strategy index.
group_composition : list[int]
    Group composition vector.

Returns
-------
float
)pbdoc",
                 py::arg("strategy"),
                 py::arg("group_composition"))

            .def("nb_strategies", &egttools::FinitePopulations::games::CRDGameTU::nb_strategies)
            .def_property_readonly("endowment", &egttools::FinitePopulations::games::CRDGameTU::endowment)
            .def_property_readonly("target", &egttools::FinitePopulations::games::CRDGameTU::target)
            .def_property_readonly("group_size", &egttools::FinitePopulations::games::CRDGameTU::group_size)
            .def_property_readonly("risk", &egttools::FinitePopulations::games::CRDGameTU::risk)
            .def_property_readonly("min_rounds", &egttools::FinitePopulations::games::CRDGameTU::min_rounds)
            .def_property_readonly("nb_states", &egttools::FinitePopulations::games::CRDGameTU::nb_states)

            .def_property_readonly(
                "strategies",
                [](egttools::FinitePopulations::games::CRDGameTU &self) -> py::list {
                    py::list out;
                    for (auto *s: self.strategies()) out.append(py::cast(s));
                    return out;
                },
                R"pbdoc(List of strategy objects participating in the game.)pbdoc")

            .def("save_payoffs",
                 &egttools::FinitePopulations::games::CRDGameTU::save_payoffs,
                 R"pbdoc(
Saves the payoff matrix to a text file.

Parameters
----------
file_name : str
    Path to the output file.
)pbdoc");

    py::class_<egttools::FinitePopulations::OneShotCRD,
                egttools::FinitePopulations::AbstractGame>(mGames, "OneShotCRD")
            .def(py::init<double, double, double, int, int>(),
                 R"pbdoc(
One-shot collective risk dilemma.

Parameters
----------
endowment : float
    Initial endowment received by all players.
cost : float
    Fraction of the endowment contributed by cooperators.
risk : float
    Probability of collective loss if the group fails to reach the threshold.
group_size : int
    Number of players in the group.
min_nb_cooperators : int
    Minimum number of cooperators needed to avoid risk.
)pbdoc",
                 py::arg("endowment"),
                 py::arg("cost"),
                 py::arg("risk"),
                 py::arg("group_size"),
                 py::arg("min_nb_cooperators"),
                 py::return_value_policy::reference_internal)

            .def("play",
                 &egttools::FinitePopulations::OneShotCRD::play,
                 R"pbdoc(
Executes a one-shot CRD round and updates payoffs for the given group composition.

Parameters
----------
group_composition : numpy.ndarray
    Number of players per strategy.
game_payoffs : numpy.ndarray
    Output vector to store payoffs for each player.
)pbdoc")

            .def("calculate_payoffs",
                 &egttools::FinitePopulations::OneShotCRD::calculate_payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
Updates the payoff matrix and cooperation level matrix for all strategy pairs.
)pbdoc")

            .def("calculate_fitness",
                 &egttools::FinitePopulations::OneShotCRD::calculate_fitness,
                 R"pbdoc(
Calculates the fitness of a strategy given a population state.

Parameters
----------
player_strategy : int
    Index of the focal strategy.
pop_size : int
    Population size.
population_state : numpy.ndarray
    Vector of strategy counts in the population.

Returns
-------
float
)pbdoc",
                 py::arg("player_strategy"),
                 py::arg("pop_size"),
                 py::arg("population_state"))

            .def("calculate_population_group_achievement",
                 &egttools::FinitePopulations::OneShotCRD::calculate_population_group_achievement,
                 R"pbdoc(
Computes the group achievement for the given population state.
)pbdoc",
                 py::arg("population_size"),
                 py::arg("population_state"))

            .def("calculate_group_achievement",
                 &egttools::FinitePopulations::OneShotCRD::calculate_group_achievement,
                 R"pbdoc(
Computes group achievement from a stationary distribution.
)pbdoc",
                 py::arg("population_size"),
                 py::arg("stationary_distribution"))

            .def("__str__", &egttools::FinitePopulations::OneShotCRD::toString)
            .def("type", &egttools::FinitePopulations::OneShotCRD::type)

            .def("payoffs",
                 &egttools::FinitePopulations::OneShotCRD::payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(Returns the expected payoff matrix.)pbdoc")

            .def("payoff",
                 &egttools::FinitePopulations::OneShotCRD::payoff,
                 R"pbdoc(Returns the payoff for a given strategy and group composition.)pbdoc",
                 py::arg("strategy"),
                 py::arg("strategy_pair"))

            .def_property_readonly("group_achievement_per_group",
                                   &egttools::FinitePopulations::OneShotCRD::group_achievements)

            .def("nb_strategies", &egttools::FinitePopulations::OneShotCRD::nb_strategies)
            .def_property_readonly("endowment", &egttools::FinitePopulations::OneShotCRD::endowment)
            .def_property_readonly("min_nb_cooperators", &egttools::FinitePopulations::OneShotCRD::min_nb_cooperators)
            .def_property_readonly("group_size", &egttools::FinitePopulations::OneShotCRD::group_size)
            .def_property_readonly("risk", &egttools::FinitePopulations::OneShotCRD::risk)
            .def_property_readonly("cost", &egttools::FinitePopulations::OneShotCRD::cost)
            .def_property_readonly("nb_states", &egttools::FinitePopulations::OneShotCRD::nb_group_compositions)
            .def("save_payoffs",
                 &egttools::FinitePopulations::OneShotCRD::save_payoffs,
                 R"pbdoc(Saves the payoff matrix to a text file.)pbdoc");

    py::class_<egttools::FinitePopulations::NPlayerStagHunt,
                egttools::FinitePopulations::AbstractGame>(mGames, "NPlayerStagHunt")
            .def(py::init<int, int, double, double>(),
                 R"pbdoc(
N-player stag hunt.

Parameters
----------
group_size : int
    Number of players in the group.
cooperation_threshold : int
    Minimum number of cooperators required to produce the collective benefit.
enhancement_factor : float
    Multiplicative factor applied to the benefit when the public good is provided.
cost : float
    Cost of cooperation.
)pbdoc",
                 py::arg("group_size"),
                 py::arg("cooperation_threshold"),
                 py::arg("enhancement_factor"),
                 py::arg("cost"),
                 py::return_value_policy::reference_internal)

            .def("play",
                 &egttools::FinitePopulations::NPlayerStagHunt::play,
                 R"pbdoc(
Simulates the game and fills in the payoff vector for a given group composition.

Parameters
----------
group_composition : numpy.ndarray
    Number of players of each strategy in the group.
game_payoffs : numpy.ndarray
    Output vector to store the resulting payoff for each player.
)pbdoc")

            .def("calculate_payoffs",
                 &egttools::FinitePopulations::NPlayerStagHunt::calculate_payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
Computes and stores the expected payoff matrix for all strategy-group combinations.
)pbdoc")

            .def("calculate_fitness",
                 &egttools::FinitePopulations::NPlayerStagHunt::calculate_fitness,
                 R"pbdoc(
Computes the fitness of a strategy given a population state.

Parameters
----------
player_strategy : int
    Index of the focal strategy.
pop_size : int
    Total number of individuals in the population.
population_state : numpy.ndarray
    Vector of strategy counts.

Returns
-------
float
)pbdoc",
                 py::arg("player_strategy"),
                 py::arg("pop_size"),
                 py::arg("population_state"))

            .def("calculate_population_group_achievement",
                 &egttools::FinitePopulations::NPlayerStagHunt::calculate_population_group_achievement,
                 R"pbdoc(
Estimates the likelihood that a random group from the population meets the cooperation threshold.
)pbdoc",
                 py::arg("population_size"),
                 py::arg("population_state"))

            .def("calculate_group_achievement",
                 &egttools::FinitePopulations::NPlayerStagHunt::calculate_group_achievement,
                 R"pbdoc(
Computes the expected collective success weighted by a stationary distribution.
)pbdoc",
                 py::arg("population_size"),
                 py::arg("stationary_distribution"))

            .def("__str__", &egttools::FinitePopulations::NPlayerStagHunt::toString)
            .def("type", &egttools::FinitePopulations::NPlayerStagHunt::type)

            .def("payoffs",
                 &egttools::FinitePopulations::NPlayerStagHunt::payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(Returns the expected payoff matrix for all strategy combinations.)pbdoc")

            .def("payoff",
                 &egttools::FinitePopulations::NPlayerStagHunt::payoff,
                 R"pbdoc(Returns the payoff of a strategy given a group composition.)pbdoc",
                 py::arg("strategy"),
                 py::arg("strategy_pair"))

            .def_property_readonly("group_achievement_per_group",
                                   &egttools::FinitePopulations::NPlayerStagHunt::group_achievements)

            .def("nb_strategies", &egttools::FinitePopulations::NPlayerStagHunt::nb_strategies)
            .def("strategies", &egttools::FinitePopulations::NPlayerStagHunt::strategies)
            .def("nb_group_configurations", &egttools::FinitePopulations::NPlayerStagHunt::nb_group_configurations)
            .def_property_readonly("group_size", &egttools::FinitePopulations::NPlayerStagHunt::group_size)
            .def_property_readonly("cooperation_threshold",
                                   &egttools::FinitePopulations::NPlayerStagHunt::cooperation_threshold)
            .def_property_readonly("enhancement_factor",
                                   &egttools::FinitePopulations::NPlayerStagHunt::enhancement_factor)
            .def_property_readonly("cost", &egttools::FinitePopulations::NPlayerStagHunt::cost)
            .def("save_payoffs",
                 &egttools::FinitePopulations::NPlayerStagHunt::save_payoffs,
                 R"pbdoc(Saves the payoff matrix to a text file.)pbdoc");

    py::class_<egttools::FinitePopulations::Matrix2PlayerGameHolder,
                egttools::FinitePopulations::AbstractGame>(mGames, "Matrix2PlayerGameHolder")
            .def(py::init<int, const Eigen::Ref<const egttools::Matrix2D> &>(),
                 R"pbdoc(
Matrix-based 2-player game holder.

Parameters
----------
nb_strategies : int
    Number of strategies used in the game.
payoff_matrix : numpy.ndarray
    Matrix containing the payoff of each strategy against all others.
)pbdoc",
                 py::arg("nb_strategies"),
                 py::arg("payoff_matrix"),
                 py::return_value_policy::reference_internal,
                 py::keep_alive<0, 2>())

            .def("play",
                 &egttools::FinitePopulations::Matrix2PlayerGameHolder::play,
                 R"pbdoc(
Executes a match given a group composition and stores the resulting payoffs.

Parameters
----------
group_composition : numpy.ndarray
    Count of each strategy in the group.
game_payoffs : numpy.ndarray
    Output vector to be filled with each player's payoff.
)pbdoc")

            .def("calculate_payoffs",
                 &egttools::FinitePopulations::Matrix2PlayerGameHolder::calculate_payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
Returns the stored payoff matrix.

Returns
-------
numpy.ndarray
    Payoff matrix.
)pbdoc")

            .def("calculate_fitness",
                 &egttools::FinitePopulations::Matrix2PlayerGameHolder::calculate_fitness,
                 R"pbdoc(
Computes the fitness of a strategy given the population configuration.

Parameters
----------
player_strategy : int
    Index of the focal strategy.
pop_size : int
    Size of the population.
population_state : numpy.ndarray
    Vector of counts of each strategy in the population.

Returns
-------
float
)pbdoc",
                 py::arg("player_strategy"),
                 py::arg("pop_size"),
                 py::arg("population_state"))

            .def("__str__", &egttools::FinitePopulations::Matrix2PlayerGameHolder::toString)
            .def("type", &egttools::FinitePopulations::Matrix2PlayerGameHolder::type)

            .def("payoffs",
                 &egttools::FinitePopulations::Matrix2PlayerGameHolder::payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(Returns the expected payoff matrix.)pbdoc")

            .def("payoff",
                 &egttools::FinitePopulations::Matrix2PlayerGameHolder::payoff,
                 R"pbdoc(Returns the payoff for a given strategy pair.)pbdoc",
                 py::arg("strategy"),
                 py::arg("strategy_pair"))

            .def("nb_strategies", &egttools::FinitePopulations::Matrix2PlayerGameHolder::nb_strategies)

            .def("update_payoff_matrix",
                 &egttools::FinitePopulations::Matrix2PlayerGameHolder::update_payoff_matrix,
                 R"pbdoc(Replaces the internal payoff matrix with a new one.)pbdoc",
                 py::arg("payoff_matrix"))

            .def("save_payoffs",
                 &egttools::FinitePopulations::Matrix2PlayerGameHolder::save_payoffs,
                 R"pbdoc(Saves the current payoff matrix to a text file.)pbdoc");

    py::class_<egttools::FinitePopulations::MatrixNPlayerGameHolder,
                egttools::FinitePopulations::AbstractGame>(mGames, "MatrixNPlayerGameHolder")
            .def(py::init<int, int, const Eigen::Ref<const egttools::Matrix2D> &>(),
                 R"pbdoc(
Matrix-based N-player game holder.

Parameters
----------
nb_strategies : int
    Number of strategies in the game.
group_size : int
    Size of the interacting group.
payoff_matrix : numpy.ndarray
    Matrix encoding payoffs for all strategy-group pairs.
)pbdoc",
                 py::arg("nb_strategies"),
                 py::arg("group_size"),
                 py::arg("payoff_matrix"),
                 py::return_value_policy::reference_internal,
                 py::keep_alive<0, 3>())

            .def("play",
                 &egttools::FinitePopulations::MatrixNPlayerGameHolder::play,
                 R"pbdoc(
Simulates the game based on a predefined payoff matrix.

Parameters
----------
group_composition : numpy.ndarray
    Number of players using each strategy in the group.
game_payoffs : numpy.ndarray
    Output vector for storing player payoffs.
)pbdoc")

            .def("calculate_payoffs",
                 &egttools::FinitePopulations::MatrixNPlayerGameHolder::calculate_payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(
Returns the internal matrix of precomputed payoffs.

Returns
-------
numpy.ndarray
)pbdoc")

            .def("calculate_fitness",
                 &egttools::FinitePopulations::MatrixNPlayerGameHolder::calculate_fitness,
                 R"pbdoc(
Computes the fitness of a strategy based on the current population state.

Parameters
----------
player_strategy : int
    Index of the strategy used by the focal player.
pop_size : int
    Population size.
population_state : numpy.ndarray
    Vector of strategy counts in the population.

Returns
-------
float
    Fitness of the focal strategy.
)pbdoc",
                 py::arg("player_strategy"),
                 py::arg("pop_size"),
                 py::arg("population_state"))

            .def("__str__", &egttools::FinitePopulations::MatrixNPlayerGameHolder::toString)
            .def("type", &egttools::FinitePopulations::MatrixNPlayerGameHolder::type)

            .def("payoffs",
                 &egttools::FinitePopulations::MatrixNPlayerGameHolder::payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(Returns the full payoff matrix.)pbdoc")

            .def("payoff",
                 &egttools::FinitePopulations::MatrixNPlayerGameHolder::payoff,
                 R"pbdoc(Returns the payoff for a strategy given a specific group configuration.)pbdoc",
                 py::arg("strategy"),
                 py::arg("strategy_pair"))

            .def("nb_strategies", &egttools::FinitePopulations::MatrixNPlayerGameHolder::nb_strategies)
            .def("group_size", &egttools::FinitePopulations::MatrixNPlayerGameHolder::group_size)
            .def("nb_group_configurations",
                 &egttools::FinitePopulations::MatrixNPlayerGameHolder::nb_group_configurations)

            .def("update_payoff_matrix",
                 &egttools::FinitePopulations::MatrixNPlayerGameHolder::update_payoff_matrix,
                 R"pbdoc(Replaces the stored payoff matrix with a new one.)pbdoc",
                 py::arg("payoff_matrix"))

            .def("save_payoffs",
                 &egttools::FinitePopulations::MatrixNPlayerGameHolder::save_payoffs,
                 R"pbdoc(Saves the payoff matrix to a text file.)pbdoc");

    py::class_<egttools::FinitePopulations::games::AbstractSpatialGame, stubs::PyAbstractSpatialGame>(
                mGames, "AbstractSpatialGame")
            .def(py::init<>(), R"pbdoc(
Abstract base class for spatially structured games.

This interface supports general spatial interaction models, where the fitness of a strategy
is computed based on a local context.
)pbdoc")

            .def("calculate_fitness",
                 &egttools::FinitePopulations::games::AbstractSpatialGame::calculate_fitness,
                 py::arg("strategy_index"),
                 py::arg("state"),
                 R"pbdoc(
Calculates the fitness of a strategy in a local interaction context.

Parameters
----------
strategy_index : int
    The strategy of the focal player.
state : numpy.ndarray
    Vector representing the local configuration.

Returns
-------
float
    The computed fitness of the strategy in the given local state.
)pbdoc")

            .def("nb_strategies", &egttools::FinitePopulations::games::AbstractSpatialGame::nb_strategies)
            .def("__str__", &egttools::FinitePopulations::games::AbstractSpatialGame::toString)
            .def("type", &egttools::FinitePopulations::games::AbstractSpatialGame::type);

    py::class_<egttools::FinitePopulations::games::NormalFormNetworkGame,
                egttools::FinitePopulations::games::AbstractSpatialGame>(mGames, "NormalFormNetworkGame")
            .def(py::init<size_t, const Eigen::Ref<const egttools::Matrix2D> &>(),
                 R"pbdoc(
Normal-form network game.

Parameters
----------
nb_rounds : int
    Number of repeated rounds for each pairwise encounter.
payoff_matrix : numpy.ndarray
    Payoff matrix specifying the outcomes for all strategy pairs.
)pbdoc",
                 py::arg("nb_rounds"),
                 py::arg("payoff_matrix"),
                 py::return_value_policy::reference_internal,
                 py::keep_alive<0, 2>())

            .def(py::init(&egttools::init_normal_form_network_game_from_python_list),
                 R"pbdoc(
Normal-form game with custom strategy list.

Parameters
----------
nb_rounds : int
    Number of rounds of interaction.
payoff_matrix : numpy.ndarray
    Payoff matrix used for each pairwise encounter.
strategies : list[AbstractNFGStrategy]
    List of strategy instances.
)pbdoc",
                 py::arg("nb_rounds"),
                 py::arg("payoff_matrix"),
                 py::arg("strategies"),
                 py::return_value_policy::reference_internal,
                 py::keep_alive<0, 2>())

            .def("calculate_payoffs",
                 &egttools::FinitePopulations::games::NormalFormNetworkGame::calculate_payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(Recalculates the expected payoff matrix based on current strategies.)pbdoc")

            .def("calculate_fitness",
                 &egttools::FinitePopulations::games::NormalFormNetworkGame::calculate_fitness,
                 py::arg("strategy_index"),
                 py::arg("state"),
                 R"pbdoc(
Computes fitness of a strategy given a neighborhood configuration.

Parameters
----------
strategy_index : int
    Strategy whose fitness will be computed.
state : numpy.ndarray
    Vector representing neighborhood strategy counts.

Returns
-------
float
    Fitness value.
)pbdoc")

            .def("calculate_cooperation_level_neighborhood",
                 &egttools::FinitePopulations::games::NormalFormNetworkGame::calculate_cooperation_level_neighborhood,
                 py::arg("strategy_index"),
                 py::arg("state"),
                 R"pbdoc(
Calculates the level of cooperation in a given neighborhood.

Parameters
----------
strategy_index : int
    Focal strategy.
state : numpy.ndarray
    Neighbor strategy counts.

Returns
-------
float
    Level of cooperation.
)pbdoc")

            .def("nb_strategies", &egttools::FinitePopulations::games::NormalFormNetworkGame::nb_strategies)
            .def("nb_rounds", &egttools::FinitePopulations::games::NormalFormNetworkGame::nb_rounds)
            .def("__str__", &egttools::FinitePopulations::games::NormalFormNetworkGame::toString)
            .def("type", &egttools::FinitePopulations::games::NormalFormNetworkGame::type)

            .def("expected_payoffs",
                 &egttools::FinitePopulations::games::NormalFormNetworkGame::expected_payoffs,
                 py::return_value_policy::reference_internal,
                 R"pbdoc(Returns the expected payoffs for each strategy.)pbdoc")

            .def("strategies",
                 [](egttools::FinitePopulations::games::NormalFormNetworkGame &self) -> py::list {
                     py::list out;
                     for (auto *s: self.strategies()) out.append(py::cast(s));
                     return out;
                 },
                 R"pbdoc(List of strategies currently active in the game.)pbdoc");

    py::class_<egttools::FinitePopulations::games::OneShotCRDNetworkGame,
                egttools::FinitePopulations::games::AbstractSpatialGame>(mGames, "OneShotCRDNetworkGame")
            .def(py::init<double, double, double, int>(),
                 R"pbdoc(
One-shot collective risk dilemma in networks.

Parameters
----------
endowment : float
    Initial endowment received by each individual.
cost : float
    Cost of contributing to the public good.
risk : float
    Probability of collective loss if the threshold is not met.
min_nb_cooperators : int
    Minimum number of cooperators required to avoid risk.
)pbdoc",
                 py::arg("endowment"),
                 py::arg("cost"),
                 py::arg("risk"),
                 py::arg("min_nb_cooperators"),
                 py::return_value_policy::reference_internal)

            .def("calculate_fitness",
                 &egttools::FinitePopulations::games::OneShotCRDNetworkGame::calculate_fitness,
                 py::arg("strategy_index"),
                 py::arg("state"),
                 R"pbdoc(
Computes the fitness of a strategy in a local neighborhood.

Parameters
----------
strategy_index : int
    The focal strategy being evaluated.
state : numpy.ndarray
    Vector representing the number of neighbors using each strategy.

Returns
-------
float
    The fitness of the strategy given the local state.
)pbdoc")

            .def("nb_strategies", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::nb_strategies)
            .def("endowment", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::endowment)
            .def("cost", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::cost)
            .def("risk", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::risk)
            .def("min_nb_cooperators", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::min_nb_cooperators)
            .def("__str__", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::toString)
            .def("type", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::type);
}
