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
    // Converts a Python list of AbstractNFGStrategy* into a C++ vector and creates a NormalFormGame.
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

    // Converts a Python list of AbstractNFGStrategy* into a C++ vector and creates a NormalFormNetworkGame.
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

    // Converts a Python list of AbstractCRDStrategy* into a C++ vector and creates a CRDGame.
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
        return std::make_unique<FinitePopulations::CRDGame>(endowment, threshold, nb_rounds,
                                                            group_size, risk, enhancement_factor, strategies_cpp);
    }

    // Converts a Python list of AbstractCRDStrategy* into a C++ vector and creates a CRDGame with Timing Uncertainty.
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
        return std::make_unique<FinitePopulations::games::CRDGameTU>(endowment, threshold, nb_rounds,
                                                                     group_size, risk, tu, strategies_cpp);
    }
} // namespace egttools

void init_games(const py::module_ &mGames) {
    // Sets a brief documentation string for the `egttools.numerical.games` submodule.
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


    py::class_<egttools::FinitePopulations::AbstractGame, stubs::PyAbstractGame>(mGames, "AbstractGame",
                R"pbdoc(
Base class for all game-theoretic models in EGTtools.

This abstract class defines the required interface for any game to be used in
evolutionary dynamics models. All concrete games must inherit from this class
and implement its methods.
)pbdoc")
            .def(py::init<>())
            .def("play", &egttools::FinitePopulations::AbstractGame::play, py::arg("group_composition"),
                 py::arg("game_payoffs"),
                 R"pbdoc(
Computes the payoff of each strategy for a given group composition.

This method modifies `game_payoffs` in-place to store the payoff of each strategy,
given a group composed of the specified number of individuals per strategy.

This avoids memory reallocation by reusing the existing array, which can
significantly speed up simulations.

Parameters
----------
group_composition : NDArray[np.int64]
    A vector of shape (n_strategies,) indicating the number of individuals
    playing each strategy in the group.
game_payoffs : NDArray[np.float64]
    A pre-allocated vector of shape (n_strategies,) which will be updated
    with the payoffs of each strategy in the group. Modified in-place.

Returns
-------
None
    This function modifies `game_payoffs` directly.

Examples
--------
>>> group = np.array([1, 2, 0])
>>> out = np.zeros_like(group, dtype=np.float64)
>>> game.play(group, out)
>>> print(out)  # Contains payoffs for each strategy in the group
)pbdoc")

            .def("calculate_payoffs", &egttools::FinitePopulations::AbstractGame::calculate_payoffs,
                 R"pbdoc(
Calculates and stores all payoffs internally for all possible group compositions.

This method must be called before computing fitness values or using the game in simulations.
)pbdoc")

            .def("calculate_fitness", &egttools::FinitePopulations::AbstractGame::calculate_fitness,
                 py::arg("strategy_index"), py::arg("pop_size"), py::arg("strategies"),
                 R"pbdoc(
Computes the fitness of a given strategy in a population.

Parameters
----------
strategy_index : int
    The index of the strategy whose fitness is being computed.
pop_size : int
    Total population size.
strategies : NDArray[np.int64]
    Vector representing the number of individuals using each strategy.

Returns
-------
float
    The computed fitness of the specified strategy.
)pbdoc")

            .def("__str__", &egttools::FinitePopulations::AbstractGame::toString,
                 R"pbdoc(
Returns a string representation of the game object.

Returns
-------
str
    A string describing the game instance.
)pbdoc")

            .def("type", &egttools::FinitePopulations::AbstractGame::type,
                 R"pbdoc(
Returns the type of the game as a string.

Returns
-------
str
    A label identifying the game type (e.g., "NormalFormGame").
)pbdoc")

            .def("payoffs", &egttools::FinitePopulations::AbstractGame::payoffs,
                 R"pbdoc(
Returns the current payoff matrix of the game.

Returns
-------
NDArray[np.float64]
    The stored payoff matrix used in the game.
)pbdoc")

            .def("payoff", &egttools::FinitePopulations::AbstractGame::payoff,
                 py::arg("strategy"), py::arg("group_composition"),
                 R"pbdoc(
Returns the expected payoff of a specific strategy in a group.

Parameters
----------
strategy : int
    Index of the focal strategy.
group_composition : NDArray[np.int64]
    Vector specifying the number of individuals using each strategy.

Returns
-------
float
    Expected payoff of the strategy in the given group context.
)pbdoc")

            .def("nb_strategies", &egttools::FinitePopulations::AbstractGame::nb_strategies,
                 R"pbdoc(
Returns the number of strategies available in the game.

Returns
-------
int
    The total number of strategies.
)pbdoc")

            .def("save_payoffs", &egttools::FinitePopulations::AbstractGame::save_payoffs,
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
         Abstract N-Player Game.

         This abstract base class represents a symmetric N-player game in which each strategy's
         fitness is computed as the expected payoff over all group compositions in a population.

         Notes
         -----
         Subclasses must implement the `play` and `calculate_payoffs` methods.
         The following attributes are expected:
         - `self.nb_strategies_` (int): number of strategies.
         - `self.payoffs_` (numpy.ndarray): of shape (nb_strategies, nb_group_configurations).

         Parameters
         ----------
         nb_strategies : int
             Total number of strategies in the game.
         group_size : int
             Size of the interacting group.
         )pbdoc",
                 py::arg("nb_strategies"), py::arg("group_size"),
                 py::return_value_policy::reference_internal)

            .def("play", &egttools::FinitePopulations::AbstractNPlayerGame::play,
                 R"pbdoc(
         Executes the game for a given group composition and fills the payoff vector.

         Parameters
         ----------
         group_composition : List[int] | numpy.ndarray[int]
             The number of players of each strategy in the group.
         game_payoffs : List[float] | numpy.ndarray[float]
             Output container where the payoff of each player will be written.
         )pbdoc",
                 py::arg("group_composition"), py::arg("game_payoffs"))

            .def("calculate_payoffs", &egttools::FinitePopulations::AbstractNPlayerGame::calculate_payoffs,
                 R"pbdoc(
         Computes and returns the full payoff matrix.

         Returns
         -------
         numpy.ndarray
             A matrix with expected payoffs. Each row represents a strategy,
             each column a group configuration.
         )pbdoc")

            .def("calculate_fitness", &egttools::FinitePopulations::AbstractNPlayerGame::calculate_fitness,
                 R"pbdoc(
         Computes the fitness of a given strategy in a population state.

         Parameters
         ----------
         strategy_index : int
             The strategy of the focal player.
         pop_size : int
             Total population size (excluding the focal player).
         strategies : numpy.ndarray[uint64]
             The population state as a strategy count vector.

         Returns
         -------
         float
             Fitness of the focal strategy in the given state.
         )pbdoc",
                 py::arg("strategy_index"), py::arg("pop_size"), py::arg("strategies"))

            .def("__str__", &egttools::FinitePopulations::AbstractNPlayerGame::toString,
                 R"pbdoc(
         Returns a string representation of the game.

         Returns
         -------
         str
         )pbdoc")

            .def("type", &egttools::FinitePopulations::AbstractNPlayerGame::type,
                 R"pbdoc(
         Returns the string identifier for the game.

         Returns
         -------
         str
         )pbdoc")

            .def("payoffs", &egttools::FinitePopulations::AbstractNPlayerGame::payoffs,
                 R"pbdoc(
         Returns the payoff matrix.

         Returns
         -------
         numpy.ndarray
             The matrix of shape (nb_strategies, nb_group_configurations).
         )pbdoc")

            .def("payoff", &egttools::FinitePopulations::AbstractNPlayerGame::payoff,
                 R"pbdoc(
         Returns the payoff of a strategy in a given group context.

         Parameters
         ----------
         strategy : int
             The strategy index.
         group_composition : list[int] or numpy.ndarray[int]
             The group configuration.

         Returns
         -------
         float
             The corresponding payoff.
         )pbdoc",
                 py::arg("strategy"), py::arg("group_composition"))

            .def("update_payoff", &egttools::FinitePopulations::AbstractNPlayerGame::update_payoff,
                 R"pbdoc(
         Updates an entry in the payoff matrix.

         Parameters
         ----------
         strategy_index : int
             Index of the strategy (row).
         group_configuration_index : int
             Index of the group composition (column).
         value : float | numpy.float64
             The new payoff value.
         )pbdoc",
                 py::arg("strategy_index"), py::arg("group_configuration_index"), py::arg("value"))

            .def("nb_strategies", &egttools::FinitePopulations::AbstractNPlayerGame::nb_strategies,
                 R"pbdoc(
         Returns the number of strategies in the game.

         Returns
         -------
         int
         )pbdoc")

            .def("group_size", &egttools::FinitePopulations::AbstractNPlayerGame::group_size,
                 R"pbdoc(
         Returns the size of the group.

         Returns
         -------
         int
         )pbdoc")

            .def("nb_group_configurations",
                 &egttools::FinitePopulations::AbstractNPlayerGame::nb_group_configurations,
                 R"pbdoc(
         Returns the number of distinct group configurations.

         Returns
         -------
         int
         )pbdoc")

            .def("save_payoffs", &egttools::FinitePopulations::AbstractNPlayerGame::save_payoffs,
                 R"pbdoc(
         Saves the payoff matrix to a text file.

         Parameters
         ----------
         file_name : str
             Destination file path.
         )pbdoc",
                 py::arg("file_name"));

    // Binding for NormalFormGame class.
    py::class_<egttools::FinitePopulations::NormalFormGame, egttools::FinitePopulations::AbstractGame>(
                mGames, "NormalFormGame")
            .def(py::init<size_t, const Eigen::Ref<const egttools::Matrix2D> &>(), R"pbdoc(
        Normal Form Game with two actions.

        Implements a repeated symmetric 2-player game based on a payoff matrix.

        Parameters
        ----------
        nb_rounds : int
            Number of rounds played by each strategy pair.
        payoff_matrix : numpy.ndarray of shape (2, 2)
            Payoff matrix where entry (i, j) gives the payoff of strategy i against j.
        )pbdoc", py::arg("nb_rounds"), py::arg("payoff_matrix"),
                 py::return_value_policy::reference_internal, py::keep_alive<0, 2>())

            .def(py::init(&egttools::init_normal_form_game_from_python_list), R"pbdoc(
        Normal Form Game initialized with custom strategy classes.

        This constructor allows using any number of strategies, defined in Python as subclasses
        of AbstractNFGStrategy.

        Parameters
        ----------
        nb_rounds : int
            Number of rounds in the repeated game.
        payoff_matrix : numpy.ndarray
            Payoff matrix of shape (nb_actions, nb_actions).
        strategies : list[AbstractNFGStrategy]
            List of strategy instances.
        )pbdoc",
                 py::arg("nb_rounds"), py::arg("payoff_matrix"), py::arg("strategies"),
                 py::return_value_policy::reference_internal)

            .def("play", &egttools::FinitePopulations::NormalFormGame::play, R"pbdoc(
        Executes a game round and stores the resulting payoffs.

        Parameters
        ----------
        group_composition : list[int] or numpy.ndarray[int]
            Composition of the pairwise game (usually 2 players).
        game_payoffs : list[float] or numpy.ndarray[float]
            Output array to store individual payoffs.
        )pbdoc", py::arg("group_composition"), py::arg("game_payoffs"))

            .def("calculate_payoffs", &egttools::FinitePopulations::NormalFormGame::calculate_payoffs, R"pbdoc(
        Calculates the expected payoff matrix for all strategy pairs.

        Returns
        -------
        numpy.ndarray of shape (nb_strategies, nb_strategies)
            Matrix of expected payoffs between strategies.
        )pbdoc")

            .def("calculate_fitness", &egttools::FinitePopulations::NormalFormGame::calculate_fitness, R"pbdoc(
        Computes the fitness of a strategy in a given population state.

        Parameters
        ----------
        player_strategy : int
            Index of the focal strategy.
        population_size : int
            Total number of individuals (excluding the focal player).
        population_state : numpy.ndarray[numpy.uint64]
            Strategy counts in the population.

        Returns
        -------
        float
            Fitness of the focal strategy.
        )pbdoc", py::arg("player_strategy"), py::arg("population_size"), py::arg("population_state"))

            .def("__str__", &egttools::FinitePopulations::NormalFormGame::toString, R"pbdoc(
        Returns a string representation of the NormalFormGame instance.

        Returns
        -------
        str
        )pbdoc")

            .def("type", &egttools::FinitePopulations::NormalFormGame::type, R"pbdoc(
        Returns the type identifier of the game.

        Returns
        -------
        str
        )pbdoc")

            .def("payoffs", &egttools::FinitePopulations::NormalFormGame::payoffs, R"pbdoc(
        Returns the payoff matrix.

        Returns
        -------
        numpy.ndarray of shape (nb_strategies, nb_strategies)
        )pbdoc", py::return_value_policy::reference_internal)

            .def("payoff", &egttools::FinitePopulations::NormalFormGame::payoff, R"pbdoc(
        Returns the payoff for a given strategy in a specific match-up.

        Parameters
        ----------
        strategy : int
            Index of the strategy used by the player.
        strategy_pair : list[int]
            List of two integers representing the strategies in the match-up.

        Returns
        -------
        float
        )pbdoc", py::arg("strategy"), py::arg("strategy_pair"))

            .def("expected_payoffs", &egttools::FinitePopulations::NormalFormGame::expected_payoffs,
                 R"pbdoc(Returns the matrix of expected payoffs between strategies.)pbdoc",
                 py::return_value_policy::reference_internal)

            .def("nb_strategies", &egttools::FinitePopulations::NormalFormGame::nb_strategies,
                 R"pbdoc(Number of strategies in the game.)pbdoc")

            .def_property_readonly("nb_rounds", &egttools::FinitePopulations::NormalFormGame::nb_rounds,
                                   R"pbdoc(Number of rounds per interaction.)pbdoc")

            .def_property_readonly("nb_states", &egttools::FinitePopulations::NormalFormGame::nb_states,
                                   R"pbdoc(Number of unique pairwise strategy combinations.)pbdoc")

            .def_property_readonly("strategies", &egttools::FinitePopulations::NormalFormGame::strategies,
                                   R"pbdoc(List of strategies participating in the game.)pbdoc")

            .def("save_payoffs", &egttools::FinitePopulations::NormalFormGame::save_payoffs, R"pbdoc(
        Saves the payoff matrix to a text file.

        Parameters
        ----------
        file_name : str
            File path where the matrix will be saved.
        )pbdoc");

    py::class_<egttools::FinitePopulations::CRDGame, egttools::FinitePopulations::AbstractGame>(mGames, "CRDGame")
            .def(py::init(&egttools::init_crd_game_from_python_list), R"pbdoc(
        Collective Risk Dilemma (CRD) Game.

        This implementation allows defining arbitrary strategies using instances of AbstractCRDStrategy.

        Based on:
        Milinski et al. (2008). The collective-risk social dilemma and the prevention of simulated
        dangerous climate change. PNAS, 105(7), 2291–2294.

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
            Multiplier for successful cooperation (may add surplus).
        strategies : list[AbstractCRDStrategy]
            List of strategy instances.
        )pbdoc",
                 py::arg("endowment"), py::arg("threshold"), py::arg("nb_rounds"), py::arg("group_size"),
                 py::arg("risk"), py::arg("enhancement_factor"), py::arg("strategies"),
                 py::return_value_policy::reference_internal, py::keep_alive<0, 7>())

            .def("play", &egttools::FinitePopulations::CRDGame::play, R"pbdoc(
        Plays a single round of the CRD game for the specified group composition.

        Parameters
        ----------
        group_composition : list[int] or numpy.ndarray[int]
            Number of players using each strategy.
        game_payoffs : list[float] or numpy.ndarray[float]
            Output vector to store player payoffs.
        )pbdoc")

            .def("calculate_payoffs", &egttools::FinitePopulations::CRDGame::calculate_payoffs, R"pbdoc(
        Computes the expected payoffs for each strategy under all group configurations.

        Returns
        -------
        numpy.ndarray of shape (nb_strategies, nb_group_configurations)
        )pbdoc")

            .def("calculate_fitness", &egttools::FinitePopulations::CRDGame::calculate_fitness, R"pbdoc(
        Calculates the fitness of a strategy in a given population state.

        Parameters
        ----------
        player_strategy : int
            Index of the focal strategy.
        pop_size : int
            Total population size (excluding focal).
        population_state : numpy.ndarray[numpy.uint64]
            Vector of strategy counts.

        Returns
        -------
        float
        )pbdoc",
                 py::arg("player_strategy"), py::arg("pop_size"), py::arg("population_state"))

            .def("calculate_population_group_achievement",
                 &egttools::FinitePopulations::CRDGame::calculate_population_group_achievement,
                 R"pbdoc(Calculates group achievement for the population at a given state.)pbdoc",
                 py::arg("population_size"), py::arg("population_state"))

            .def("calculate_group_achievement", &egttools::FinitePopulations::CRDGame::calculate_group_achievement,
                 R"pbdoc(Calculates group achievement given a stationary distribution.)pbdoc",
                 py::arg("population_size"), py::arg("stationary_distribution"))

            .def("calculate_polarization", &egttools::FinitePopulations::CRDGame::calculate_polarization,
                 R"pbdoc(Computes contribution polarization relative to the fair contribution (E/2).)pbdoc",
                 py::arg("population_size"), py::arg("population_state"))

            .def("calculate_polarization_success",
                 &egttools::FinitePopulations::CRDGame::calculate_polarization_success,
                 R"pbdoc(Computes contribution polarization among successful groups.)pbdoc",
                 py::arg("population_size"), py::arg("population_state"))

            .def("__str__", &egttools::FinitePopulations::CRDGame::toString,
                 R"pbdoc(Returns a string representation of the CRDGame.)pbdoc")
            .def("type", &egttools::FinitePopulations::CRDGame::type, R"pbdoc(Returns the type of the game.)pbdoc")

            .def("payoffs", &egttools::FinitePopulations::CRDGame::payoffs,
                 R"pbdoc(Returns the payoff matrix for all strategies and group configurations.)pbdoc")

            .def("payoff", &egttools::FinitePopulations::CRDGame::payoff, R"pbdoc(
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
        )pbdoc", py::arg("strategy"), py::arg("group_composition"))

            .def("nb_strategies", &egttools::FinitePopulations::CRDGame::nb_strategies,
                 R"pbdoc(Number of strategies in the game.)pbdoc")

            .def_property_readonly("endowment", &egttools::FinitePopulations::CRDGame::endowment,
                                   R"pbdoc(Initial endowment for each player.)pbdoc")

            .def_property_readonly("target", &egttools::FinitePopulations::CRDGame::target,
                                   R"pbdoc(Collective target to avoid risk.)pbdoc")

            .def_property_readonly("group_size", &egttools::FinitePopulations::CRDGame::group_size,
                                   R"pbdoc(Number of players per group.)pbdoc")

            .def_property_readonly("risk", &egttools::FinitePopulations::CRDGame::risk,
                                   R"pbdoc(Probability of losing endowment if the target is not met.)pbdoc")

            .def_property_readonly("enhancement_factor", &egttools::FinitePopulations::CRDGame::enhancement_factor,
                                   R"pbdoc(Multiplier applied to payoffs if the target is met.)pbdoc")

            .def_property_readonly("nb_rounds", &egttools::FinitePopulations::CRDGame::nb_rounds,
                                   R"pbdoc(Number of rounds in the game.)pbdoc")

            .def_property_readonly("nb_states", &egttools::FinitePopulations::CRDGame::nb_states,
                                   R"pbdoc(Number of distinct population states.)pbdoc")

            .def_property_readonly("strategies", &egttools::FinitePopulations::CRDGame::strategies,
                                   R"pbdoc(List of strategy instances in the game.)pbdoc")

            .def("save_payoffs", &egttools::FinitePopulations::CRDGame::save_payoffs, R"pbdoc(
        Saves the payoff matrix to a file.

        Parameters
        ----------
        file_name : str
            Output file path.
        )pbdoc");

    py::class_<egttools::FinitePopulations::games::CRDGameTU, egttools::FinitePopulations::AbstractGame>(
                mGames, "CRDGameTU")
            .def(init(&egttools::init_crd_tu_game_from_python_list), R"pbdoc(
        Collective Risk Dilemma with Timing Uncertainty (CRDGameTU).

        This game extends the CRD setting proposed by Milinski et al. (2008) by introducing
        uncertainty about the number of rounds before a potential disaster. Timing uncertainty
        is captured using a TimingUncertainty object.

        In this version, players contribute resources over multiple rounds to a common pool.
        If the collective threshold is reached before the disaster occurs, the group avoids the risk.
        Otherwise, each player faces a predefined probability of losing their remaining endowment.

        Parameters
        ----------
        endowment : int
            Initial endowment of each player.
        threshold : int
            Collective target required to avoid risk.
        nb_rounds : int
            Maximum number of rounds (before which a disaster may occur).
        group_size : int
            Number of players per group.
        risk : float
            Probability of failure if the target is not met.
        tu : TimingUncertainty
            An object modeling timing uncertainty.
        strategies : list[AbstractCRDStrategy]
            List of strategy instances.
        )pbdoc",
                 py::arg("endowment"), py::arg("threshold"), py::arg("nb_rounds"),
                 py::arg("group_size"), py::arg("risk"), py::arg("tu"), py::arg("strategies"),
                 py::return_value_policy::reference_internal, py::keep_alive<0, 7>())

            .def("play", &egttools::FinitePopulations::games::CRDGameTU::play, R"pbdoc(
        Executes one iteration of the CRD game using a specific group composition.

        This method calculates the payoffs for each player based on their strategy
        and the current group composition under timing uncertainty.

        Parameters
        ----------
        group_composition : list[int] or numpy.ndarray[int]
            Number of players per strategy in the group.
        game_payoffs : list[float] or numpy.ndarray[float]
            Output vector for player payoffs.
        )pbdoc")

            .def("calculate_payoffs", &egttools::FinitePopulations::games::CRDGameTU::calculate_payoffs, R"pbdoc(
        Computes the expected payoffs for each strategy across all group configurations.

        Returns
        -------
        numpy.ndarray
            Matrix of shape (nb_strategies, nb_group_configurations).
        )pbdoc")

            .def("calculate_fitness", &egttools::FinitePopulations::games::CRDGameTU::calculate_fitness, R"pbdoc(
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
                 py::arg("player_strategy"), py::arg("pop_size"), py::arg("population_state"))

            .def("calculate_population_group_achievement",
                 &egttools::FinitePopulations::games::CRDGameTU::calculate_population_group_achievement,
                 R"pbdoc(Calculates group achievement for a given population state.)pbdoc",
                 py::arg("population_size"), py::arg("population_state"))

            .def("calculate_group_achievement",
                 &egttools::FinitePopulations::games::CRDGameTU::calculate_group_achievement,
                 R"pbdoc(Calculates group achievement based on a stationary distribution.)pbdoc",
                 py::arg("population_size"), py::arg("stationary_distribution"))

            .def("calculate_polarization", &egttools::FinitePopulations::games::CRDGameTU::calculate_polarization,
                 py::call_guard<py::gil_scoped_release>(),
                 R"pbdoc(Computes contribution polarization in a given population state.)pbdoc",
                 py::arg("population_size"), py::arg("population_state"))

            .def("calculate_polarization_success",
                 &egttools::FinitePopulations::games::CRDGameTU::calculate_polarization_success,
                 py::call_guard<py::gil_scoped_release>(),
                 R"pbdoc(Computes contribution polarization among successful groups.)pbdoc",
                 py::arg("population_size"), py::arg("population_state"))

            .def("__str__", &egttools::FinitePopulations::games::CRDGameTU::toString,
                 R"pbdoc(Returns a string representation of the CRDGameTU.)pbdoc")
            .def("type", &egttools::FinitePopulations::games::CRDGameTU::type,
                 R"pbdoc(Returns the type identifier of the game.)pbdoc")

            .def("payoffs", &egttools::FinitePopulations::games::CRDGameTU::payoffs,
                 R"pbdoc(Returns the matrix of expected payoffs.)pbdoc")

            .def("payoff", &egttools::FinitePopulations::games::CRDGameTU::payoff, R"pbdoc(
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
        )pbdoc", py::arg("strategy"), py::arg("group_composition"))

            .def("nb_strategies", &egttools::FinitePopulations::games::CRDGameTU::nb_strategies,
                 R"pbdoc(Number of strategies in the game.)pbdoc")

            .def_property_readonly("endowment", &egttools::FinitePopulations::games::CRDGameTU::endowment,
                                   R"pbdoc(Initial endowment per player.)pbdoc")

            .def_property_readonly("target", &egttools::FinitePopulations::games::CRDGameTU::target,
                                   R"pbdoc(Target that the group must reach.)pbdoc")

            .def_property_readonly("group_size", &egttools::FinitePopulations::games::CRDGameTU::group_size,
                                   R"pbdoc(Size of the group.)pbdoc")

            .def_property_readonly("risk", &egttools::FinitePopulations::games::CRDGameTU::risk,
                                   R"pbdoc(Probability of losing endowment if the target is not met.)pbdoc")

            .def_property_readonly("min_rounds", &egttools::FinitePopulations::games::CRDGameTU::min_rounds,
                                   R"pbdoc(Minimum number of rounds the game will run.)pbdoc")

            .def_property_readonly("nb_states", &egttools::FinitePopulations::games::CRDGameTU::nb_states,
                                   R"pbdoc(Number of possible population states.)pbdoc")

            .def_property_readonly("strategies", &egttools::FinitePopulations::games::CRDGameTU::strategies,
                                   R"pbdoc(List of strategy objects participating in the game.)pbdoc")

            .def("save_payoffs", &egttools::FinitePopulations::games::CRDGameTU::save_payoffs, R"pbdoc(
        Saves the payoff matrix to a text file.

        Parameters
        ----------
        file_name : str
            Path to the output file.
        )pbdoc");

    py::class_<egttools::FinitePopulations::OneShotCRD, egttools::FinitePopulations::AbstractGame>(
                mGames, "OneShotCRD")
            .def(py::init<double, double, double, int, int>(), R"pbdoc(
        One-Shot Collective Risk Dilemma (CRD).

        This implementation models the one-shot version of the CRD introduced in:
        Santos, F. C., & Pacheco, J. M. (2011).
        "Risk of collective failure provides an escape from the tragedy of the commons."
        PNAS, 108(26), 10421–10425.

        A single group of `group_size` players must decide whether to contribute to a public good.
        Cooperators (Cs) contribute a fraction `cost` of their `endowment`, while Defectors (Ds) contribute nothing.
        If the number of cooperators is greater than or equal to `min_nb_cooperators`, all players avoid the risk
        and receive their remaining endowment. Otherwise, each player loses their remaining endowment with
        probability `risk`.

        Parameters
        ----------
        endowment : float
            The initial endowment received by all players.
        cost : float
            The fraction of the endowment that Cooperators contribute (in [0, 1]).
        risk : float
            Probability of collective loss if the group fails to reach the threshold.
        group_size : int
            Number of players in the group.
        min_nb_cooperators : int
            Minimum number of cooperators needed to avoid risk.
        )pbdoc",
                 py::arg("endowment"), py::arg("cost"), py::arg("risk"),
                 py::arg("group_size"), py::arg("min_nb_cooperators"),
                 py::return_value_policy::reference_internal)

            .def("play", &egttools::FinitePopulations::OneShotCRD::play, R"pbdoc(
        Executes a one-shot CRD round and updates payoffs for the given group composition.

        Assumes two strategies: Defectors (index 0) and Cooperators (index 1).

        The payoffs are computed as follows:

        .. math::
            \Pi_{D}(k) = b\{\theta(k-M)+ (1-r)[1 - \theta(k-M)]\}

            \Pi_{C}(k) = \Pi_{D}(k) - cb

            \text{where } \theta(x) = \begin{cases} 0 & x < 0 \\ 1 & x \geq 0 \end{cases}

        Parameters
        ----------
        group_composition : list[int] or numpy.ndarray[int]
            Number of players per strategy.
        game_payoffs : list[float] or numpy.ndarray[float]
            Output vector to store payoffs for each player.
        )pbdoc")

            .def("calculate_payoffs", &egttools::FinitePopulations::OneShotCRD::calculate_payoffs, R"pbdoc(
        Updates the payoff matrix and cooperation level matrix for all strategy pairs.

        This method precomputes and stores the expected payoff for each strategy given every
        possible group composition, allowing for faster access in subsequent simulations.
        )pbdoc")

            .def("calculate_fitness", &egttools::FinitePopulations::OneShotCRD::calculate_fitness, R"pbdoc(
        Calculates the fitness of a strategy given a population state.

        Assumes the focal player is excluded from the population state.

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
                 py::arg("player_strategy"), py::arg("pop_size"), py::arg("population_state"))

            .def("calculate_population_group_achievement",
                 &egttools::FinitePopulations::OneShotCRD::calculate_population_group_achievement,
                 R"pbdoc(
        Computes the group achievement for the given population state.

        This metric captures the expected probability that a randomly formed group from the population
        will reach the collective contribution threshold.

        Parameters
        ----------
        population_size : int
            Total number of individuals.
        population_state : numpy.ndarray
            Vector of counts of each strategy in the population.

        Returns
        -------
        float
            The probability that a randomly formed group avoids the collective risk.
        )pbdoc",
                 py::arg("population_size"), py::arg("population_state"))

            .def("calculate_group_achievement",
                 &egttools::FinitePopulations::OneShotCRD::calculate_group_achievement,
                 R"pbdoc(
        Computes group achievement from a stationary distribution.

        This method evaluates the probability that a group avoids risk across all states
        weighted by their probability in the stationary distribution.

        Parameters
        ----------
        population_size : int
            Total population size.
        stationary_distribution : numpy.ndarray[float]
            Stationary distribution over population states.

        Returns
        -------
        float
            Weighted average group success probability.
        )pbdoc",
                 py::arg("population_size"), py::arg("stationary_distribution"))

            .def("__str__", &egttools::FinitePopulations::OneShotCRD::toString)
            .def("type", &egttools::FinitePopulations::OneShotCRD::type)
            .def("payoffs", &egttools::FinitePopulations::OneShotCRD::payoffs,
                 R"pbdoc(Returns the expected payoff matrix.)pbdoc")
            .def("payoff", &egttools::FinitePopulations::OneShotCRD::payoff,
                 R"pbdoc(Returns the payoff for a given strategy and group composition.)pbdoc",
                 py::arg("strategy"), py::arg("strategy pair"))
            .def_property_readonly("group_achievement_per_group",
                                   &egttools::FinitePopulations::OneShotCRD::group_achievements)
            .def("nb_strategies", &egttools::FinitePopulations::OneShotCRD::nb_strategies,
                 R"pbdoc(Number of strategies in the game.)pbdoc")
            .def_property_readonly("endowment", &egttools::FinitePopulations::OneShotCRD::endowment,
                                   R"pbdoc(Initial endowment per player.)pbdoc")
            .def_property_readonly("min_nb_cooperators",
                                   &egttools::FinitePopulations::OneShotCRD::min_nb_cooperators,
                                   R"pbdoc(Minimum number of cooperators required to avoid risk.)pbdoc")
            .def_property_readonly("group_size", &egttools::FinitePopulations::OneShotCRD::group_size,
                                   R"pbdoc(Number of players per group.)pbdoc")
            .def_property_readonly("risk", &egttools::FinitePopulations::OneShotCRD::risk,
                                   R"pbdoc(Probability of collective failure.)pbdoc")
            .def_property_readonly("cost", &egttools::FinitePopulations::OneShotCRD::cost,
                                   R"pbdoc(Fraction of endowment contributed by cooperators.)pbdoc")
            .def_property_readonly("nb_states", &egttools::FinitePopulations::OneShotCRD::nb_group_compositions,
                                   R"pbdoc(Number of distinct group compositions.)pbdoc")
            .def("save_payoffs", &egttools::FinitePopulations::OneShotCRD::save_payoffs,
                 R"pbdoc(Saves the payoff matrix to a text file.)pbdoc");

    py::class_<egttools::FinitePopulations::NPlayerStagHunt, egttools::FinitePopulations::AbstractGame>(
                mGames, "NPlayerStagHunt")
            .def(py::init<int, int, double, double>(), R"pbdoc(
        N-Player Stag Hunt (NPSH).

        This game models a public goods scenario where individuals may choose to cooperate (hunt stag)
        or defect (hunt hare). Only if a sufficient number of players cooperate is the public good provided.

        Based on the model described in:
        Pacheco, J. M., Vasconcelos, V. V., & Santos, F. C. (2014).
        "Evolutionary dynamics of collective action in N-person stag hunt dilemmas."
        Journal of Theoretical Biology, 350, 61–68.

        Parameters
        ----------
        group_size : int
            Number of players in the group (N).
        cooperation_threshold : int
            Minimum number of cooperators (M) required to produce the collective benefit.
        enhancement_factor : float
            Multiplicative factor applied to the benefit when the public good is provided.
        cost : float
            Cost of cooperation.
        )pbdoc",
                 py::arg("group_size"), py::arg("cooperation_threshold"),
                 py::arg("enhancement_factor"), py::arg("cost"),
                 py::return_value_policy::reference_internal)

            .def("play", &egttools::FinitePopulations::NPlayerStagHunt::play, R"pbdoc(
        Simulates the game and fills in the payoff vector for a given group composition.

        Parameters
        ----------
        group_composition : list[int] or numpy.ndarray[int]
            Number of players of each strategy in the group.
        game_payoffs : list[float] or numpy.ndarray[float]
            Output vector to store the resulting payoff for each player.
        )pbdoc")

            .def("calculate_payoffs", &egttools::FinitePopulations::NPlayerStagHunt::calculate_payoffs, R"pbdoc(
        Computes and stores the expected payoff matrix for all strategy-group combinations.

        Also updates internal cooperation level metrics for use in diagnostics and analysis.
        )pbdoc")

            .def("calculate_fitness", &egttools::FinitePopulations::NPlayerStagHunt::calculate_fitness, R"pbdoc(
        Computes the fitness of a strategy given a population state.

        Parameters
        ----------
        player_strategy : int
            Index of the focal strategy.
        pop_size : int
            Total number of individuals in the population.
        population_state : numpy.ndarray
            Vector of strategy counts (excluding the focal individual).

        Returns
        -------
        float
        )pbdoc",
                 py::arg("player_strategy"), py::arg("pop_size"), py::arg("population_state"))

            .def("calculate_population_group_achievement",
                 &egttools::FinitePopulations::NPlayerStagHunt::calculate_population_group_achievement,
                 R"pbdoc(
        Estimates the likelihood that a random group from the population meets the cooperation threshold.

        This value can serve as a proxy for expected success of collective actions.

        Parameters
        ----------
        population_size : int
            Total number of individuals in the population.
        population_state : numpy.ndarray
            Vector of counts for each strategy.

        Returns
        -------
        float
        )pbdoc",
                 py::arg("population_size"), py::arg("population_state"))

            .def("calculate_group_achievement",
                 &egttools::FinitePopulations::NPlayerStagHunt::calculate_group_achievement,
                 R"pbdoc(
        Computes the expected collective success weighted by a stationary distribution.

        Parameters
        ----------
        population_size : int
            Total population size.
        stationary_distribution : numpy.ndarray[float]
            Stationary distribution over population states.

        Returns
        -------
        float
            Average group success probability.
        )pbdoc",
                 py::arg("population_size"), py::arg("stationary_distribution"))

            .def("__str__", &egttools::FinitePopulations::NPlayerStagHunt::toString)
            .def("type", &egttools::FinitePopulations::NPlayerStagHunt::type)
            .def("payoffs", &egttools::FinitePopulations::NPlayerStagHunt::payoffs,
                 R"pbdoc(Returns the expected payoff matrix for all strategy combinations.)pbdoc")
            .def("payoff", &egttools::FinitePopulations::NPlayerStagHunt::payoff,
                 R"pbdoc(Returns the payoff of a strategy given a group composition.)pbdoc",
                 py::arg("strategy"), py::arg("strategy pair"))
            .def_property_readonly("group_achievement_per_group",
                                   &egttools::FinitePopulations::NPlayerStagHunt::group_achievements)
            .def("nb_strategies", &egttools::FinitePopulations::NPlayerStagHunt::nb_strategies,
                 R"pbdoc(Number of strategies involved in the game.)pbdoc")
            .def("strategies", &egttools::FinitePopulations::NPlayerStagHunt::strategies,
                 R"pbdoc(Returns the list of strategy names used in the game.)pbdoc")
            .def("nb_group_configurations", &egttools::FinitePopulations::NPlayerStagHunt::nb_group_configurations,
                 R"pbdoc(Number of unique group compositions.)pbdoc")
            .def_property_readonly("group_size", &egttools::FinitePopulations::NPlayerStagHunt::group_size,
                                   R"pbdoc(Size of the player group in each game round.)pbdoc")
            .def_property_readonly("cooperation_threshold",
                                   &egttools::FinitePopulations::NPlayerStagHunt::cooperation_threshold,
                                   R"pbdoc(Minimum number of cooperators required to succeed.)pbdoc")
            .def_property_readonly("enhancement_factor",
                                   &egttools::FinitePopulations::NPlayerStagHunt::enhancement_factor,
                                   R"pbdoc(Factor by which collective benefit is multiplied when successful.)pbdoc")
            .def_property_readonly("cost", &egttools::FinitePopulations::NPlayerStagHunt::cost,
                                   R"pbdoc(Cost paid by each cooperator.)pbdoc")
            .def("save_payoffs", &egttools::FinitePopulations::NPlayerStagHunt::save_payoffs,
                 R"pbdoc(Saves the payoff matrix to a text file.)pbdoc");

    py::class_<egttools::FinitePopulations::Matrix2PlayerGameHolder, egttools::FinitePopulations::AbstractGame>(
                mGames, "Matrix2PlayerGameHolder")
            .def(py::init<int, const Eigen::Ref<const egttools::Matrix2D> &>(), R"pbdoc(
        Matrix-based 2-Player Game Holder.

        Stores the expected payoffs between strategies in a 2-player game.
        This class is useful for simulations where the payoff matrix is externally computed
        and fixed, enabling fast fitness calculations without recomputation.

        Parameters
        ----------
        nb_strategies : int
            Number of strategies used in the game.
        payoff_matrix : numpy.ndarray[float64[m, m]]
            Matrix containing the payoff of each strategy against all others.
        )pbdoc",
                 py::arg("nb_strategies"), py::arg("payoff_matrix"),
                 py::return_value_policy::reference_internal, py::keep_alive<0, 2>())

            .def("play", &egttools::FinitePopulations::Matrix2PlayerGameHolder::play, R"pbdoc(
        Executes a match given a group composition and stores the resulting payoffs.

        Parameters
        ----------
        group_composition : list[int] or numpy.ndarray[int]
            Count of each strategy in the group (typically 2 players).
        game_payoffs : list[float] or numpy.ndarray[float]
            Output vector to be filled with each player's payoff.
        )pbdoc")

            .def("calculate_payoffs", &egttools::FinitePopulations::Matrix2PlayerGameHolder::calculate_payoffs,
                 R"pbdoc(
        Returns the stored payoff matrix.

        Returns
        -------
        numpy.ndarray
            Payoff matrix of shape (nb_strategies, nb_strategies).
        )pbdoc")

            .def("calculate_fitness", &egttools::FinitePopulations::Matrix2PlayerGameHolder::calculate_fitness,
                 R"pbdoc(
        Computes the fitness of a strategy given the population configuration.

        Assumes the focal player is not included in the population state.

        Parameters
        ----------
        player_type : int
            Index of the focal strategy.
        pop_size : int
            Size of the population.
        population_state : numpy.ndarray
            Vector of counts of each strategy in the population.

        Returns
        -------
        float
        )pbdoc",
                 py::arg("player_strategy"), py::arg("pop_size"), py::arg("population_state"))

            .def("__str__", &egttools::FinitePopulations::Matrix2PlayerGameHolder::toString)
            .def("type", &egttools::FinitePopulations::Matrix2PlayerGameHolder::type)
            .def("payoffs", &egttools::FinitePopulations::Matrix2PlayerGameHolder::payoffs,
                 R"pbdoc(Returns the expected payoff matrix.)pbdoc")
            .def("payoff", &egttools::FinitePopulations::Matrix2PlayerGameHolder::payoff,
                 R"pbdoc(Returns the payoff for a given strategy pair.)pbdoc",
                 py::arg("strategy"), py::arg("strategy pair"))
            .def("nb_strategies", &egttools::FinitePopulations::Matrix2PlayerGameHolder::nb_strategies,
                 R"pbdoc(Returns the number of strategies in the game.)pbdoc")
            .def("update_payoff_matrix",
                 &egttools::FinitePopulations::Matrix2PlayerGameHolder::update_payoff_matrix,
                 R"pbdoc(Replaces the internal payoff matrix with a new one.)pbdoc",
                 py::arg("payoff_matrix"))
            .def("save_payoffs", &egttools::FinitePopulations::Matrix2PlayerGameHolder::save_payoffs,
                 R"pbdoc(Saves the current payoff matrix to a text file.)pbdoc");

    py::class_<egttools::FinitePopulations::MatrixNPlayerGameHolder, egttools::FinitePopulations::AbstractGame>(
                mGames, "MatrixNPlayerGameHolder")
            .def(py::init<int, int, const Eigen::Ref<const egttools::Matrix2D> &>(), R"pbdoc(
        Matrix-based N-Player Game Holder.

        Stores a fixed matrix of expected payoffs for N-player games, where each entry corresponds
        to the expected payoff for a strategy across possible group configurations.

        This class enables efficient fitness evaluations for evolutionary simulations without
        re-computing payoffs.

        Parameters
        ----------
        nb_strategies : int
            Number of strategies in the game.
        group_size : int
            Size of the interacting group.
        payoff_matrix : numpy.ndarray[float64[m, n]]
            Matrix of shape (nb_strategies, nb_group_configurations) encoding payoffs for all strategy-group pairs.
        )pbdoc",
                 py::arg("nb_strategies"), py::arg("group_size"), py::arg("payoff_matrix"),
                 py::return_value_policy::reference_internal, py::keep_alive<0, 3>())

            .def("play", &egttools::FinitePopulations::MatrixNPlayerGameHolder::play, R"pbdoc(
        Simulates the game based on a predefined payoff matrix.

        Parameters
        ----------
        group_composition : list[int] or numpy.ndarray[int]
            Number of players using each strategy in the group.
        game_payoffs : list[float] or numpy.ndarray[float]
            Output vector for storing player payoffs.
        )pbdoc")

            .def("calculate_payoffs", &egttools::FinitePopulations::MatrixNPlayerGameHolder::calculate_payoffs,
                 R"pbdoc(
        Returns the internal matrix of precomputed payoffs.

        Returns
        -------
        numpy.ndarray
            Matrix of shape (nb_strategies, nb_group_configurations).
        )pbdoc")

            .def("calculate_fitness", &egttools::FinitePopulations::MatrixNPlayerGameHolder::calculate_fitness,
                 R"pbdoc(
        Computes the fitness of a strategy based on the current population state.

        Parameters
        ----------
        player_strategy : int
            Index of the strategy used by the focal player.
        pop_size : int
            Population size (excluding focal player).
        population_state : numpy.ndarray
            Vector of strategy counts in the population.

        Returns
        -------
        float
            Fitness of the focal strategy.
        )pbdoc",
                 py::arg("player_strategy"), py::arg("pop_size"), py::arg("population_state"))

            .def("__str__", &egttools::FinitePopulations::MatrixNPlayerGameHolder::toString)
            .def("type", &egttools::FinitePopulations::MatrixNPlayerGameHolder::type)
            .def("payoffs", &egttools::FinitePopulations::MatrixNPlayerGameHolder::payoffs,
                 R"pbdoc(Returns the full payoff matrix.)pbdoc")
            .def("payoff", &egttools::FinitePopulations::MatrixNPlayerGameHolder::payoff,
                 R"pbdoc(Returns the payoff for a strategy given a specific group configuration.)pbdoc",
                 py::arg("strategy"), py::arg("strategy pair"))
            .def("nb_strategies", &egttools::FinitePopulations::MatrixNPlayerGameHolder::nb_strategies,
                 R"pbdoc(Number of strategies defined in the game.)pbdoc")
            .def("group_size", &egttools::FinitePopulations::MatrixNPlayerGameHolder::group_size,
                 R"pbdoc(Size of the player group.)pbdoc")
            .def("nb_group_configurations",
                 &egttools::FinitePopulations::MatrixNPlayerGameHolder::nb_group_configurations,
                 R"pbdoc(Number of distinct group configurations supported by the matrix.)pbdoc")
            .def("update_payoff_matrix",
                 &egttools::FinitePopulations::MatrixNPlayerGameHolder::update_payoff_matrix,
                 R"pbdoc(Replaces the stored payoff matrix with a new one.)pbdoc",
                 py::arg("payoff_matrix"))
            .def("save_payoffs", &egttools::FinitePopulations::MatrixNPlayerGameHolder::save_payoffs,
                 R"pbdoc(Saves the payoff matrix to a text file.)pbdoc");


    py::class_<egttools::FinitePopulations::games::AbstractSpatialGame, stubs::PyAbstractSpatialGame>(
                mGames, "AbstractSpatialGame")
            .def(py::init<>(), R"pbdoc(
        Abstract base class for spatially structured games.

        This interface supports general spatial interaction models, where the fitness of a strategy
        is computed based on a local context (e.g., neighborhood composition).

        This is typically used in network-based or spatial grid environments.

        Note
        ----
        This interface is still under active development and may change in future versions.
        )pbdoc")

            .def("calculate_fitness", &egttools::FinitePopulations::games::AbstractSpatialGame::calculate_fitness,
                 py::arg("strategy_index"), py::arg("state"),
                 R"pbdoc(
        Calculates the fitness of a strategy in a local interaction context.

        Parameters
        ----------
        strategy_index : int
            The strategy of the focal player.
        state : numpy.ndarray[int]
            Vector representing the local configuration (e.g., neighbor counts).

        Returns
        -------
        float
            The computed fitness of the strategy in the given local state.
        )pbdoc")

            .def("nb_strategies", &egttools::FinitePopulations::games::AbstractSpatialGame::nb_strategies,
                 R"pbdoc(Returns the number of strategies in the spatial game.)pbdoc")

            .def("__str__", &egttools::FinitePopulations::games::AbstractSpatialGame::toString,
                 R"pbdoc(String representation of the spatial game.)pbdoc")

            .def("type", &egttools::FinitePopulations::games::AbstractSpatialGame::type,
                 R"pbdoc(Identifier for the type of spatial game.)pbdoc");

    py::class_<egttools::FinitePopulations::games::NormalFormNetworkGame,
                egttools::FinitePopulations::games::AbstractSpatialGame>(mGames, "NormalFormNetworkGame")
            .def(py::init<size_t, const Eigen::Ref<const egttools::Matrix2D> &>(), R"pbdoc(
        Normal Form Network Game.

        This class implements a networked version of a standard two-player normal form game.
        Agents are placed in a network or spatial structure and interact only with their neighbors,
        using a fixed payoff matrix to compute outcomes.

        Each agent selects an action (or strategy), and receives payoffs based on pairwise interactions
        with all neighbors. Repeated interactions are supported, allowing for multi-round strategies
        and dynamic behaviors such as conditional cooperation.

        Parameters
        ----------
        nb_rounds : int
            Number of repeated rounds for each pairwise encounter.
        payoff_matrix : numpy.ndarray[float64[m, m]]
            Payoff matrix specifying the outcomes for all strategy pairs.
        )pbdoc",
                 py::arg("nb_rounds"), py::arg("payoff_matrix"),
                 py::return_value_policy::reference_internal, py::keep_alive<0, 2>())

            .def(py::init(&egttools::init_normal_form_network_game_from_python_list), R"pbdoc(
        Normal Form Game with custom strategy list.

        This constructor initializes a network game with a custom list of strategies.
        Each strategy must be a pointer to a subclass of AbstractNFGStrategy.

        Parameters
        ----------
        nb_rounds : int
            Number of rounds of interaction.
        payoff_matrix : numpy.ndarray[float64[m, m]]
            Payoff matrix used for each pairwise encounter.
        strategies : List[egttools.behaviors.AbstractNFGStrategy]
            A list of strategy pointers.
        )pbdoc",
                 py::arg("nb_rounds"), py::arg("payoff_matrix"), py::arg("strategies"),
                 py::return_value_policy::reference_internal, py::keep_alive<0, 2>())

            .def("calculate_payoffs", &egttools::FinitePopulations::games::NormalFormNetworkGame::calculate_payoffs,
                 R"pbdoc(Recalculates the expected payoff matrix based on current strategies.)pbdoc")

            .def("calculate_fitness", &egttools::FinitePopulations::games::NormalFormNetworkGame::calculate_fitness,
                 py::arg("strategy_index"), py::arg("state"),
                 R"pbdoc(
        Computes fitness of a strategy given a neighborhood configuration.

        Parameters
        ----------
        strategy_index : int
            Strategy whose fitness will be computed.
        state : numpy.ndarray[int]
            Vector representing neighborhood strategy counts.

        Returns
        -------
        float
            Fitness value.
        )pbdoc")

            .def("calculate_cooperation_level_neighborhood",
                 &egttools::FinitePopulations::games::NormalFormNetworkGame::calculate_cooperation_level_neighborhood,
                 py::arg("strategy_index"), py::arg("state"),
                 R"pbdoc(
        Calculates the level of cooperation in a given neighborhood.

        Useful when using multi-round or conditional strategies where past actions
        may affect behavior.

        Parameters
        ----------
        strategy_index : int
            Focal strategy.
        state : numpy.ndarray[int]
            Neighbor strategy counts.

        Returns
        -------
        float
            Level of cooperation.
        )pbdoc")

            .def("nb_strategies", &egttools::FinitePopulations::games::NormalFormNetworkGame::nb_strategies,
                 R"pbdoc(Number of strategies available.)pbdoc")

            .def("nb_rounds", &egttools::FinitePopulations::games::NormalFormNetworkGame::nb_rounds,
                 R"pbdoc(Number of repeated rounds per encounter.)pbdoc")

            .def("__str__", &egttools::FinitePopulations::games::NormalFormNetworkGame::toString,
                 R"pbdoc(String representation of the game object.)pbdoc")

            .def("type", &egttools::FinitePopulations::games::NormalFormNetworkGame::type,
                 R"pbdoc(Returns the type identifier of the game.)pbdoc")

            .def("expected_payoffs", &egttools::FinitePopulations::games::NormalFormNetworkGame::expected_payoffs,
                 R"pbdoc(Returns the expected payoffs for each strategy.)pbdoc")

            .def("strategies", &egttools::FinitePopulations::games::NormalFormNetworkGame::strategies,
                 R"pbdoc(List of strategies currently active in the game.)pbdoc");

    py::class_<egttools::FinitePopulations::games::OneShotCRDNetworkGame,
                egttools::FinitePopulations::games::AbstractSpatialGame>(mGames, "OneShotCRDNetworkGame")
            .def(py::init<double, double, double, int>(), R"pbdoc(
        One-Shot Collective Risk Dilemma in Networks.

        This game implements the one-shot version of the Collective Risk Dilemma in spatial or networked settings,
        following the formulation in:

        Santos, F. C., & Pacheco, J. M. (2011).
        "Risk of collective failure provides an escape from the tragedy of the commons."
        Proceedings of the National Academy of Sciences, 108(26), 10421–10425.

        The game is played once by each group. Cooperation is costly, and a threshold number of
        cooperators is required to avoid collective loss. Spatial interaction allows strategies to
        propagate based on local fitness.

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
                 py::arg("endowment"), py::arg("cost"), py::arg("risk"), py::arg("min_nb_cooperators"),
                 py::return_value_policy::reference_internal)

            .def("calculate_fitness", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::calculate_fitness,
                 py::arg("strategy_index"), py::arg("state"),
                 R"pbdoc(
        Computes the fitness of a strategy in a local neighborhood.

        The neighborhood composition determines whether the group reaches the threshold required
        to avoid risk. If successful, players keep their endowment; otherwise, they lose it with probability `risk`.

        Parameters
        ----------
        strategy_index : int
            The focal strategy being evaluated.
        state : numpy.ndarray[int]
            Vector representing the number of neighbors using each strategy.

        Returns
        -------
        float
            The fitness of the strategy given the local state.
        )pbdoc")

            .def("nb_strategies", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::nb_strategies,
                 R"pbdoc(Returns the number of strategies available in the game.)pbdoc")

            .def("endowment", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::endowment,
                 R"pbdoc(Returns the initial endowment for each player.)pbdoc")

            .def("cost", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::cost,
                 R"pbdoc(Returns the cost of cooperation.)pbdoc")

            .def("risk", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::risk,
                 R"pbdoc(Returns the probability of losing endowment if cooperation fails.)pbdoc")

            .def("min_nb_cooperators",
                 &egttools::FinitePopulations::games::OneShotCRDNetworkGame::min_nb_cooperators,
                 R"pbdoc(Returns the minimum number of cooperators required to prevent collective loss.)pbdoc")

            .def("__str__", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::toString,
                 R"pbdoc(String representation of the game.)pbdoc")

            .def("type", &egttools::FinitePopulations::games::OneShotCRDNetworkGame::type,
                 R"pbdoc(Returns the identifier for the game type.)pbdoc");
}
