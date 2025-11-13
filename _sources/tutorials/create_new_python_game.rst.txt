.. _create-new-python-game:

========================================
Creating a New Game Directly in Python
========================================

This guide explains how to create a **new game purely in Python** using EGTtools, without needing to modify or compile C++ code.

If you want to create a new C++ game, see instead: :ref:`create-new-cpp-game`.

Implementing a Game from Scratch: the `AbstractGame` Class
----------------------------------------------------------

The `Game` class is central to the EGTtools library: it defines the environment where strategic interactions happen.
All games must extend the `AbstractGame` abstract base class.

A new game must implement the following methods:

🛠️ **Required Methods**:

- `play`
- `calculate_payoffs`
- `calculate_fitness`
- `__str__`
- `nb_strategies`
- `type`
- `payoffs`
- `payoff`
- `save_payoffs`

Below is a description of the class and the purpose of each method:

.. code-block:: python

    class AbstractGame:
        def play(self, group_composition: Union[List[int], numpy.ndarray], game_payoffs: numpy.ndarray) -> None:
            """Calculate payoffs for each strategy inside a group."""
            pass

        def calculate_payoffs(self) -> np.ndarray:
            """Set up the payoff matrix for all possible group compositions."""
            pass

        def calculate_fitness(self, strategy_index: int, pop_size: int, population_state: numpy.ndarray) -> float:
            """Return the fitness of a strategy given the current population state."""
            pass

        def __str__(self) -> str:
            """Return a string representation of the game."""
            pass

        def nb_strategies(self) -> int:
            """Return the number of available strategies."""
            pass

        def type(self) -> str:
            """Return a string describing the game type."""
            pass

        def payoffs(self) -> np.ndarray:
            """Return the payoff matrix."""
            pass

        def payoff(self, strategy: int, group_configuration: List[int]) -> float:
            """Return the payoff of a strategy for a given group composition."""
            pass

        def save_payoffs(self, file_name: str) -> None:
            """Save the payoff matrix and game parameters to a file."""
            pass

Simplifying Game Implementation: `AbstractTwoPLayerGame` and `AbstractNPlayerGame`
-----------------------------------------------------------------------------------

In many cases, the fitness of a strategy is simply its expected payoff at a given state.
For this reason, EGTtools provides two simplified abstract classes:

📚 **Simplified Base Classes**:

- `egttools.games.AbstractTwoPLayerGame` — for two-player games
- `egttools.games.AbstractNPlayerGame` — for N-player games

When using these classes, you only need to implement:

- `play`
- `calculate_payoffs`

Example: The N-Player Stag Hunt Game
------------------------------------

Here is an example of how to implement the **N-player Stag Hunt Game**:

.. code-block:: python

    from egttools.games import AbstractNPlayerGame
    from egttools import sample_simplex
    import numpy as np
    from typing import Union, List

    class NPlayerStagHunt(AbstractNPlayerGame):

        def __init__(self, group_size, enhancement_factor, cooperation_threshold, cost):
            self.group_size_ = group_size
            self.enhancement_factor_ = enhancement_factor
            self.cooperation_threshold_ = cooperation_threshold
            self.cost_ = cost
            self.strategies = ['Defect', 'Cooperate']
            self.nb_strategies_ = 2
            super().__init__(self.nb_strategies_, self.group_size_)

        def play(self, group_composition: Union[List[int], np.ndarray], game_payoffs: np.ndarray) -> None:
            if group_composition[0] == 0:
                game_payoffs[0] = 0
                game_payoffs[1] = self.cost_ * (self.enhancement_factor_ - 1)
            elif group_composition[1] == 0:
                game_payoffs[0] = 0
                game_payoffs[1] = 0
            else:
                payoff = (group_composition[1] * self.enhancement_factor_) / self.group_size_ \
                         if group_composition[1] >= self.cooperation_threshold_ else 0
                game_payoffs[0] = payoff
                game_payoffs[1] = payoff - self.cost_

        def calculate_payoffs(self) -> np.ndarray:
            payoffs_container = np.zeros(shape=(self.nb_strategies_,), dtype=np.float64)
            for i in range(self.nb_group_configurations_):
                group_composition = sample_simplex(i, self.group_size_, self.nb_strategies_)
                self.play(group_composition, payoffs_container)
                for strategy_index, strategy_payoff in enumerate(payoffs_container):
                    self.payoffs_[strategy_index, i] = strategy_payoff
                payoffs_container[:] = 0
            return self.payoffs_

Already Have a Precomputed Payoff Matrix?
------------------------------------------

If you have a ready-made payoff matrix, you can avoid writing a full class. Use:

📦 **Matrix-based Holders**:

- `Matrix2PlayerGameHolder` — for 2-player games (requires a square matrix)
- `MatrixNPlayerGameHolder` — for N-player games (requires a `(nb_strategies, nb_group_configurations)` matrix)

.. note::

    - Calculate the number of group configurations using `egttools.calculate_nb_states(group_size, nb_strategies)`.
    - Use `egttools.sample_simplex(index, group_size, nb_strategies)` to retrieve the group configuration for a column.

🚨 **Important**:
When a strategy is not present in a group configuration, its expected payoff must be set to **0**.

You can find a full example here: :doc:`examples/hawk_dove_dynamics`

List of Implemented Games
--------------------------

- **NormalFormGame** — Iterated matrix games.
- **PGG** — Public Goods Game with group-based cooperation.
- **OneShotCRD** — One-shot Collective Risk Dilemma :cite:t:`santosRiskCollectiveFailure2011`.
- **CRDGame** — Collective Risk Dilemma over multiple rounds :cite:t:`milinski2008collective`.
- **NPlayerStagHunt** — N-player Stag Hunt Game :cite:t:`Pacheco2009`.

See also: :doc:`create_new_behaviors` to learn how to extend or create new strategies.
