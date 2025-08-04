.. _creating-new-strategies:

Creating New Strategies
========================

EGTtools allows you to define your own strategies for evolutionary simulations.
You can either:
- Create custom standalone strategies for your own custom games,
- Extend existing games by creating new strategies based on the provided Abstract classes.

---

Standalone Custom Strategies
-----------------------------

If you are implementing your **own game** from scratch, you do not need to depend on EGTtools' internal strategy system.
You can simply define strategies as you like (e.g., by setting custom rules, payoffs, or dynamics).

For example:

.. code-block:: python

    class MyCustomStrategy:
        def decide_action(self, state):
            # Define behavior here
            return "cooperate" if state < 0.5 else "defect"

This is useful when designing **entirely new environments or games**.

---

Extending Existing Games
-------------------------

If you want your strategy to **work with an existing EGTtools game** (e.g., `CRD`, `CPR`, `NormalForm` games),
you must inherit from the corresponding **Abstract Strategy** class provided in `egttools.behaviors`.

Available Abstract classes include:

- `AbstractCPRStrategy` — for strategies interacting with Common Pool Resource games.
- `AbstractCRDStrategy` — for strategies interacting with Common Resource Dilemma games.
- `AbstractNormalFormStrategy` — for strategies interacting with Normal Form games.

Each Abstract class defines the methods your strategy must implement.

Example: Defining a New CPR Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose you want to create a new strategy for the Common Pool Resource (CPR) game.

First, inherit from `AbstractCPRStrategy`:

.. code-block:: python

    from egttools.behaviors.CPR.abstract_cpr_strategy import AbstractCPRStrategy

    class AlwaysConserveCPR(AbstractCPRStrategy):
        def __init__(self):
            super().__init__()

        def act(self, state):
            """Always act conservatively, regardless of the state."""
            return 0  # 0 could represent 'conserve'

This strategy can now be used inside the CPR game framework.

Example: Defining a New Normal Form Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Normal Form games:

.. code-block:: python

    from egttools.behaviors.NormalForm.abstract_normal_form_strategy import AbstractNormalFormStrategy

    class TitForTat(AbstractNormalFormStrategy):
        def __init__(self):
            super().__init__()
            self.last_opponent_move = None

        def reset(self):
            self.last_opponent_move = None

        def play(self, own_payoffs, opponent_payoffs):
            if self.last_opponent_move is None:
                return 0  # cooperate initially
            return self.last_opponent_move

        def receive_result(self, own_action, opponent_action):
            self.last_opponent_move = opponent_action

This strategy will mimic the opponent's previous move, a classic in repeated games.

---

Implementation Notes
---------------------

- Strategies must correctly implement all **abstract methods** defined in the parent class.
- C++ and Python strategies are **treated the same way** once exposed via pybind11 bindings.
- Strategies can be stateful (keep internal memory) or stateless (fixed behaviors).

If you miss implementing a required method, you will receive a `TypeError` when trying to instantiate your strategy.

---

.. note::
    EGTtools' strategy system is designed to be flexible.
    You are free to create rich, complex strategies with memory, randomness, and learning — not just static behaviors.

.. note::
    In future versions, we plan to provide templates and utility classes to simplify the creation of memory-based strategies.
