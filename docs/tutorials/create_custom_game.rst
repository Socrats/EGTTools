.. _custom-game-guide:

=============================
Creating a Custom Game in EGTtools
=============================

This guide explains how to implement and register a custom game in EGTtools. It extends the original tutorial
with structured advice, required method summaries, and integration tips for the Python interface.

Overview
--------
All game classes in EGTtools must implement a core interface derived from `AbstractGame`. Depending on your model,
you may subclass `AbstractNPlayerGame`, `AbstractSpatialGame`, or a more specific game holder like `MatrixNPlayerGameHolder`.

Required Methods
----------------
A custom game should implement the following methods:

.. code-block:: cpp

    void play(const std::vector<int>& group_composition,
              std::vector<double>& game_payoffs) override;

    Matrix2D calculate_payoffs() override;

    double calculate_fitness(int strategy_index,
                             int pop_size,
                             const Eigen::Ref<const VectorInt>& strategies) override;

    std::string toString() const override;

    int nb_strategies() const override;

    std::string type() const override;

    Matrix2D payoffs() const override;

    double payoff(int strategy,
                  const std::vector<int>& group_composition) const override;

    void save_payoffs(const std::string& filename) const override;

Registering Your Game in Python
-------------------------------
Use `pybind11` to expose your class in the `init_games()` function:

.. code-block:: cpp

    py::class_<MyCustomGame, AbstractGame>(m, "MyCustomGame")
        .def(py::init<int, double>())
        .def("play", &MyCustomGame::play)
        .def("calculate_fitness", &MyCustomGame::calculate_fitness)
        .def("__str__", &MyCustomGame::toString);

Once compiled, your game is available via:

.. code-block:: python

    from egttools.numerical.games import MyCustomGame
    game = MyCustomGame(nb_strategies=2, intensity=0.5)
    print(game.payoffs())

Testing and Integration
------------------------
- Check that the shape of the payoff matrix is consistent: `(nb_strategies, nb_group_configurations)`
- Validate simulation compatibility using numerical tools like `PairwiseComparisonNumerical`
- Use `.save_payoffs("out.txt")` to inspect output

Suggested Future Documentation
------------------------------

We recommend extending the documentation with:

- **Game Design Patterns**: Contrast 2-player, N-player, symmetric/asymmetric models
- **Strategy Integration Guide**: How to subclass and inject strategies into custom games
- **Examples Gallery**: Realistic annotated samples of CRD, public goods, volunteer's dilemma, etc.
- **Advanced Custom Pipelines**: Creating hybrid simulations using user-defined replicator rules or spatial networks

See Also
--------
- :ref:`api-abstract-game`
- :ref:`tutorials`
