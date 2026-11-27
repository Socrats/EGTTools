.. _create-new-cpp-game:

===================================
Creating a New Game in C++ (EGTtools)
===================================

This guide explains how to implement and register a **custom game written in C++** for use within EGTtools, following the latest API conventions.

If you instead want to create a game **purely in Python**, see the guide: :ref:`create-new-python-game`.

Overview
--------

In EGTtools, all C++ game classes must inherit from the `AbstractGame` interface.
Depending on your game's structure, you may choose to subclass:

- `AbstractNPlayerGame` — for general N-player games
- `AbstractSpatialGame` — for spatially structured or networked games
- `MatrixNPlayerGameHolder` — for matrix-based N-player interactions

Each base class provides tailored support for different types of evolutionary game models.

Required Methods
----------------

A custom game implemented in C++ must override the following methods:

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

To expose your C++ game class to Python via `pybind11`, add it to the `init_games()` binding function like this:

.. code-block:: cpp

    py::class_<MyCustomGame, AbstractGame>(m, "MyCustomGame")
        .def(py::init<int, double>())
        .def("play", &MyCustomGame::play)
        .def("calculate_payoffs", &MyCustomGame::calculate_payoffs)
        .def("calculate_fitness", &MyCustomGame::calculate_fitness)
        .def("__str__", &MyCustomGame::toString)
        .def("nb_strategies", &MyCustomGame::nb_strategies)
        .def("type", &MyCustomGame::type)
        .def("payoffs", &MyCustomGame::payoffs)
        .def("payoff", &MyCustomGame::payoff)
        .def("save_payoffs", &MyCustomGame::save_payoffs);

After compiling EGTtools, your new game class is accessible from Python:

.. code-block:: python

    from egttools.numerical.games import MyCustomGame
    game = MyCustomGame(nb_strategies=2, intensity=0.5)
    print(game.payoffs())

Testing and Integration
------------------------

When validating your new game:

- ✔ Ensure the payoff matrix shape is correct: `(nb_strategies, nb_group_configurations)`.
- ✔ Run numerical tests using classes like `PairwiseComparisonNumerical`.
- ✔ Use `.save_payoffs("out.txt")` and manually inspect the output.
- ✔ Try basic simulations to ensure evolutionary dynamics work without crashes.

Related Tutorials
-----------------

- 📖 See also: :ref:`create-new-python-game` — *Create a game directly in Python, without C++ compilation*.