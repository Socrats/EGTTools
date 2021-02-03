//
// Created by Elias Fernandez on 2019-02-11.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <egttools/SeedGenerator.h>
#include <egttools/finite_populations/PairwiseMoran.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>
#include <egttools/finite_populations/games/NormalFormGame.h>
#include <egttools/LruCache.hpp>

namespace py = pybind11;
using namespace egttools;
using PairwiseComparison = egttools::FinitePopulations::PairwiseMoran<egttools::Utils::LRUCache<std::string, double>>;

PYBIND11_MODULE(numerical, m) {
  m.doc() = R"pbdoc(
        EGTtools: Efficient methods for modeling and studying Social Dynamics and Game Theory.
        This library is written in C++ (with python bindings) and pure Python.

        Note:
        Soon this library will be separated into two modules: EGTtools and LTtools which will contain
        methods for analysing social dynamics with evolutionary game theory (EGT) and Learning Theory (LT).
        The latter will be mostly focused on reinforcement learning (RL). These modules will be
        issues both separately and joined in the Dyrwin library.
        -----------------------
        .. currentmodule:: EGTtools
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

  // Use this function to get access to the singleton
  py::class_<Random::SeedGenerator, std::unique_ptr<Random::SeedGenerator, py::nodelete>>(m, "Random")
      .def("init", []() {
        return std::unique_ptr<Random::SeedGenerator, py::nodelete>(&Random::SeedGenerator::getInstance());
      }, R"pbdoc(
            Initializes the random seed generator from random_device.
           )pbdoc")
      .def("init", [](unsigned long int seed) {
        auto instance = std::unique_ptr<Random::SeedGenerator, py::nodelete>(&Random::SeedGenerator::getInstance());
        instance->setMainSeed(seed);
        return instance;
      }, R"pbdoc(
            Initializes the random seed generator from seed.
           )pbdoc")
      .def_property_readonly_static("seed_", [](const py::object&) {
                                      return egttools::Random::SeedGenerator::getInstance().getMainSeed();
                                    }, R"pbdoc(Returns current seed)pbdoc"
      )
      .def_static("generate", []() {
        return egttools::Random::SeedGenerator::getInstance().getSeed();
      },"generates a random seed")
      .def_static("seed", [](unsigned long int seed){
        egttools::Random::SeedGenerator::getInstance().setMainSeed(seed);
      });

  py::class_<egttools::FinitePopulations::AbstractGame>(m, "AbstractGame")
      .def("play", &egttools::FinitePopulations::AbstractGame::play)
      .def("calculate_payoffs", &egttools::FinitePopulations::AbstractGame::calculate_payoffs)
      .def("calculate_fitness", &egttools::FinitePopulations::AbstractGame::calculate_fitness)
      .def("to_string", &egttools::FinitePopulations::AbstractGame::toString)
      .def("type", &egttools::FinitePopulations::AbstractGame::type)
      .def("payoffs", &egttools::FinitePopulations::AbstractGame::payoffs)
      .def("payoff", &egttools::FinitePopulations::AbstractGame::payoff)
      .def_property_readonly("nb_strategies", &egttools::FinitePopulations::AbstractGame::nb_strategies)
      .def("save_payoffs", &egttools::FinitePopulations::AbstractGame::save_payoffs);

  m.def("calculate_state",
        static_cast<size_t (*)(const size_t &, const egttools::Factors &)>(&egttools::FinitePopulations::calculate_state),
        "calculates an index given a simplex state",
        py::arg("group_size"), py::arg("group_composition"));
  m.def("calculate_state",
        static_cast<size_t (*)(const size_t &,
                               const Eigen::Ref<const egttools::VectorXui> &)>(&egttools::FinitePopulations::calculate_state),
        "calculates an index given a simplex state",
        py::arg("group_size"), py::arg("group_composition"));
  m.def("sample_simplex",
        static_cast<egttools::VectorXui (*)(size_t, const size_t &, const size_t &)>(&egttools::FinitePopulations::sample_simplex),
        "returns a point in the simplex given an index", py::arg("index"), py::arg("pop_size"),
        py::arg("nb_strategies"));
  m.def("calculate_nb_states",
        &egttools::starsBars,
        "calculates the number of states (combinations) of the members of a group in a subgroup.",
        py::arg("group_size"), py::arg("sub_group_size"));

  py::class_<egttools::FinitePopulations::NormalFormGame, egttools::FinitePopulations::AbstractGame>(m, "NormalFormGame")
      .def(py::init<size_t, const Eigen::Ref<const Matrix2D> &>(), "Normal Form Game", py::arg("nb_rounds"),
           py::arg("payoff_matrix"))
      .def("play", &egttools::FinitePopulations::NormalFormGame::play)
      .def("calculate_payoffs", &egttools::FinitePopulations::NormalFormGame::calculate_payoffs,
           "updates the internal payoff and coop_level matrices by calculating the payoff of each strategy "
           "given any possible strategy pair")
      .def("calculate_fitness", &egttools::FinitePopulations::NormalFormGame::calculate_fitness,
           "calculates the fitness of an individual of a given strategy given a population state."
           "It always assumes that the population state does not contain the current individual",
           py::arg("player_strategy"),
           py::arg("pop_size"), py::arg("population_state"))
      .def("to_string", &egttools::FinitePopulations::NormalFormGame::toString)
      .def("type", &egttools::FinitePopulations::NormalFormGame::type)
      .def("payoffs", &egttools::FinitePopulations::NormalFormGame::payoffs)
      .def("payoff", &egttools::FinitePopulations::NormalFormGame::payoff,
           "returns the payoff of a strategy given a strategy pair.", py::arg("strategy"),
           py::arg("strategy pair"))
      .def("expected_payoffs", &egttools::FinitePopulations::NormalFormGame::expected_payoffs, "returns the expected payoffs of each strategy vs another")
      .def_property_readonly("nb_strategies", &egttools::FinitePopulations::NormalFormGame::nb_strategies)
      .def_property_readonly("nb_rounds", &egttools::FinitePopulations::NormalFormGame::nb_rounds)
      .def_property_readonly("nb_states", &egttools::FinitePopulations::NormalFormGame::nb_states)
      .def("save_payoffs", &egttools::FinitePopulations::NormalFormGame::save_payoffs);

  py::class_<PairwiseComparison>(m, "PairwiseMoran")
      .def(py::init<size_t, egttools::FinitePopulations::AbstractGame &, size_t>(),
           "Runs a moran process with pairwise comparison and calculates fitness according to game",
           py::arg("pop_size"), py::arg("game"), py::arg("cache_size"), py::keep_alive<1, 3>())
      .def("evolve",
           static_cast<egttools::VectorXui (PairwiseComparison::*)(size_t, double, double,
                                                                   const Eigen::Ref<const egttools::VectorXui> &)>( &PairwiseComparison::evolve ),
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
}