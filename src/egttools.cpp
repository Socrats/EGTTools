//
// Created by Elias Fernandez on 2019-02-11.
//

#include <egttools/SeedGenerator.h>
#include <egttools/finite_populations/games/NormalFormGame.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <egttools/Data.hpp>
#include <egttools/LruCache.hpp>
#include <egttools/finite_populations/PairwiseMoran.hpp>
#include <egttools/finite_populations/behaviors/NFGStrategies.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>

namespace py = pybind11;
using namespace egttools;
using PairwiseComparison = egttools::FinitePopulations::PairwiseMoran<egttools::Utils::LRUCache<std::string, double>>;

PYBIND11_MODULE(numerical, m) {
    m.doc() = R"pbdoc(
        EGTtools: Efficient Evolutionary Game Theory models and methods.
        This library is written in C++ (with python bindings) and pure Python.

        Note:
        This model is part a larger library named Dyrwin which also includes other methods to model
        social dynamics which use learning theory (e.g., Reinforcement Learning), and methods to analyze, process
        and model experimental data.
        -----------------------
        .. currentmodule:: egttools
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    // Use this function to get access to the singleton
    py::class_<Random::SeedGenerator, std::unique_ptr<Random::SeedGenerator, py::nodelete>>(m, "Random")
            .def(
                    "init", []() {
                        return std::unique_ptr<Random::SeedGenerator, py::nodelete>(&Random::SeedGenerator::getInstance());
                    },
                    R"pbdoc(
            Initializes the random seed generator from random_device.
           )pbdoc")
            .def(
                    "init", [](unsigned long int seed) {
                        auto instance = std::unique_ptr<Random::SeedGenerator, py::nodelete>(&Random::SeedGenerator::getInstance());
                        instance->setMainSeed(seed);
                        return instance;
                    },
                    R"pbdoc(
            Initializes the random seed generator from seed.
           )pbdoc")
            .def_property_readonly_static(
                    "seed_", [](const py::object &) {
                        return egttools::Random::SeedGenerator::getInstance().getMainSeed();
                    },
                    R"pbdoc(Returns current seed)pbdoc")
            .def_static(
                    "generate", []() {
                        return egttools::Random::SeedGenerator::getInstance().getSeed();
                    },
                    "generates a random seed")
            .def_static("seed", [](unsigned long int seed) {
                egttools::Random::SeedGenerator::getInstance().setMainSeed(seed);
            });

    // Now we define a submodule
    auto mGames = m.def_submodule("games");
    auto mBehaviors = m.def_submodule("behaviors");
    auto mNF = m.def_submodule("NormalForm");
    auto mNFTwoActions = m.def_submodule("TwoActions");
    auto mData = m.def_submodule("DataStructures");

    py::class_<egttools::FinitePopulations::AbstractGame>(mGames, "AbstractGame")
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

    py::class_<egttools::FinitePopulations::NormalFormGame, egttools::FinitePopulations::AbstractGame>(mGames, "NormalFormGame")
            .def(py::init<size_t, const Eigen::Ref<const Matrix2D> &>(),
                 "Normal Form Game. This constructor assumes that there are only two possible strategies and two possible actions.",
                 py::arg("nb_rounds"),
                 py::arg("payoff_matrix"))
            .def(py::init<size_t, const Eigen::Ref<const Matrix2D> &, const egttools::FinitePopulations::StrategyVector &>(),
                 "Normal Form Game", py::arg("nb_rounds"),
                 py::arg("payoff_matrix"), py::arg("strategies"))
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
            .def_property_readonly("strategies", &egttools::FinitePopulations::NormalFormGame::strategies)
            .def("save_payoffs", &egttools::FinitePopulations::NormalFormGame::save_payoffs);

    py::class_<egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNF, "AbstractNFGStrategy")
            .def("get_action", &egttools::FinitePopulations::behaviors::AbstractNFGStrategy::get_action)
            .def("type", &egttools::FinitePopulations::behaviors::AbstractNFGStrategy::type);

    py::class_<egttools::FinitePopulations::behaviors::twoActions::Cooperator,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Cooperator")
            .def(py::init<>(), "This strategy always cooperates.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::Cooperator::get_action)
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::Cooperator::type);

    py::class_<egttools::FinitePopulations::behaviors::twoActions::Defector,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Defector")
            .def(py::init<>(), "This strategy always defects.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::Defector::get_action)
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::Defector::type);

    py::class_<egttools::FinitePopulations::behaviors::twoActions::RandomPlayer,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Random")
            .def(py::init<>(), "This players chooses cooperation with uniform random probability.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::get_action)
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::type);

    py::class_<egttools::FinitePopulations::behaviors::twoActions::TitForTat,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "TFT")
            .def(py::init<>(), "Tit for Tat: Cooperates in the first round and imitates the opponent's move thereafter.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::TitForTat::get_action)
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::TitForTat::type);

    py::class_<egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "SuspiciousTFT")
            .def(py::init<>(), "Defects on the first round and imitates its opponent's previous move thereafter.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::get_action)
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::type);

    py::class_<egttools::FinitePopulations::behaviors::twoActions::GenerousTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "GenerousTFT")
            .def(py::init<double, double, double, double>(),
                 "Cooprates on the first round and after its opponent "
                 "cooperates. Following a defection,it cooperates with probability\n"
                 "@f[ p(R,P,T,S) = min{1 - \\frac{T-R}{R-S}, \\frac{R-P}{T-P}} @f]\n"
                 "where R, P, T and S are the reward, punishment, temptation and "
                 "suckers payoffs.",
                 py::arg("R"), py::arg("P"),
                 py::arg("T"), py::arg("S"))
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::get_action)
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::type);

    py::class_<egttools::FinitePopulations::behaviors::twoActions::GradualTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "GradualTFT")
            .def(py::init<>(),
                 "TFT with two differences:\n"
                 "(1) it increases the string of punishing defection responses "
                 "with each additional defection by its opponent\n"
                 "(2) it apologizes for each string of defections"
                 "by cooperating in the subsequent two rounds.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::GradualTFT::get_action)
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::GradualTFT::type);

    py::class_<egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "ImperfectTFT")
            .def(py::init<double>(),
                 "Imitates opponent as in TFT, but makes mistakes "
                 "with :param error_probability.",
                 py::arg("error_probability"))
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::get_action)
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::type);

    py::class_<egttools::FinitePopulations::behaviors::twoActions::TFTT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "TFTT")
            .def(py::init<>(), "Tit for 2 tats: Defects if defected twice.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::TFTT::get_action)
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::TFTT::type);

    py::class_<egttools::FinitePopulations::behaviors::twoActions::TTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "TTFT")
            .def(py::init<>(), "2 Tits for tat: Defects twice if defected.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::TTFT::get_action)
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::TTFT::type);

    py::class_<egttools::FinitePopulations::behaviors::twoActions::GRIM,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "GRIM")
            .def(py::init<>(),
                 "Grim (Trigger): Cooperates until its opponent has defected once, "
                 "and then defects for the rest of the game.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::GRIM::get_action)
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::GRIM::type);

    py::class_<egttools::FinitePopulations::behaviors::twoActions::Pavlov,
            egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Pavlov")
            .def(py::init<>(),
                 "Win-stay loose-shift: Cooperates if it and its opponent moved alike in"
                 "previous move and defects if they moved differently.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::Pavlov::get_action)
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::Pavlov::type);

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
}