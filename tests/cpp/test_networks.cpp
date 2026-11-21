// Network structures (Network: asynchronous updates; NetworkSync: synchronous updates) and the
// evolvers that run them (NetworkEvolver, GeneralPopulationEvolver).
//
// These tests check the rules of the update (state bookkeeping, fitness from the neighbourhood,
// one change per asynchronous step, absorbing states). Comparisons of the dynamics with an
// independent reference implementation are NET-01.
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>
#include <egttools/finite_populations/evolvers/GeneralPopulationEvolver.hpp>
#include <egttools/finite_populations/evolvers/NetworkEvolver.hpp>
#include <egttools/finite_populations/games/NormalFormNetworkGame.h>
#include <egttools/finite_populations/structure/Network.hpp>
#include <egttools/finite_populations/structure/NetworkSync.hpp>

using Catch::Matchers::WithinAbs;
using egttools::Matrix2D;
using egttools::VectorXui;
using NodeDictionary = egttools::FinitePopulations::structure::NodeDictionary;
using Game = egttools::FinitePopulations::games::NormalFormNetworkGame;
using AsyncNetwork = egttools::FinitePopulations::structure::Network<Game>;
using SyncNetwork = egttools::FinitePopulations::structure::NetworkSync<Game>;
using egttools::FinitePopulations::evolvers::NetworkEvolver;

namespace {
    NodeDictionary complete_graph(const int n) {
        NodeDictionary graph;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (i != j) graph[i].push_back(j);
        return graph;
    }

    // A ring of n nodes with one extra node (index n) that has no neighbours.
    NodeDictionary ring_with_isolated_node(const int n) {
        NodeDictionary graph;
        for (int i = 0; i < n; ++i) graph[i] = {(i + n - 1) % n, (i + 1) % n};
        graph[n] = {};
        return graph;
    }

    Matrix2D prisoners_dilemma() {
        Matrix2D A(2, 2);
        A << 3.0, 0.0, 5.0, 1.0;
        return A;
    }

    template<typename Structure>
    VectorXui count_strategies(const Structure &structure, const int nb_strategies) {
        VectorXui counts = VectorXui::Zero(nb_strategies);
        for (const int s: structure.population_strategies()) counts(s) += 1;
        return counts;
    }
}// namespace

TEST_CASE("Network initialize_state places the requested number of each strategy", "[networks]") {
    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);
    Game game(1, prisoners_dilemma());
    AsyncNetwork network(2, 1.0, 0.0, complete_graph(10), game);
    VectorXui state(2);
    state << 3, 7;
    network.initialize_state(state);
    CHECK(network.population_size() == 10);
    CHECK(network.mean_population_state() == state);
    CHECK(count_strategies(network, 2) == state);
}

TEST_CASE("Network fitness is the total payoff against the neighbours", "[networks]") {
    // NormalFormNetworkGame: a node's fitness is the sum of A(own, neighbour) over its neighbours.
    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);
    const Matrix2D A = prisoners_dilemma();
    Game game(1, A);
    const auto graph = ring_with_isolated_node(6);
    AsyncNetwork network(2, 1.0, 0.0, graph, game);
    VectorXui state(2);
    state << 3, 4;
    network.initialize_state(state);
    const auto strategies = network.population_strategies();
    double max_error = 0.0;
    for (int node = 0; node < network.population_size(); ++node) {
        double expected = 0.0;
        for (const int neighbour: graph.at(node)) expected += A(strategies[node], strategies[neighbour]);
        max_error = std::max(max_error, std::abs(network.calculate_fitness(node) - expected));
    }
    CHECK(max_error < 1e-14);
}

TEST_CASE("Network asynchronous updates change at most one node per step", "[networks]") {
    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);
    Game game(1, prisoners_dilemma());
    AsyncNetwork network(2, 1.0, 0.05, complete_graph(12), game);
    VectorXui state(2);
    state << 6, 6;
    network.initialize_state(state);
    int wrong_bookkeeping = 0, multiple_changes = 0;
    for (int step = 0; step < 5000; ++step) {
        const auto before = network.population_strategies();
        network.update_population();
        const auto after = network.population_strategies();
        int changed = 0;
        for (size_t i = 0; i < before.size(); ++i) changed += before[i] != after[i];
        multiple_changes += changed > 1;
        wrong_bookkeeping += network.mean_population_state() != count_strategies(network, 2);
    }
    CHECK(multiple_changes == 0);
    CHECK(wrong_bookkeeping == 0);
}

TEST_CASE("Network without mutation: monomorphic states and isolated nodes never change", "[networks]") {
    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);
    Game game(1, prisoners_dilemma());
    AsyncNetwork monomorphic(2, 1.0, 0.0, complete_graph(8), game);
    VectorXui all_zero(2);
    all_zero << 8, 0;
    monomorphic.initialize_state(all_zero);
    for (int step = 0; step < 2000; ++step) monomorphic.update_population();
    CHECK(monomorphic.mean_population_state() == all_zero);

    // A node without neighbours has no role model, so it keeps its strategy.
    AsyncNetwork ring(2, 5.0, 0.0, ring_with_isolated_node(6), game);
    VectorXui mixed(2);
    mixed << 3, 4;
    ring.initialize_state(mixed);
    const int isolated_strategy = ring.population_strategies()[6];
    for (int step = 0; step < 2000; ++step) ring.update_population();
    CHECK(ring.population_strategies()[6] == isolated_strategy);
}

TEST_CASE("NetworkSync updates keep the bookkeeping consistent", "[networks]") {
    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);
    Game game(1, prisoners_dilemma());
    SyncNetwork network(2, 1.0, 0.05, complete_graph(10), game);
    VectorXui state(2);
    state << 5, 5;
    network.initialize_state(state);
    CHECK(network.mean_population_state() == state);
    int wrong_bookkeeping = 0;
    for (int step = 0; step < 500; ++step) {
        network.update_population();
        wrong_bookkeeping += network.mean_population_state() != count_strategies(network, 2);
        wrong_bookkeeping += network.mean_population_state().sum() != 10;
    }
    CHECK(wrong_bookkeeping == 0);

    // Without mutation a monomorphic population stays monomorphic.
    SyncNetwork monomorphic(2, 1.0, 0.0, complete_graph(10), game);
    VectorXui all_one(2);
    all_one << 0, 10;
    monomorphic.initialize_state(all_one);
    for (int step = 0; step < 200; ++step) monomorphic.update_population();
    CHECK(monomorphic.mean_population_state() == all_one);
}

TEST_CASE("NetworkEvolver::run records the state after the transitory period", "[networks][evolvers]") {
    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);
    Game game(1, prisoners_dilemma());
    AsyncNetwork network(2, 1.0, 0.01, complete_graph(10), game);
    VectorXui state(2);
    state << 5, 5;
    const auto trajectory = NetworkEvolver::run(300, 100, state, network);
    REQUIRE(trajectory.rows() == 200);
    REQUIRE(trajectory.cols() == 2);
    CHECK((trajectory.rowwise().sum().array() == 10).all());
    CHECK_THROWS_AS(NetworkEvolver::run(100, 100, state, network), std::invalid_argument);
}

TEST_CASE("NetworkEvolver::evolve without mutation keeps a monomorphic state", "[networks][evolvers]") {
    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);
    Game game(1, prisoners_dilemma());
    SyncNetwork network(2, 1.0, 0.0, complete_graph(10), game);
    VectorXui state(2);
    state << 10, 0;
    const VectorXui final_state = NetworkEvolver::evolve(100, state, network);
    CHECK(final_state == state);
}

TEST_CASE("GeneralPopulationEvolver::run returns states of the right size", "[networks][evolvers]") {
    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);
    Game game(1, prisoners_dilemma());
    AsyncNetwork network(2, 1.0, 0.01, complete_graph(10), game);
    egttools::FinitePopulations::evolvers::GeneralPopulationEvolver evolver(network);
    const auto trajectory = evolver.run(1100, 100);
    REQUIRE(trajectory.cols() == 2);
    CHECK(trajectory.rows() > 0);
    CHECK((trajectory.rowwise().sum().array() == 10).all());
}
