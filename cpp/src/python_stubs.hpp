/** Copyright (c) 2020-2021  Elias Fernandez
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
#pragma once
#ifndef EGTTOOLS_PYTHONSTUBS_HPP
#define EGTTOOLS_PYTHONSTUBS_HPP

#include <egttools/Types.h>

#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/behaviors/AbstractCRDStrategy.hpp>
#include <egttools/finite_populations/behaviors/AbstractNFGStrategy.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>
#include <egttools/finite_populations/games/AbstractNPlayerGame.hpp>
#include <egttools/finite_populations/games/AbstractSpatialGame.hpp>
#include <egttools/finite_populations/structure/AbstractNetworkStructure.hpp>
#include <egttools/finite_populations/structure/AbstractStructure.hpp>

namespace py = pybind11;

namespace stubs {

    using PayoffVector = egttools::FinitePopulations::PayoffVector;
    using GroupPayoffs = egttools::FinitePopulations::GroupPayoffs;

    class PyAbstractGame : public egttools::FinitePopulations::AbstractGame {
    public:
        /* Inherit the constructors */
        using egttools::FinitePopulations::AbstractGame::AbstractGame;

        /* Trampoline (need one for each virtual function) */
        void play(const egttools::FinitePopulations::StrategyCounts &group_composition,
                  PayoffVector &game_payoffs) override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    void,                                      /* Return type */
                    egttools::FinitePopulations::AbstractGame, /* Parent class */
                    play,                                      /* Name of function in C++ (must match Python name) */
                    group_composition, game_payoffs            /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        const GroupPayoffs &calculate_payoffs() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    GroupPayoffs &,                            /* Return type */
                    egttools::FinitePopulations::AbstractGame, /* Parent class */
                    calculate_payoffs                          /* Name of function in C++ (must match Python name) */
                                                               /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        double calculate_fitness(const int &strategy_index, const size_t &pop_size,
                                 const Eigen::Ref<const egttools::VectorXui> &strategies) override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    double,                                    /* Return type */
                    egttools::FinitePopulations::AbstractGame, /* Parent class */
                    calculate_fitness,                         /* Name of function in C++ (must match Python name) */
                    strategy_index, pop_size, strategies       /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        [[nodiscard]] size_t nb_strategies() const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    size_t,                                    /* Return type */
                    egttools::FinitePopulations::AbstractGame, /* Parent class */
                    nb_strategies                              /* Name of function in C++ (must match Python name) */
                                                               /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        [[nodiscard]] std::string toString() const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE_NAME(
                    std::string,                               /* Return type */
                    egttools::FinitePopulations::AbstractGame, /* Parent class */
                    "__str__",                                 /* Python name */
                    toString                                   /* Name of function in C++ (must match Python name) */
                                                               /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        [[nodiscard]] std::string type() const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    std::string,                               /* Return type */
                    egttools::FinitePopulations::AbstractGame, /* Parent class */
                    type                                       /* Name of function in C++ (must match Python name) */
                                                               /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        [[nodiscard]] const GroupPayoffs &payoffs() const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    GroupPayoffs &,                            /* Return type */
                    egttools::FinitePopulations::AbstractGame, /* Parent class */
                    payoffs                                    /* Name of function in C++ (must match Python name) */
                                                               /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        [[nodiscard]] double payoff(int strategy,
                                    const egttools::FinitePopulations::StrategyCounts &group_composition) const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    double,                                    /* Return type */
                    egttools::FinitePopulations::AbstractGame, /* Parent class */
                    payoff,                                    /* Name of function in C++ (must match Python name) */
                    strategy, group_composition                /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        void save_payoffs(std::string file_name) const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    void,                                      /* Return type */
                    egttools::FinitePopulations::AbstractGame, /* Parent class */
                    save_payoffs,                              /* Name of function in C++ (must match Python name) */
                    file_name                                  /* Argument(s) */
            );
        }
    };

    class PyAbstractNPlayerGame : public egttools::FinitePopulations::AbstractNPlayerGame {
    public:
        /* Inherit the constructors */
        using egttools::FinitePopulations::AbstractNPlayerGame::AbstractNPlayerGame;

        /* Trampoline (need one for each virtual function) */
        void play(const egttools::FinitePopulations::StrategyCounts &group_composition,
                  PayoffVector &game_payoffs) override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    void,                                             /* Return type */
                    egttools::FinitePopulations::AbstractNPlayerGame, /* Parent class */
                    play,                                             /* Name of function in C++ (must match Python name) */
                    group_composition, game_payoffs                   /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        const GroupPayoffs &calculate_payoffs() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    GroupPayoffs &,                                   /* Return type */
                    egttools::FinitePopulations::AbstractNPlayerGame, /* Parent class */
                    calculate_payoffs                                 /* Name of function in C++ (must match Python name) */
                                                                      /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        double calculate_fitness(const int &strategy_index, const size_t &pop_size,
                                 const Eigen::Ref<const egttools::VectorXui> &strategies) override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE(
                    double,                                           /* Return type */
                    egttools::FinitePopulations::AbstractNPlayerGame, /* Parent class */
                    calculate_fitness,                                /* Name of function in C++ (must match Python name) */
                    strategy_index, pop_size, strategies              /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        [[nodiscard]] size_t nb_strategies() const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE(
                    size_t,                                           /* Return type */
                    egttools::FinitePopulations::AbstractNPlayerGame, /* Parent class */
                    nb_strategies                                     /* Name of function in C++ (must match Python name) */
                                                                      /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        [[nodiscard]] size_t group_size() const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE(
                    size_t,                                           /* Return type */
                    egttools::FinitePopulations::AbstractNPlayerGame, /* Parent class */
                    group_size                                        /* Name of function in C++ (must match Python name) */
                                                                      /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        [[nodiscard]] int64_t nb_group_configurations() const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE(
                    size_t,                                           /* Return type */
                    egttools::FinitePopulations::AbstractNPlayerGame, /* Parent class */
                    nb_group_configurations                           /* Name of function in C++ (must match Python name) */
                                                                      /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        [[nodiscard]] std::string toString() const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_NAME(
                    std::string,                                      /* Return type */
                    egttools::FinitePopulations::AbstractNPlayerGame, /* Parent class */
                    "__str__",                                        /* Python name */
                    toString                                          /* Name of function in C++ (must match Python name) */
                                                                      /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        [[nodiscard]] std::string type() const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE(
                    std::string,                                      /* Return type */
                    egttools::FinitePopulations::AbstractNPlayerGame, /* Parent class */
                    type                                              /* Name of function in C++ (must match Python name) */
                                                                      /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        [[nodiscard]] const GroupPayoffs &payoffs() const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE(
                    GroupPayoffs &,                                   /* Return type */
                    egttools::FinitePopulations::AbstractNPlayerGame, /* Parent class */
                    payoffs                                           /* Name of function in C++ (must match Python name) */
                                                                      /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        [[nodiscard]] double payoff(int strategy,
                                    const egttools::FinitePopulations::StrategyCounts &group_composition) const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE(
                    double,                                           /* Return type */
                    egttools::FinitePopulations::AbstractNPlayerGame, /* Parent class */
                    payoff,                                           /* Name of function in C++ (must match Python name) */
                    strategy, group_composition                       /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        void update_payoff(int strategy_index, int group_configuration_index, double value) override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE(
                    void,                                             /* Return type */
                    egttools::FinitePopulations::AbstractNPlayerGame, /* Parent class */
                    update_payoff,                                    /* Name of function in C++ (must match Python name) */
                    strategy_index, group_configuration_index, value  /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        void save_payoffs(std::string file_name) const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE(
                    void,                                             /* Return type */
                    egttools::FinitePopulations::AbstractNPlayerGame, /* Parent class */
                    save_payoffs,                                     /* Name of function in C++ (must match Python name) */
                    file_name                                         /* Argument(s) */
            );
        }
    };

    class PyAbstractNFGStrategy : public egttools::FinitePopulations::behaviors::AbstractNFGStrategy {
    public:
        /* Inherit the constructors */
        using egttools::FinitePopulations::behaviors::AbstractNFGStrategy::AbstractNFGStrategy;

        /* Trampoline (need one for each virtual function) */
        int get_action(size_t time_step, int action_prev) override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    size_t,                                                      /* Return type */
                    egttools::FinitePopulations::behaviors::AbstractNFGStrategy, /* Parent class */
                    get_action,                                                  /* Name of function in C++ (must match Python name) */
                    time_step, action_prev                                       /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        std::string type() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    std::string,                                                 /* Return type */
                    egttools::FinitePopulations::behaviors::AbstractNFGStrategy, /* Parent class */
                    type                                                         /* Name of function in C++ (must match Python name) */
                                                                                 /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        bool is_stochastic() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    bool,                                                        /* Return type */
                    egttools::FinitePopulations::behaviors::AbstractNFGStrategy, /* Parent class */
                    is_stochastic                                                /* Name of function in C++ (must match Python name) */
                                                                                 /* Argument(s) */
            );
        }
    };

    class PyAbstractCRDStrategy : public egttools::FinitePopulations::behaviors::AbstractCRDStrategy {
    public:
        /* Inherit the constructors */
        using egttools::FinitePopulations::behaviors::AbstractCRDStrategy::AbstractCRDStrategy;

        /* Trampoline (need one for each virtual function) */
        int get_action(size_t time_step, int action_prev) override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    int,                                                         /* Return type */
                    egttools::FinitePopulations::behaviors::AbstractCRDStrategy, /* Parent class */
                    get_action,                                                  /* Name of function in C++ (must match Python name) */
                    time_step, action_prev                                       /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        std::string type() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    std::string,                                                 /* Return type */
                    egttools::FinitePopulations::behaviors::AbstractCRDStrategy, /* Parent class */
                    type                                                         /* Name of function in C++ (must match Python name) */
                                                                                 /* Argument(s) */
            );
        }
    };

    class PyAbstractSpatialGame : public egttools::FinitePopulations::games::AbstractSpatialGame {
        /* Inherit the constructors */
        using egttools::FinitePopulations::games::AbstractSpatialGame::AbstractSpatialGame;

        /* Trampoline (need one for each virtual function) */
        double calculate_fitness(int strategy_index, egttools::VectorXui &state) override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    double,                                                  /* Return type */
                    egttools::FinitePopulations::games::AbstractSpatialGame, /* Parent class */
                    calculate_fitness,                                       /* Name of function in C++ (must match Python name) */
                    strategy_index,                                          /* Argument(s) */
                    state);
        }

        /* Trampoline (need one for each virtual function) */
        [[nodiscard]] int nb_strategies() const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    int,                                                     /* Return type */
                    egttools::FinitePopulations::games::AbstractSpatialGame, /* Parent class */
                    nb_strategies                                            /* Name of function in C++ (must match Python name) */
                                                                             /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        [[nodiscard]] std::string toString() const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    std::string,                                             /* Return type */
                    egttools::FinitePopulations::games::AbstractSpatialGame, /* Parent class */
                    toString                                                 /* Name of function in C++ (must match Python name) */
                                                                             /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        [[nodiscard]] std::string type() const override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    std::string,                                             /* Return type */
                    egttools::FinitePopulations::games::AbstractSpatialGame, /* Parent class */
                    type                                                     /* Name of function in C++ (must match Python name) */
                                                                             /* Argument(s) */
            );
        }
    };

    class PyAbstractStructure : public egttools::FinitePopulations::structure::AbstractStructure {
    public:
        /* Inherit the constructors */
        using egttools::FinitePopulations::structure::AbstractStructure::AbstractStructure;

        /* Trampoline (need one for each virtual function) */
        void initialize() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    void,                                                      /* Return type */
                    egttools::FinitePopulations::structure::AbstractStructure, /* Parent class */
                    initialize,                                                /* Name of function in C++ (must match Python name) */
                                                                               /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        void update_population() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    void,                                                      /* Return type */
                    egttools::FinitePopulations::structure::AbstractStructure, /* Parent class */
                    update_population,                                         /* Name of function in C++ (must match Python name) */
                                                                               /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        egttools::VectorXui &mean_population_state() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    egttools::VectorXui &,                                     /* Return type */
                    egttools::FinitePopulations::structure::AbstractStructure, /* Parent class */
                    mean_population_state,                                     /* Name of function in C++ (must match Python name) */
                                                                               /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        int nb_strategies() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    int,                                                       /* Return type */
                    egttools::FinitePopulations::structure::AbstractStructure, /* Parent class */
                    nb_strategies,                                             /* Name of function in C++ (must match Python name) */
                                                                               /* Argument(s) */
            );
        }
    };

    class PyAbstractNetworkStructure : public egttools::FinitePopulations::structure::AbstractNetworkStructure {
    public:
        /* Inherit the constructors */
        using egttools::FinitePopulations::structure::AbstractNetworkStructure::AbstractNetworkStructure;

        /* Trampoline (need one for each virtual function) */
        void initialize() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    void,                                                             /* Return type */
                    egttools::FinitePopulations::structure::AbstractNetworkStructure, /* Parent class */
                    initialize,                                                       /* Name of function in C++ (must match Python name) */
                                                                                      /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        void initialize_state(egttools::VectorXui &state) override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    void,                                                             /* Return type */
                    egttools::FinitePopulations::structure::AbstractNetworkStructure, /* Parent class */
                    initialize_state,                                                 /* Name of function in C++ (must match Python name) */
                    state                                                             /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        void update_population() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    void,                                                             /* Return type */
                    egttools::FinitePopulations::structure::AbstractNetworkStructure, /* Parent class */
                    update_population,                                                /* Name of function in C++ (must match Python name) */
                                                                                      /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        void update_node(int node) override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    void,                                                             /* Return type */
                    egttools::FinitePopulations::structure::AbstractNetworkStructure, /* Parent class */
                    update_node,                                                      /* Name of function in C++ (must match Python name) */
                    node                                                              /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        egttools::Vector &calculate_average_gradient_of_selection() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    egttools::Vector &,                                               /* Return type */
                    egttools::FinitePopulations::structure::AbstractNetworkStructure, /* Parent class */
                    calculate_average_gradient_of_selection,                          /* Name of function in C++ (must match Python name) */
                                                                                      /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        egttools::Vector &calculate_average_gradient_of_selection_and_update_population() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    egttools::Vector &,                                               /* Return type */
                    egttools::FinitePopulations::structure::AbstractNetworkStructure, /* Parent class */
                    calculate_average_gradient_of_selection_and_update_population,    /* Name of function in C++ (must match Python name) */
                                                                                      /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        egttools::VectorXui &mean_population_state() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    egttools::VectorXui &,                                            /* Return type */
                    egttools::FinitePopulations::structure::AbstractNetworkStructure, /* Parent class */
                    mean_population_state,                                            /* Name of function in C++ (must match Python name) */
                                                                                      /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        int nb_strategies() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    int,                                                              /* Return type */
                    egttools::FinitePopulations::structure::AbstractNetworkStructure, /* Parent class */
                    nb_strategies,                                                    /* Name of function in C++ (must match Python name) */
                                                                                      /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        int population_size() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    int,                                                              /* Return type */
                    egttools::FinitePopulations::structure::AbstractNetworkStructure, /* Parent class */
                    population_size,                                                  /* Name of function in C++ (must match Python name) */
                                                                                      /* Argument(s) */
            );
        }

        /* Trampoline (need one for each virtual function) */
        egttools::FinitePopulations::structure::NodeDictionary &network() override {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERRIDE_PURE(
                    egttools::FinitePopulations::structure::NodeDictionary &,         /* Return type */
                    egttools::FinitePopulations::structure::AbstractNetworkStructure, /* Parent class */
                    network,                                                          /* Name of function in C++ (must match Python name) */
                                                                                      /* Argument(s) */
            );
        }
    };

}// namespace stubs

#endif//EGTTOOLS_PYTHONSTUBS_HPP