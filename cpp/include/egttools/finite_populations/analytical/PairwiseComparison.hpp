/** Copyright (c) 2019-2022  Elias Fernandez
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
#ifndef EGTTOOLS_FINITEPOPULATIONS_ANALYTICAL_PAIRWISECOMPARISON_HPP
#define EGTTOOLS_FINITEPOPULATIONS_ANALYTICAL_PAIRWISECOMPARISON_HPP

#include <egttools/Distributions.h>
#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>

#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>
#include <stdexcept>

#if defined(_OPENMP)
#include <egttools/OpenMPUtils.hpp>
#endif

namespace egttools::FinitePopulations::analytical {
    /**
     * @brief Provides analytical methods to study evolutionary dynamics in finite populations
     * with the Pairwise Comparison rule.
     */
    class PairwiseComparison {
    public:
        PairwiseComparison(int population_size, egttools::FinitePopulations::AbstractGame &game);

        ~PairwiseComparison() = default;

        SparseMatrix2D calculate_transition_matrix(double beta, double mu);

//        double calculate_gradient_of_selection(double beta, VectorXui state);

        // setters
        void update_population_size(int population_size);
        void update_game(egttools::FinitePopulations::AbstractGame &game);

        // getters
        [[nodiscard]] int nb_strategies() const;
        [[nodiscard]] int64_t nb_states() const;
        [[nodiscard]] int population_size() const;
        [[nodiscard]] const egttools::FinitePopulations::AbstractGame &game() const;

    private:
        int population_size_, nb_strategies_;
        int64_t nb_states_;
        egttools::FinitePopulations::AbstractGame &game_;
    };
}// namespace egttools::FinitePopulations::analytical

#endif//EGTTOOLS_FINITEPOPULATIONS_ANALYTICAL_PAIRWISECOMPARISON_HPP
