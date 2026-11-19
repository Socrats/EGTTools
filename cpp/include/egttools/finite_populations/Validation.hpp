/** Copyright (c) 2019-2026  Elias Fernandez
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
#ifndef EGTTOOLS_FINITEPOPULATIONS_VALIDATION_HPP
#define EGTTOOLS_FINITEPOPULATIONS_VALIDATION_HPP

/**
 * @file Validation.hpp
 * @brief Checks for arguments that users pass from Python.
 *
 * Rules (roadmap MEM-09):
 *  - Only methods that users call directly validate their arguments. Kernels that the library
 *    calls in inner loops (e.g. a game's calculate_fitness) document their preconditions instead,
 *    and their Python bindings validate, so internal calls pay nothing.
 *  - An index outside [0, S) throws std::out_of_range (Python IndexError); any other invalid
 *    argument throws std::invalid_argument (Python ValueError). Messages name the argument and the
 *    valid range.
 */

#include <cstdint>
#include <stdexcept>
#include <string>

#include <egttools/Types.h>

namespace egttools::FinitePopulations {
    /**
     * @brief Checks that `index` is a strategy index in [0, nb_strategies).
     * @throws std::out_of_range otherwise.
     */
    inline void validate_strategy_index(const int64_t index, const int64_t nb_strategies, const char *name) {
        if (index < 0 || index >= nb_strategies)
            throw std::out_of_range(std::string(name) + " must be a strategy index in [0, " +
                                    std::to_string(nb_strategies) + "), got " + std::to_string(index));
    }

    /**
     * @brief Checks the strategy pair of a fixation computation: both indices valid and different.
     * @throws std::out_of_range for an invalid index, std::invalid_argument if they are equal.
     */
    inline void validate_invader_resident(const int64_t invader, const int64_t resident,
                                          const int64_t nb_strategies) {
        validate_strategy_index(invader, nb_strategies, "the invading strategy index");
        validate_strategy_index(resident, nb_strategies, "the resident strategy index");
        if (invader == resident)
            throw std::invalid_argument("the invading and resident strategies must be different, got " +
                                        std::to_string(invader) + " for both");
    }

    /**
     * @brief Checks a population state: one count per strategy, each between 0 and Z, summing to Z.
     *
     * Each entry is bounded before summing, so values that wrap around in unsigned arithmetic
     * (e.g. a negative number converted to an unsigned count) cannot add up to Z.
     *
     * @throws std::invalid_argument otherwise.
     */
    inline void validate_population_state(const Eigen::Ref<const VectorXui> &state, const int64_t nb_strategies,
                                          const int64_t population_size, const char *name = "state") {
        if (state.size() != nb_strategies)
            throw std::invalid_argument(std::string(name) + " must have one entry per strategy (" +
                                        std::to_string(nb_strategies) + "), got " + std::to_string(state.size()));
        const auto z = static_cast<std::uint64_t>(population_size);
        std::uint64_t total = 0;
        for (Eigen::Index i = 0; i < state.size(); ++i) {
            const auto count = static_cast<std::uint64_t>(state(i));
            if (count > z)
                throw std::invalid_argument(std::string(name) + " entries must be between 0 and the population size " +
                                            std::to_string(population_size) + ", got " + std::to_string(count) +
                                            " at index " + std::to_string(i));
            total += count;
        }
        if (total != z)
            throw std::invalid_argument(std::string(name) + " entries must sum to the population size " +
                                        std::to_string(population_size) + ", got " + std::to_string(total));
    }
}// namespace egttools::FinitePopulations

#endif//EGTTOOLS_FINITEPOPULATIONS_VALIDATION_HPP
