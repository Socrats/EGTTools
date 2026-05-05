/** Copyright (c) 2019-2021  Elias Fernandez
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

#include <egttools/finite_populations/PairwiseComparisonTransitionOperator.hpp>

#include <stdexcept>
#include <string>

namespace egttools::FinitePopulations {

    PairwiseComparisonTransitionOperator::PairwiseComparisonTransitionOperator(
        const size_t population_size,
        AbstractGame &game,
        const double beta,
        const double mu)
        : pop_size_(population_size),
          nb_strategies_(game.nb_strategies()),
          game_(&game),
          beta_(beta),
          mu_(mu) {

        if (population_size < 2)
            throw std::invalid_argument("population_size must be >= 2");
        if (game.nb_strategies() < 2)
            throw std::invalid_argument("game must have at least 2 strategies");
        if (beta < 0.0)
            throw std::invalid_argument("beta must be >= 0");
        if (mu < 0.0 || mu > 1.0)
            throw std::invalid_argument("mu must be in [0, 1]");

        nb_states_ = static_cast<int64_t>(
            egttools::starsBars(population_size, game.nb_strategies()));

        inv_Z_   = 1.0 / static_cast<double>(pop_size_);
        inv_Zm1_ = 1.0 / static_cast<double>(pop_size_ - 1);
        one_minus_mu_ = 1.0 - mu_;
        mutation_probability_ = (nb_strategies_ > 2)
                                    ? mu_ / static_cast<double>(nb_strategies_ - 1)
                                    : mu_;
    }

    int64_t PairwiseComparisonTransitionOperator::size() const {
        return nb_states_;
    }

    void PairwiseComparisonTransitionOperator::apply_transpose(
        const Eigen::Ref<const Vector> &x,
        Eigen::Ref<Vector> y) const {

        if (x.size() != nb_states_ || y.size() != nb_states_)
            throw std::invalid_argument(
                "apply_transpose: x and y must have length " +
                std::to_string(nb_states_));

        y.setZero();
        VectorXui current(static_cast<int64_t>(nb_strategies_));

        for (int64_t src = 0; src < nb_states_; ++src) {
            sample_simplex(static_cast<size_t>(src), pop_size_,
                           nb_strategies_, current);

            const double xi = x(src);
            const double total_offdiag = enumerate_transitions_(
                current, [&](const int64_t dest, const double prob) {
                    y(dest) += prob * xi;
                });

            y(src) += (1.0 - total_offdiag) * xi;
        }
    }

    void PairwiseComparisonTransitionOperator::apply(
        const Eigen::Ref<const Vector> &x,
        Eigen::Ref<Vector> y) const {

        if (x.size() != nb_states_ || y.size() != nb_states_)
            throw std::invalid_argument(
                "apply: x and y must have length " + std::to_string(nb_states_));

        y.setZero();
        VectorXui current(static_cast<int64_t>(nb_strategies_));

        for (int64_t src = 0; src < nb_states_; ++src) {
            sample_simplex(static_cast<size_t>(src), pop_size_,
                           nb_strategies_, current);

            double ys = 0.0;
            const double total_offdiag = enumerate_transitions_(
                current, [&](const int64_t dest, const double prob) {
                    ys += prob * x(dest);
                });

            y(src) = ys + (1.0 - total_offdiag) * x(src);
        }
    }

    void PairwiseComparisonTransitionOperator::apply_residual(
        const Eigen::Ref<const Vector> &x,
        Eigen::Ref<Vector> y) const {

        if (x.size() != nb_states_ || y.size() != nb_states_)
            throw std::invalid_argument(
                "apply_residual: x and y must have length " +
                std::to_string(nb_states_));

        apply_transpose(x, y);
        y = x - y;
    }

} // namespace egttools::FinitePopulations
