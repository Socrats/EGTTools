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
#include <vector>

#if HAS_ARPACK
#include <arpack/arpack.hpp>
#endif

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

    Vector PairwiseComparisonTransitionOperator::compute_stationary_distribution(
        double tol, size_t max_iter) const {

        Vector pi     = Vector::Constant(nb_states_, 1.0 / static_cast<double>(nb_states_));
        Vector pi_new(nb_states_);

        for (size_t iter = 0; iter < max_iter; ++iter) {
            apply_transpose(pi, pi_new);
            pi_new /= pi_new.sum();
            if ((pi_new - pi).cwiseAbs().sum() < tol)
                return pi_new;
            pi.swap(pi_new);
        }

        throw std::runtime_error(
            "compute_stationary_distribution: not converged after " +
            std::to_string(max_iter) + " iterations");
    }

#if HAS_ARPACK
    Vector PairwiseComparisonTransitionOperator::compute_stationary_arpack(
        double tol, int ncv, int max_iter) const {

        const int n   = static_cast<int>(nb_states_);
        const int nev = 1;

        // Krylov subspace size: at least 2*nev+1; rule of thumb max(2*nev+1, 20).
        if (ncv <= 0)
            ncv = std::min(std::max(2 * nev + 1, 20), n);
        else
            ncv = std::min(ncv, n);

        if (ncv <= nev)
            throw std::invalid_argument(
                "compute_stationary_arpack: ncv must be > nev");

        // Workspace sizes for non-symmetric real (dnaupd/dneupd).
        const int lworkl = 3 * ncv * ncv + 6 * ncv;

        std::vector<double> resid(n, 1.0 / n);
        std::vector<double> v(n * ncv, 0.0);
        std::vector<double> workd(3 * n, 0.0);
        std::vector<double> workl(lworkl, 0.0);
        std::vector<int>    iparam(11, 0);
        std::vector<int>    ipntr(14, 0);  // 14 for non-symmetric

        iparam[0] = 1;         // ishift: exact shifts
        iparam[2] = max_iter;  // maxitr
        iparam[6] = 1;         // mode 1: standard eigenproblem A*x = λ*x

        int ido = 0, info = 0;

        do {
            arpack::naupd(ido, arpack::bmat::identity, n,
                          arpack::which::largest_magnitude, nev,
                          tol, resid.data(), ncv, v.data(), n,
                          iparam.data(), ipntr.data(),
                          workd.data(), workl.data(), lworkl, info);

            if (ido == -1 || ido == 1) {
                // Apply P^T: y = P^T * x (entirely in C++, no Python roundtrip)
                Eigen::Map<const Vector> x(workd.data() + ipntr[0] - 1, n);
                Eigen::Map<Vector>       y(workd.data() + ipntr[1] - 1, n);
                apply_transpose(x, y);
            }
        } while (ido != 99);

        if (info < 0)
            throw std::runtime_error(
                "compute_stationary_arpack: dnaupd error code " +
                std::to_string(info));

        // Extract eigenvector: dneupd returns real + imaginary parts.
        std::vector<double> dr(nev + 1, 0.0), di(nev + 1, 0.0);
        std::vector<double> z(n * (nev + 1), 0.0);
        std::vector<double> workev(3 * ncv, 0.0);
        std::vector<int>    select(ncv, 0);

        int rvec = 1;
        arpack::neupd(rvec, arpack::howmny::ritz_vectors,
                      select.data(), dr.data(), di.data(),
                      z.data(), n, 0.0, 0.0, workev.data(),
                      arpack::bmat::identity, n,
                      arpack::which::largest_magnitude, nev,
                      tol, resid.data(), ncv, v.data(), n,
                      iparam.data(), ipntr.data(),
                      workd.data(), workl.data(), lworkl, info);

        if (info != 0)
            throw std::runtime_error(
                "compute_stationary_arpack: dneupd error code " +
                std::to_string(info));

        // The leading eigenvector (real part) is in z[0..n-1].
        Eigen::Map<Vector> pi(z.data(), n);
        pi = pi.cwiseAbs();
        pi /= pi.sum();
        return pi;
    }
#endif // HAS_ARPACK

} // namespace egttools::FinitePopulations
