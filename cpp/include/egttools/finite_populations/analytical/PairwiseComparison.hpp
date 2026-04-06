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
#include <egttools/Types.h>

#include <egttools/utils/ThreadSafeLRUCache.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>
#include <tuple>

#if (HAS_BOOST)
#include <boost/multiprecision/cpp_dec_float.hpp>
#endif

#if defined(_OPENMP)
#include <egttools/OpenMPExtensions.hpp>
#endif

namespace egttools::FinitePopulations::analytical {
#if (HAS_BOOST)
    using cpp_dec_float_100 = boost::multiprecision::cpp_dec_float_100;
#endif
    using FitnessCacheKey = std::uint64_t;
    using Cache = egttools::Utils::ThreadSafeLRUCache<FitnessCacheKey, double>;

    /**
     * @brief Analytical tools for finite-population evolutionary dynamics under the pairwise comparison rule.
     *
     * This class studies a well-mixed population of fixed size @f$Z@f$, whose state is represented
     * by a vector of strategy counts
     * @f[
     * x = (x_1, \dots, x_n), \qquad \sum_{i=1}^n x_i = Z,
     * @f]
     * where @f$n@f$ is the number of strategies.
     *
     * Given a game defining the fitness of each strategy in each population state, this class provides
     * methods to:
     * - construct the full Markov transition matrix with mutation,
     * - compute gradients of selection with and without mutation,
     * - compute pairwise fixation probabilities,
     * - construct the reduced Small Mutation Limit (SML) Markov chain.
     *
     * Under the pairwise comparison rule, if an individual using strategy @f$i@f$ compares with an
     * individual using strategy @f$j@f$, the probability that @f$i@f$ imitates @f$j@f$ is
     * typically given by the Fermi kernel
     * @f[
     * p_{i \to j}(x)
     * =
     * \frac{1}{1 + \exp[-\beta (f_j(x) - f_i(x))]},
     * @f]
     * where @f$\beta \ge 0@f$ is the intensity of selection and @f$f_i(x)@f$ is the fitness of
     * strategy @f$i@f$ in state @f$x@f$.
     */
    class PairwiseComparison {
    public:
        /**
         * @brief Constructs a pairwise-comparison process for a fixed population size and game.
         *
         * The game must implement the fitness of each strategy as a function of the current population
         * state. The number of strategies is inferred from the game, and the number of population states
         * is determined by the stars-and-bars formula
         * @f[
         * |\mathcal{S}| = \binom{Z + n - 1}{n - 1},
         * @f]
         * where @f$Z@f$ is the population size and @f$n@f$ is the number of strategies.
         *
         * @note The game object is stored by reference. Updating the underlying game currently requires
         * constructing a new PairwiseComparison object.
         *
         * @param population_size Size @f$Z@f$ of the population.
         * @param game Game defining the fitness landscape over population states.
         */
        PairwiseComparison(int population_size, egttools::FinitePopulations::AbstractGame &game);

        /**
         * @brief Constructs a pairwise-comparison process with an explicit fitness-cache size.
         *
         * This overload is identical to the constructor above, but also allows controlling the size
         * of the internal LRU cache used to store previously computed fitness values.
         *
         * @param population_size Size @f$Z@f$ of the population.
         * @param game Game defining the fitness landscape over population states.
         * @param cache_size Maximum number of cached fitness evaluations.
         */
        PairwiseComparison(int population_size, egttools::FinitePopulations::AbstractGame &game, size_t cache_size);

        ~PairwiseComparison() = default;

        /**
         * @brief Pre-computes fitness values along all simplex edges.
         *
         * This method is useful when repeated pairwise fixation calculations are needed, since fixation
         * probabilities only depend on edge states involving two strategies at a time.
         */
        void pre_calculate_edge_fitnesses();

        /**
         * @brief Computes the full transition matrix of the finite-population Markov chain.
         *
         * The full chain evolves on the set of all population states
         * @f[
         * \mathcal{S} = \left\{x \in \mathbb{N}^n : \sum_{i=1}^n x_i = Z \right\}.
         * @f]
         * Each off-diagonal entry corresponds to a one-step transition in which one individual changes
         * strategy, so the destination state differs from the source state by @f$+1@f$ in one strategy
         * and @f$-1@f$ in another. The diagonal is then chosen so that each row sums to one.
         *
         * Mutation is incorporated directly in the transition probabilities. In particular, for a transition
         * in which strategy @f$j@f$ decreases by one individual and strategy @f$i@f$ increases by one
         * individual, the corresponding probability combines:
         * - imitation/selection, weighted by @f$(1-\mu)@f$,
         * - mutation, weighted by @f$\mu@f$.
         *
         * For large state spaces, explicitly storing this matrix may become prohibitively expensive in
         * memory. In such cases, dimensional reduction methods such as the Small Mutation Limit (SML)
         * are usually preferable.
         *
         * @param beta Intensity of selection @f$\beta@f$.
         * @param mu Mutation probability @f$\mu@f$.
         * @return Sparse transition matrix of size @f$|\mathcal{S}| \times |\mathcal{S}|@f$.
         */
        SparseMatrix2D calculate_transition_matrix(double beta, double mu);

        /**
         * @brief Computes the gradient of selection without mutation for a given population state.
         *
         * Let @f$x = (x_1,\dots,x_n)@f$ denote the current state. This method returns the expected
         * net one-step change in the strategy counts due only to selection, i.e. without mutation.
         *
         * For each strategy @f$i@f$, the returned quantity is
         * @f[
         * g_i(x)
         * =
         * \frac{1}{n}
         * \sum_{j \ne i}
         * \left[
         * T_{j \to i}^{\mathrm{sel}}(x) - T_{i \to j}^{\mathrm{sel}}(x)
         * \right],
         * @f]
         * where @f$T_{j \to i}^{\mathrm{sel}}(x)@f$ is the probability that one individual of strategy
         * @f$j@f$ is replaced by one individual of strategy @f$i@f$ under pure pairwise comparison.
         *
         * Under the Fermi rule, this local net flux can be written as
         * @f[
         * T_{j \to i}^{\mathrm{sel}}(x) - T_{i \to j}^{\mathrm{sel}}(x)
         * =
         * \frac{x_i x_j}{Z(Z-1)}
         * \tanh\!\left(\frac{\beta}{2}(f_i(x)-f_j(x))\right).
         * @f]
         *
         * The resulting vector is tangent to the simplex, i.e.
         * @f[
         * \sum_{i=1}^n g_i(x) = 0.
         * @f]
         *
         * @param beta Intensity of selection @f$\beta@f$.
         * @param state Population state @f$x@f$, given as strategy counts.
         * @return Vector of size @f$n@f$ containing the selection gradient at the given state.
         */
        Vector calculate_gradient_of_selection(double beta, const Eigen::Ref<const VectorXui> &state) const;

        /**
        * @brief Computes the gradient of selection with mutation for a given population state.
        *
        * This method returns the expected one-step drift of the strategy counts when both pairwise comparison
        * and mutation are active.
        *
        * Let @f$x = (x_1,\dots,x_n)@f$ be the current population state, with population size @f$Z@f$ and
        * @f$n@f$ strategies. The returned gradient is
        * @f[
        * g_i^{(\mu)}(x)
        * =
        * (1-\mu)\, g_i(x)
        * +
        * \frac{\mu_{\mathrm{eff}}}{nZ}\left(Z - n x_i\right),
        * @f]
        * where @f$g_i(x)@f$ is the mutation-free gradient returned by
        * `calculate_gradient_of_selection`, and @f$\mu_{\mathrm{eff}}@f$ is the effective mutation
        * probability towards one specific alternative strategy, as defined by
        * `effective_mutation_probability_`.
        *
        * The first term is the selection contribution, scaled by @f$(1-\mu)@f$, while the second term is
        * the mutation drift induced by uniform mutation towards the other strategies.
        *
        * As in the mutation-free case, the resulting vector is tangent to the simplex:
        * @f[
        * \sum_{i=1}^n g_i^{(\mu)}(x) = 0.
        * @f]
        *
        * @note This method is kept separate from `calculate_gradient_of_selection` to avoid introducing
        * additional branching or overloading overhead in a function that may be called repeatedly in tight loops.
        *
        * @param beta Intensity of selection @f$\beta@f$.
        * @param mu Mutation probability @f$\mu@f$.
        * @param state Population state @f$x@f$, given as strategy counts.
        * @return Vector of size @f$n@f$ containing the gradient with mutation at the given state.
        */
        Vector calculate_gradient_of_selection_with_mutation(double beta,
                                                             double mu,
                                                             const Eigen::Ref<const VectorXui> &state) const;

        /**
         * @brief Calculates the fixation probability of a mutant strategy in a resident population.
         *
         * This method considers the one-dimensional edge of the simplex involving only the invading
         * and resident strategies. It returns the probability that a single individual using the
         * invading strategy eventually takes over a population initially composed of residents.
         *
         * Formally, this is the probability that the birth-death chain on states
         * @f$k = 0, 1, \dots, Z@f$ reaches @f$k=Z@f$ before @f$k=0@f$, starting from @f$k=1@f$,
         * where @f$k@f$ is the number of invaders.
         *
         * @param index_invading_strategy Index of the invading strategy.
         * @param index_resident_strategy Index of the resident strategy.
         * @param beta Intensity of selection @f$\beta@f$.
         * @return Fixation probability of one invader in a resident population.
         */
        double calculate_fixation_probability(int index_invading_strategy, int index_resident_strategy, double beta);

        /**
         * @brief Computes the reduced Markov chain and fixation matrix under the Small Mutation Limit.
         *
         * In the Small Mutation Limit (SML), mutations are assumed sufficiently rare that the population
         * is almost always monomorphic before the next mutation occurs. The full dynamics can then be
         * approximated by a Markov chain on the @f$n@f$ monomorphic states only.
         *
         * If the current monomorphic population uses strategy @f$i@f$, the probability of transitioning
         * to monomorphic strategy @f$j@f$ is proportional to the fixation probability of one mutant
         * @f$j@f$ in a resident population of @f$i@f$:
         * @f[
         * T_{ij}^{\mathrm{SML}}
         * =
         * \frac{\rho_{ij}}{n-1},
         * \qquad i \ne j,
         * @f]
         * where @f$\rho_{ij}@f$ is the fixation probability of one @f$j@f$ mutant in a population of @f$i@f$.
         * The diagonal terms are set so that each row sums to one:
         * @f[
         * T_{ii}^{\mathrm{SML}} = 1 - \sum_{j \ne i} T_{ij}^{\mathrm{SML}}.
         * @f]
         *
         * The returned fixation matrix satisfies:
         * - `fixation_probabilities(i, j)` = probability that one mutant @f$j@f$ fixates in a population of @f$i@f$.
         *
         * @param beta Intensity of selection @f$\beta@f$.
         * @return Tuple containing:
         *         - the reduced SML transition matrix,
         *         - the matrix of pairwise fixation probabilities.
         */
        std::tuple<Matrix2D, Matrix2D> calculate_transition_and_fixation_matrix_sml(double beta);

        // setters

        /**
         * @brief Updates the population size and recomputes dependent dimensions.
         *
         * Changing the population size changes the state space size
         * @f$\binom{Z+n-1}{n-1}@f$, so any cached or precomputed quantities that depend on @f$Z@f$
         * should be considered specific to the new population size only.
         *
         * @param population_size New population size @f$Z@f$.
         */
        void update_population_size(int population_size);

        // getters

        /**
         * @brief Returns the number of strategies.
         *
         * @return Number of strategies @f$n@f$.
         */
        [[nodiscard]] int nb_strategies() const;

        /**
         * @brief Returns the number of population states.
         *
         * This is the cardinality of the simplex lattice
         * @f[
         * |\mathcal{S}| = \binom{Z+n-1}{n-1}.
         * @f]
         *
         * @return Number of states in the full Markov chain.
         */
        [[nodiscard]] int64_t nb_states() const;

        /**
         * @brief Returns the population size.
         *
         * @return Population size @f$Z@f$.
         */
        [[nodiscard]] int population_size() const;

        /**
         * @brief Returns the underlying game.
         *
         * @return Reference to the game used to evaluate fitnesses.
         */
        [[nodiscard]] const AbstractGame &game() const;

    private:
        int population_size_, nb_strategies_;
        size_t cache_size_;
        int64_t nb_states_;
        AbstractGame &game_;

        Cache cache_;

        /**
         * @brief Computes the local pairwise contribution to the mutation-free gradient.
         *
         * For a pair of distinct strategies, this method evaluates the antisymmetric local contribution
         * to the gradient associated with the transition
         * @f[
         * x \mapsto x + e_{\text{increasing}} - e_{\text{decreasing}},
         * @f]
         * where one individual of `decreasing_strategy` is replaced by one individual of
         * `increasing_strategy`.
         *
         * Under the Fermi rule, the corresponding local net contribution is
         * @f[
         * \frac{x_i x_j}{Z(Z-1)}
         * \tanh\!\left(\frac{\beta}{2}(f_i(x)-f_j(x))\right),
         * @f]
         * with @f$i =@f$ `increasing_strategy` and @f$j =@f$ `decreasing_strategy`.
         *
         * @param decreasing_strategy Index of the strategy that decreases by one individual.
         * @param increasing_strategy Index of the strategy that increases by one individual.
         * @param beta Intensity of selection @f$\beta@f$.
         * @param state Current population state.
         * @return Local mutation-free net flux from `decreasing_strategy` to `increasing_strategy`.
         */
        inline double calculate_local_gradient_(int decreasing_strategy, int increasing_strategy, double beta,
                                                VectorXui &state) const;

        /**
        * @brief Returns the effective mutation probability towards one specific alternative strategy.
        *
        * In the transition rule used by this class, mutation is first triggered with probability @f$\mu@f$.
        * Conditional on mutation, the offspring adopts one of the other @f$n-1@f$ strategies uniformly at random.
        * Therefore, the probability of mutating from a focal strategy into one specific alternative strategy is
        * @f[
        * \mu_{\mathrm{eff}} =
        * \begin{cases}
        * \mu, & n = 2, \\[4pt]
        * \mu / (n - 1), & n > 2,
        * \end{cases}
        * @f]
        * where @f$n@f$ is the number of strategies.
        *
        * This quantity is used in the mutation contribution to the drift/gradient.
        *
        * @param mu Mutation probability @f$\mu@f$.
        * @return Effective mutation probability towards one specific alternative strategy.
        */
        inline double effective_mutation_probability_(double mu) const;;

        /**
         * @brief Computes the fitness of one strategy in a given population state.
         *
         * This method delegates the actual fitness computation to the underlying game and may use the
         * internal LRU cache to avoid recomputing previously requested values.
         *
         * @param strategy_index Index of the focal strategy.
         * @param state Current population state.
         * @param state_index Integer index associated with the population state.
         * @return Fitness of the focal strategy in the given state.
         */
        inline double calculate_fitness_(int strategy_index,
                                         const VectorXui &state,
                                         int64_t state_index);
    };
} // namespace egttools::FinitePopulations::analytical

#endif//EGTTOOLS_FINITEPOPULATIONS_ANALYTICAL_PAIRWISECOMPARISON_HPP
