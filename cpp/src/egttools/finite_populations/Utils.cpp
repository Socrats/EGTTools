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

#include <egttools/finite_populations/Utils.hpp>

double egttools::FinitePopulations::fermi(double beta, double a, double b) {
    return 1 / (1 + std::exp(beta * (a - b)));
}

double egttools::FinitePopulations::contest_success(double z, double a, double b) {
    double tmp = 1 / z;
    double tmp1 = std::pow(a, tmp);
    return tmp1 / (tmp1 + std::pow(b, tmp));
}

double egttools::FinitePopulations::contest_success(double a, double b) {
    if (a > b) return 1.0;
    else return 0.0;
}

size_t egttools::FinitePopulations::calculate_state(const size_t &group_size, const egttools::Factors &current_group) {
    size_t retval = 0;
    auto remaining = group_size;

    // In order to find the index for the input combination, we are basically
    // counting the number of combinations we have 'behind us", and we're going
    // to be the next. So for example if we have 10 combinations behind us,
    // we're going to be number 11.
    //
    // We do this recursively, element by element. For each element we count
    // the number of combinations we left behind. If data[i] is the highest
    // possible (i.e. it accounts for all remaining points), then it is the
    // first and we didn't miss anything.
    //
    // Otherwise we count how many combinations we'd have had with the max (1),
    // add it to the number, and decrease the h. Then we try again: are we
    // still lower? If so, count again how many combinations we'd have had with
    // this number (size() - 1). And so on, until we match the number we have.
    //
    // Then we go to the next element, considering the subarray of one element
    // less (thus the size() - i), and we keep going.
    //
    // Note that by using this algorithm the last element in the array is never
    // needed (since it is determined by the others), and additionally when we
    // have no remaining elements to parse we can just break.
    for (size_t i = 0; i < current_group.size() - 1; ++i) {
        auto h = remaining;
        while (h > current_group[i]) {
            retval += egttools::starsBars(remaining - h, current_group.size() - i - 1);
            --h;
        }
        if (remaining == current_group[i])
            break;
        remaining -= current_group[i];
    }

    return retval;
}

size_t
egttools::FinitePopulations::calculate_state(const size_t &group_size,
                                             const Eigen::Ref<const egttools::VectorXui> &current_group) {
    size_t retval = 0;
    auto remaining = group_size;

    // In order to find the index for the input combination, we are basically
    // counting the number of combinations we have 'behind us", and we're going
    // to be the next. So for example if we have 10 combinations behind us,
    // we're going to be number 11.
    //
    // We do this recursively, element by element. For each element we count
    // the number of combinations we left behind. If data[i] is the highest
    // possible (i.e. it accounts for all remaining points), then it is the
    // first and we didn't miss anything.
    //
    // Otherwise we count how many combinations we'd have had with the max (1),
    // add it to the number, and decrease the h. Then we try again: are we
    // still lower? If so, count again how many combinations we'd have had with
    // this number (size() - 1). And so on, until we match the number we have.
    //
    // Then we go to the next element, considering the subarray of one element
    // less (thus the size() - i), and we keep going.
    //
    // Note that by using this algorithm the last element in the array is never
    // needed (since it is determined by the others), and additionally when we
    // have no remaining elements to parse we can just break.
    for (long int i = 0; i < current_group.size() - 1; ++i) {
        auto h = remaining;
        while (h > current_group(i)) {
            retval += egttools::starsBars<size_t>(remaining - h, current_group.size() - i - 1);
            --h;
        }
        if (remaining == current_group(i))
            break;
        remaining -= current_group(i);
    }

    return retval;
}

egttools::VectorXui
egttools::FinitePopulations::sample_simplex(size_t i, const size_t &pop_size, const size_t &nb_strategies) {
    // To be able to infer the multidimensional state from the index
    // we apply a recursive algorithm that will complete a vector of size
    // nb_strategies from right to left

    egttools::VectorXui state = egttools::VectorXui::Zero(static_cast<int64_t>(nb_strategies));
    auto remaining = pop_size;

    for (signed long a = 0; a < static_cast<int64_t>(nb_strategies); ++a) {
        // reset the state container
        for (size_t j = remaining; j > 0; --j) {
            auto count = egttools::starsBars<size_t>(remaining - j, nb_strategies - a - 1);
            if (i >= count) {
                i -= count;
            } else {
                state(a) = j;
                remaining -= j;
                break;
            }
        }
    }
    return state;
}

void egttools::FinitePopulations::sample_simplex(size_t i, const size_t &pop_size, const size_t &nb_strategies,
                                                 egttools::VectorXui &state) {
    // To be able to infer the multidimensional state from the index
    // we apply a recursive algorithm that will complete a vector of size
    // nb_strategies from right to left

    auto remaining = pop_size;

    for (signed long a = 0; a < static_cast<signed long>(nb_strategies); ++a) {
        // reset the state container
        state(a) = 0;
        for (size_t j = remaining; j > 0; --j) {
            auto count = egttools::starsBars<size_t>(remaining - j, nb_strategies - a - 1);
            if (i >= count) {
                i -= count;
            } else {
                state(a) = j;
                remaining -= j;
                break;
            }
        }
    }
}

void egttools::FinitePopulations::sample_simplex(size_t i, const size_t &pop_size, const size_t &nb_strategies,
                                                 std::vector<size_t> &state) {
    // To be able to infer the multidimensional state from the index
    // we apply a recursive algorithm that will complete a vector of size
    // nb_strategies from right to left

    auto remaining = pop_size;

    for (size_t a = 0; a < nb_strategies; ++a) {
        // reset the state container
        state[a] = 0;
        for (size_t j = remaining; j > 0; --j) {
            auto count = egttools::starsBars(remaining - j, nb_strategies - a - 1);
            if (i >= count) {
                i -= count;
            } else {
                state[a] = j;
                remaining -= j;
                break;
            }
        }
    }
}
