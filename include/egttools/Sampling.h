//
// Created by Elias Fernandez on 21/01/2020.
//

#ifndef EGTTOOLS_INCLUDE_SAMPLING_H_
#define EGTTOOLS_INCLUDE_SAMPLING_H_

#include <unordered_set>

namespace egttools { namespace sampling {

/**
 * @brief Samples a set of \p k elements from a set of \p N elements without replacement.
 *
 * There is no type checking in this template! However you should only use either int, size_t,
 * unsigned (and other non-float numeric types). It is not implemented/tested for arrays
 * and other data/class structures.
 *
 * DISCLAIMER: This function has been obtained from a Stack overflow post and is based on the algorithm
 * by Robert Floyd that can be found at:
 * https://www.nowherenearithaca.com/2013/05/robert-floyds-tiny-and-beautiful.html.
 *
 * @tparam T : type of the set elements
 * @tparam G : type of the random generator
 * @param N : size of the set to sample from
 * @param k : size of the sample to generate
 * @param gen : random generator
 * @return : an unordered set containing the sample of size \p k.
 */
template<typename T=int, typename G>
std::unordered_set<T> sample_without_replacement(int N, int k, G &gen) {
  std::unordered_set<T> elements;
  for (int r = N - k; r < N; ++r) {
    auto v = std::uniform_int_distribution<T>(1, r)(gen);

    // there are two cases.
    // v is not in candidates ===> add it
    // v is in candidates ===> well, r is definitely not,
    // because this is the first interaction in the loop
    // that we could've picked something that big.

    if (!elements.insert(v).second) {
      elements.insert(r);
    }
    return elements;
  }
}

/**
 * @brief Samples a set of \p k elements from a set of \p N elements without replacement.
 *
 * This version of the method requires an unordered_set passed by reference. This can considerably
 * speed up the execution in some cases.
 *
 * There is no type checking in this template! However you should only use either int, size_t,
 * unsigned, float or double (and other numeric types). It is not implemented/tested for arrays
 * and other data/class structures.
 *
 * DISCLAIMER: This function has been obtained from a Stack overflow post and is based on the algorithm
 * by Robert Floyd that can be found at:
 * https://www.nowherenearithaca.com/2013/05/robert-floyds-tiny-and-beautiful.html.
 *
 * @tparam T : type of the set elements
 * @tparam G : type of the random generator
 * @param N : size of the set to sample from
 * @param k : size of the sample to generate
 * @param container : reference to unordered_set container to store the sample
 * @param gen : random generator
 * @return : an unordered set containing the sample of size \p k.
 */
template<typename T=int, typename G>
void sample_without_replacement(int N, int k, std::unordered_set<T> &container, G &gen) {
  for (int r = N - k; r < N; ++r) {
    auto v = std::uniform_int_distribution<T>(0, r)(gen);

    // there are two cases.
    // v is not in candidates ===> add it
    // v is in candidates ===> well, r is definitely not,
    // because this is the first interaction in the loop
    // that we could've picked something that big.

    if (!container.insert(v).second) {
      container.insert(r);
    }
  }
}

/**
 * @brief Samples a set of \p k elements from a set of \p N elements without replacement.
 *
 * This function returns a vector containing the sample.
 *
 * There is no type checking in this template! However you should only use either int, size_t,
 * unsigned (and other non-float numeric types). It is not implemented/tested for arrays
 * and other data/class structures.
 *
 * DISCLAIMER: This function has been obtained from a Stack overflow post and is based on the algorithm
 * by Robert Floyd that can be found at:
 * https://www.nowherenearithaca.com/2013/05/robert-floyds-tiny-and-beautiful.html.
 *
 * @tparam T : type of the set elements
 * @tparam G : type of the random generator
 * @param N : size of the set to sample from
 * @param k : size of the sample to generate
 * @param gen : random generator
 * @return : an unordered set containing the sample of size \p k.
 */
template<typename T=int, typename G>
std::vector<T> sample_without_replacement(T N, T k, G &gen) {
  std::unordered_set<T> elements = sample_without_replacement(N, k, gen);

  // Now we need to shuffle the set and store it in a vector in order
  // to obtain a true random set
  std::vector<T> result(elements.begin(), elements.end());
  std::shuffle(result.begin(), result.end(), gen);
  return result;
}

/**
 * @brief Samples a set of \p k elements from a set of \p N elements without replacement.
 *
 * This version of the method requires an unordered_set passed by reference. This can considerably
 * speed up the execution in some cases. It also requires a vector passed by reference, where
 * the final sample will be store. It assumes that the passed vector has the same size as the
 * unordered set (\p k)!
 *
 * There is no type checking in this template! However you should only use either int, size_t,
 * unsigned, float or double (and other numeric types). It is not implemented/tested for arrays
 * and other data/class structures.
 *
 * DISCLAIMER: This function has been obtained from a Stack overflow post and is based on the algorithm
 * by Robert Floyd that can be found at:
 * https://www.nowherenearithaca.com/2013/05/robert-floyds-tiny-and-beautiful.html.
 *
 * @tparam T : type of the set elements
 * @tparam G : type of the random generator
 * @param N : size of the set to sample from
 * @param k : size of the sample to generate
 * @param container : reference to unordered_set container to store the sample
 * @param gen : random generator
 * @return : an unordered set containing the sample of size \p k.
 */
template<typename T=int, typename G>
void sample_without_replacement(T N, T k, std::vector<T> &sample_container, std::unordered_set<T> &container, G &gen) {
  sample_without_replacement(N, k, container, gen);

  // Now we need to shuffle the set and store it in a vector in order
  // to obtain a true random set
  int i = 0;
  for (auto& it: container) {
    sample_container[i] = it.second;
  }
  std::shuffle(sample_container.begin(), sample_container.end(), gen);
}

/**
 * @brief Samples a set of \p k elements from a set of \p N elements without replacement.
 *
 * This version of the method requires an unordered_set passed by reference. This can considerably
 * speed up the execution in some cases. It also requires a vector passed by reference, where
 * the final sample will be store. It assumes that the passed vector has the same size as the
 * unordered set (\p k)!
 *
 * There is no type checking in this template! However you should only use either int, size_t,
 * unsigned, float or double (and other numeric types). It is not implemented/tested for arrays
 * and other data/class structures.
 *
 * DISCLAIMER: This function has been obtained from a Stack overflow post and is based on the algorithm
 * by Robert Floyd that can be found at:
 * https://www.nowherenearithaca.com/2013/05/robert-floyds-tiny-and-beautiful.html.
 *
 * @tparam T : type of the set elements
 * @tparam C : container for the final sample (std::vector, egttools::RL::PopContainer, ...)
 * @tparam G : type of the random generator
 * @param N : size of the set to sample from
 * @param k : size of the sample to generate
 * @param container : reference to unordered_set container to store the sample
 * @param gen : random generator
 * @return : an unordered set containing the sample of size \p k.
 */
template<typename T=int, typename C, typename G>
void sample_without_replacement(T N, T k, C &sample_container, std::unordered_set<T> &container, G &gen) {
  sample_without_replacement(N, k, container, gen);

  // Now we need to shuffle the set and store it in a vector in order
  // to obtain a true random set
  int i = 0;
  for (auto& it: container) {
    sample_container[i] = it.second;
  }
  std::shuffle(sample_container.begin(), sample_container.end(), gen);
}

} }

#endif //DYRWIN_INCLUDE_SAMPLING_H_
