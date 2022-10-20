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

#include <egttools/SeedGenerator.h>

using namespace egttools::Random;

SeedGenerator::SeedGenerator() {
  std::random_device sys_rand;
  _rng_seed = sys_rand();
  _rng_engine.seed(_rng_seed);
}

SeedGenerator::SeedGenerator(unsigned long int seed) {
  _rng_seed = seed;
  _rng_engine.seed(seed);
}

SeedGenerator &SeedGenerator::getInstance() {
  static SeedGenerator _instance;
  return _instance;
}

unsigned long int SeedGenerator::getSeed() {
  // wrapping up the generator with uniform distribution helps guarantee a good quality for the seed
  std::uniform_int_distribution<unsigned long int> distribution(0, std::numeric_limits<unsigned>::max());
  return distribution(_rng_engine);
}
void SeedGenerator::setMainSeed(unsigned long seed) {
  _rng_seed = seed;
  _rng_engine.seed(_rng_seed);
}

std::mt19937_64 * egttools::Random::thread_local_generator() {
        static thread_local std::mt19937_64 *generator = nullptr;
        if (!generator) generator = new std::mt19937_64(SeedGenerator::getInstance().getSeed());
        return generator;
}
