//
// Created by Elias Fernandez on 17/05/2018.
//

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
