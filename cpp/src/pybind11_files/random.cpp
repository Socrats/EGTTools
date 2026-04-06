/** Copyright (c) 2022-2026  Elias Fernandez
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

#include "random.hpp"

using namespace egttools;

void init_random(py::module_ &m) {
    py::module_ random_mod = m.def_submodule(
        "Random",
        "Utilities for random seed generation."
    );

    random_mod.def(
        "init",
        []() {
            Random::SeedGenerator::getInstance();
        },
        R"pbdoc(
Initialize the random seed generator using system entropy.

This initializes the singleton seed generator with a seed obtained from the
system random device.

See Also
--------
egttools.Random.init_with_seed
egttools.Random.seed
egttools.Random.current_seed
)pbdoc"
    );

    random_mod.def(
        "init_with_seed",
        [](const unsigned long int seed) {
            auto &instance = Random::SeedGenerator::getInstance();
            instance.setMainSeed(seed);
        },
        py::arg("seed"),
        R"pbdoc(
Initialize the random seed generator with a fixed seed.

This allows fully reproducible stochastic simulations.

Parameters
----------
seed : int
    Seed used to initialize the internal random number generator.

See Also
--------
egttools.Random.init
egttools.Random.seed
egttools.Random.current_seed
)pbdoc"
    );

    random_mod.def(
        "seed",
        [](unsigned long int seed) {
            Random::SeedGenerator::getInstance().setMainSeed(seed);
        },
        py::arg("seed"),
        R"pbdoc(
Reset the random seed generator with a new seed.

Parameters
----------
seed : int
    New seed value.

See Also
--------
egttools.Random.init
egttools.Random.init_with_seed
egttools.Random.current_seed
)pbdoc"
    );

    random_mod.def(
        "generate",
        []() {
            return Random::SeedGenerator::getInstance().getSeed();
        },
        R"pbdoc(
Generate a new pseudo-random seed.

Returns
-------
int
    Pseudo-random integer that can be used as a seed for other generators.

See Also
--------
egttools.Random.current_seed
)pbdoc"
    );

    random_mod.def(
        "current_seed",
        []() {
            return Random::SeedGenerator::getInstance().getMainSeed();
        },
        R"pbdoc(
Return the current main seed.

Returns
-------
int
    Seed currently used by the seed generator.

See Also
--------
egttools.Random.init
egttools.Random.init_with_seed
egttools.Random.seed
)pbdoc"
    );
}
