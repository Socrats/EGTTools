/** Copyright (c) 2022-2025  Elias Fernandez
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
    py::module_ random_mod = m.def_submodule("Random", "Module for random seed generation.");

    random_mod.def(
        "init", []() {
            // Initialize with random_device
            Random::SeedGenerator::getInstance();
        },
        R"pbdoc(
        Initialize the random seed generator using system entropy.

        This initializes the singleton seed generator with a seed from the system's random device.
        Should be called once at the beginning of your program for reproducible randomization.

        See Also
        --------
        egttools.Random.seed
    )pbdoc"
    );

    random_mod.def(
        "init_with_seed", [](const unsigned long int seed) {
            auto &instance = Random::SeedGenerator::getInstance();
            instance.setMainSeed(seed);
        },
        py::arg("seed"),
        R"pbdoc(
        Initialize the random seed generator with a fixed seed.

        This initializes the singleton seed generator using a user-specified integer seed.
        This enables full reproducibility when running stochastic simulations.

        Parameters
        ----------
        seed : int
            Seed for initializing the internal random number generator.

        See Also
        --------
        egttools.Random.seed
    )pbdoc"
    );

    random_mod.def(
        "seed", [](unsigned long int seed) {
            Random::SeedGenerator::getInstance().setMainSeed(seed);
        },
        py::arg("seed"),
        R"pbdoc(
        Reset the random seed generator with a new seed.

        This resets the internal random number generator of the singleton instance
        using the given seed. Useful to change the outcome of the stochastic processes.

        Parameters
        ----------
        seed : int
            New seed value.
    )pbdoc"
    );

    random_mod.def(
        "generate", []() {
            return Random::SeedGenerator::getInstance().getSeed();
        },
        R"pbdoc(
        Generate a new random seed.

        Returns a pseudo-random integer based on the internal state of the generator.
        Can be used to seed other generators in a reproducible way.

        Returns
        -------
        int
            A pseudo-random integer usable as a seed.
    )pbdoc"
    );

    random_mod.def(
        "current_seed", []() {
            return Random::SeedGenerator::getInstance().getMainSeed();
        },
        R"pbdoc(
        Get the seed currently in use.

        Returns the main seed used to initialize the random number generator.

        Returns
        -------
        int
            The current seed value.
    )pbdoc"
    );
}
