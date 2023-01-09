# Copyright (c) 2019-2021  Elias Fernandez
#
# This file is part of EGTtools.
#
# EGTtools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EGTtools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EGTtools.  If not, see <http://www.gnu.org/licenses/>

"""
The code used in here has been adapted from https://github.com/YannickJadoul/Parselmouth/blob/master/setup.py
"""

import io
import os
import re
import shlex
import sys

try:
    from skbuild import setup
except ImportError:
    print("Please update pip to pip 10 or greater, or a manually install the PEP 518 requirements in pyproject.toml",
          file=sys.stderr)
    raise

from setuptools import find_packages


def find_version():
    with io.open(os.path.join(os.path.dirname(__file__), "cpp/src", "version.h"), encoding='utf8') as f:
        version_file = f.read()
    version_match = re.search(r"^#define EGTTOOLS_VERSION ([\da-z.]+)$", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    version=find_version(),
    # packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    packages=['egttools', 'egttools.numerical', 'egttools.numerical.structure', 'egttools.analytical',
              'egttools.plotting', 'egttools.games',
              'egttools.behaviors',
              'egttools.behaviors.CRD', 'egttools.behaviors.NormalForm', 'egttools.behaviors.NormalForm.TwoActions',
              'egttools.behaviors.CPR',
              'egttools.helpers',
              'egttools.distributions',
              'egttools.datastructures'
              ],
    package_dir={'egttools': "src/egttools",
                 'egttools.numerical': "src/egttools/numerical",
                 'egttools.numerical.structure': 'src/egttools/numerical/structure',
                 'egttools.analytical': "src/egttools/analytical",
                 'egttools.plotting': "src/egttools/plotting", 'egttools.games': "src/egttools/games",
                 'egttools.behaviors': "src/egttools/behaviors", 'egttools.behaviors.CRD': "src/egttools/behaviors/CRD",
                 'egttools.behaviors.NormalForm': "src/egttools/behaviors/NormalForm",
                 'egttools.behaviors.NormalForm.TwoActions': "src/egttools/behaviors/NormalForm/TwoActions",
                 'egttools.behaviors.CPR': "src/egttools/behaviors/CPR",
                 'egttools.helpers': "src/egttools/helpers",
                 'egttools.distributions': "src/egttools/distributions",
                 'egttools.datastructures': "src/egttools/datastructures"
                 },
    cmake_args=shlex.split(os.environ.get('EGTTOOLS_EXTRA_CMAKE_ARGS', '')),
    cmake_install_dir="src/egttools/numerical",
    cmake_with_sdist=True,
)
