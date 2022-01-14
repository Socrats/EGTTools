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


def patched_windows_platform_init(self):
    import textwrap
    from skbuild.platform_specifics.windows import WindowsPlatform, CMakeVisualStudioCommandLineGenerator, \
        CMakeVisualStudioIDEGenerator

    super(WindowsPlatform, self).__init__()

    self._vs_help = textwrap.dedent(
        """Building Windows wheels for requires Microsoft 
        Visual Studio 2017 or 2019: https://visualstudio.microsoft.com/vs/""").strip()

    supported_vs_years = [("2019", "v141"), ("2017", "v141")]
    for vs_year, vs_toolset in supported_vs_years:
        self.default_generators.extend([
            CMakeVisualStudioCommandLineGenerator("Ninja", vs_year, vs_toolset),
            CMakeVisualStudioIDEGenerator(vs_year, vs_toolset),
            CMakeVisualStudioCommandLineGenerator("NMake Makefiles", vs_year, vs_toolset),
            CMakeVisualStudioCommandLineGenerator("NMake Makefiles JOM", vs_year, vs_toolset)
        ])


import skbuild.platform_specifics.windows

skbuild.platform_specifics.windows.WindowsPlatform.__init__ = patched_windows_platform_init


def find_version():
    with io.open(os.path.join(os.path.dirname(__file__), "src", "version.h"), encoding='utf8') as f:
        version_file = f.read()
    version_match = re.search(r"^#define EGTTOOLS_VERSION ([0-9a-z.]+)$", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    version=find_version(),
    packages=['egttools', 'egttools.analytical', 'egttools.plotting', 'egttools.games',
              'egttools.behaviors',
              'egttools.behaviors.CRD', 'egttools.behaviors.NormalForm', 'egttools.behaviors.NormalForm.TwoActions',
              'egttools.helpers',
              'egttools.distributions',
              'egttools.datastructures'
              ],
    package_dir={'egttools': "egttools",
                 'egttools.analytical': "egttools/analytical",
                 'egttools.plotting': "egttools/plotting", 'egttools.games': "egttools/games",
                 'egttools.behaviors': "egttools/behaviors", 'egttools.behaviors.CRD': "egttools/behaviors/CRD",
                 'egttools.behaviors.NormalForm': "egttools/behaviors/NormalForm",
                 'egttools.behaviors.NormalForm.TwoActions': "egttools/behaviors/NormalForm/TwoActions",
                 'egttools.helpers': "egttools/helpers",
                 'egttools.distributions': "egttools/distributions",
                 'egttools.datastructures': "egttools/datastructures"
                 },
    # py_modules={'utils'},
    cmake_args=shlex.split(os.environ.get('EGTTOOLS_EXTRA_CMAKE_ARGS', '')),
    cmake_install_dir="egttools",
)
