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


def transform_to_valid_windows_path(input_path):
    # Remove the initial backslash if it exists
    if input_path.startswith('\\'):
        input_path = input_path[1:]
    # Replace the first part (\d) with the corresponding drive letter (D:)
    if input_path[1] == '\\':  # Check if the second character is a backslash
        input_path = input_path[0] + ':' + input_path[1:]
    # Replace remaining backslashes with forward slashes or leave as is for valid Windows format
    valid_path = input_path.replace('\\', '\\')

    valid_path = os.path.normpath(valid_path)
    return valid_path


cmake_args = shlex.split(os.environ.get('EGTTOOLS_EXTRA_CMAKE_ARGS', ''))

IS_HPC = os.environ.get('HPC', 'OFF')

if IS_HPC == 'OFF':
    # Try to find the vcpkg path
    vcpkg_path = os.environ.get('VCPKG_PATH', '')

    if not vcpkg_path:
        # Assume vcpkg is in the project root if not explicitly set
        project_root = os.path.dirname(os.path.abspath(__file__))
        default_vcpkg_path = os.path.join(project_root, 'vcpkg')
        if os.path.exists(default_vcpkg_path):
            vcpkg_path = default_vcpkg_path
        else:
            print("Warning: VCPKG_PATH not set and no vcpkg folder found — CMake may fail", file=sys.stderr)

    vcpkg_toolchain_file = os.path.normpath(os.path.join(vcpkg_path, 'vcpkg', 'scripts',
                                                         'buildsystems', 'vcpkg.cmake'))
    if os.name == 'nt':
        vcpkg_toolchain_file = transform_to_valid_windows_path(vcpkg_toolchain_file)

    cmake_args.append(f'-DCMAKE_TOOLCHAIN_FILE={vcpkg_toolchain_file}')
else:
    cmake_args.append(f'-DHPC=ON')


def find_version():
    with io.open(os.path.join(os.path.dirname(__file__), "cpp/src", "version.h"), encoding='utf8') as f:
        version_file = f.read()
    version_match = re.search(r'#define EGTTOOLS_VERSION_STRING\s+"([\d.]+)"', version_file)
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
    cmake_args=cmake_args,
    cmake_install_dir="src/egttools/numerical",
    cmake_with_sdist=False,
    include_package_data=True,  # required to honor MANIFEST.in
    package_data={"egttools": ["*.pyi"]},  # or use a glob: ["**/*.pyi"]
)
