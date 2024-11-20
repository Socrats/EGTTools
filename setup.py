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
import subprocess
import sys
import shutil

try:
    from skbuild import setup
    from setuptools import find_packages
    from setuptools.command.build import build as _build
except ImportError:
    print("Please update pip to pip 10 or greater, or a manually install the PEP 518 requirements in pyproject.toml",
          file=sys.stderr)
    raise


def find_folder_recursively(start_path, folder_name):
    """
    Recursively search for a folder with the specified name starting from the given path.

    :param start_path: The root directory to start the search.
    :param folder_name: The name of the folder to find.
    :return: A list of full paths to matching folders.
    """
    matching_folders = []
    for root, dirs, files in os.walk(start_path):
        if folder_name in dirs:
            matching_folders.append(os.path.join(root, folder_name))
    return matching_folders


# class sdist(_sdist):
    # def run(self):
    #     if not os.path.exists("vcpkg"):
    #         print("Cloning vcpkg...")
    #         subprocess.check_call(["git", "clone", "https://github.com/microsoft/vcpkg.git"])
    #         subprocess.check_call(["./vcpkg/bootstrap-vcpkg.sh"])
    #         print("vcpkg is ready.")
    #     print("Installing vcpkg dependencies...")
    #     subprocess.check_call(["./vcpkg/vcpkg", "install"])
    #
    #     # # Path to vcpkg installed libraries
    #     # vcpkg_lib_path = os.path.join('vcpkg_installed')
    #     # package_lib_path = os.path.join('external')
    #     #
    #     # found_folders = find_folder_recursively(os.getcwd(), "vcpkg_installed")
    #     #
    #     # if type(found_folders) is not list or len(found_folders) == 0:
    #     #     raise FileNotFoundError("Could not find the vcpkg_installed folder. Make sure you have run the install_vcpkg.sh script.")
    #     # elif len(found_folders) > 1:
    #     #     # Ensure the package lib directory exists
    #     #     os.makedirs(package_lib_path, exist_ok=True)
    #     #
    #     #     # Copy vcpkg libraries to the package directory
    #     #     if os.path.isdir(found_folders[0]):
    #     #         # Recursively copy subdirectories
    #     #         shutil.copytree(found_folders[0], package_lib_path, dirs_exist_ok=True)
    #     #     else:
    #     #         # Copy files
    #     #         shutil.copy2(found_folders[0], package_lib_path)
    #
    #     # Run the original sdist command
    #     _sdist.run(self)

class build(_build):
    def run(self):
        if not os.path.exists("vcpkg"):
            print("Cloning vcpkg...")
            subprocess.check_call(["git", "clone", "https://github.com/microsoft/vcpkg.git"])
            subprocess.check_call(["./vcpkg/bootstrap-vcpkg.sh"])
            print("vcpkg is ready.")
        print("Installing vcpkg dependencies...")
        subprocess.check_call(["./vcpkg/vcpkg", "install"])

        # Run the original build_ext command
        _build.run(self)

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
    cmake_args=shlex.split(
        os.environ.get('EGTTOOLS_EXTRA_CMAKE_ARGS', '')),
    cmake_install_dir="src/egttools/numerical",
    cmake_with_sdist=False,
    cmdclass={'build': build},
)
