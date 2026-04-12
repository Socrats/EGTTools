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

import os
import shlex
import shutil
import sys

def _version():
    # conda-build (and other sdist builds without a .git) set EGTTOOLS_VERSION
    # from the recipe.  Honour it so that the installed Python metadata matches
    # the conda package version even when git is unavailable.
    env_version = os.environ.get("EGTTOOLS_VERSION", "").strip()
    if env_version:
        return env_version
    try:
        from setuptools_scm import get_version as _get_scm_version
        return _get_scm_version(
            root=".",
            relative_to=__file__,
            local_scheme="no-local-version",
            fallback_version="0.0.0",
            write_to="src/egttools/_version.py",
        )
    except Exception:
        return "0.0.0"

try:
    from skbuild import setup
    from skbuild.constants import CMAKE_INSTALL_DIR as _CMAKE_INSTALL_DIR
except ImportError:
    print("Please update pip to pip 10 or greater, or a manually install the PEP 518 requirements in pyproject.toml",
          file=sys.stderr)
    raise


def _sync_python_sources_to_staging() -> None:
    """
    Propagate hand-edited Python source files into the scikit-build staging tree
    before scikit-build's developer-mode copy runs.

    Background
    ----------
    When ``setup.py build_ext --inplace`` is invoked, scikit-build detects
    developer mode (``build_ext_inplace=True``) and copies every Python file
    listed in ``package_data`` FROM ``_skbuild/<plat>/cmake-install/`` BACK TO
    ``src/``.  This copy happens inside ``setup()`` — *before* CMake runs —
    so any edits made to ``src/*.py`` after the last build are silently
    overwritten by the stale staging-tree versions.

    Fix: scan ``src/egttools/`` for ``.py`` files that are *newer* than their
    counterpart in the cmake-install staging tree and copy them there first.
    scikit-build then propagates the fresh versions back to ``src/``, keeping
    everything consistent.

    This function is intentionally a no-op when the staging tree does not
    exist yet (clean first-time builds).
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.join(project_root, "src", "egttools")
    staging_root = os.path.join(_CMAKE_INSTALL_DIR(), "src", "egttools")

    if not os.path.isdir(staging_root):
        return  # Clean build — staging tree doesn't exist yet; nothing to sync.

    for dirpath, _dirnames, filenames in os.walk(src_root):
        rel_dir = os.path.relpath(dirpath, src_root)
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            src_file = os.path.join(dirpath, fname)
            dst_file = os.path.join(staging_root, rel_dir, fname)
            if os.path.exists(dst_file) and os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)


_sync_python_sources_to_staging()


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

SKIP_VCPKG = os.environ.get('SKIP_VCPKG', 'OFF')

if SKIP_VCPKG == 'OFF':
    # Try to find the vcpkg path
    vcpkg_path = os.environ.get('VCPKG_PATH', '')

    if not vcpkg_path:
        # Assume vcpkg is in the project root if not explicitly set.
        # VCPKG_PATH must be the project root (not the vcpkg subdir), because
        # the toolchain file is constructed as vcpkg_path/vcpkg/scripts/buildsystems/vcpkg.cmake.
        project_root = os.path.dirname(os.path.abspath(__file__))
        default_vcpkg_path = os.path.join(project_root, 'vcpkg')
        if os.path.exists(default_vcpkg_path):
            vcpkg_path = project_root
        else:
            print("Warning: VCPKG_PATH not set and no vcpkg folder found — CMake may fail", file=sys.stderr)

    vcpkg_toolchain_file = os.path.normpath(os.path.join(vcpkg_path, 'vcpkg', 'scripts',
                                                         'buildsystems', 'vcpkg.cmake'))
    if os.name == 'nt':
        vcpkg_toolchain_file = transform_to_valid_windows_path(vcpkg_toolchain_file)

    cmake_args.append(f'-DCMAKE_TOOLCHAIN_FILE={vcpkg_toolchain_file}')
else:
    cmake_args.append(f'-DSKIP_VCPKG=ON')


setup(
    version=_version(),
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
    package_data={
        "egttools": ["*.pyi", "py.typed"],
        "egttools.numerical": ["*.so", "*.dylib", "*.pyd", "*.pyi", "lib/*.dylib", "lib/*.so",
                               "numerical_/*.pyi", "numerical_/**/*.pyi",
                               "egttools_build_info.txt"],
    },
)
