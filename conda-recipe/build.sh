#!/usr/bin/env bash
set -euxo pipefail

# Remove any stale scikit-build cmake cache copied from the developer's source
# tree (it may contain CMAKE_TOOLCHAIN_FILE pointing to vcpkg).
rm -rf _skbuild

# Remove stale pre-built extension modules from the developer's tree.
# conda-build copies all source files (including .gitignore-d ones), so old
# *.so / *.dylib files from local builds would otherwise be bundled into the
# wheel alongside the freshly-compiled extension, causing import failures when
# the test environment uses a different Python version.
rm -f src/egttools/numerical/*.so src/egttools/numerical/*.dylib src/egttools/numerical/lib/*.dylib

# Tell setup.py to skip vcpkg toolchain injection (it reads SKIP_VCPKG directly).
export SKIP_VCPKG=ON

# scikit-build reads CMAKE_ARGS from the environment and forwards them to cmake.
# We also pass CMAKE_PREFIX_PATH via EGTTOOLS_EXTRA_CMAKE_ARGS, which setup.py
# reads explicitly with shlex.split — this is more reliable than relying on
# scikit-build's own CMAKE_ARGS parsing when conda-build also sets CMAKE_ARGS.
export CMAKE_ARGS="-DSKIP_VCPKG=ON -DCMAKE_PREFIX_PATH=${PREFIX}"
export EGTTOOLS_EXTRA_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${PREFIX}"

echo "[egttools] PREFIX=${PREFIX}"
echo "[egttools] CMAKE_ARGS=${CMAKE_ARGS}"

$PYTHON -m pip install . --no-build-isolation -vvv
