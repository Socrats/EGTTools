#!/usr/bin/env bash
set -euxo pipefail

# Remove any stale scikit-build cmake cache copied from the developer's source
# tree (it may contain CMAKE_TOOLCHAIN_FILE pointing to vcpkg).
rm -rf _skbuild

# Remove stale pre-built extension modules from the developer's tree.
# conda-build copies all source files (including .gitignore-d ones), so old
# *.so / *.dylib files from local builds would otherwise be bundled into the
# package alongside the freshly-compiled extension, causing import failures when
# the test environment uses a different Python version.
rm -f src/egttools/numerical/*.so src/egttools/numerical/*.dylib src/egttools/numerical/lib/*.dylib

# Tell setup.py to skip vcpkg toolchain injection (it reads SKIP_VCPKG directly).
export SKIP_VCPKG=ON

# Feature flags of this variant (set in meta.yaml build/script_env).
FEATURE_ARGS="-DEGTTOOLS_ENABLE_ARPACK=${EGTTOOLS_ENABLE_ARPACK:-OFF} -DEGTTOOLS_ENABLE_PETSC=${EGTTOOLS_ENABLE_PETSC:-OFF}"

# scikit-build reads CMAKE_ARGS from the environment; keep conda-build's own arguments.
# setup.py also reads EGTTOOLS_EXTRA_CMAKE_ARGS explicitly, which is the more reliable path.
export CMAKE_ARGS="${CMAKE_ARGS:-} -DSKIP_VCPKG=ON -DCMAKE_PREFIX_PATH=${PREFIX} ${FEATURE_ARGS}"
export EGTTOOLS_EXTRA_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${PREFIX} ${FEATURE_ARGS}"

echo "[egttools] PREFIX=${PREFIX}"
echo "[egttools] CMAKE_ARGS=${CMAKE_ARGS}"

$PYTHON -m pip install . --no-build-isolation --no-deps -vv
