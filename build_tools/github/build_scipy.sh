#!/bin/bash

set -e
set -x

# OpenMP is not present on macOS by default
if [[ "$RUNNER_OS" == "macOS" && "$BUILD_ARCH" != "x86_64" ]]; then
  # We need to build scipy and numpy from source
  brew install openblas gfortran
  export OPENBLAS=$(brew --prefix openblas)
  export GFORTRAN=$(brew --prefix gfortran)
  export CIBW_ENVIRONMENT="$CIBW_ENVIRONMENT OPENBLAS=$OPENBLAS GFORTRAN=$GFORTRAN"
  echo "OPENBLAS=$(brew --prefix openblas)" >> "$GITHUB_ENV"
  echo "GFORTRAN=$(brew --prefix gfortran)" >> "$GITHUB_ENV"

  python -m pip install cython pybind11
  python -m pip install --no-binary :all: --no-use-pep517 numpy

  # then compile scipy
  python -m pip install --no-binary :all: --no-use-pep517 scipy
fi
