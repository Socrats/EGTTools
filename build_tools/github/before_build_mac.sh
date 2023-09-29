#!/bin/bash

set -e
set -x

python -m pip install --upgrade --user pip cmake

# Install extra libraries for parallel processing
brew install gfortran
brew install openblas
brew install libomp
brew install boost

if [[ $(uname) == "Darwin" ]]; then
  if [[ "$CIBW_BUILD" == *-macosx_arm64 || "$CIBW_BUILD" == *-macosx_universal2:arm64 ]]; then
    brew reinstall libomp
    export MACOSX_DEPLOYMENT_TARGET=12.0
    export EGTTOOLS_EGTTOOLS_EXTRA_CMAKE_ARGS="-DMACOSX_DEPLOYMENT_TARGET=12.0 -DLIBOMP_DIR='$(brew --prefix libomp)'"
    echo "MACOSX_DEPLOYMENT_TARGET=12.0" >>"$GITHUB_ENV"

  else
    brew reinstall libomp
    export MACOSX_DEPLOYMENT_TARGET=10.15
    export EGTTOOLS_EGTTOOLS_EXTRA_CMAKE_ARGS="-DMACOSX_DEPLOYMENT_TARGET=10.15 -DLIBOMP_DIR='$(brew --prefix libomp)'"
    echo "MACOSX_DEPLOYMENT_TARGET=10.15" >>"$GITHUB_ENV"
  fi
fi

# First we download the correct eigen3 version
curl -O https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
tar xf eigen-3.3.9.tar.gz
mv eigen-3.3.9 eigen3

# Install eigen 3
cd eigen3
mkdir "build"
cd build
cmake ..
make install
