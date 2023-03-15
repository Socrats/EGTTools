#!/bin/bash

# Code copied from scikit-learn

set -e
set -x

python -m pip install --upgrade --user pip cmake

# First we download the correct eigen3 version
curl -O https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
tar xf eigen-3.3.9.tar.gz
mv eigen-3.3.9 eigen3

# Download Boost 1.80.0
curl -L https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.tar.gz -o boost_1_80_0.tar.gz
tar xf boost_1_80_0.tar.gz
mv boost_1_80_0 boost


if [[ $(uname) == "Darwin" ]]; then
  if [[ "$CIBW_BUILD" == *-macosx_arm64 ||  "$CIBW_BUILD" == *-macosx_universal2:arm64 ]]; then
    export MACOSX_DEPLOYMENT_TARGET=12.0
    echo "MACOSX_DEPLOYMENT_TARGET=12.0" >> "$GITHUB_ENV"

    # Install boost
    cd boost
    ./bootstrap.sh
    ./b2 -j8 architecture=arm64 address-model=64 install
    cd ..
  else
    export MACOSX_DEPLOYMENT_TARGET=10.15
    echo "MACOSX_DEPLOYMENT_TARGET=10.15" >> "$GITHUB_ENV"

    # Install boost
    cd boost
    ./bootstrap.sh
    ./b2 -j8 architecture=x86 address-model=64 install
    cd ..
  fi
fi
#  ln -s "$(which cmake)" /usr/local/bin/cmake

# Install eigen 3
cd eigen3
mkdir "build"
cd build
cmake ..
make install

# Install extra libraries
brew install gfortran
brew install openblas
