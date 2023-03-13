#!/bin/bash

# Code copied from scikit-learn

set -e
set -x

brew install eigen
brew install gfortran
brew install openblas
brew install libomp
brew install boost

if [[ $(uname) == "Darwin" ]]; then
  if [[ "$CIBW_BUILD" == *-macosx_arm64 ||  "$CIBW_BUILD" == *-macosx_universal2 ]]; then
    export MACOSX_DEPLOYMENT_TARGET=12.0
    export EGTTOOLS_EXTRA_CMAKE_ARGS='-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=12.0'
  else
    export MACOSX_DEPLOYMENT_TARGET=10.9
    export EGTTOOLS_EXTRA_CMAKE_ARGS='-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.9'
  fi
fi
