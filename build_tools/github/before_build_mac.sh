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
    echo "MACOSX_DEPLOYMENT_TARGET=12.0" >> "$GITHUB_ENV"
  else
    export MACOSX_DEPLOYMENT_TARGET=10.15
    echo "MACOSX_DEPLOYMENT_TARGET=10.15" >> "$GITHUB_ENV"
  fi
fi
