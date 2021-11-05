#!/bin/bash

set -e
set -x

brew update
brew install eigen
echo "EIGEN=$(brew --prefix eigen)" >> $GITHUB_ENV
brew install gfortran
echo "GFORTRAN=$(brew --prefix gfortran)" >> $GITHUB_ENV
brew install openblas
echo "OPENBLAS=$(brew --prefix openblas)" >> $GITHUB_ENV
brew install libomp
echo "LIBOMP=$(brew --prefix libomp)" >> $GITHUB_ENV