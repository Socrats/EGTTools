#!/bin/bash

set -e
set -x

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # The Linux test environment is run in a Docker container and
    # it is not possible to copy the test configuration file (yet)


    # First we download the correct eigen3 version
    wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
    tar xvzf eigen-3.3.9.tar.gz
    mv eigen-3.3.9 eigen3

    EIGEN_PATH=$(realpath eigen3)

    export CFLAGS="$CFLAGS -I$EIGEN_PATH/include"
    export CXXFLAGS="$CXXFLAGS -I$EIGEN_PATH/include"
    export Eigen3_DIR="$EIGEN_PATH/cmake"
fi