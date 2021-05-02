#!/bin/bash

set -e
set -x

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # The Linux test environment is run in a Docker container and
    # it is not possible to copy the test configuration file (yet)

    pip install cmake
    ln -s "$(which cmake)" /usr/local/bin/cmake

    # First we download the correct eigen3 version and build it
    curl -O  https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
    tar xf eigen-3.3.9.tar.gz
    mv eigen-3.3.9 eigen3
    cd eigen3
    mkdir "build"
    cd build
    cmake ..
    make install


#    EIGEN_PATH="$(cd "$(dirname "$1")"; pwd -P)/$(basename "$1")eigen3"
#
#    export CFLAGS="$CFLAGS -I$EIGEN_PATH/include"
#    export CXXFLAGS="$CXXFLAGS -I$EIGEN_PATH/include"
#    export Eigen3_DIR="$EIGEN_PATH/cmake/"
#    echo "Eigen3_DIR=$Eigen3_DIR" >> CIBW_ENVIRONMENT
fi