#!/bin/bash

set -e
set -x

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  # The Linux test environment is run in a Docker container and
  # it is not possible to copy the test configuration file (yet)

  pip install cmake
  ln -s "$(which cmake)" /usr/local/bin/cmake

  # First we download the correct eigen3 version and build it
  curl -O https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
  tar xf eigen-3.3.9.tar.gz
  mv eigen-3.3.9 eigen3
  cd eigen3
  mkdir "build"
  cd build
  cmake ..
  make install
elif [[ "$RUNNER_OS" == "Windows" ]]; then
  #  curl.exe --output eigen-3.3.9.tar.gz --url https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
  #  tar -zxvf eigen-3.3.9.tar.gz
  #  ren eigen-3.3.9 eigen3
  #  export EIGEN_INCLUDE_DIR="eigen3"
  # First we download the correct eigen3 version and build it
  curl -O https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
  tar xf eigen-3.3.9.tar.gz
  mv eigen-3.3.9 eigen3
  EIGEN3_INCLUDE_DIR="$(pwd)/eigen3"
  export EIGEN3_INCLUDE_DIR
  echo "EIGEN3_INCLUDE_DIR=$EIGEN3_INCLUDE_DIR"
  echo "EIGEN3_INCLUDE_DIR=$EIGEN3_INCLUDE_DIR" >> "$GITHUB_ENV"
  export CIBW_ENVIRONMENT_WINDOWS="$CIBW_ENVIRONMENT EIGEN3_INCLUDE_DIR=$EIGEN3_INCLUDE_DIR"
  export CMAKE_MODULE_PATH="$CMAKE_MODULE_PATH EIGEN3_INCLUDE_DIR$"
  export CIBW_ENVIRONMENT_WINDOWS="$CIBW_ENVIRONMENT_WINDOWS CMAKE_MODULE_PATH=$CMAKE_MODULE_PATH --config Release"
fi
