#!/bin/bash

set -e
set -x

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  # The Linux test environment is run in a Docker container and
  # it is not possible to copy the test configuration file (yet)

  python -m pip install --upgrade --user pip cmake
#  ln -s "$(which cmake)" /usr/local/bin/cmake

  # First we download the correct eigen3 version and build it
  curl -O https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
  tar xf eigen-3.3.9.tar.gz
  mv eigen-3.3.9 eigen3
  cd eigen3
  mkdir "build"
  cd build
  cmake ..
  make install
#  yum install -y eigen3-devel
#  echo "Eigen3_DIR='/usr/include/eigen3'" >> $GITHUB_ENV
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
  export CMAKE_MODULE_PATH="$CMAKE_MODULE_PATH $EIGEN3_INCLUDE_DIR$"
  export CIBW_ENVIRONMENT_WINDOWS="$CIBW_ENVIRONMENT_WINDOWS CMAKE_MODULE_PATH=$CMAKE_MODULE_PATH '--config Release'"
elif [[ "$RUNNER_OS" == "macOS" ]]; then
  # Make sure to use a libomp version binary compatible with the oldest
  # supported version of the macos SDK as libomp will be vendored into the
  # scikit-learn wheels for macos. The list of bottles can be found at:
  # https://formulae.brew.sh/api/formula/libomp.json. Currently, the oldest
  # supported macos version is: High Sierra / 10.13. When upgrading this, be
  # sure to update the MACOSX_DEPLOYMENT_TARGET environment variable in
  # wheels.yml accordingly.
  if [[ "$BUILD_ARCH" == "macosx_x86_64"  ]]; then
    wget https://homebrew.bintray.com/bottles/libomp-11.0.0.high_sierra.bottle.tar.gz
    brew install libomp-11.0.0.high_sierra.bottle.tar.gz
    export MACOSX_DEPLOYMENT_TARGET=10.13
    export CIBW_ENVIRONMENT="$CIBW_ENVIRONMENT MACOSX_DEPLOYMENT_TARGET=10.13"
  else
    export MACOSX_DEPLOYMENT_TARGET=10.15
    export CIBW_ENVIRONMENT="$CIBW_ENVIRONMENT MACOSX_DEPLOYMENT_TARGET=10.15"
    brew install libomp
  fi
  export CC=/usr/bin/clang
  export CXX=/usr/bin/clang++
  export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
  export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
  export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
  export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib -L/usr/local/opt/libomp/lib -lomp"

  brew install eigen
  brew install openblas

  export LDFLAGS="$LDFLAGS -L/usr/local/opt/openblas/lib"
  export CPPFLAGS="$CPPFLAGS -I/usr/local/opt/openblas/include"
fi
