#!/bin/bash

set -e
set -x

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

# Download and install Boost 1.80.0
yum check-update
yum list available boost\*
sudo apt-get install libboost-all-dev

#sudo apt-get update && sudo apt-get install -yq libboost1.80-dev
#yum search boost
#curl -O https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.tar.gz
#tar xf boost_1_80_0.tar.gz
#mv boost_1_80_0 boost
#cd boost
#./bootstrap.sh
#./bjam install
#yum install boost-devel
