#!/bin/bash

set -e
set -x

#choco install eigen

#curl.exe --output eigen-3.3.9.tar.gz --url https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
#tar -zxvf eigen-3.3.9.tar.gz
#ren eigen-3.3.9 Eigen3
#move eigen3 "C:/Program Files (x86)/Eigen3"
#export Eigen3_DIR='C:/Program Files (x86)/Eigen3'
#echo "Eigen3_DIR='C:/Program Files (x86)/Eigen3'" >> $GITHUB_ENV
export EGTTOOLS_EXTRA_CMAKE_ARGS='-DSKIP_OPENMP:BOOL=TRUE'
#echo "EGTTOOLS_EXTRA_CMAKE_ARGS='-DSKIP_OPENMP:BOOL=TRUE'" >> $GITHUB_ENV

#choco install boost-msvc-14.3

