#!/bin/bash

# Set LIBOMP location
export EGTTOOLS_EXTRA_CMAKE_ARGS="-DLIBOMP_DIR='/opt/homebrew/Caskroom/miniforge/base/envs/egtenv'"
python -m build
pip install -e .
cp _skbuild/cmake-install/numerical/numerical_.cpython-310-darwin.so src/egttools/numerical/
