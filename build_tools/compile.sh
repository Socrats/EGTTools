#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <path_to_egttools> <path_to_openmp> <path_to_conda_env>"
  exit 1
fi

EGTTOOLS_PATH=$1
OPENMP_PATH=$2
CONDA_ENV_PATH=$3

# Activate the conda environment
source "$CONDA_ENV_PATH/bin/activate"

# Set the necessary environment variables
export VCPKG_PATH=$EGTTOOLS_PATH
export EGTTOOLS_EXTRA_CMAKE_ARGS="-DLIBOMP_DIR='$OPENMP_PATH' -DLAPACK_DIR='$CONDA_ENV_PATH'"

# Install egttools using pip
pip install "$EGTTOOLS_PATH"

# Deactivate the conda environment
conda deactivate
