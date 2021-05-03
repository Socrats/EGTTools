#!/bin/bash

set -e
set -x

# Check if python is installed
python "$1/build_tools/github/check_if_egttools_is_installed.py"

#pip install pytest-github-actions-annotate-failures
#pytest "$1/tests"
pytest -m pytest