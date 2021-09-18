#!/bin/bash

set -e
set -x

# Check if python is installed
python "$1/build_tools/github/check_if_egttools_is_installed.py"

#pip install pytest-github-actions-annotate-failures
#python -m pytest "$1/tests" -s --import-mode=importlib

nosetests "$1/tests"