#!/bin/bash

set -e
set -x

python -m pip install --upgrade pip cibuildwheel twine

# The version of the built dependencies are specified
# in the pyproject.toml file, while the tests are run
# against the most recent version of the dependencies
python -m cibuildwheel --output-dir wheelhouse

