#!/usr/bin/env bash
set -euxo pipefail

export CMAKE_ARGS="-DSKIP_VCPKG=ON"

$PYTHON -m pip install . --no-build-isolation -vvv
