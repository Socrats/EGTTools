# docker/memcheck/Dockerfile — run the EGTtools tests under Valgrind memcheck.
#
# Build (from the repository root, with the pybind11 submodule checked out):
#   docker build -f docker/memcheck/Dockerfile -t egttools-memcheck .
#
# Run the C++ test programs under Valgrind (exit code 1 if Valgrind reports errors):
#   docker run --rm egttools-memcheck            # same as: docker run --rm egttools-memcheck cpp
#
# Run the Python test suite under Valgrind (slow; writes valgrind-python.txt):
#   docker run --rm -v "$PWD/out:/workspace/EGTtools/out" egttools-memcheck python
#
# Dependencies come from Ubuntu packages (SKIP_VCPKG=ON); OpenMP is disabled because its
# runtime produces false positives under Valgrind.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build valgrind \
        libeigen3-dev libboost-dev libopenblas-dev liblapack-dev \
        python3 python3-dev python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/EGTtools
COPY . /workspace/EGTtools

RUN python3 -m venv /workspace/venv
ENV PATH=/workspace/venv/bin:$PATH \
    SKIP_VCPKG=ON \
    EGTTOOLS_VERSION=0.0.0

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy scipy matplotlib seaborn networkx plotly pytest \
        "scikit-build>=0.13.1" "cmake>=3.27,<4" ninja "setuptools-scm>=7"

# Debug build of the C++ test programs.
RUN cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSKIP_VCPKG=ON -DUSE_OPENMP=OFF \
        -DBUILD_NUMERICAL_TESTS=ON -DPython_EXECUTABLE=/workspace/venv/bin/python && \
    cmake --build build

# Python extension, built in place for the Python mode.
RUN EGTTOOLS_EXTRA_CMAKE_ARGS="-DUSE_OPENMP=OFF -DBUILD_NUMERICAL_TESTS=OFF" \
    python setup.py build_ext --inplace

ENTRYPOINT ["bash", "docker/memcheck/run.sh"]
CMD ["cpp"]
