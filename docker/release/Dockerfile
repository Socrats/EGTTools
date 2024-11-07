# Use an official Ubuntu base image
FROM ubuntu:24.04

ARG BOOST_VERSION=1.84.0
ARG CMAKE_VERSION=3.30.1
ARG NUM_JOBS=8

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    software-properties-common \
    wget \
    curl \
    language-pack-en \
    locales \
    locales-all \
    libtool \
    pkg-config \
    ca-certificates \
    libssl-dev \
    vim \
    git \
    gcc \
    g++ \
    clang \
    gdb \
    valgrind \
    cmake \
    make \
    ninja-build \
    autoconf \
    automake \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libomp-dev \
    libeigen3-dev \
    dos2unix \
    tar \
    rsync \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-dev \
    python3-venv \
    && apt-get clean

# System locale
# Important for UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8


# Install CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz && \
    tar xzf cmake-${CMAKE_VERSION}.tar.gz && \
    cd cmake-${CMAKE_VERSION} && \
    ./bootstrap && \
    make -j${NUM_JOBS} && \
    make install && \
    rm -rf /tmp/*

# Install Boost
# https://www.boost.org/doc/libs/1_80_0/more/getting_started/unix-variants.html
RUN cd /tmp && \
    BOOST_VERSION_MOD=$(echo $BOOST_VERSION | tr . _) && \
    wget https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_MOD}.tar.bz2 && \
    tar --bzip2 -xf boost_${BOOST_VERSION_MOD}.tar.bz2 && \
    cd boost_${BOOST_VERSION_MOD} && \
    ./bootstrap.sh --prefix=/usr/local && \
    ./b2 install && \
    rm -rf /tmp/*

# Set the working directory
WORKDIR /EGTtools

# Copy the project files into the Docker image
COPY . /EGTtools

# Install virtual environment
RUN python3 -m venv egtenv
# Set python executable from virtual environment
#RUN #bash egtenv/bin/activate
#
# Install Python dependencies
RUN egtenv/bin/pip3 install --upgrade pip build pytest
RUN egtenv/bin/pip3 install -r requirements.txt

# Build the project
RUN egtenv/bin/python3 -m build
#
## Install the project
RUN egtenv/bin/pip3 install .
#
## Run tests
RUN egtenv/bin/pytest tests

RUN valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./tests/cpp/test_PairwiseMoran_run.cpp

# Default command
CMD ["/bin/bash"]