FROM ubuntu:24.04

ARG BOOST_VERSION=1.84.0
ARG CMAKE_VERSION=3.30.1
ARG NUM_JOBS=8

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# System locale
# Important for UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# Install necessary packages, including Valgrind
RUN apt-get update && \
    apt-get install -y \
    valgrind cmake g++ ninja-build \
    libeigen3-dev \
    libboost-all-dev \
    libomp-dev \
    python3 \
    python3-dev \
    && apt-get clean

# Copy EGTTools source into container
WORKDIR /workspace/EGTtools
COPY . /workspace/EGTtools

# Install dependencies with vcpkg or any other method
# (Assuming vcpkg is already set up and dependencies are listed)
#ENV VCPKG_FORCE_SYSTEM_BINARIES=1
#RUN ./vcpkg/bootstrap-vcpkg.sh
#RUN ./vcpkg/vcpkg install

# Build EGTTools
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
RUN cmake --build build

# Run Valgrind on tests or example usage of EGTTools
CMD ["valgrind", "--leak-check=full", "--show-leak-kinds=all", "./build/tests/testPMrun"]