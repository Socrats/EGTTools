# Use Ubuntu 24.04 as the base image
FROM ubuntu:latest

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

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y \
        valgrind cmake g++ ninja-build \
        libeigen3-dev \
        libboost-all-dev \
        libomp-dev \
        python3 \
        python3-dev \
        python3-wheel \
        python3-dev \
        python3-venv \
        && apt-get clean && \
        rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install virtual environment
RUN python3 -m venv egtenv

# Copy EGTTools source into container
WORKDIR /workspace/EGTtools
COPY . /workspace/EGTtools

# Install Python dependencies
RUN ../egtenv/bin/pip3 install --no-cache-dir --upgrade -U pip build pytest tqdm
RUN ../egtenv/bin/pip3 install --no-cache-dir -r requirements.txt

# build EGTTools
RUN ../egtenv/bin/python3 -m build

# install EGTTools
RUN ../egtenv/bin/pip3 install .

ENV PYTHONMALLOC=debug

## Run tests
#RUN ../egtenv/bin/pytest tests

# return to main workspace
#WORKDIR /workspace

## clean directory
#RUN rm -r EGTtools

## Copy the notebook to the workspace directory
#COPY /docker/notebook/homework2-Solution.ipynb /workspace/

# Expose Jupyter Lab port
#EXPOSE 8888

# Run Valgrind on tests or example usage of EGTTools
CMD ["valgrind", "--leak-check=full", "--show-leak-kinds=all", "--track-origins=yes", "--verbose", "--log-file=valgrind-out.txt", "../egtenv/bin/python3", "-m", "pytest", "tests"]
