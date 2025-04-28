#!/usr/bin/env bash
set -e
set -x

# Detect if running inside manylinux container
if [ -f /etc/redhat-release ] || [ -d /opt/_internal ]; then
  echo "[install_blas_openmp.sh] Detected manylinux container — installing OpenBLAS and LAPACK."

  # Try using yum or dnf to install OpenBLAS dev
  if command -v yum &>/dev/null; then
    yum install -y openblas-devel lapack-devel
  elif command -v dnf &>/dev/null; then
    dnf install -y openblas-devel lapack-devel
  else
    echo "[install_blas_openmp.sh] Warning: No yum/dnf found. Skipping OpenBLAS/LAPACK install."
  fi

  exit 0
fi

# Detect platform
unameOut="$(uname -s)"
case "${unameOut}" in
Linux*) platform=Linux ;;
Darwin*) platform=Mac ;;
CYGWIN* | MINGW* | MSYS*) platform=Windows ;;
*) platform="UNKNOWN:${unameOut}" ;;
esac

echo "Detected platform: ${platform}"

if [[ "$platform" == "Linux" ]]; then
  echo "Installing system libraries for Linux..."

  if [ "$(id -u)" -eq 0 ]; then
    apt-get update -y
    xargs -a build_tools/github/apt-packages.txt apt-get install -y --no-install-recommends
  else
    sudo apt-get update -y
    xargs -a build_tools/github/apt-packages.txt sudo apt-get install -y --no-install-recommends
  fi

elif [[ "$platform" == "Mac" ]]; then
  echo "Installing system libraries for macOS..."

  if ! command -v brew &>/dev/null; then
    echo "Homebrew not found! Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi

  brew update
  xargs brew install <build_tools/github/brew-packages.txt

else
  echo "No system libraries needed for Windows (handled by MSVC)"
fi