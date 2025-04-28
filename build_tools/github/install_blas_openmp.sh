#!/usr/bin/env bash
set -e
set -x

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     platform=Linux;;
    Darwin*)    platform=Mac;;
    CYGWIN*|MINGW*|MSYS*) platform=Windows;;
    *)          platform="UNKNOWN:${unameOut}"
esac

echo "Detected platform: ${platform}"

if [[ "$platform" == "Linux" ]]; then
    echo "Installing system libraries for Linux..."
    sudo apt-get update -y
    xargs -a build_tools/github/apt-packages.txt sudo apt-get install -y --no-install-recommends

elif [[ "$platform" == "Mac" ]]; then
    echo "Installing system libraries for macOS..."

    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found! Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    brew update
    xargs brew install < build_tools/github/brew-packages.txt

else
    echo "No system libraries needed for Windows (handled by MSVC)"
fi