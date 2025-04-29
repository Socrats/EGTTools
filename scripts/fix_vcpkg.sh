#!/usr/bin/env bash
# -------------------------------------------------------------
# Script to fully clean and reinitialize the vcpkg submodule
# Safe to run anytime after a git merge or submodule corruption
# -------------------------------------------------------------

set -e # Exit immediately if any command fails

echo "[fix_vcpkg.sh] Cleaning and fixing vcpkg submodule..."

# Deinitialize vcpkg
git submodule deinit -f vcpkg || true

# Remove old vcpkg folder
rm -rf vcpkg

# Reinitialize
git submodule update --init --recursive

# Fetch all tags/branches inside vcpkg
cd vcpkg
git fetch --all
cd ..

echo "[fix_vcpkg.sh] vcpkg submodule fully restored."
