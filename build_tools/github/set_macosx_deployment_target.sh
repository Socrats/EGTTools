#!/usr/bin/env bash
set -e
set -x

# Detect architecture
arch_name="$(uname -m)"
echo "Detected architecture: ${arch_name}"

if [[ "$arch_name" == "arm64" ]]; then
  export MACOSX_DEPLOYMENT_TARGET=14.0
  echo "MACOSX_DEPLOYMENT_TARGET=14.0" >> $GITHUB_ENV
  echo "Set MACOSX_DEPLOYMENT_TARGET=14.0 for ARM64"
else
  export MACOSX_DEPLOYMENT_TARGET=13.0
  echo "MACOSX_DEPLOYMENT_TARGET=13.0" >> $GITHUB_ENV
  echo "Set MACOSX_DEPLOYMENT_TARGET=13.0 for Intel x86_64"
fi