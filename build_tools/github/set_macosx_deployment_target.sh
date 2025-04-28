#!/usr/bin/env bash
set -e
set -x

# Detect architecture
arch_name="$(uname -m)"
echo "Detected architecture: ${arch_name}"

if [[ "$arch_name" == "arm64" ]]; then
  export MACOSX_DEPLOYMENT_TARGET=14.0
  echo "Set MACOSX_DEPLOYMENT_TARGET=14.0 for ARM64"
else
  export MACOSX_DEPLOYMENT_TARGET=11.0
  echo "Set MACOSX_DEPLOYMENT_TARGET=11.0 for Intel x86_64"
fi