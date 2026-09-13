#!/bin/bash
# Build the image. Usage: bash singularity/build.sh [out.sif]
set -e
cd "$(dirname "$0")/.."
out="${1:-singularity/hira.sif}"
singularity build --force --fakeroot "$out" singularity/hira.def
echo "Built $out"
[ "$out" = "singularity/hira.sif" ] || echo "Non-default path -- set HIRA_SIF=$(cd "$(dirname "$out")" && pwd)/$(basename "$out") in .env"
