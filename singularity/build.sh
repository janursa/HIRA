#!/bin/bash
# Build an image. Usage: bash singularity/build.sh [out.sif] [def]
set -e
cd "$(dirname "$0")/.."
out="${1:-singularity/hira.sif}"
singularity build --force --fakeroot "$out" "singularity/${2:-hira.def}"
echo "Built $out"
[ "$out" = "singularity/hira.sif" ] || echo "Non-default path -- set HIRA_SIF=$(cd "$(dirname "$out")" && pwd)/$(basename "$out") in .env"
