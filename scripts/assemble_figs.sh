#!/bin/bash
# Merged/assembled figures. Each script reads persisted stats -- no pipeline rerun.
# Usage: bash scripts/assemble_figs.sh
set -e

[ -f .env ] && set -a && source .env && set +a

for f in scripts/assemble_figs/*.py; do
    echo "---------------------------------------------------------- $(basename "$f") ----------------------------------------------------------"
    python "$f"
done
