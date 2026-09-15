#!/usr/bin/env bash
# Push a history-free, data-free snapshot of HEAD to the CiiM org mirror.
# The org repo is a snapshot, not a clone: no history, no *.h5ad. So a plain
# push can't work -- we rebuild the single commit and force-push it.
# ponytail: force-push rewrites the mirror each time; fine while nobody
# commits directly to the org repo. If they do, switch to a tracked branch.
set -euo pipefail

REMOTE=git@github.com:CiiM-Bioinformatics-group/HIRA.git
SRC=$(git rev-parse --show-toplevel)
mkdir -p "$SRC/temp"
SUB=$(git -C "$SRC" rev-parse HEAD:GRNimmuneClock)
SNAP=$(mktemp -d "$SRC/temp/ciim-snap.XXXXXX")
trap 'rm -rf "$SNAP"' EXIT

git -C "$SRC" archive HEAD | tar -x -C "$SNAP"
find "$SNAP" -name '*.h5ad' -delete
find "$SNAP" -name '__pycache__' -type d -prune -exec rm -rf {} +

cd "$SNAP"
git init -q -b main
git add -A
git update-index --add --cacheinfo 160000,"$SUB",GRNimmuneClock
git commit -q -m "HIRA snapshot of $(git -C "$SRC" rev-parse --short HEAD)

Test data fixtures (*.h5ad) excluded; full history at github.com/janursa/HIRA."
git push -q --force "$REMOTE" main
echo "pushed $(git rev-parse --short HEAD) -> $REMOTE"
