#!/bin/bash
# Refresh everything GRNimmuneClock ships (GRNs, example, models, held-out data, aging
# stats) from the current results_folder. Run after src/clock/run_train.py retrains.
set -e

[ -f .env ] && set -a && source .env && set +a

python src/clock/build_package_data.py "$@"
