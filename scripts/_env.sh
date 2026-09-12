# Sourced by every script in scripts/. Resolves HIRA_DIR from this file's location,
# loads .env, and routes `python` through the Singularity image. Set HIRA_SIF empty
# in .env to use the host/conda interpreter instead.

_hira_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -f "$_hira_root/.env" ] && set -a && . "$_hira_root/.env" && set +a
export HIRA_DIR="$_hira_root"   # repo location always wins over whatever .env says
: "${HIRA_BASE_DIR:?not set -- copy .env.example to .env and fill it in}"
export PYTHONPATH="$(dirname "$_hira_root"):$_hira_root/GRNimmuneClock${PYTHONPATH:+:$PYTHONPATH}"

# Opt-in SLURM job mail. #SBATCH lines can't read env vars, so pass it at submit time:
# every `sbatch` call site uses `sbatch $SBATCH_MAIL ...`.
export SBATCH_MAIL=${HIRA_SBATCH_MAIL:+--mail-type=END,FAIL --mail-user=$HIRA_SBATCH_MAIL}

# Default to the image built by singularity/build.sh; HIRA_SIF= (empty) opts out.
: "${HIRA_SIF=$_hira_root/singularity/hira.sif}"

if [ -n "$HIRA_SIF" ]; then
    [ -f "$HIRA_SIF" ] || { echo "HIRA_SIF=$HIRA_SIF not found -- run: bash singularity/build.sh (or set HIRA_SIF= in .env to use conda)" >&2; exit 1; }
    HIRA_BINDS="$HIRA_DIR,$HIRA_BASE_DIR"
    for _d in "${HIRA_RAW_DIR:-}" "${TASK_GRN_BENCHMARK_DIR:-}"; do
        [ -n "$_d" ] && [ -d "$_d" ] && HIRA_BINDS="$HIRA_BINDS,$_d"
    done
    export HIRA_BINDS
    python() { singularity exec -B "$HIRA_BINDS" "$HIRA_SIF" python "$@"; }
    python3() { python "$@"; }
fi
