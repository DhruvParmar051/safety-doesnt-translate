#!/bin/bash
# The primary judge (Aya Expanse 32B) on the full H200: labels each model's saved replies, then
# rebuilds its metrics with both judges. Waits in the queue for the slice jobs from submit.sh
# that are still queued or running (afterok: it never starts if one of them fails), then for the H200.
# The 32B must already be downloaded on the login node (setup is done by scripts/hpc/auto.sh).
#   scripts/hpc/submit_h200.sh [models...]   (default: global drifted fire water earth)
set -euo pipefail
cd "$(dirname "$0")/../.."
models=("${@:-global drifted fire water earth}")
commands=()
for model in ${models[*]}; do
  commands+=("judge $model --judge aya" "metrics $model")
done
pending=()
for id in $(cat runs/.slice_jobs 2>/dev/null); do
  if squeue -h -j "$id" 2>/dev/null | grep -q .; then
    pending+=("$id")
  elif ! sacct -j "$id" -n -X -o State | grep -q COMPLETED; then
    echo "Slice job $id did not complete; fix and resubmit it before the H200 judge." >&2
    exit 1
  fi
done
depend=()
if [ ${#pending[@]} -gt 0 ]; then
  depend=(--dependency="afterok:$(IFS=:; echo "${pending[*]}")")
fi
export STAGE=judge_primary   # sbatch passes the environment on
sbatch --account=f202591001 --partition=h200-fullgpu --gres=gpu:h200:1 --time=12:00:00 \
  "${depend[@]}" --job-name=sdt-judge-32b scripts/hpc/run.sbatch "${commands[@]}"
