#!/bin/bash
# Smoke test of phases 3-5 on one slice before the full chain: every step once, on the current
# data and the drifted model, with no runs/.done markers. scripts/hpc/chain.sh archives what this
# leaves (runs/locate, runs/repair, repair models) before the real runs.
#   sbatch --account=<account> scripts/hpc/smoke_phases.sbatch ["sdt ..." ...]   (default: every step)
#SBATCH --job-name=sdt-smoke-345
#SBATCH --partition=rtx6000-mig
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc/env.sh
for folder in $(python scripts/hpc/hub.py folders global base translator); do
  rsync -a --copy-unsafe-links --exclude '*.incomplete' "master:$HF_HOME/$folder/" "$HF_HOME/$folder/"
done
rsync -a "master:$MODELS_DIR/" "$MODELS_DIR/"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
trap 'rsync -a "$MODELS_DIR/" "master:$MODELS_DIR/"' EXIT

step() {  # runs one step, records PASS/FAIL and its time, carries on either way
  local start=$SECONDS
  echo "== $(date '+%T') $*"
  if "$@"; then echo "SMOKE PASS ($((SECONDS - start))s): $*"; else echo "SMOKE FAIL ($((SECONDS - start))s): $*"; fi
}

if [ $# -gt 0 ]; then  # given steps only, e.g. after fixing some
  for command in "$@"; do
    # shellcheck disable=SC2086  # each argument is a whole command line
    step $command
  done
else
  step sdt locate directions --models global drifted
  step sdt locate components --models global drifted
  step sdt locate circuitkit
  step sdt repair task-arithmetic --value 0.5
  step sdt repair circuit-patch --value 5
  step sdt repair resta --value 0.5
  # One-shot needs Global's judged calib replies; the Aya judge needs the H200, so this smoke test
  # labels them by their opening words instead. The chain judges them properly and overwrites this.
  step sdt generate global --split calib
  step python scripts/hpc/smoke_fake_calib_labels.py
  step sdt repair oneshot --value 0.005
  for mode in uniform mixed bf16; do
    step sdt compress "$mode" --model repair-task-arithmetic-0.5
  done
  step python scripts/hpc/smoke_generate.py repair-task-arithmetic-0.5-4bit-uniform \
    repair-task-arithmetic-0.5-4bit-mixed repair-task-arithmetic-0.5-bf16-hf repair-oneshot-0.005
fi
echo "== $(date '+%T') smoke done"
