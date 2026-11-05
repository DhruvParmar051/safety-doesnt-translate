#!/bin/bash
# Runs `sdt` commands in order on one 24 GB MIG slice; each argument is one command, e.g.
#   sbatch --account=f202591001 scripts/hpc/run.sbatch "data translate" "data build"
# A finished command leaves runs/.done/<command>, so resubmitting the same job resumes it.
# STAGE lists the configured models to copy to the GPU node (default: all), e.g.
#   STAGE="translator judge_secondary" sbatch ...
# BEST=<drifted model> replaces @BEST@ in the commands with that model's best repair, chosen when
# the job starts (scripts/hpc/best_repair.py), e.g. "compress uniform --model @BEST@".
#SBATCH --job-name=sdt
#SBATCH --partition=rtx6000-mig
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc/env.sh

# Copy weights, the capability dataset and our own models from the login node ("master").
# shellcheck disable=SC2086  # STAGE is a list of names
for folder in $(python scripts/hpc/hub.py folders ${STAGE:-}); do
  rsync -a --copy-unsafe-links --exclude '*.incomplete' \
    "master:$HF_HOME/$folder/" "$HF_HOME/$folder/"
done
rsync -a "master:$MODELS_DIR/" "$MODELS_DIR/"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# Models a step creates (drifted, repairs) go back to the login node, even if a later step fails.
trap 'rsync -a "$MODELS_DIR/" "master:$MODELS_DIR/"' EXIT

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p runs/.done
if [ -n "${BEST:-}" ]; then
  best=$(python scripts/hpc/best_repair.py "$BEST")
  echo "== best repair of $BEST: $best"
fi
for command in "$@"; do
  command=${command//@BEST@/${best:-@BEST@}}
  marker="runs/.done/${command// /_}"
  if [ -e "$marker" ]; then
    echo "== skip (done): sdt $command"
    continue
  fi
  echo "== $(date '+%F %T') sdt $command"
  # shellcheck disable=SC2086  # each argument is a whole command line
  sdt $command
  touch "$marker"
done
echo "== $(date '+%F %T') all done"
