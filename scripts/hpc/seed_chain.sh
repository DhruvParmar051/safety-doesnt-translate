#!/usr/bin/env bash
# A second training seed for the strong drift, and the best cheap repair on it: does the drift
# (and the repair) repeat, or was the first fine-tune lucky?
#
#   bash scripts/hpc/seed_chain.sh [seed]      # default seed 1; the project seed is 0
#
# Two jobs, about 1.5 hours in all:
#   1. slice: fine-tune drifted-harmful-s<seed>, evaluate it, one-shot 0.01 on it, evaluate that
#   2. H200:  Aya judge and metrics for both (afterok:1)
# Same data, same hyperparameters as drifted-harmful; only the LoRA init and data order change.
# Resubmitting skips finished steps (runs/.done markers).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/hpc/env.sh
seed=${1:-1}
src=drifted-harmful-s$seed
fix=repair-harmful-s$seed-oneshot-0.01  # repair_name(): the source's tag goes after "repair-"
acct=--account=f202591001
job=scripts/hpc/run.sbatch
lg="--judge llama_guard"
submit() { sbatch --parsable $acct "$@"; }

train=$(STAGE="global judge_secondary" submit --time=4:00:00 --job-name=sdt-seed$seed $job \
  "finetune --variant harmful --seed $seed" "evaluate $src $lg" \
  "repair oneshot --value 0.01 --source $src" "evaluate $fix $lg")
judge=$(STAGE=judge_primary submit --partition=h200-fullgpu --gres=gpu:h200:1 --time=2:00:00 \
  --dependency=afterok:$train --job-name=sdt-seed$seed-judge $job \
  "judge $src --judge aya" "metrics $src" "judge $fix --judge aya" "metrics $fix")

printf '%s\n' "seed-train $train" "seed-judge $judge" | tee "runs/.seed${seed}_jobs"
squeue -u "$USER"
