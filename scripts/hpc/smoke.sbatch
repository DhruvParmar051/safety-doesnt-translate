#!/bin/bash
# Checks a 24 GB MIG slice before the real jobs: copying from the login node, offline datasets, and
# two vLLM engines in one process (the second only fits if the first released its memory).
#   SMOKE_MODEL=Qwen/Qwen3-8B sbatch --account=<account> scripts/hpc/smoke.sbatch
# SMOKE_MODEL is any ~8B model already in the login node's hub cache (default: the translator).
#SBATCH --job-name=sdt-smoke
#SBATCH --partition=rtx6000-mig
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc/env.sh

model=${SMOKE_MODEL:-$(python -c 'from sdt.config import load_config; print(load_config().model_path("translator"))')}
dataset=$(python -c 'from sdt.config import load_config; print(load_config()["capability"]["dataset"])')
for folder in "hub/models--${model//\//--}" "hub/datasets--${dataset//\//--}"; do
  rsync -a --copy-unsafe-links --exclude '*.incomplete' \
    "master:$HF_HOME/$folder/" "$HF_HOME/$folder/"
done
echo "== copy from login node: ok"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

python scripts/hpc/smoke_check.py "$model" "$dataset"
echo "== smoke test passed"
