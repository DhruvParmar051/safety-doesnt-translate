#!/bin/bash
# Phases 1-2 on 24 GB MIG slices: data first, then the drift fine-tune and the audits on three
# slices at once, judged by the secondary judge only. The primary judge (Aya Expanse 32B) needs
# the H200: scripts/hpc/submit_h200.sh. Run scripts/hpc/setup_login.sh first.
set -euo pipefail
cd "$(dirname "$0")/../.."
account=--account=<account>
job=scripts/hpc/run.sbatch
# Everything except the primary judge, which these slices cannot hold.
export STAGE="global base fire water earth translator judge_secondary"   # sbatch passes the environment on
judge="--judge llama_guard"

data=$(sbatch --parsable $account --job-name=sdt-data --time=6:00:00 $job \
  "data translate" "data build")
after=--dependency=afterok:$data
{
  sbatch --parsable $account $after --job-name=sdt-drift $job finetune "evaluate drifted $judge"
  sbatch --parsable $account $after --job-name=sdt-eval-a $job \
    "evaluate global $judge" "evaluate fire $judge"
  sbatch --parsable $account $after --job-name=sdt-eval-b $job \
    "evaluate water $judge" "evaluate earth $judge"
} > runs/.slice_jobs   # submit_h200.sh waits for these
squeue -u "$USER"
