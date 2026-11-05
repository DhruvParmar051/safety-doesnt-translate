#!/bin/bash
# Login node only (it has internet, GPU nodes do not): downloads and the CPU-only data steps.
# Run once, and again if /tmp/$USER was cleaned (files unused for 10 days are deleted).
# Downloads what the 24 GB slices need; the 32B judge comes later (scripts/hpc/auto.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/hpc/env.sh
SLICE_MODELS="global base fire water earth translator judge_secondary"

# The link drops now and then; a retry resumes from the files already downloaded.
for attempt in 1 2 3 4 5; do
  # shellcheck disable=SC2086  # a list of names
  python scripts/hpc/hub.py download $SLICE_MODELS && break
  [ "$attempt" = 5 ] && exit 1
  echo "download failed (attempt $attempt); retrying in 5 min"
  sleep 300
done
[ -s data/raw/aya_redteaming.jsonl ] || sdt data sources
[ -s data/processed/task_train.jsonl ] || sdt data task
echo "Login node ready. Submit with scripts/hpc/submit.sh"
