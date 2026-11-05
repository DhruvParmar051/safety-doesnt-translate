#!/bin/bash
# Login node, unattended: setup, then the slice jobs, then the 32B judge's download, then the
# H200 judge job (it waits in the queue for the slice jobs and for the H200).
# Safe to start while an earlier setup_login.sh is still running.
#   setsid nohup scripts/hpc/auto.sh > logs/auto.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
while pgrep -u "$USER" -f scripts/hpc/setup_login.sh >/dev/null; do sleep 60; done

# Re-running setup is cheap when it already finished (cached files, finished steps skipped).
if ! scripts/hpc/setup_login.sh; then
  echo "$(date '+%F %T') setup failed; nothing submitted (see above)"
  exit 1
fi
source scripts/hpc/env.sh
if [ -e runs/.submitted ]; then
  echo "$(date '+%F %T') already submitted; not submitting again"
else
  echo "$(date '+%F %T') submitting the slice jobs"
  scripts/hpc/submit.sh && touch runs/.submitted
fi
downloaded=no
for attempt in 1 2 3 4 5; do
  python scripts/hpc/hub.py download judge_primary && downloaded=yes && break
  echo "32B download failed (attempt $attempt); retrying in 5 min"
  sleep 300
done
if [ "$downloaded" = no ]; then
  echo "$(date '+%F %T') 32B download failed; H200 job not submitted"
  exit 1
fi
if [ -e runs/.submitted_h200 ]; then
  echo "$(date '+%F %T') H200 job already submitted"
else
  echo "$(date '+%F %T') submitting the H200 judge job"
  scripts/hpc/submit_h200.sh && touch runs/.submitted_h200
fi
squeue -u "$USER"
echo "$(date '+%F %T') done"
