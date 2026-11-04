#!/usr/bin/env bash
# Follow-up experiments on the strong drift, after the main chain (chain.sh):
#   D  circuit patch with more components (k = 30, 60): too few components, or the wrong ones?
#   C  CircuitKIT EAP-IG on drifted-harmful, to compare with Global's circuit like for like
#   F  translate-then-judge in sw, bn, fil: does Aya miss harm because it cannot read the language?
#   B  the same harmful drift on Fire, Water, Earth: does more data in a language protect it?
#
#   bash scripts/hpc/extras_chain.sh
#
# Slices take the fine-tune-free work, the H200 the 32B judge, CircuitKIT and the fine-tunes.
# Resubmitting skips finished steps (runs/.done markers).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/hpc/env.sh
acct=--account=f202591001
job=scripts/hpc/run.sbatch
h200=(--partition=h200-fullgpu --gres=gpu:h200:1)
slices="global base fire water earth judge_secondary"
lg="--judge llama_guard"
dh=drifted-harmful
submit() { sbatch --parsable $acct "$@"; }

# D: rank 60 components (top_components), then patch 30 and 60 of them.
rm -f "runs/.done/locate_components_--models_global_$dh"
patch=$(STAGE=$slices submit --time=4:00:00 --job-name=sdt-x-patch $job \
  "locate components --models global $dh" \
  "repair circuit-patch --value 30 --source $dh" "evaluate repair-harmful-circuit-patch-30 $lg" \
  "repair circuit-patch --value 60 --source $dh" "evaluate repair-harmful-circuit-patch-60 $lg")
# C
ckit=$(STAGE=global submit "${h200[@]}" --time=2:00:00 --job-name=sdt-x-circuitkit $job \
  "locate circuitkit --models $dh")
# F: translate on a slice (NLLB in 8-bit), judge the English on the H200.
trans=$(STAGE="translator" submit --time=3:00:00 --job-name=sdt-x-translate $job \
  "via-english translate $dh" "via-english translate repair-harmful-oneshot-0.01")
# B: three fine-tunes on the H200, evaluated on a slice.
btrain=$(STAGE="fire water earth" submit "${h200[@]}" --time=4:00:00 --job-name=sdt-x-regional $job \
  "finetune --variant harmful --base fire" "finetune --variant harmful --base water" \
  "finetune --variant harmful --base earth")
beval=$(STAGE=$slices submit --time=4:00:00 --dependency=afterok:$btrain --job-name=sdt-x-regional-eval $job \
  "evaluate $dh-fire $lg" "evaluate $dh-water $lg" "evaluate $dh-earth $lg")
# Aya judge for everything above (afterany: a failed part leaves the others to be judged). A job
# stops at its first failing step, so the newest code (translate-then-judge) goes last.
cmds=()
for m in repair-harmful-circuit-patch-30 repair-harmful-circuit-patch-60 "$dh-fire" "$dh-water" "$dh-earth"; do
  cmds+=("judge $m --judge aya" "metrics $m")
done
for m in "$dh-via-en" repair-harmful-oneshot-0.01-via-en; do cmds+=("judge $m --judge aya"); done
cmds+=("via-english report $dh" "via-english report repair-harmful-oneshot-0.01")
judge=$(STAGE=judge_primary submit "${h200[@]}" --time=3:00:00 \
  --dependency=afterany:$trans:$patch:$beval --job-name=sdt-x-judge $job "${cmds[@]}")

printf '%s\n' "x-patch $patch" "x-circuitkit $ckit" "x-translate $trans" "x-regional $btrain" \
  "x-regional-eval $beval" "x-judge $judge" | tee runs/.extras_jobs
squeue -u "$USER"
