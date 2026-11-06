#!/bin/bash
# The whole experiment as one Slurm dependency chain, from the data rebuild to the report.
# Login node, once:  scripts/hpc/chain.sh   (job ids go to runs/.chain_jobs)
#
#   data (Aya 101 in 8-bit) ───────────┬─ eval-a: global, drifted, global calib ─┐
#                                      ├─ eval-b: fire, water, earth, drifted-harmful ─┴─ judge-1 (H200) ─┐
#                                      ├─ locate: directions, components (both drifts) ──────────────────┤
#                                      └─ circuitkit (H200; a cross-check, nothing waits for it)
#   repairs of drifted-harmful: weight edits (after data, locate) + calib-based (after judge-1)
#       ── judge-2a (H200) ── compress best ── judge-3 (H200) ── report
#   repairs of drifted: the same two jobs ── judge-2b (H200)
#
# Only the Aya 32B judge and CircuitKIT need the H200; the rest fits a 24 GB slice (configs/hpc.yaml).
# afterok: a job starts only if the jobs it needs succeeded. Locate may fail without stopping the
# repairs (afterany): only circuit-patch and mixed compression need it, and they run last.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/hpc/env.sh
acct=--account=${SDT_ACCOUNT:?set SDT_ACCOUNT in scripts/hpc/local.env}
job=scripts/hpc/run.sbatch
h200=(--partition=h200-fullgpu --gres=gpu:h200:1)
slices="global base fire water earth judge_secondary"
lg="--judge llama_guard"

# The previous run used the old translations: keep it, and clear the markers of every step that redoes it.
old=runs/archive-$(date +%Y%m%d-%H%M)
mkdir -p "$old"
for d in generations judgements metrics locate repair compress; do
  [ -e "runs/$d" ] && mv "runs/$d" "$old/"
done
# Repairs from the smoke test: every one the chain uses is made again before it is read.
mkdir -p "$MODELS_DIR-archive/$(basename "$old")"
for m in "$MODELS_DIR"/repair-*; do [ -e "$m" ] && mv "$m" "$MODELS_DIR-archive/$(basename "$old")/"; done
find runs/.done -type f ! -name 'finetune' ! -name 'finetune_--variant_harmful' -delete
echo "Archived the previous run to $old"

submit() { sbatch --parsable $acct "$@"; }

data=$(STAGE=translator submit --time=4:00:00 --job-name=sdt-data $job \
  "data translate" "data build")
eval_a=$(STAGE=$slices submit --dependency=afterok:$data --job-name=sdt-eval-a $job \
  "evaluate global $lg" "evaluate drifted $lg" "generate global --split calib")
eval_b=$(STAGE=$slices submit --dependency=afterok:$data --job-name=sdt-eval-b $job \
  "evaluate fire $lg" "evaluate water $lg" "evaluate earth $lg" "evaluate drifted-harmful $lg")
locate=$(STAGE=$slices submit --dependency=afterok:$data --job-name=sdt-locate $job \
  "locate directions --models global drifted drifted-harmful fire water earth" \
  "locate components --models global drifted-harmful" \
  "locate components --models global drifted")
# CircuitKIT's EAP-IG needs more than a slice for long prompts; a cross-check nothing else waits for.
circuitkit=$(STAGE=global submit "${h200[@]}" --time=3:00:00 --dependency=afterok:$data \
  --job-name=sdt-circuitkit $job "locate circuitkit")

judge=()
for m in global drifted drifted-harmful fire water earth; do judge+=("judge $m --judge aya" "metrics $m"); done
judge1=$(STAGE=judge_primary submit "${h200[@]}" --time=6:00:00 --dependency=afterok:$eval_a:$eval_b \
  --job-name=sdt-judge-1 $job "${judge[@]}" "judge global --split calib --judge aya")

repairs() {  # $1 = drifted model; prints the repaired model names (as repair_name() makes them)
  local tag=""
  [ "$1" != drifted ] && tag="${1#drifted-}-"
  echo "repair-${tag}task-arithmetic-0.5 repair-${tag}task-arithmetic-1 repair-${tag}oneshot-0.005" \
    "repair-${tag}oneshot-0.01 repair-${tag}resta-0.5 repair-${tag}safety-ft repair-${tag}circuit-patch-10" \
    "repair-${tag}interpolate-0.5"
}
repair_jobs() {  # $1 = drifted model; prints the ids of its two repair jobs
  local src=$1 names weights=() calib=()
  names=($(repairs "$src"))
  # Weight edits need no judged calib data, so they need not wait for the H200.
  weights+=("repair task-arithmetic --value 0.5 --source $src" "repair task-arithmetic --value 1.0 --source $src")
  weights+=("repair resta --value 0.5 --source $src")
  # Interpolation back toward Global: the fair weight-space baseline (task arithmetic adds all of
  # instruction tuning again, since Global - Base is more than safety).
  weights+=("repair interpolate --value 0.5 --source $src")
  for n in "${names[0]}" "${names[1]}" "${names[4]}" "${names[7]}"; do weights+=("evaluate $n $lg"); done
  # Needs locate's components, so it goes last: if locate failed, only this part fails.
  weights+=("repair circuit-patch --value 10 --source $src" "evaluate ${names[6]} $lg")
  STAGE=$slices submit --dependency=afterok:$data,afterany:$locate \
    --job-name="sdt-repair-w-${src}" $job "${weights[@]}"
  # One-shot and the safety fine-tune learn from Global's refusals, judged by the Aya judge.
  calib+=("repair oneshot --value 0.005 --source $src" "repair oneshot --value 0.01 --source $src")
  calib+=("repair safety-ft --source $src")
  for n in "${names[2]}" "${names[3]}" "${names[5]}"; do calib+=("evaluate $n $lg"); done
  STAGE=$slices submit --dependency=afterok:$judge1 --job-name="sdt-repair-c-${src}" $job "${calib[@]}"
}
judge_job() {  # $1 = drifted model, $2 = weight-edit repair job, $3 = calib repair job
  local cmds=()
  # circuit-patch comes last: if it is missing (locate failed), every other repair is judged first.
  for n in $(repairs "$1"); do cmds+=("judge $n --judge aya" "metrics $n"); done
  STAGE=judge_primary submit "${h200[@]}" --time=6:00:00 --dependency=afterany:$2,afterok:$3 \
    --job-name="sdt-judge-$1" $job "${cmds[@]}"
}
rep_h=($(repair_jobs drifted-harmful))
judge2a=$(judge_job drifted-harmful "${rep_h[0]}" "${rep_h[1]}")
rep_d=($(repair_jobs drifted))
judge2b=$(judge_job drifted "${rep_d[0]}" "${rep_d[1]}")

# Phase 5 on the best repair of the strong drift; bf16-hf is the same weights on the 4-bit backend.
# afterany: judge-2a fails only on a missing circuit-patch, and the best repair is chosen from the
# repairs it did judge.
compress=$(BEST=drifted-harmful STAGE=$slices submit --dependency=afterany:$judge2a --job-name=sdt-compress $job \
  "compress bf16 --model @BEST@" "compress uniform --model @BEST@" "compress mixed --model @BEST@" \
  "evaluate @BEST@-bf16-hf $lg" "evaluate @BEST@-4bit-uniform $lg" "evaluate @BEST@-4bit-mixed $lg")
judge3=$(BEST=drifted-harmful STAGE=judge_primary submit "${h200[@]}" --time=3:00:00 \
  --dependency=afterok:$compress --job-name=sdt-judge-3 $job \
  "judge @BEST@-bf16-hf --judge aya" "metrics @BEST@-bf16-hf" \
  "judge @BEST@-4bit-uniform --judge aya" "metrics @BEST@-4bit-uniform" \
  "judge @BEST@-4bit-mixed --judge aya" "metrics @BEST@-4bit-mixed")
# Without --models, `report compression` would chart every model with metrics.
report=$(BEST=drifted-harmful STAGE=global submit --dependency=afterany:$judge3:$judge2b --time=1:00:00 \
  --job-name=sdt-report $job "report all" \
  "report compression --models @BEST@-bf16-hf @BEST@-4bit-uniform @BEST@-4bit-mixed")

printf '%s\n' "data $data" "eval-a $eval_a" "eval-b $eval_b" "locate $locate" "circuitkit $circuitkit" "judge-1 $judge1" \
  "repair-harmful ${rep_h[*]}" "judge-2a $judge2a" "repair-drifted ${rep_d[*]}" "judge-2b $judge2b" \
  "compress $compress" "judge-3 $judge3" "report $report" | tee runs/.chain_jobs
squeue -u "$USER"
