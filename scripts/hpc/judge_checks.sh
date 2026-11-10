#!/usr/bin/env bash
# Is the Swahili gap the judge, or the translator? Three checks on the 32B judge:
#   1  same content: English replies translated into a language, and back. The judge seeing less
#      in the translation but the same after the round trip means it misses harm that is there.
#   2  Global's replies through translation: does the translator turn safe replies harmful?
#   3  Russian and Arabic, to have every language judged in English.
#   4  English prompt, original reply: is it the reply or the question the judge misreads?
# Also the Spanish and Hindi control of translate-then-judge, so a fresh run gets all of it.
#
#   bash scripts/hpc/judge_checks.sh
#
# Two jobs, about 1.5 hours: translate on a slice, then the Aya judge on the H200 (afterok).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/hpc/env.sh
acct=--account=${SDT_ACCOUNT:?set SDT_ACCOUNT in scripts/hpc/local.env}
job=scripts/hpc/run.sbatch
dh=drifted-harmful
langs="sw fil es"
all="sw bn fil es hi ru ar"
submit() { sbatch --parsable $acct "$@"; }

# This judges a longer file than the finished run did, so it must not count as done.
rm -f "runs/.done/judge_$dh-via-en_--judge_aya"
for f in runs/generations/$dh-via-en.jsonl runs/judgements/$dh-via-en.aya.jsonl \
  runs/metrics/$dh.via_english.json; do
  [ -e "$f.5lang.bak" ] || cp "$f" "$f.5lang.bak"
done

translate=("via-english translate global --langs sw fil" "via-english translate $dh --langs ru ar"
  "via-english into $dh --langs $langs")
for lang in $langs; do translate+=("via-english translate $dh-into-$lang --langs $lang"); done
for lang in $langs; do translate+=("judge $dh-into-$lang --judge llama_guard"); done
trans=$(STAGE="translator judge_secondary" submit --time=4:00:00 --job-name=sdt-check-translate \
  $job "${translate[@]}")

cmds=("judge global-via-en --judge aya" "judge $dh-via-en --judge aya")
for lang in $langs; do
  cmds+=("judge $dh-into-$lang --judge aya" "judge $dh-into-$lang-via-en --judge aya")
done
cmds+=("via-english report global --langs sw fil" "via-english report $dh --langs $all"
  "via-english same-content $dh --langs $langs" "via-english english-prompt $dh"
  "judge $dh-en-prompt --judge aya" "via-english prompt-check $dh")
judge=$(STAGE=judge_primary submit --partition=h200-fullgpu --gres=gpu:h200:1 --time=2:00:00 \
  --dependency=afterok:$trans --job-name=sdt-check-judge $job "${cmds[@]}")

printf '%s\n' "check-translate $trans" "check-judge $judge" | tee runs/.check_jobs
squeue -u "$USER"
