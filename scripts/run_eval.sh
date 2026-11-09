#!/bin/bash
# One lm-eval run of LLaDA (LLaDA-8B-Instruct or LLaDA-1.5) with any decoder of the paper.
# Usage: bash scripts/run_eval.sh <run_name> <task> <num_fewshot> <model_args> [extra lm_eval args, e.g. --limit 20]
# Results go to results/<run_name>/ (lm-eval results, per-question samples, speed.jsonl, nfe.jsonl).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
NAME=$1; TASK=$2; SHOTS=$3; MARGS=$4; shift 4
OUT=$REPO/results/$NAME
mkdir -p "$OUT"
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_METRICS_CACHE=$OUT/.metrics   # per-run cache: code_eval writes a temp file at a fixed path
cd "$REPO/measured_coupling/llada"
python eval_llada.py --tasks "$TASK" --num_fewshot "$SHOTS" --batch_size 1 \
  --confirm_run_unsafe_code --model llada_dist \
  --model_args "${MARGS},show_speed=True,outp_path=$OUT/speed.jsonl" \
  --output_path "$OUT" --log_samples "$@"
