#!/bin/bash
# One lm-eval run of Dream-v0-Instruct-7B with any decoder of the paper.
# Usage: bash scripts/run_eval_dream.sh <run_name> <task> <num_fewshot> <model_args> [extra lm_eval args]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
NAME=$1; TASK=$2; SHOTS=$3; MARGS=$4; shift 4
OUT=$REPO/results/$NAME
mkdir -p "$OUT"
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_METRICS_CACHE=$OUT/.metrics
cd "$REPO/measured_coupling/dream"
python eval.py --tasks "$TASK" --num_fewshot "$SHOTS" --batch_size 1 \
  --confirm_run_unsafe_code --model dream \
  --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,${MARGS},show_speed=True,outp_path=$OUT/speed.jsonl" \
  --output_path "$OUT" --log_samples "$@"
