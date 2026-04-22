#!/usr/bin/env bash
# 1-optimizer-step-per-super-epoch baseline: --single-subtree on global radix
# (NOT per-subtree) format. All 65 root-children collapse into one optimizer step.
# Compares against per-root-child baseline (65 steps/SE at lr=3e-3 → 14.59).
#
# Since 1-step/SE is 65× less weight-movement per SE, a much higher LR or
# far more epochs would be needed to match. We test a small LR sweep.

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT" || { echo "Cannot cd to PROJECT_ROOT=$PROJECT_ROOT"; exit 1; }
if [ ! -f Justfile ]; then
    echo "PROJECT_ROOT=$PROJECT_ROOT doesn't look like the microgpt project root" >&2
    exit 1
fi

OUT="rnd/radix-saturation/logs"
mkdir -p "$OUT"
N_RUNS=3
EVAL_POS=16384
INIT_CKPT="data/input.random.model"
TRIE=/tmp/agpt_input_d8_radix  # d=16 global needs 9.5GB KV, doesn't fit; use d=8
BIN="bin/agpt_train_ngram"

GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

eval_ppl () {
    local checkpoint="$1"
    bin/perplexity --model "$checkpoint" --file data/input.txt \
        --max-positions $EVAL_POS --backend cublas 2>&1 \
      | awk '/^Perplexity:/ {print $2}'
}

run_1step () {
    local label="$1"; local lr="$2"; local epochs="$3"
    local cfg="$OUT/${label}.json"
    cat > "$cfg" <<EOF
{
  "label": "$label",
  "tool": "$BIN",
  "trie_dir": "$TRIE",
  "trie_format": "global-radix",
  "init_checkpoint": "$INIT_CKPT",
  "epochs": $epochs,
  "lr": $lr,
  "optimizer": "rmsprop",
  "rmsprop_beta": 0.999,
  "lr_schedule": "constant",
  "single_subtree": true,
  "entropy_lambda": 1.0,
  "mass_weight": "linear",
  "eval_positions": $EVAL_POS,
  "eval_backend": "cublas",
  "n_runs": $N_RUNS,
  "steps_per_super_epoch": 1,
  "git_hash": "$GIT_HASH",
  "timestamp": "$(date -Iseconds)"
}
EOF
    for i in $(seq 1 $N_RUNS); do
        local work="/tmp/1step_${label}_r${i}.model"
        cp "$INIT_CKPT" "$work"
        local log="$OUT/${label}_r${i}.log"
        local t0=$SECONDS
        $BIN \
            --model "$work" \
            --trie-dir "$TRIE" \
            --save "$work" \
            --epochs "$epochs" \
            --lr "$lr" \
            --optimizer rmsprop --rmsprop-beta 0.999 \
            --lr-schedule constant \
            --single-subtree --entropy-lambda 1.0 \
            --mass-weight linear \
            > "$log" 2>&1
        local rc=$?
        local elapsed=$((SECONDS - t0))
        local ppl=$(eval_ppl "$work")
        echo "$label  run $i  PPL = $ppl  time = ${elapsed}s  (rc=$rc)"
    done
}

echo "=============================================================="
echo "1-optimizer-step/super-epoch baseline (global radix + --single-subtree)"
echo "  init: $INIT_CKPT  trie: $TRIE  git: $GIT_HASH"
echo "  baseline ref (d=8, 3 SE, 65 steps/SE lr=3e-3): 17.99 mean"
echo "=============================================================="

# 1 epoch across different LRs. At 1 step/SE, per-SE movement is 65× less
# than baseline, so test aggressive LRs.
run_1step 1step-d8-lr-3e-3-ep1     3e-3   1
run_1step 1step-d8-lr-1e-2-ep1     1e-2   1
run_1step 1step-d8-lr-3e-2-ep1     3e-2   1
run_1step 1step-d8-lr-1e-1-ep1     1e-1   1
run_1step 1step-d8-lr-3e-1-ep1     3e-1   1

echo ""
echo "Done."
