#!/usr/bin/env bash
# Radix saturation vs. PPL — from-scratch (random-init) sweep.
# AGPT at d=8/16/32 with --mass-weight linear, plus SGD at matched
# seq_len. Measures convergence from data/input.random.model.

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT" || { echo "Cannot cd to PROJECT_ROOT=$PROJECT_ROOT"; exit 1; }
if [ ! -f Justfile ]; then
    echo "PROJECT_ROOT=$PROJECT_ROOT doesn't look like the microgpt project root" >&2
    exit 1
fi

OUT="rnd/radix-saturation/logs"
RES="rnd/radix-saturation/results"
mkdir -p "$OUT" "$RES"
N_RUNS=3
EVAL_POS=16384
INIT_CKPT="data/input.random.model"
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

if [ ! -f "$INIT_CKPT" ]; then
    echo "Missing $INIT_CKPT. Build it first:" >&2
    echo "  bin/microgpt --file data/input.txt --model $INIT_CKPT --steps 1 --lr 0 \\" >&2
    echo "      --seed 42 --d-model 64 --n-layers 2 --seq-len 128 --backend cublas" >&2
    exit 1
fi

eval_ppl () {
    local checkpoint="$1"
    bin/perplexity --model "$checkpoint" --file data/input.txt \
        --max-positions $EVAL_POS --backend cublas 2>&1 \
      | awk '/^Perplexity:/ {print $2}'
}

write_cfg () {
    local cfg="$1"
    shift
    cat > "$cfg" <<EOF
$@
EOF
}

run_agpt () {
    local label="$1"; local trie="$2"; local mode="$3"; local epochs="$4"
    local cfg="$OUT/${label}.json"
    cat > "$cfg" <<EOF
{
  "label": "$label",
  "tool": "bin/agpt_train (via scripts/agpt_train_best.sh)",
  "trie_dir": "$trie",
  "init_checkpoint": "$INIT_CKPT",
  "epochs": $epochs,
  "lr": 3e-3,
  "optimizer": "rmsprop",
  "rmsprop_beta": 0.999,
  "lr_schedule": "warmup-cosine",
  "warmup_epochs": 1,
  "single_subtree": true,
  "entropy_lambda": 1.0,
  "mass_weight": "$mode",
  "eval_positions": $EVAL_POS,
  "eval_backend": "cublas",
  "n_runs": $N_RUNS,
  "git_hash": "$GIT_HASH",
  "timestamp": "$(date -Iseconds)"
}
EOF
    local flag=""
    [[ "$mode" != "off" ]] && flag="--mass-weight $mode"
    for i in $(seq 1 $N_RUNS); do
        local log="$OUT/${label}_r${i}.log"
        local save="/tmp/sat_${label}_r${i}.model"
        local t0=$SECONDS
        bash scripts/agpt_train_best.sh \
            --model-init "$INIT_CKPT" \
            --trie-dir "$trie" \
            --epochs "$epochs" \
            --save-best "$save" \
            --eval-positions $EVAL_POS \
            -- \
            --lr 3e-3 --optimizer rmsprop --rmsprop-beta 0.999 \
            --lr-schedule warmup-cosine --warmup-epochs 1 \
            --single-subtree --entropy-lambda 1.0 \
            $flag > "$log" 2>&1
        local rc=$?
        local elapsed=$((SECONDS - t0))
        local best_ppl
        best_ppl=$(grep -E "^=== Best:" "$log" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')
        if [[ -z "$best_ppl" ]]; then
            echo "$label  run $i  FAILED (rc=$rc, ${elapsed}s) — see $log"
        else
            echo "$label  run $i  best PPL = $best_ppl  time = ${elapsed}s  (rc=$rc)"
        fi
    done
}

run_sgd () {
    local label="$1"; local seq_len="$2"; local steps="$3"
    local cfg="$OUT/${label}.json"
    cat > "$cfg" <<EOF
{
  "label": "$label",
  "tool": "bin/microgpt",
  "init_checkpoint": "$INIT_CKPT",
  "seq_len": $seq_len,
  "steps": $steps,
  "lr": 3e-3,
  "eval_positions": $EVAL_POS,
  "eval_backend": "cublas",
  "n_runs": $N_RUNS,
  "git_hash": "$GIT_HASH",
  "timestamp": "$(date -Iseconds)"
}
EOF
    for i in $(seq 1 $N_RUNS); do
        local work="/tmp/sat_${label}_r${i}.model"
        cp "$INIT_CKPT" "$work"
        local log="$OUT/${label}_r${i}.log"
        local t0=$SECONDS
        bin/microgpt --file data/input.txt --model "$work" \
            --seq-len "$seq_len" --steps "$steps" \
            --lr 3e-3 --backend cublas \
            > "$log" 2>&1
        local rc=$?
        local elapsed=$((SECONDS - t0))
        local ppl=$(eval_ppl "$work")
        echo "$label  run $i  PPL = $ppl  time = ${elapsed}s  (rc=$rc)"
    done
}

echo "=============================================================="
echo "Radix saturation vs PPL — from-scratch"
echo "  init: $INIT_CKPT"
echo "  git:  $GIT_HASH"
echo "=============================================================="

echo ""
echo "-- AGPT linear, 10 super-epochs --"
run_agpt sat-d8-linear-ep3   /tmp/agpt_input_d8_radix_pst   linear 3
run_agpt sat-d16-linear-ep3  /tmp/agpt_input_d16_radix_pst  linear 3
run_agpt sat-d32-linear-ep3  /tmp/agpt_input_d32_per        linear 3

echo ""
echo "-- SGD at 50k steps (roughly matched wall-clock) --"
run_sgd sat-sgd-s16-50k 16 50000
run_sgd sat-sgd-s32-50k 32 50000

echo ""
echo "All done. Summarize:"
echo "  grep -H 'PPL' $OUT/sat-*_r*.log | head"
