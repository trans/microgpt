#!/usr/bin/env bash
# LR sweep for partition-depth=2 (bigram) at d=16, from random init.
# Baseline: partition-depth=1 at lr=3e-3 → 14.59 mean (from radix-saturation).
# Goal: find the LR where bigram can beat per-root-child.

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
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
BIN="bin/agpt_train_ngram"

if [ ! -x "$BIN" ]; then
    echo "Missing $BIN. Build it from the agpt-ngram-partition branch." >&2
    exit 1
fi

eval_ppl () {
    local checkpoint="$1"
    bin/perplexity --model "$checkpoint" --file data/input.txt \
        --max-positions $EVAL_POS --backend cublas 2>&1 \
      | awk '/^Perplexity:/ {print $2}'
}

# Use agpt_train_best-style best-of-epochs selection to match the rest of
# the experiment. We write a temporary wrapper that tracks each epoch's
# PPL and picks the best.
run_lr () {
    local label="$1"; local partition="$2"; local lr="$3"; local sched="$4"
    local cfg="$OUT/${label}.json"
    cat > "$cfg" <<EOF
{
  "label": "$label",
  "tool": "$BIN",
  "trie_dir": "/tmp/agpt_input_d16_radix_pst",
  "init_checkpoint": "$INIT_CKPT",
  "partition_depth": $partition,
  "epochs": 3,
  "lr": $lr,
  "lr_schedule": "$sched",
  "warmup_epochs": 1,
  "optimizer": "rmsprop",
  "rmsprop_beta": 0.999,
  "single_subtree": true,
  "entropy_lambda": 1.0,
  "mass_weight": "linear",
  "eval_positions": $EVAL_POS,
  "eval_backend": "cublas",
  "n_runs": $N_RUNS,
  "git_hash": "$GIT_HASH",
  "timestamp": "$(date -Iseconds)"
}
EOF
    for i in $(seq 1 $N_RUNS); do
        local work="/tmp/lrsweep_${label}_r${i}.model"
        local log="$OUT/${label}_r${i}.log"
        cp "$INIT_CKPT" "$work"
        local t0=$SECONDS
        $BIN \
            --model "$work" \
            --trie-dir /tmp/agpt_input_d16_radix_pst \
            --epochs 3 \
            --save-every 1 \
            --save "$work" \
            --lr "$lr" \
            --optimizer rmsprop --rmsprop-beta 0.999 \
            --lr-schedule "$sched" --warmup-epochs 1 \
            --single-subtree --entropy-lambda 1.0 \
            --mass-weight linear \
            --partition-depth "$partition" \
            > "$log" 2>&1
        local rc=$?
        local elapsed=$((SECONDS - t0))
        # Evaluate each per-epoch checkpoint and pick the best.
        local best_ppl=""
        local best_ep=""
        for ep in 1 2 3; do
            local ckpt="${work}.ep${ep}"
            [ -f "$ckpt" ] || continue
            local p=$(eval_ppl "$ckpt")
            echo "  $label r${i} ep${ep}: PPL = $p" >> "$log"
            if [ -z "$best_ppl" ] || awk "BEGIN {exit !($p < $best_ppl)}"; then
                best_ppl="$p"
                best_ep="$ep"
            fi
        done
        echo "$label  run $i  best PPL = $best_ppl (ep${best_ep})  time = ${elapsed}s  (rc=$rc)"
    done
}

echo "=============================================================="
echo "N-gram partition LR sweep (d=16, partition-depth=2, 3 super-epochs)"
echo "  baseline: partition-depth=1 lr=3e-3 warmup-cosine → 14.59 mean"
echo "  git: $GIT_HASH"
echo "=============================================================="

echo ""
echo "-- partition-depth=2 LR sweep (constant schedule) --"
run_lr p2-lr-3e-5    2 3e-5 constant
run_lr p2-lr-1e-4    2 1e-4 constant
run_lr p2-lr-3e-4    2 3e-4 constant
run_lr p2-lr-1e-3    2 1e-3 constant

echo ""
echo "Done."
