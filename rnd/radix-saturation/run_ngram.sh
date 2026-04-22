#!/usr/bin/env bash
# N-gram partition comparison at d=16.
# Compares: partition_depth=1 (per-root-child, 65 steps/SE) vs
#           partition_depth=2 (bigram, ~1139 steps/SE).
# Uses bin/agpt_train_ngram (agpt-ngram-partition branch) so it can run
# alongside the main sweep without disturbing bin/agpt_train.

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
    echo "Missing $BIN. Build it from the agpt-ngram-partition branch:" >&2
    echo "  /opt/cuda/bin/nvcc -O2 src/cuda/agpt_train.cu src/cuda/kernels.cu -lcublas -o $BIN" >&2
    exit 1
fi

eval_ppl () {
    local checkpoint="$1"
    bin/perplexity --model "$checkpoint" --file data/input.txt \
        --max-positions $EVAL_POS --backend cublas 2>&1 \
      | awk '/^Perplexity:/ {print $2}'
}

run_ngram () {
    local label="$1"; local trie="$2"; local partition="$3"; local epochs="$4"
    local lr="$5"; local sched="$6"
    local cfg="$OUT/${label}.json"
    cat > "$cfg" <<EOF
{
  "label": "$label",
  "tool": "$BIN",
  "trie_dir": "$trie",
  "init_checkpoint": "$INIT_CKPT",
  "partition_depth": $partition,
  "epochs": $epochs,
  "lr": $lr,
  "optimizer": "rmsprop",
  "rmsprop_beta": 0.999,
  "lr_schedule": "$sched",
  "warmup_epochs": 1,
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
        local work="/tmp/ngram_${label}_r${i}.model"
        cp "$INIT_CKPT" "$work"
        local log="$OUT/${label}_r${i}.log"
        local t0=$SECONDS
        $BIN \
            --model "$work" \
            --trie-dir "$trie" \
            --epochs "$epochs" \
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
        local ppl=$(eval_ppl "$work")
        echo "$label  run $i  PPL = $ppl  time = ${elapsed}s  (rc=$rc)"
    done
}

echo "=============================================================="
echo "N-gram partition comparison"
echo "  init: $INIT_CKPT"
echo "  bin:  $BIN"
echo "  git:  $GIT_HASH"
echo "=============================================================="

# At partition-depth > 1 the warmup-cosine horizon is undercounted
# (see commit note). Use constant LR for bigram/trigram to isolate
# the partition effect.
echo ""
echo "-- d=16 partition-depth=1 (baseline; 65 steps/SE) --"
run_ngram sat-d16-p1 /tmp/agpt_input_d16_radix_pst 1 3 3e-3 warmup-cosine

echo ""
echo "-- d=16 partition-depth=2 (bigram; ~1139 steps/SE) --"
# Lower LR because more Adam steps / same total "gradient movement".
# 3e-3 / 17 ≈ 1.7e-4; start at 3e-4 for a first pass, tune if needed.
run_ngram sat-d16-p2 /tmp/agpt_input_d16_radix_pst 2 3 3e-4 constant

echo ""
echo "Done."
