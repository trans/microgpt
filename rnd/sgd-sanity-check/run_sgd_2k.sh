#!/usr/bin/env bash
# Minimal SGD-only sanity run at 2000 steps for seq_len=16 and seq_len=32.
# Mirrors run.sh but only the 2K configs, three runs each.

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT" || { echo "Cannot cd to PROJECT_ROOT=$PROJECT_ROOT"; exit 1; }
if [ ! -f Justfile ]; then
    echo "PROJECT_ROOT=$PROJECT_ROOT doesn't look like the microgpt project root" >&2
    exit 1
fi

set -u -o pipefail

OUT="rnd/sgd-sanity-check/logs"
RES="rnd/sgd-sanity-check/results"
mkdir -p "$OUT" "$RES"
N_RUNS=3
EVAL_POS=16384
INIT_CKPT="data/input.agpt.model"
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

eval_ppl () {
    local checkpoint="$1"
    bin/perplexity --model "$checkpoint" --file data/input.txt \
        --max-positions $EVAL_POS --backend cublas 2>&1 \
      | awk '/^Perplexity:/ {print $2}'
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
        local work="/tmp/sgd_sanity_${label}_r${i}.model"
        cp "$INIT_CKPT" "$work"
        local log="$OUT/${label}_r${i}.log"
        local t0=$SECONDS
        bin/microgpt --file data/input.txt --model "$work" \
            --seq-len "$seq_len" --steps "$steps" \
            --lr 3e-4 --backend cublas \
            > "$log" 2>&1
        local rc=$?
        local elapsed=$((SECONDS - t0))
        local ppl=$(eval_ppl "$work")
        echo "$label  run $i  PPL = $ppl  time = ${elapsed}s  (rc=$rc)"
    done
}

echo "=== SGD 2K sanity (s16 and s32) ==="
run_sgd sgd-s16-2000 16 2000
run_sgd sgd-s32-2000 32 2000
echo "Done."
