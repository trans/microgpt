#!/usr/bin/env bash
# Phase 2: deterministic per-root-child baseline at d=32 (post Wk/Wv fix).
# Same structure as Phase 1 at d=16.

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT" || { echo "Cannot cd to PROJECT_ROOT=$PROJECT_ROOT"; exit 1; }
if [ ! -f Justfile ]; then
    echo "PROJECT_ROOT=$PROJECT_ROOT doesn't look like the microgpt project root" >&2
    exit 1
fi

OUT="rnd/post-fix-baseline/logs"
mkdir -p "$OUT"
N_RUNS=3
EVAL_POS=16384
INIT_CKPT="data/input.random.model"
TRIE=/tmp/agpt_input_d32_radix
BIN="bin/agpt_train"
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

eval_ppl () {
    bin/perplexity --model "$1" --file data/input.txt \
        --max-positions $EVAL_POS --backend cublas 2>&1 \
      | awk '/^Perplexity:/ {print $2}'
}

run_det () {
    local label="$1"; local lr="$2"; local sched="$3"; local warm="$4"; local wd="$5"
    for i in $(seq 1 $N_RUNS); do
        local work="/tmp/pf_d32_${label}_r${i}.model"
        cp "$INIT_CKPT" "$work"
        local log="$OUT/${label}_r${i}.log"
        local t0=$SECONDS
        local warm_arg=""
        if [ "$warm" != "0" ]; then warm_arg="--warmup-epochs $warm"; fi
        local wd_arg=""
        if [ "$wd" != "0" ]; then wd_arg="--weight-decay $wd"; fi
        $BIN --model "$work" --trie-dir "$TRIE" --save "$work" \
            --epochs 3 --lr "$lr" \
            --optimizer rmsprop --rmsprop-beta 0.999 \
            --lr-schedule "$sched" $warm_arg $wd_arg \
            --entropy-lambda 1.0 --mass-weight linear \
            --no-accumulate \
            > "$log" 2>&1
        local rc=$?
        local elapsed=$((SECONDS - t0))
        local ppl=$(eval_ppl "$work")
        echo "$label  run $i  PPL = $ppl  time = ${elapsed}s  (rc=$rc)"
    done
}

echo "=============================================================="
echo "Phase 2: post-fix d=32 deterministic baseline (3 SE × 65 steps)"
echo "  init: $INIT_CKPT  trie: $TRIE  git: $GIT_HASH"
echo "  pre-fix paper reference: 13.17  |  pre-fix linear-mass: 13.36"
echo "=============================================================="

run_det d32-det-const-lr3e-3         3e-3  constant       0 0
run_det d32-det-const-lr1e-3         1e-3  constant       0 0
run_det d32-det-wc-lr3e-3            3e-3  warmup-cosine  1 0
run_det d32-det-wc-lr1e-3            1e-3  warmup-cosine  1 0
run_det d32-det-wc-lr3e-4-wd0.01     3e-4  warmup-cosine  1 0.01
run_det d32-det-wc-lr1e-3-wd0.01     1e-3  warmup-cosine  1 0.01

echo ""
echo "Done."
