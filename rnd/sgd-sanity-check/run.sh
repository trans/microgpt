#!/usr/bin/env bash
# Matrix runner for SGD vs. AGPT sanity-check.
#
# Runs each config 3× and captures PPL. Logs per-config in logs/.
# Assumes the following already exist:
#   bin/microgpt  (built via `just build` or `just build-cuda`)
#   bin/agpt_train  (built via `just build-agpt-train`)
#   bin/perplexity  (built via `just build-perplexity`)
#   /tmp/agpt_input_d16_radix_pst  (per-subtree trie)
#   /tmp/agpt_input_d32_per        (per-subtree trie)
#   data/input.agpt.model          (shared init checkpoint)

set -eu -o pipefail

cd "$(dirname "$0")/.."  # project root
OUT="rnd/sgd-sanity-check/logs"
RES="rnd/sgd-sanity-check/results"
mkdir -p "$OUT" "$RES"
N_RUNS=3
EVAL_POS=16384

eval_ppl () {
    local checkpoint="$1"
    bin/perplexity --model "$checkpoint" --file data/input.txt \
        --max-positions $EVAL_POS --backend openblas 2>&1 \
      | awk '/^Perplexity:/ {print $2}'
}

run_agpt () {
    local label="$1"; local trie="$2"; shift 2
    for i in $(seq 1 $N_RUNS); do
        local ckpt="/tmp/sgd_sanity_${label}_r${i}.model"
        cp data/input.agpt.model "$ckpt.work"
        local log="$OUT/${label}_r${i}.log"
        bin/agpt_train --model "$ckpt.work" --trie-dir "$trie" \
            --epochs 3 --lr 3e-3 --optimizer rmsprop --rmsprop-beta 0.999 \
            --lr-schedule warmup-cosine --warmup-epochs 1 \
            --single-subtree --entropy-lambda 1.0 \
            --save "$ckpt.work" \
            "$@" > "$log" 2>&1
        local ppl=$(eval_ppl "$ckpt.work")
        echo "$label  run $i  PPL = $ppl"
    done
}

run_sgd () {
    local label="$1"; local seq_len="$2"; local steps="$3"
    for i in $(seq 1 $N_RUNS); do
        local ckpt="/tmp/sgd_sanity_${label}_r${i}.model"
        local log="$OUT/${label}_r${i}.log"
        # microgpt's plain window training (no --agpt)
        bin/microgpt --file data/input.txt \
            --seq-len "$seq_len" --steps "$steps" \
            --d-model 64 --n-heads 4 --n-layers 2 --d-ff 256 \
            --lr 3e-3 --backend openblas \
            --save "$ckpt" > "$log" 2>&1
        local ppl=$(eval_ppl "$ckpt")
        echo "$label  run $i  PPL = $ppl"
    done
}

echo "=== SGD at seq_len=16, varying step budgets ==="
run_sgd sgd-s16-195   16  195
run_sgd sgd-s16-1000  16  1000
run_sgd sgd-s16-5000  16  5000
run_sgd sgd-s16-20000 16  20000

echo ""
echo "=== SGD at seq_len=128, varying step budgets ==="
run_sgd sgd-s128-195   128 195
run_sgd sgd-s128-1000  128 1000
run_sgd sgd-s128-5000  128 5000

echo ""
echo "=== AGPT d=16 baseline ==="
run_agpt agpt-d16       /tmp/agpt_input_d16_radix_pst
run_agpt agpt-d16-mass  /tmp/agpt_input_d16_radix_pst --mass-weight

echo ""
echo "=== AGPT d=32 baseline ==="
run_agpt agpt-d32       /tmp/agpt_input_d32_per
run_agpt agpt-d32-mass  /tmp/agpt_input_d32_per --mass-weight

echo ""
echo "Done. Logs in $OUT/. Summarize with:"
echo "  grep -H 'Best:\\|PPL' $OUT/*.log"
