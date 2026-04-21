#!/usr/bin/env bash
# Reproduce the d=16 blending experiment (n=6 per config).
#
# Must run on the `agpt-root-loop` branch (tip ≥ ce4cc3e).
# Requires: bin/agpt_train built with blending support, data/input.agpt.model,
# /tmp/agpt_input_d16_radix_pst (built via `just build-agpt-train` and then
# the --agpt-build-index pipeline in bin/microgpt).

set -eu -o pipefail

RUNS=6
OUT_DIR="$(dirname "$0")/results"
mkdir -p "$OUT_DIR"

for config in baseline blend; do
    echo "=== d=16 per-subtree, $config, $RUNS runs ==="
    for i in $(seq 1 $RUNS); do
        MODEL_TMP=$(mktemp -d)/work.model
        EXTRA=""
        [[ "$config" == "blend" ]] && EXTRA="--blend-alpha 1.0"
        bash scripts/agpt_train_best.sh \
            --model-init data/input.agpt.model \
            --trie-dir /tmp/agpt_input_d16_radix_pst \
            --epochs 3 \
            --save-best "$MODEL_TMP" \
            --eval-positions 16384 \
            -- \
            --lr 3e-3 --optimizer rmsprop --rmsprop-beta 0.999 \
            --lr-schedule warmup-cosine --warmup-epochs 1 \
            --single-subtree --entropy-lambda 1.0 \
            $EXTRA 2>&1 | tee "$OUT_DIR/${config}_run${i}.log" | grep -E "Best:|ep[0-9]: PPL"
        echo "---"
    done
done

echo ""
echo "Logs saved to $OUT_DIR/"
echo "Extract best PPLs with: grep -H 'Best:' $OUT_DIR/*.log"
