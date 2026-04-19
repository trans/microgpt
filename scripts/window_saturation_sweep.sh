#!/usr/bin/env bash
# Saturated window-training baseline sweep for the AGPT paper.
#
# IMPORTANT: uses --backend openblas, not cublas. The cublas train_step path
# in bin/microgpt produces a loss that goes down but a model whose generation
# output is gibberish and whose held-out PPL is worse than random (≈170 on
# a vocab=65 corpus). See docs/known-bugs-cublas-training.md. Openblas is
# slower (CPU) but correct — the paper's window baselines must use it.
#
# Trains the same 108k-param model at seq_len ∈ {128, 512, 1024, 2048, 4096}
# for 10k steps each (adjusted down at longer contexts to fit hardware budget),
# then evaluates held-out PPL on data/input.txt tail.
#
# Writes results to /tmp/window_sweep_results.tsv.
set -eu -o pipefail

LOG=/tmp/window_sweep.log
RESULTS=/tmp/window_sweep_results.tsv
echo -e "seq_len\tsteps\ttrain_time_sec\tfinal_loss\tppl" > "$RESULTS"

run_one () {
    local sl=$1
    local steps=$2
    local cap=$3
    local model="/tmp/wsw_sl${sl}.model"
    rm -f "$model"

    echo "=== seq_len=$sl, $steps steps, cap=${cap}GiB ===" | tee -a "$LOG"
    local start=$(date +%s)
    MICROGPT_MAT_CAP_GIB=$cap bin/microgpt --file data/input.txt \
        --steps $steps --seq-len $sl --backend openblas \
        --model "$model" --seed 42 2>&1 | tail -3 | tee -a "$LOG"
    local train_time=$(($(date +%s) - start))
    echo "train time: ${train_time}s" | tee -a "$LOG"

    local final_loss=$(grep "^Final avg loss:" "$LOG" | tail -1 | awk '{print $4}')

    echo "--- measuring PPL ---" | tee -a "$LOG"
    local ppl=$(bin/perplexity --model "$model" --file data/input.txt \
                --max-positions 32768 --backend openblas 2>&1 \
              | awk '/^Perplexity:/ {print $2}')
    echo "seq_len=$sl PPL=$ppl" | tee -a "$LOG"

    echo -e "${sl}\t${steps}\t${train_time}\t${final_loss}\t${ppl}" >> "$RESULTS"
    echo
}

run_one 128  10000 3
run_one 512  10000 3
run_one 1024 10000 3
run_one 2048 10000 8
run_one 4096 10000 8

echo "=== SUMMARY ==="
column -t -s $'\t' "$RESULTS"
