#!/usr/bin/env bash
# Saturated window-training baseline sweep for the AGPT paper.
#
# Uses --backend cublas for GPU speed. The save-stale-weights bug that was
# silently making cublas training effectively train random-init models was
# fixed 2026-04-19 (see docs/known-bugs-cublas-training.md).
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
        --steps $steps --seq-len $sl --backend cublas \
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

run_one 16   10000 3
run_one 32   10000 3
run_one 128  10000 3
run_one 512  10000 3
run_one 1024 10000 3
run_one 2048 10000 8
run_one 4096 10000 8

echo "=== SUMMARY ==="
column -t -s $'\t' "$RESULTS"
