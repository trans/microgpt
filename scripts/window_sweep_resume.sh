#!/usr/bin/env bash
# Resume the saturated window-baseline sweep with fixes:
#  - seq_len 16 and 32 added for fair AGPT d=16/32 comparison
#  - MAT_CAP raised (previous 8 GiB was insufficient at seq=2048 10k steps)
set -eu -o pipefail

LOG=/tmp/window_sweep_resume.log
RESULTS=/tmp/window_sweep_results.tsv

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

# Short-context baselines for fair AGPT comparison at d=16/32.
run_one 16   10000 3
run_one 32   10000 3
# Continuation of the long-context run that crashed at cap=8.
run_one 2048 10000 16
run_one 4096 10000 16

echo "=== SUMMARY (full table so far) ==="
column -t -s $'\t' "$RESULTS"
