#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_FILE="${1:-data/input.txt}"
STEPS="${STEPS:-50}"
BACKEND="${BACKEND:-crystal}"
SEQ_LEN="${SEQ_LEN:-128}"
D_MODEL="${D_MODEL:-64}"
N_LAYERS="${N_LAYERS:-2}"
AGPT_STARTS="${AGPT_STARTS:-20000}"
AGPT_OFFSET="${AGPT_OFFSET:-0}"
AGPT_PROGRESS="${AGPT_PROGRESS:-0}"
SEED="${SEED:-1234}"
KEEP_INDEX="${KEEP_INDEX:-0}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

WINDOW_OUT="$TMP_DIR/window.out"
AGPT_BUILD_OUT="$TMP_DIR/agpt_build.out"
AGPT_TRAIN_OUT="$TMP_DIR/agpt_train.out"
WINDOW_MODEL="$TMP_DIR/window.model"
AGPT_MODEL="$TMP_DIR/agpt.model"
AGPT_INDEX="$TMP_DIR/agpt.index"

if [[ ! -x bin/microgpt ]]; then
  echo "bin/microgpt not found; build it first." >&2
  exit 1
fi

now_ms() {
  python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
}

elapsed_ms() {
  local start_ms="$1"
  local end_ms="$2"
  echo $((end_ms - start_ms))
}

extract_loss() {
  local file="$1"
  grep "Final avg loss" "$file" | tail -n 1 | awk '{print $4}'
}

extract_generation() {
  local file="$1"
  awk '
    /^Final generation:/ {capture=1; next}
    capture && NF { print; exit }
  ' "$file"
}

extract_build_ms() {
  local file="$1"
  grep "Trie build:" "$file" | tail -n 1 | awk '{print $3}'
}

extract_load_ms() {
  local file="$1"
  grep "Trie load:" "$file" | tail -n 1 | awk '{print $3}'
}

echo "Building AGPT index..."
agpt_build_start="$(now_ms)"
bin/microgpt "$DATA_FILE" \
  --agpt \
  --backend "$BACKEND" \
  --seed "$SEED" \
  --seq-len "$SEQ_LEN" \
  --d-model "$D_MODEL" \
  --n-layers "$N_LAYERS" \
  --no-save \
  --build-only \
  --agpt-max-starts "$AGPT_STARTS" \
  --agpt-start-offset "$AGPT_OFFSET" \
  --agpt-progress "$AGPT_PROGRESS" \
  --agpt-save-index "$AGPT_INDEX" \
  --model "$AGPT_MODEL" > "$AGPT_BUILD_OUT" 2>&1
agpt_build_end="$(now_ms)"

echo "Running window baseline..."
window_start="$(now_ms)"
bin/microgpt "$DATA_FILE" \
  --backend "$BACKEND" \
  --seed "$SEED" \
  --seq-len "$SEQ_LEN" \
  --d-model "$D_MODEL" \
  --n-layers "$N_LAYERS" \
  --steps "$STEPS" \
  --no-save \
  --model "$WINDOW_MODEL" > "$WINDOW_OUT" 2>&1
window_end="$(now_ms)"

echo "Running AGPT training..."
agpt_train_start="$(now_ms)"
bin/microgpt "$DATA_FILE" \
  --agpt \
  --backend "$BACKEND" \
  --seed "$SEED" \
  --seq-len "$SEQ_LEN" \
  --d-model "$D_MODEL" \
  --n-layers "$N_LAYERS" \
  --steps "$STEPS" \
  --no-save \
  --agpt-load-index "$AGPT_INDEX" \
  --model "$AGPT_MODEL" > "$AGPT_TRAIN_OUT" 2>&1
agpt_train_end="$(now_ms)"

window_elapsed="$(elapsed_ms "$window_start" "$window_end")"
agpt_build_elapsed="$(elapsed_ms "$agpt_build_start" "$agpt_build_end")"
agpt_train_elapsed="$(elapsed_ms "$agpt_train_start" "$agpt_train_end")"

window_loss="$(extract_loss "$WINDOW_OUT")"
agpt_loss="$(extract_loss "$AGPT_TRAIN_OUT")"
window_gen="$(extract_generation "$WINDOW_OUT")"
agpt_gen="$(extract_generation "$AGPT_TRAIN_OUT")"
agpt_build_ms="$(extract_build_ms "$AGPT_BUILD_OUT")"
agpt_load_ms="$(extract_load_ms "$AGPT_TRAIN_OUT")"
agpt_index_size="$(du -h "$AGPT_INDEX" | awk '{print $1}')"

printf "\n%-12s %-8s %-12s %-12s %-12s\n" "Mode" "Steps" "Loss" "Wall ms" "Index"
printf "%-12s %-8s %-12s %-12s %-12s\n" "-----------" "--------" "------------" "------------" "------------"
printf "%-12s %-8s %-12s %-12s %-12s\n" "window" "$STEPS" "${window_loss:-n/a}" "$window_elapsed" "-"
printf "%-12s %-8s %-12s %-12s %-12s\n" "agpt" "$STEPS" "${agpt_loss:-n/a}" "$agpt_train_elapsed" "$agpt_index_size"

printf "\n%-20s %s\n" "AGPT build wall ms" "$agpt_build_elapsed"
printf "%-20s %s\n" "AGPT reported build" "${agpt_build_ms:-n/a}"
printf "%-20s %s\n" "AGPT reported load" "${agpt_load_ms:-n/a}"
printf "%-20s %s\n" "Data file" "$DATA_FILE"
printf "%-20s %s\n" "Backend" "$BACKEND"
printf "%-20s %s\n" "Seed" "$SEED"
printf "%-20s %s\n" "Seq len" "$SEQ_LEN"
printf "%-20s %s\n" "AGPT starts" "$AGPT_STARTS"
printf "%-20s %s\n" "AGPT offset" "$AGPT_OFFSET"

printf "\nWindow sample:\n%s\n" "${window_gen:-<none>}"
printf "\nAGPT sample:\n%s\n" "${agpt_gen:-<none>}"

if [[ "$KEEP_INDEX" == "1" ]]; then
  KEEP_PATH="$ROOT_DIR/agpt.compare.index"
  cp "$AGPT_INDEX" "$KEEP_PATH"
  echo
  echo "Kept AGPT index at $KEEP_PATH"
fi
