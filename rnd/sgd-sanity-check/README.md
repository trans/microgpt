# SGD vs AGPT Sanity Check

**Status**: in progress.

**Code**: main (no new code needed — `bin/microgpt` without `--agpt` does
SGD window training; `bin/agpt_train` does AGPT). Any new flags we add
would go on a branch.

## Hypothesis

If the D-trie captures the corpus's conditional distributions faithfully, and
AGPT's gradient aggregation is mathematically equivalent to "sum the gradient
contributions from all corpus positions sharing the same context," then AGPT
and standard SGD over corpus positions should converge to **similar held-out
PPL at matched compute budget**.

If AGPT is *significantly better* at matched compute: the trie aggregation is
adding something (prefix-sharing speedup, or the equal-per-context weighting is
actually beneficial).

If SGD is better: either there's a bug in AGPT's gradient aggregation OR the
equal-per-context weighting is actively hurting us and we should use
`--mass-weight`.

If they match: AGPT is behaving as specified — the aggregation is mathematically
sound. This is the primary sanity check.

## The weighting question (why this isn't just a re-run)

SGD with uniform corpus-position sampling trains in proportion to corpus
frequency — common D-grams get sampled often, rare ones rarely. Each gradient
step is weighted implicitly by `f(context)`.

AGPT's default behavior is **equal weight per radix node**, regardless of how
many corpus positions that node represents. So:
- `AGPT-default`: sums `∇L` across unique contexts with equal weight each.
- `AGPT --mass-weight`: weights by `log(1 + edge_mass)` — partial frequency
  restoration.
- `SGD`: weights proportional to corpus frequency (linear in count).

These are THREE different effective loss functions, even though they're
training the same model on the same corpus. Comparing them cleanly is part
of this experiment.

## What we'll measure

1. **Raw held-out PPL** at each config, matched at:
   - Same wall-clock time.
   - Same number of optimizer steps.
   - Same number of gradient samples (corpus positions covered).
2. **Convergence speed** (PPL vs. training time curve).
3. **Sensitivity to weighting scheme** (AGPT default vs. AGPT mass-weighted vs. SGD uniform).

## Configs

All runs use the same model architecture:
- d_model=64, n_heads=4, n_layers=2, d_ff=256, head_dim=16, vocab_size=65.
- lr=3e-3, rmsprop (β=0.999), warmup-cosine schedule.

### Variants to compare

| Label | Tool | Mode | Context length |
|---|---|---|---|
| sgd-s16 | `bin/microgpt` | plain window training | seq_len=16 |
| sgd-s128 | `bin/microgpt` | plain window training | seq_len=128 |
| agpt-d16 | `bin/agpt_train` | AGPT default | d=16 |
| agpt-d16-mass | `bin/agpt_train` | AGPT + `--mass-weight` | d=16 |
| agpt-d32 | `bin/agpt_train` | AGPT default | d=32 |
| agpt-d32-mass | `bin/agpt_train` | AGPT + `--mass-weight` | d=32 |

### Matched-compute scheme

AGPT d=16 at 3 super-epochs = 195 optimizer steps. Baseline.
AGPT d=32 at 3 super-epochs = 195 optimizer steps.
SGD will be run at various step budgets (195, 1000, 5000, 20000) to see where
its PPL curve sits relative to AGPT's.

## Files

- `run.sh` — runs the matrix of configs.
- `logs/` — raw training output per config.
- `results/summary.md` — condensed PPL comparison.

## Decisions made (documentable so we don't re-litigate)

- **Training data**: `data/input.txt` (Shakespeare, 1.11M chars).
- **Eval**: `bin/perplexity --file data/input.txt --max-positions 16384 --backend openblas`.
  Uses the same file for eval — relies on perplexity tool's held-out-by-default
  convention. (Worth double-checking: is the eval really held-out?)
- **Model init**: fresh weights per config (not loaded from a common checkpoint)
  — otherwise we'd be measuring fine-tuning dynamics, not the training-from-scratch
  behavior we want.
- **Single-subtree mode**: for AGPT (matches the memory's winning recipe).
