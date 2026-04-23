# Post-Fix AGPT Baseline Re-establishment

**Context**: Commit `1c858c0` fixed a fundamental bug where Wk, Wv, and all
7 biases (wq_b, wk_b, wv_b, wo_b, l1_b, l2_b, out_b) had been silently
frozen at random initialization across the entire project's training
history. AGPT was effectively a "random-features-attention" architecture.

With attention now fully trainable, every prior hyperparameter — learning
rate, weight decay, warmup, mass-weight, sampler choice, super-epochs —
may have shifted. We need fresh baselines before building new features on
top.

## Reference pre-fix numbers (for sanity comparison)

At matched Lightning config (3 SE × 260 samples, L3 p_stop=0.3, mass=linear):

| depth | pre-fix PPL | pre-fix LR |
|---|---|---|
| d=16 | 15.38 | 2e-4 |
| d=32 | 12.07 | 2e-4 |

Note: the pre-fix model had frozen attention weights — those numbers are
**not a correctness target**. We aim to equal or beat them with a
correctly-trained model, which may require different LR/regularization.

## Preliminary post-fix findings (single runs)

| config | PPL |
|---|---|
| d=16 wc-lr3e-4-wd0.01 3SE | 15.08 (beats pre-fix 15.38) |
| d=16 const-lr1e-4-wd0.01 3SE | 15.42 |
| d=32 wc-lr3e-4-wd0.01 3SE | 12.86 |
| d=32 wc-lr2e-4-wd0.01 3SE | 13.20 |
| d=32 wc-lr3e-4-wd0.01 6SE | 12.93 (plateaued) |

d=32 hasn't reached pre-fix 12.07 yet. LR and schedule need more sweep.

## Plan

### Phase 1: deterministic d=16 baseline

Paper-style per-root-child (3 SE × 65 steps, `--no-accumulate`). Sweep lr
and weight-decay. Reference: paper's pre-fix 15.28.

### Phase 2: deterministic d=32 baseline

Same, at d=32 (paper's pre-fix 13.17 / 13.36).

### Phase 3: Lightning vs deterministic

Pick best det-LR from phases 1-2, then run Lightning at matched budgets
(260, 650 samples/SE). Find Lightning's best lr.

### Phase 4: mass-weight re-check

pre-fix: `--mass-weight linear` won everywhere. Re-sweep {off, log, sqrt,
linear} at best lr for both d=16 and d=32. With trainable K/V the winner
may shift.
