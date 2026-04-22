# rnd/ — Research / Experiment Directory

Each experiment lives in its own subdirectory:

```
rnd/
  <experiment-name>/
    README.md              # hypothesis, recipe, status, branch pointer
    run.sh                 # reproducible runner (optional)
    results/               # raw logs (optional)
    analysis.py / .md      # post-hoc analysis (optional)
    notes.md               # running notes (optional)
    findings.md            # summary of what we learned (optional)
```

## Convention

- **Code changes** for an experiment go on a **branch** (off main, never merged).
  The rnd/ entry's `README.md` cites the branch name and relevant commit hashes.
- **Data, scripts, notes, writeups** go in the rnd/ subdirectory, on `main`.
- The rnd/ entry persists even after the branch is archived — so the experiment's
  story (hypothesis + results + links to the code at that point in time) lives
  permanently in main.

## Current experiments

| Directory | Status | Code branch | Summary |
|---|---|---|---|
| [convergence](convergence/) | complete | main | Trie path probability convergence across corpus sizes. Paper published. |
| [mass-conservation](mass-conservation/) | complete | main | Suffix tree mass conservation (terminal + cutoff decomposition). |
| [root-loop](root-loop/) | complete — K=2 negative | `agpt-root-loop` | Virtual-tree leaf-to-root training (K>1). Didn't beat K=1 baseline at d=16. |
| [blending](blending/) | complete — d=16 win | `agpt-root-loop` | Suffix-depth blending at radix endpoints. +0.32 PPL at d=16; no help at d=32. |
| [sparsity-profile](sparsity-profile/) | complete | main | Per-depth sparsity profile for d=8/16/32. Cap absorbs 97.9% of chars at d=32; radix endpoints plateau at ~1.67M. |
| [sgd-sanity-check](sgd-sanity-check/) | in progress | main | Compare SGD vs AGPT at matched compute; probe weighting choice. |
| [radix-saturation](radix-saturation/) | in progress | main | From-scratch PPL curve vs. radix-trie saturation. d=8/16/32 → does PPL asymptote as endpoints do? |
