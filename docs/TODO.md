# TODO

## AGPT

- Replace the current `--agpt-max-starts` prefix-of-file safety cap with a full-corpus strategy.
- Candidate directions:
  - distribute capped starts across the whole corpus instead of taking only the first `N`
  - support randomized or epoch-rotated start subsets
  - remove the need for the cap with a more compact prefix/suffix index
