# Suffix Tree Mass Conservation Notes

## Problem Context

Given a corpus (e.g. `"ABACACB"`), we construct a suffix tree by inserting all suffixes:

```
ABACACB
BACACB
ACACB
CACB
ACB
CB
B
```

Each node represents a prefix, and edges carry frequency counts.

(In our codebase this is realized as a *prefix* trie built from
sliding-start positions — when the start range covers every position
in the corpus, the result is a suffix tree of the corpus.)

---

## Key Observation

Naively, one might expect:

    incoming mass = sum(outgoing edge mass)

This is **not generally true** for a finite corpus.

### Why?

Because some suffixes **terminate at nodes** without continuing.

Example:

- `"B"` appears twice:
  - `"BACACB"` → contributes to edge `B → A`
  - `"B"` → terminates at `B`

So:

    count(B) = 2
    outgoing(B) = 1
    missing mass = 1 (termination)

---

## Correct Conservation Law

For any node `v`:

    count(v) = sum(child_edge_counts) + terminal_count(v)

Where:

- `count(v)` = number of suffixes passing through node `v`
- `child_edge_counts` = frequencies of outgoing edges
- `terminal_count(v)` = number of suffixes ending exactly at `v`

---

## Making Conservation Exact

To eliminate "dead-end" mass, introduce an explicit end token:

    <EOT>

Then suffixes become:

```
ABACACB<EOT>
BACACB<EOT>
...
B<EOT>
```

Now:

    count(v) = sum(child_edge_counts)

because termination is represented as:

    v → <EOT>

---

## Effect of Max Depth (Truncation)

If the tree has a max depth `D`, then some suffixes are **artificially cut off**.

This introduces additional "missing mass":

    cutoff_count(v)

Updated conservation:

    count(v) = sum(child_edge_counts)
             + terminal_count(v)
             + cutoff_count(v)

Where:

- `terminal_count(v)` = real end of suffix
- `cutoff_count(v)` = truncated due to depth limit

---

## Interpretation

- **Terminal mass** = real property of finite corpus
- **Cutoff mass** = modeling artifact (lossy)

Both appear as missing outgoing mass unless explicitly tracked.

---

## Recommended Node Structure

Each node should store:

```
struct Node {
    count: int                  # visits to this node
    children: Map<Token, int>  # edge frequencies
    terminal_count: int        # suffix ends here
    cutoff_count: int          # truncated here (optional)
}
```

Invariant:

    count = sum(children.values)
          + terminal_count
          + cutoff_count

---

## Summary

- Mass is conserved when termination is explicitly modeled
- Missing mass is not an error — it represents:
  - real suffix endings
  - or artificial truncation
- Adding `<EOT>` (and optionally `<TRUNC>`) restores strict conservation

---

## Builder self-checks (added)

If the construction is correct, the following hold and are cheap to
verify:

- **Global check.** Summing all `terminal_count(v)` across the tree
  equals `N` (total number of inserted suffixes = corpus length).
  Every suffix must end somewhere.
- **Cutoff shape.** When max depth is `D`, sliding-start construction
  over a corpus of length `N` yields approximately `N − D` suffixes
  long enough to reach depth `D`. Summing `cutoff_count(v)` across all
  depth-`D` nodes recovers this number exactly.
- **Invariant at every node.** `count(v) == sum(children.values) +
  terminal_count(v) + cutoff_count(v)` should be asserted during
  build, not only computed at query time.

These three together catch off-by-one errors in the builder
immediately — a mis-tallied suffix always breaks at least one.

## `<EOT>` vs. `<TRUNC>` — practical recommendation (added)

- **`<EOT>` is the cheap, natural win.** Many tokenizers already
  include an end-of-text token. Treating it as just another token
  converts `terminal_count` into ordinary child-edge mass, so
  exact conservation holds for the finite-corpus part without
  special handling anywhere in the trie code.

- **`<TRUNC>` as an explicit model-facing token is a philosophically
  defensible but a practically awkward choice.** If the model learns
  to emit `<TRUNC>`, you're asking it to model your arbitrary depth
  cap as if it were a property of the data. Much cleaner: track
  `cutoff_count` as node-side bookkeeping (non-emitted), and use it
  only in analysis or loss-weighting, never as a training target.

## Connection to the path-probability convergence experiment

This decomposition sharpens the interpretation of our mass-1 path
tails (see `rnd/mass-filter-findings.md`). Currently the builder
lumps terminal and cutoff mass together as "edges we don't see." In
that frame, we could not distinguish:

- a mass-1 path that terminates at depth 7 because the corpus ends
  there (real, unavoidable);
- a mass-1 path that hits the depth cap at d=16 with more corpus to
  go (artifact, fixable by deeper D).

With explicit `terminal_count` and `cutoff_count` tracking, the
convergence analysis can re-slice the noise attribution: compute ρ
over paths with zero cutoff-mass vs. paths whose mass is partly or
fully cutoff. The mass-filter experiment suggests the difference will
be small (rank correlation is robust), but having the decomposition
makes the claim testable rather than assumed.

## Follow-up TODO

Rebuild the convergence tries with `terminal_count` and
`cutoff_count` tracked per node (and the three self-checks above
asserted during build). Re-generate the per-path CSVs with additional
columns `terminal_a/terminal_b`, `cutoff_a/cutoff_b`, then re-run
`rnd/mass_filter_experiment.py` with slices that separate the two
noise sources.

This does not invalidate any existing result in the convergence paper
— it refines the interpretation and gives a cleaner framework for
follow-up work.
