# Root-Loop Implementation Plan

Branch: `agpt-root-loop`
Goal: extend current AGPT training to context K·D while preserving
AGPT aggregation and avoiding classical-backoff sparse regions.

## Construction (to confirm before coding)

Within each D-segment of the virtual tree, we walk the D-trie from
some seed token through D steps. At the D-boundary, the leaf token
becomes the seed for the next segment.

**Open detail:** what determines the distribution *within* a segment
at segment-relative depth j (0 < j ≤ D)?

- **Option M1 (Markov-1 everywhere):** at every step, the next
  token's distribution depends only on the immediately previous
  token, via the D-trie's root-child→grandchild transitions. Simple.
  Uses only the shallowest layer of the D-trie throughout.

- **Option Mj (Markov-cycle-relative-depth, default for root-loop):**
  at segment depth j, distribution depends on the last j tokens
  *within the current segment*. Uses progressively deeper D-trie
  layers within a segment, resetting to Markov-1 at each D-boundary.
  This uses the D-trie's structure fully within a segment while
  still avoiding crossing into sparse deep regions.

Your example was consistent with both, since D=2 didn't exercise the
distinction. I believe **Mj** is what you want — it exploits the
full D-trie within each segment and resets at the boundary, which
matches "root-looping" more literally. M1 throws away the intra-
segment D-trie structure.

Confirm: **use Mj.** Let me know if M1 was what you meant instead.

## Training structure

For each root-child subtree (as in current AGPT), train on the
virtual tree subtree rooted at that root-child. The virtual tree
subtree extends beyond the D-trie subtree via recursive D-segment
attachment.

At each virtual-tree node V in the subtree:

- Forward input = V's virtual-ancestor chain (≤ K·D tokens).
- Loss target at V = D-trie lookup at (V's current segment-relative
  depth) within V's current segment.
- Gradient aggregates at V level.
- Subtree-scoped Adam step: one step per root-child subtree, as today.

## Virtual-ancestor construction

For a radix node N at D-trie depth d_N, its virtual-ancestor chain
at virtual depth (k-1)·D + d_N (= its k-th virtual occurrence) is:

```
[ segment 1 tokens (D of them),
  segment 2 tokens (D of them, seeded by segment 1's leaf),
  ...
  segment (k-1) tokens (D of them),
  segment k's first d_N tokens = N's D-trie ancestor chain ]
```

Each segment's D tokens are a D-trie path from the segment's seed
token. Segment 1's seed is the D-trie root; segments 2..k−1 have
seeds determined by the previous segment's leaf.

## What's the canonical virtual position for N?

A radix node N has multiple potential virtual positions (one per
cycle k ∈ {1, 2, ..., K}). For training, we need to pick one — or
train on all.

**Option A: Train only at cycle 1 (no extension).** This is current
AGPT. No gain.

**Option B: Train at cycle K only.** The deepest virtual position
for N. Gives the transformer the longest context. Requires
constructing K−1 prior segments. Need to pick the canonical prior
segments.

**Option C: Train at all cycles.** K forward/backward passes per
radix node (at cycles 1, 2, ..., K). K× compute.

**Option D: Pick cycle k at random per pass.** Like Option C but
stochastically sampled. Same expected gradient as Option C, 1/K
the compute.

**Option D is probably the right starting choice** — random cycle
per training pass gives the transformer exposure to contexts of all
different positional offsets without K× compute.

## What's the canonical prior for a chosen cycle > 1?

When we train N at virtual cycle k > 1, we need to fill in the
k−1 prior segments. Options:

**Option i: First-occurrence in corpus.** For each N, find the first
corpus position where N's D-gram appears with at least (k−1)·D
corpus tokens before it. Use the actual tokens at those prior
positions as the prior segments. **Cost:** one pass over corpus to
index. **Pro:** prior is natural (real corpus text). **Con:**
arbitrary choice among many occurrences; training/inference may
still mismatch for other occurrences.

**Option ii: Markov-1-greedy from root.** Starting at D-trie root,
at each step pick the highest-probability transition; at D-boundary
the leaf seeds the next segment. Deterministic; always the same
prior for all cycle-k training of N *with the same final D-gram*.
**Pro:** fully canonical, always defined. **Con:** prior is a
specific synthetic path, not a real corpus sequence.

**Option iii: Zero-padding.** Prior = (k−1)·D pad tokens. Gives the
transformer "nothing" for the prior. Test whether just having the
positional encoding extend helps.

**My recommendation**: **Option ii (Markov-1-greedy) for the first
experiment.** Deterministic, cheap, fully automatic. If it works,
we have a result. If it doesn't, Option i is the more realistic
but costlier fallback.

## Open questions before coding

1. **Confirm Mj vs M1** for within-segment distribution (default
   Mj).
2. **Confirm Option D** (random cycle per pass) for cycle selection
   (I advocate this; user may want Option B for determinism).
3. **Confirm Option ii** (Markov-1-greedy prior) for canonical prior
   (advocates for this; flag Option i as fallback).
4. **K**: how many cycles? `K = seq_len / D`. For seq_len=128 and
   D=32, K=4. For seq_len=128 and D=16, K=8. Start with small K.
5. **Aggregation**: how does the subtree-scoped Adam step integrate
   with random cycle sampling? I think: one random cycle per
   radix-node-forward-pass, gradients aggregate into the subtree's
   one Adam step. Confirm.

## Implementation outline (for current CUDA trainer)

Once construction confirmed:

1. **Virtual ancestor construction (CPU side, one-shot per radix trie
   load)**: for each radix node, pre-compute the canonical prior for
   each potential cycle 1..K. Store as array of int32 token-ids. Cost
   per radix node: O(K·D·V) where V is vocab size (due to
   Markov-1-greedy which iterates over all tokens at each step).

2. **Training inner loop modification**: at each radix-node forward
   pass:
   - Sample random cycle k ∈ {1..K}.
   - Concatenate: [prior[k], N's D-trie ancestors, N's edge tokens] →
     pad or trim to K·D total length.
   - Run forward through transformer over this full input.
   - Compute loss against N's endpoint_counts at the endpoint.
   - Accumulate gradient as today.

3. **RoPE cache**: must extend to K·D positions. If seq_len=K·D is
   already 128 and RoPE is sized to 128, no change.

4. **KV cache**: size remains total_edge_chars in current AGPT. The
   virtual ancestors are mostly in the prior[] arrays, not in the
   D-trie's char buffer. They feed the transformer's position
   directly; no KV-cache allocation needed for them.

5. **Optimizer state**: no change. Same Adam/RMSProp structure.

6. **Aggregation semantics**: one forward/backward per radix node
   per training step; one Adam step per root-child subtree. Same as
   today.

Estimated LOC: 200-400 in agpt_train.cu, plus a precomputation pass
in the Crystal-side builder for canonical priors.

## Comparison plan

Against the following baselines on same model / same corpus / same
compute budget:

1. Plain window training (bin/microgpt, seq_len=K·D).
2. Current AGPT at D=K·D (if memory allows).
3. Current AGPT at D (what we have).

The clean story we're testing: does root-loop AGPT at (D, K·D)
match or exceed AGPT at D=K·D (which blows up memory for large K·D),
while being cheaper than window training?

---

## Waiting on your confirmation of the four open questions above
before starting the code changes.
