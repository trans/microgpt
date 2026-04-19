# AGPT: Branching-Aware Training for Character-Level Language Models

**Author:** Thomas Sawyer &nbsp; `transfire@gmail.com`
**Status:** Technical preprint, v1 &nbsp; · &nbsp; **Date:** 2026-04-19
**Code:** https://github.com/trans/microgpt

---

## Abstract

Standard next-token language-model training draws random windows from the
corpus and presents them to the optimizer as independent examples. Each
character of the corpus contributes many near-identical gradients —
once per window it appears in — and the optimizer's job is largely to
average over these redundancies.

We propose **AGPT** (Aggregated-Gradient Pretraining): train instead on
a character-level prefix trie of the corpus. Every unique prefix appears
exactly once; each training example is a branching endpoint, weighted
by its corpus frequency. The trie's structure becomes the training
schedule, and a single Adam step per root-child subtree takes the place
of many redundant window-based steps.

On a 4.6 MB Shakespeare corpus with a 108k-parameter transformer, AGPT
trained on a radix-compressed trie at depth 32 reaches held-out PPL
**13.17** in **195 Adam steps** (21 minutes on a single consumer GPU).
Standard window-based training on the same model/data reaches PPL
**17.68** in **2000 SGD steps** at matched context (seq_len=32) — a
26% relative improvement with an order of magnitude fewer optimizer
updates.

The contributions are: (1) a formulation of trie training with a
subtree-scoped Adam step as the factorization-aware training unit;
(2) a radix-compressed + per-subtree file format that makes depth-32
training fit in a 15 GB-RAM consumer workstation; (3) a bigram subtree
partition that further reduces peak memory 6×, unlocking deeper d;
(4) an auto-LR scaler that lets a single base-LR hyperparameter hold
across subtree granularities. All results are reproducible from the
referenced code.

## 1. Motivation

Consider a character-level corpus of N tokens. Window training with
sequence length L produces up to N − L training examples; many of these
share long prefix substrings. For a next-token objective at position i,
the optimizer sees:

- The same `(prefix → next)` event many times (once per window it
  occurs in)
- Different prefix contexts in different windows

In the limit of a small corpus or deep shared structure, the net
gradient at a given prefix is a weighted sum of the contributions from
every window that contains it. Rather than ask the optimizer to
reconstruct that sum through sampling, AGPT computes it explicitly:
*one* gradient per unique prefix, weighted by frequency.

This is the same principle behind *natural-gradient* and
*Fisher-information* methods — exploit known structure in the data
rather than sampling around it. The trie is that structure made
explicit at the character level.

## 2. The AGPT training unit

### 2.1 Trie as training schedule

Build a character-level prefix trie of the corpus to depth d. Each
internal node N stores:

- The token emitted at N
- The count distribution over next-tokens observed at N
- Children (one per distinct next-token actually seen)

A **branching endpoint** is a node with ≥ 2 distinct next-tokens in its
count distribution. Non-branching (unary) positions carry no training
signal — the next token is deterministic given the prefix — so they
can be collapsed.

### 2.2 Radix compression

Collapse every unary chain into a single radix node with a
multi-character edge. The node's metadata becomes:

```
radix_id, parent_radix_id, first_char_depth,
edge_len L, edge_tokens[L],
edge_mass (prefix count at head of edge),
entries[] (next-token counts at endpoint)
```

Radix compression is lossless: the expanded character sequence is
identical to the original trie. At d=32 on our corpus the radix trie
has 1.67M nodes representing a 27M-node uncompressed trie — a 16.2×
reduction in node count.

### 2.3 Subtree-scoped Adam steps

The **AGPT training unit** is a root-child subtree. A root-child is a
depth-1 radix node (its edge starts at character position 0); its
subtree is itself plus all descendants. Each subtree is trained in
isolation:

1. Forward pass: compute Q/K/V, RoPE-position, attention, FFN, and
   next-token logits at every endpoint in the subtree.
2. Loss: cross-entropy weighted by endpoint count (i.e., prefix
   frequency in the corpus).
3. Backward pass: accumulate gradients across all chunks of the
   subtree, with the model parameters held fixed through the whole
   subtree.
4. **One Adam/RMSProp step** per subtree.

Holding parameters fixed through the forward-backward of the full
subtree is not an optimization detail — it is the factorization that
makes the one-shot gradient equal to the sum over all windows that
share this prefix. Firing the optimizer mid-subtree re-introduces the
staleness it was built to eliminate.

## 3. Memory scaling

### 3.1 The KV cache is the bottleneck

A transformer's K/V cache must cover every character position attended
to by any query in the subtree. The global cache size is:

```
KV_bytes = total_edge_chars × d_model × 2 (K+V) × n_layers × bytes
```

At Shakespeare d=32: 27M × 64 × 4 × 2 × 4 = 27.6 GB. Doesn't fit on
consumer hardware (15 GB RAM + 15 GB swap). Deeper d scales linearly
in `total_edge_chars`.

Radix compression reduces *node* count by ~16× but does not reduce
`total_edge_chars` — each character in an edge still consumes a KV
slot.

### 3.2 Per-subtree training

The insight: at training time, attention at a query position only
reaches characters that are *ancestors of that position*. Different
root-child subtrees share no character positions — the 'A' tree and the
'B' tree never look at each other's character positions.

Therefore: allocate the KV cache **sized to one subtree's edge
characters**, one subtree at a time. On Shakespeare d=32, peak
per-subtree KV drops from 27.6 GB to **4.0 GB** (6.9× reduction).

### 3.3 Bigram subtree partition

The unigram per-subtree scheme bottlenecks on the largest
root-child — in Shakespeare that is the space character (' '),
responsible for ~30% of corpus occurrences. At d=48+ the space subtree
alone would exceed the available RAM budget.

Bigram partitioning splits each root-child subtree by the second
character of the path. For vocab V, there are up to V² subtrees.
At Shakespeare d=32: 1,465 bigram subtrees (vs. 65 unigram),
peak per-subtree KV 677 MB (vs. 4 GB unigram, 6× further reduction).

Because each bigram subtree is self-contained (contains its own
ancestor chain), no cross-subtree coordination is needed at training
time — trivially parallelizable across a cluster.

### 3.4 Auto-LR scaling

A subtler consequence of increasing subtree count: each super-epoch
now fires more Adam steps. With 1465 bigram subtrees instead of 65
unigram, naive lr=3e-3 overshoots in a single super-epoch. Empirically
the product `lr × steps_per_super_epoch` is approximately invariant at
a fixed depth. We therefore scale:

```
lr_effective = base_lr × (REFERENCE_STEPS / steps_per_super_epoch)
```

with REFERENCE_STEPS = 65 (the unigram-d=16 calibration point). This
lets a single base_lr hold across unigram, bigram, and future
finer-grained partitions without re-tuning per config.

## 4. Empirical results

### 4.1 Setup

- Corpus: Shakespeare `input.txt`, 4.62 MB, 65-character vocabulary.
- Model: 2-layer transformer, d_model=64, 4 heads × 16 head-dim,
  d_ff=256, seq_len=128. 108,481 parameters.
- Hardware: single 8 GB GPU + 15 GB system RAM, 15 GB swap.
- Eval: held-out corpus tail, 32,768 positions scored with sliding
  seq_len=128 window. Backend: OpenBLAS (bit-identical to CUDA in
  current build).

### 4.2 Comparison vs. standard window training

All numbers from the same codebase; same model architecture; same
corpus.

| Model | Context | Optimizer steps | Held-out PPL |
|---|---:|---:|---:|
| Random init | — | 0 | 144.2 |
| Window seq=16 | 16 | 2,000 | 19.76 |
| **AGPT d=16**, RMSProp+warmup-cosine | 16 | 50 | **15.28** |
| Window seq=32 | 32 | 2,000 | 17.68 |
| **AGPT d=32**, per-subtree, 3 super-epochs | 32 | 195 | **13.17** |
| **AGPT d=32 bigram**, auto-LR, 1 super-epoch | 32 | 1,465 | **13.30** |
| Window seq=128 | 128 | 2,000 | 11.64 |

At matched context d=16, AGPT reaches PPL 15.28 in 50 Adam steps vs.
window's 19.76 in 2000 steps (20% better PPL, 40× fewer steps). At
d=32 the ratio widens to 26% and 10× respectively. Window training at
seq=128 reaches lower PPL (11.64) but requires 4× the effective context
and substantially more wall-clock time.

### 4.3 Super-epoch sensitivity (d=32 unigram)

| Super-epochs | Adam steps | Held-out PPL |
|---:|---:|---:|
| 1 | 65 | 13.71 |
| 2 | 130 | 13.18 |
| **3** | **195** | **13.17** |
| 5 | 325 | 14.84 |
| 10 | 650 | 18.27 |
| 50 | 3,250 | 37.34 |

The AGPT objective converges faster than window training, and is
correspondingly easier to over-train: 50 super-epochs catastrophically
overfits. We provide an external best-checkpoint wrapper
(`agpt_train_best.sh`) that writes per-super-epoch checkpoints and
selects the best by held-out PPL — removing the need to hand-tune the
super-epoch count.

### 4.4 Radix + pruning numbers

| Config | radix nodes | total edge chars | Peak KV (d=32) | PPL |
|---|---:|---:|---:|---:|
| Uncompressed leveled trie | 27.0 M | — | n/a | n/a |
| Radix compression | 1.67 M | 27.0 M | 4.0 GB | 13.17 |
| Radix + prune m<2, d≥4 | 0.55 M | 1.17 M | 175 MB | 16.07 |

Radix compression gives a 16× node reduction for free. Pruning by
prefix mass trades quality for memory: 23× additional reduction at
~3 PPL cost. On small corpora the cost is real (rare paths are a
significant fraction of branching signal). On billion-token corpora
we expect the cost to collapse — long-tail mass=1 paths become
essentially noise — but this is not yet empirically verified.

## 5. Implementation notes

Full implementation in Crystal (trie builder, CLI) and CUDA C++
(training engine) at the referenced repository. Key files:

- `src/agpt/streaming_radix_builder.cr` — radix trie + per-subtree
  file format + bigram partition + pruning.
- `src/cuda/agpt_train.cu` — training engine, including per-subtree
  loop, auto-LR scaling, and full GPU-state cleanup on each
  subtree boundary.
- `scripts/agpt_train_best.sh` — external best-PPL wrapper.

## 6. Limitations and future work

Honestly stated:

1. **Scale**. All results are on a 4.6 MB char-level corpus with a
   108k-parameter model. Modern LLM training is ~9 orders of magnitude
   larger in parameter count. This paper's claims are that AGPT beats
   window training *on a small corpus at matched context*; generalization
   of the advantage to BPE vocabularies and billion-token corpora is an
   open empirical question.

2. **KV-cache scaling to BPE**. At Opus-family hyperparameters
   (d_model ≈ 16k, layers ≈ 100) a 1B-token corpus would require
   petabytes of aggregate KV memory across all subtree files.
   Per-subtree training makes this distributable, but the per-node
   memory budget still requires aggressive pruning + finer partitions
   (trigram or beyond).

3. **Pruning quality at scale**. Section 4.4's pruning results on
   Shakespeare cost 3 PPL. We hypothesize near-free pruning at larger
   scale (most long paths truly unique = noise), but this is unvalidated.

4. **Generation quality**. This paper reports PPL only; subjective
   quality of generated text has not been systematically compared.

5. **Window baseline saturation**. The window baseline ran for 2,000
   steps. A fully saturated window baseline at matched compute would be
   a stronger comparison. The AGPT advantage is likely larger on
   compute-matched ground than on step-matched ground, but this is
   worth confirming.

6. **Peer comparison**. No head-to-head with tokenizer-free neighbors
   (ByT5, MambaByte, etc.) has been run.

The natural next validation is **WikiText-103 scale with a real BPE
tokenizer on a cluster-grade GPU** — approximately 2 orders of magnitude
larger than the experiment here, achievable on a single A100 for
< $100 of cloud compute. This would establish whether the AGPT
advantage holds at BPE vocab sizes and 100M-token corpora, which
would be the meaningful first step toward frontier scale.

## 7. Priority claim

The specific contributions of this work, for priority purposes:

1. The subtree-scoped Adam-step invariant as the factorization-aware
   training unit for trie-structured language model training.
2. The radix-compressed, per-subtree-file format for memory-scalable
   trie training.
3. The bigram (and by extension n-gram) subtree partition for
   unlocking deeper d at fixed memory budget.
4. The `lr × steps_per_super_epoch = constant` scaling heuristic
   enabling a single base-LR hyperparameter across subtree
   granularities.
5. Frequency-pruned trie construction past a depth threshold, with
   the observed tradeoff profile.

First public disclosure of these mechanisms is the referenced
repository's git history.

---

*Corrections, discussion, and collaboration inquiries welcome. The
repository is MIT-licensed and open to pull requests.*
