# Emergent Ventures application — AGPT (Aggregated-Gradient Pretraining)

**Applicant:** Thomas Sawyer &nbsp;·&nbsp; `transfire@gmail.com`
**Draft:** 2026-04-19
**Repo:** https://github.com/trans/microgpt
**Paper (preprint):** https://github.com/trans/microgpt/blob/main/docs/agpt-paper.md

---

## The project in one paragraph

Standard language-model training draws random windows from a corpus
and has the optimizer stochastically reconstruct the count-weighted
sum of gradients implicit in the corpus. That sum is statically
computable from a prefix trie of the corpus — one gradient per
distinct prefix, not one per sampled window. I have implemented this
end-to-end (CUDA trainer, radix-compressed trie, per-subtree
KV-cache scoping, bigram partition, auto-LR scaling) and validated
the idea on a Shakespeare corpus. In early experiments at matched
context length, the trie-based method reaches lower held-out
perplexity with roughly 10× fewer optimizer steps than a standard
window-training baseline. The next-stage experiment is a stricter
matched-compute comparison at larger scale.

**This is not a vague proposal awaiting a team; it is a working
system awaiting scale validation.**

## What's demonstrated vs. what's unknown

**Demonstrated** (small-corpus experiment, 108k-parameter char-level
model, consumer GPU):

- Gradient factorization is real and implementable — the trie
  gradient is mathematically equivalent to the window-training
  gradient sum (§5 of the paper derives this).
- At matched context, trie-based training reaches lower held-out
  perplexity than a conventional window baseline with roughly 10×
  fewer optimizer steps (trie runs use RMSProp; window uses Adam —
  both adaptive).
- The method fits depth-32 training in a consumer-memory budget
  through per-subtree KV allocation, and scales to deeper depth
  through bigram partitioning.

**Unknown** (what the grant enables validating):

- Does the per-step advantage hold at BPE vocabulary sizes?
  Char-level vocab is 65 tokens; BPE is 50k+. The branching structure
  at the trie root changes qualitatively.
- Does it hold at 100M-token and billion-token corpora? At scale,
  the mass-1 long tail becomes noise and pruning becomes close to
  free — a possible asymmetric advantage for trie-based training.
- Can AGPT compete with conventional window training at matched
  model size and compute on a mid-scale benchmark?

## What the grant buys

A one-month cloud experiment at ~100× current scale:

- **Corpus**: WikiText-103 (~100M tokens) with a standard BPE
  tokenizer (GPT-2 vocab).
- **Model**: ~10M parameters, 6-layer transformer.
- **Hardware**: single A100-class GPU on spot pricing.
- **Comparison**: AGPT vs. a standard window-training baseline at
  matched model size and compute.
- **Output**: updated paper with scale results, arXiv v2,
  and submission to a workshop (NeurIPS ML-Sys, ICLR Tiny Papers,
  or similar).

Estimated compute cost: **$500-1,500** (cloud spot, ~100-300
A100-hours).

## Funding ask

**$15,000** to cover:

- Cloud compute budget ($1,500)
- 3 months of personal runway to focus on the experiments and
  writeup full-time ($13,500 — bare-bones; I am currently without
  other income)

Deliverables in 3 months:

1. Mid-scale experiment result (BPE, 100M tokens, ~10M params).
2. Updated paper posted to arXiv.
3. Workshop submission.
4. Open-source release of the cloud-ready training code.

## Why me, why now

I'm an independent software engineer and researcher who has been
working on this for several months. The full implementation —
radix-compressed trie builder, CUDA training engine, per-subtree
memory scaling infrastructure, bigram partition, auto-LR, held-out
PPL evaluation, and the theoretical paper — is already working and
public. What's missing is the compute and runway to run the scale
validation that would determine whether this is a genuine
methodological advance or a small-corpus artifact.

This is a good fit for EV's model of supporting independent
researchers who've shipped something concrete but need a small,
targeted push to answer an important open question.

## What success looks like

Even a negative result would be publishable, because the method is
concrete and the experiment cleanly identifies whether the advantage
survives scale. The memory-scaling infrastructure (per-subtree
allocation, bigram partitioning, and trie construction tooling) is a
reusable contribution either way.

## Why this is EV-shaped

- Independent researcher without institutional backing or current
  funding.
- Working implementation, not a pitch deck.
- Clear, cheap experiment that would resolve the key open question.
- Small-dollar ask with concrete deliverables.
- If the method scales, it could become a new training primitive for
  large language models.
  
