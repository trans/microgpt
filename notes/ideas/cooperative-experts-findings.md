# Cooperative Expert Ensembles with Algorithmic Priors

## Abstract

We present a cooperative expert architecture where multiple small transformer
experts communicate through a shared, bandwidth-constrained stream. We show
that (1) algorithmic experts — hand-coded modules injecting domain knowledge —
dramatically improve performance when mixed with learned experts, (2) the
communication stream can be heavily compressed with minimal performance loss,
and (3) downstream transformer experts learn to restructure their
representations around injected algorithmic signal.

## 1. Architecture

### 1.1 Cooperative Ensemble

The system consists of N experts sharing a stream vector `s` of dimension `b`.
Each expert reads the current stream, computes its forward pass, and writes a
delta back to the stream. Experts process sequentially: E0 → E1 → ... → EN.

Each transformer expert (d_model=64, n_layers=3) projects from stream to its
internal dimension via learned projection matrices. A router network assigns
weights for logit-level aggregation across transformer experts.

### 1.2 Algorithmic Experts

Algorithmic experts (bigram tables, trigram tables, calculators) occupy the E0
slot and contribute *only* through the stream — they are excluded from the
router and produce no logits for aggregation. This clean separation means:

- The router covers only transformer experts
- Algorithmic experts inject signal that downstream experts can use or ignore
- No gradient flows through the algorithmic expert itself

### 1.3 Calculator Expert

The calculator expert parses addition problems from the token stream
("X + Y = Z"), computes the correct answer, and writes peaked probability
distributions at answer positions into the stream. At non-answer positions it
writes uniform distributions. This gives downstream experts a strong prior on
where the answer is and what it should be.

## 2. Experiments

All experiments use character-level tokenization. Shakespeare text for language
modeling; a shuffled dataset of all addition pairs 0+0 through 99+99 (10,000
lines) for arithmetic.

### 2.1 Algorithmic Priors Accelerate Learning

**Shakespeare, 10k steps, 4×d64n3 experts, b=64:**

| E0 Type    | Loss  | Notes                                  |
|------------|-------|----------------------------------------|
| None       | 1.800 | Pure transformer baseline              |
| Bigram     | 1.777 | -0.023 from statistical prior          |
| Trigram    | 1.753 | Richer context helps more              |

At 100k steps the advantage narrows — transformers eventually learn what
n-grams provide for free — but the algorithmic prior never hurts and provides
substantial early acceleration.

### 2.2 Calculator Expert on Addition

**Shuffled addition dataset, 10k steps, b=64:**

| Model              | Loss  | Eval Correct | Answer Loss |
|--------------------|-------|--------------|-------------|
| 4×d64n3 (no calc)  | 0.781 | 0/7          | —           |
| 5e + calculator    | 0.719 | 9–13/14*     | 0.003       |

*Range across runs due to initialization variance.

The baseline model cannot generalize addition at all (0/7 correct). With the
calculator expert, answer-position loss drops to 0.003 — effectively solved.
The remaining 0.72 overall loss is irreducible entropy from predicting random
operands (equation positions average 0.97 loss).

### 2.3 Out-of-Distribution Generalization

The calculator model correctly computes problems never seen in training:

| Problem       | Output | Correct |
|---------------|--------|---------|
| 123 + 456 =   | 579    | Yes     |
| 999 + 1 =     | 1000   | Yes     |
| 100 + 100 =   | 200    | Yes*    |

*Achieved on RTX 5090 run (13/14 overall).

This is by construction — the calculator expert implements the algorithm, not a
learned approximation. The transformers learn to trust and propagate its signal.

### 2.4 Stream Bandwidth Compression

**Shakespeare, 100k steps, 4×d64n3 experts:**

| Stream dim (b) | Ratio | Params | Loss  | Δ Loss |
|----------------|-------|--------|-------|--------|
| 64 (b=d)       | 1.0   | 667k   | 1.337 | —      |
| 32 (b=d/2)     | 0.5   | 651k   | 1.344 | +0.007 |
| 16 (b=d/4)     | 0.25  | 642k   | 1.380 | +0.043 |

**Shakespeare, 10k steps, extreme compression:**

| Stream dim (b) | Loss  |
|----------------|-------|
| 64             | 1.800 |
| 32             | 1.771 |
| 16             | 1.798 |
| 8              | 1.830 |

Halving the stream bandwidth costs only 0.007 loss at convergence. Even an
8-dimensional stream (8:1 compression ratio) maintains reasonable performance.
The experts learn to compress their coordination signal into a narrow
bottleneck.

**Addition dataset, calculator expert, 10k steps:**

| Stream dim | Loss  | Eval Correct |
|------------|-------|--------------|
| 64         | 0.719 | 12/14        |
| 32         | 0.721 | 5/7          |

The bandwidth constraint has minimal effect on loss but some effect on
generalization quality, suggesting the stream carries nuanced signal beyond
raw answer injection.

### 2.5 Stream Dynamics

We measure how experts use the stream via L2 norms and cosine metrics:

- **E0 (calculator) stream norm**: ~20 (consistent across training)
- **Transformer stream norms**: ~130–190 (grow during training)
- **E1 directional change (Δ)**: ~0.80 — E1 massively restructures the stream
  after reading E0's signal
- **E0 cosine vs final stream**: ~0.08–0.12 — E0's signal survives in the
  final stream direction, confirming downstream experts preserve it

The high Δ for E1 is key: it shows E1 isn't ignoring E0's contribution but
actively building on it. The low but nonzero cosine similarity shows the
original signal persists through the full expert chain.

## 3. Key Findings

1. **Algorithmic experts work as stream-only contributors.** They need no
   router weight, no logit output — pure stream injection is sufficient for
   downstream experts to leverage their signal.

2. **The stream is a genuine communication channel.** Experts don't just
   accumulate residuals; E1's 0.80 directional change shows active
   restructuring based on upstream signal.

3. **Bandwidth can be heavily compressed.** Halving the stream dimension costs
   <1% performance. This has implications for distributed expert systems where
   communication bandwidth is expensive.

4. **Algorithmic priors provide "free" signal.** Domain knowledge injected via
   algorithmic experts accelerates learning and enables generalization that
   pure neural approaches cannot achieve (e.g., out-of-distribution
   arithmetic).

5. **The architecture separates concerns cleanly.** What can be computed
   algorithmically is computed; what must be learned is learned. The neural
   components adapt around the algorithmic ones, not the other way around.

## 4. Open Questions

- **Scale**: Do these findings hold with larger experts (d=256, d=512)?
  Larger vocabularies? Harder tasks?
- **Per-position routing**: Current router assigns global weights. Per-token
  routing could allow experts to specialize on different parts of the sequence.
- **Multiple algorithmic experts**: What happens with several domain-specific
  algorithms (calculator + spell-checker + grammar rules)?
- **Learned compression**: Could the stream projection matrices be further
  constrained (e.g., low-rank) without loss?
- **Training dynamics**: Why does the bigram advantage disappear at 100k steps?
  Is the transformer learning to replicate the bigram, or finding something
  better?
