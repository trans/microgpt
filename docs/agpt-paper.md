# Prefix-Trie Factorization of Autoregressive Gradient Descent

## 1. Introduction

Standard LLM training operates over token windows, leading to:
- redundant computation
- duplicated gradients
- high-variance updates

However, natural language data has inherent **prefix structure**.

### Key Claim

Autoregressive training can be reformulated over a **weighted prefix trie**, yielding a factorization of gradient computation.

---

## 2. From Sequences to Prefix Trie

Define:
- Prefix set: \( \mathcal{T} \)
- Edge counts: \( n(p,x) \)
- Total outgoing mass: \( N_p = \sum_x n(p,x) \)

State recurrence:
\[
h_{p\cdot x} = f_\theta(h_p, x)
\]

Loss:
\[
L = -\sum_{p \in \mathcal{T}} \sum_x n(p,x)\log \pi_\theta(x \mid h_p)
\]

### Key Insight

This is equivalent to standard training, but expressed over:
- **unique prefixes**, not token positions

---

## 3. Gradient at a Node (Local Form)

Define:
\[
e_{p,x} = \pi_{p,x} N_p - n_{p,x}
\]

\[
g_p^{\text{local}} = \sum_x e_{p,x} W_x
\]

Interpretation:
- predicted mass vs observed mass

---

## 4. Recursive Backprop over the Trie

\[
G_p = g_p^{\text{local}} + \sum_x J_{p\to p\cdot x} \cdot G_{p\cdot x}
\]

This shows:
- the trie is a **single differentiable DAG**

---

## 5. Core Identity (Centerpiece)

\[
J_p \cdot \left(\sum_{s} g_s\right)
=
\sum_{s} (J_p \cdot g_s)
\]

Where:
- \(J_p\) = prefix Jacobian
- \(g_s\) = suffix gradient contributions

### Interpretation

- RHS: standard per-path gradients
- LHS: aggregated trie gradients

### Core Statement

**The chain rule commutes with aggregation over shared prefixes.**

---

## 6. Computational Consequence

Standard:
\[
O(\text{#token occurrences})
\]

Trie:
\[
O(\text{#unique prefixes})
\]

Benefits:
- shared Jacobians
- eliminated redundancy
- fewer forward/backward passes

---

## 7. Statistical Consequence

Standard:
- stochastic, noisy gradients

Trie:
- aggregated gradients
- lower variance
- larger effective batch

### Interpretation

Each prefix update incorporates all occurrences simultaneously.

---

## 8. Interpretation as Sufficient Statistics

The weighted prefix trie acts as a:

> **sufficient representation of the corpus for next-token modeling**

---

## 9. Optional Extension: Lookahead / Suffix Influence

(Not core to the method)

- subtree aggregation
- entropy propagation
- value-like signals

This is an extension layer.

---

## 10. Implementation Considerations

- trie memory layout
- prefix compression
- batching over nodes
- compatibility with transformer KV caches

---

## 11. Experiments (Future Work)

Evaluate:
- training speed
- gradient variance
- convergence rate

---

## 12. Conclusion

This is not just an optimization.

It is a **change in the representation of the learning problem**.

---

## Thesis Statement

Autoregressive language model training can be reformulated over a weighted prefix trie, where gradient computation factorizes by commuting aggregation with the chain rule, eliminating redundant computation and yielding larger, lower-variance updates.

---

## Practical Suggestion

Before writing the full paper, create a **core note** containing only:

1. Trie formulation  
2. Loss  
3. Gradient recursion  
4. Core identity  

This serves as a stable foundation document.
