# Suffix-Depth Blending as a Training-Time Teacher (No Inference Change)

## Goal

Use a depth-limited prefix tree (max depth `D`) to construct a **blended next-token distribution**
from all suffix depths, and train a standard model to match it—**without requiring the tree at inference**.

---

## Intuition

At timestep `t`, instead of relying only on the deepest suffix (which can be sparse), we:

1. Collect **all suffix matches** from depth `D` down to root.
2. Each suffix level provides a next-token distribution.
3. **Blend** these distributions into a single, smoother target.
4. Train the model to match:
   - the true next token (hard target), and
   - the blended distribution (soft target / teacher).

Naive version: sum counts across suffixes.  
Better: weight deeper suffixes more.  
Best: **learn the blending weights**.

---

## Notation

- Tokens: `x_1, x_2, ..., x_T`
- Vocabulary size: `V`
- Max tree depth: `D`

Suffix states at time `t`:

    s_t^(k) = suffix of length k ending at t,  for k = 0..D

Tree distributions:

    P_hat_k(. | t) = P_tree(. | s_t^(k))   ∈ ℝ^V

- `k = D` → deepest (most specific, often sparse)
- `k = 0` → root (most general, robust)

---

## Blended Teacher Distribution

### Option A — Naive (counts sum)

Aggregate counts across depths:

    counts_blend = sum_k counts_k
    P_blend = normalize(counts_blend)

Simple, but treats all depths equally.

---

### Option B — Heuristic weighting

Weight by depth (longer suffix = stronger):

    w_k = α^k         (0 < α ≤ 1)

    P_blend = sum_k w_k * P_hat_k
    normalize P_blend

Better, but introduces hand-tuned knobs.

---

### Option C — Learned blending (recommended)

Let a small gating function produce per-depth weights:

    a_k(t) = Gate(z_k(t), h_t)     # scalar score per depth

    λ_k(t) = softmax(a_k(t))       # ensures λ_k ≥ 0, sum λ_k = 1

Then:

    P_blend(. | t) = sum_k λ_k(t) * P_hat_k(. | t)

#### Minimal features per depth

    z_k(t) = [
        k,                     # depth
        log(1 + count_k)       # reliability
    ]

Optional extras:

    entropy_k, max_prob_k, branching_k

Gate can be a tiny linear layer or small MLP.

---

## Training Objective

Let:

- `Q_θ(. | t)` = model prediction (no tree at inference)
- `y_t` = true next token
- `P_blend(. | t)` = blended teacher

### Loss

    L_t =
        CE(y_t, Q_θ)                       # hard target
      + α * KL(P_blend || Q_θ)             # soft target (teacher)

Where:

    CE(y, Q) = -log Q(y)

    KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))

Interpretation:

- CE: “predict the correct token”
- KL: “match the plausible distribution from the tree”

---

## Inference

**Unchanged.**

At inference:

    Q_θ(. | t)

No tree lookup required.

The tree is used only during training as a **teacher prior**.

---

## Why This Works

- Deep suffixes → precise but sparse
- Shallow suffixes → robust but vague

Blending gives:

- precision when supported by data
- fallback when deep nodes are unreliable

The model learns to internalize this behavior.

---

## Computational Considerations

Per timestep:

- Need `D + 1` suffix distributions (can reuse ancestor links)
- Tiny gating network (very cheap)
- Optional KL over vocab (can be top-k approximated)

Training-only cost → acceptable.  
Inference cost → unchanged.

---

## Minimal Implementation Sketch

```python
state = shift(prev_state, x_t)
chain = ancestors_to_root(state)   # [s^(D), ..., s^(0)]

dists = []
scores = []

for k, node in enumerate(chain):
    p_hat = node.next_dist()       # from tree
    count = node.count
    z = [k, log1p(count)]
    a = gate(z)                   # scalar
    dists.append(p_hat)
    scores.append(a)

lambdas = softmax(scores)

p_blend = sum(l * p for l, p in zip(lambdas, dists))

Q = model(x_1:t)

loss = CE(y_t, Q) + alpha * KL(p_blend, Q)
```

---

## One-Line Summary

Blend all suffix-depth distributions into a **single soft target** during training, and use KL loss to teach the model that structure—without requiring the tree at inference.
