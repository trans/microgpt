# CUDA Monolithic Kernel Plan for MicroGPT

## Problem

Current cuBLAS backend launches ~350-400 individual CUDA kernels per training step. At d_model=64, kernel launch overhead (~5-10μs each) is roughly 50% of wall time. Data bounces between GPU global memory between every operation.

## Solution: Three Phases

### Phase 1: GPU-Resident Weight Store (biggest win)

All weights stay on GPU permanently. One contiguous `cudaMalloc` holds:

```
[WEIGHTS][ADAM_M][ADAM_V][GRADIENTS][ACTIVATIONS][SAVED_STATE]
```

Crystal holds a pointer + offset table. Per-step transfers reduced to:
- Upload: input_ids + target_ids (512 bytes)
- Download: loss scalar (4 bytes)

**New kernels:**
- `fused_embedding_gather` — GPU-side token lookup (currently CPU)
- `fused_loss_softmax_grad` — softmax + CE loss + gradient in one pass (currently CPU)

**Expected speedup: 2-3x at d=64**

### Phase 2: Fused Element-wise Kernels

Replace sequences of tiny kernels with fused versions:

| Fused Kernel | Replaces | Saves |
|---|---|---|
| `fused_bias_relu` | bias_add + relu_forward | ~9 launches |
| `fused_rope_qk` | 2× rope_apply per layer | ~3 launches |
| `fused_multihead_attn_softmax` | per-head attn softmax | ~3 launches |
| `fused_weighted_aggregate` | CPU aggregation loop | ~5 launches |
| `fused_adam_bulk` | per-tensor Adam | ~25 launches |

Kernel count: ~350 → ~120-150

**Expected speedup: additional 1.5-2x**

### Phase 3: Monolithic Train Step

Single C/CUDA function orchestrates the entire cooperative forward+backward+update. Eliminates Crystal FFI overhead (~300 calls × 1μs = 300μs per step).

```c
extern "C" float cuda_cooperative_train_step(
    void* weight_store,
    int* input_ids, int* target_ids,
    float lr, int adam_t,
    CoopConfig* config,
    cublasHandle_t handle
);
```

All cuBLAS matmul calls + fused kernels inside one C function. Crystal calls it once per step.

## Data Layout

Weight store total at d=64, 4 experts: ~720KB (weights + Adam m/v). Activations + saved state: ~3MB. Trivially fits on any GPU.

## Integration

New backend flag: `--backend monolith`

```crystal
case backend
when "monolith"  then MicroGPT.use_monolithic!  # new
when "cublas"    then MicroGPT.use_cublas!       # existing
when "openblas"  then MicroGPT.use_openblas!     # existing fallback
end
```

Existing OpenBLAS/cuBLAS paths remain untouched. Monolithic is additive.

## Irreducible Floor

~75-100 cuBLAS matmul calls per step (the actual compute). These cannot be fused — cuBLAS is already optimal. Everything else gets fused around them.

## Target Performance

| Scale | Current | After Phase 1-3 |
|---|---|---|
| d=64 100k steps | ~3.5 hrs (CPU) | ~20-30 min (GPU) |
| d=256 100k steps | ~24 hrs (GPU) | ~4-6 hrs (GPU) |

## File Organization

```
src/cuda/
  kernels.cu          — element-wise kernels (extend)
  fused_kernels.cu    — fused kernels (Phase 2)
  train_step.cu       — monolithic orchestrator (Phase 3)
  weight_store.h      — offset computation, config struct
  stubs.c             — extended with new stubs

src/microgpt/
  backend.cr          — WeightStore, MonolithicCuBLASBackend
  micro_gpt.cr        — GPU paths in Embedding, OutputHead
  cooperative.cr      — GPU aggregation, stream ops
  main.cr             — --backend monolith flag
```
