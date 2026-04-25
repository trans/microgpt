# MicroGPT

A minimal Crystal/CUDA transformer components kit for building and
experimenting with character-level language models. From-scratch
attention, pluggable Crystal/OpenBLAS/cuBLAS backends, a small CLI
trainer, a perplexity evaluator, and a graphical construction kit
for designing architectures visually.

## What's here

- **[MicroGPT](src/microgpt/)** — the Crystal transformer library:
  model, dataset, attention, optimizer, three backends. The
  *Design Decisions* section below covers the architectural
  choices.
- **CLI** (`bin/microgpt`) — train and generate text from a
  character-level corpus.
- **Perplexity evaluator** (`bin/perplexity`) — score any saved
  checkpoint on held-out text.
- **Cloud GPU runner** (`src/cloud/`, `bin/cloud`) — Vast.ai
  rental wrapper for remote training jobs.
- **Construction kit** (`src/construction_kit/`, Svelte frontend in
  `src/construction_kit/frontend/`) — an experimental graphical UI
  for composing transformer architectures from typed components.
  Partially implemented.

## MicroGPT library — Design Decisions

### Data Chunking

Sequential walk through the text with configurable stride. Default stride equals
`seq_len`, giving non-overlapping chunks with full coverage per epoch.

- ~1.1M tokens in tinyshakespeare, vocab size 65 (character-level).
- At seq_len=32, one epoch = ~34k steps.
- No random sampling — every token is seen exactly once per epoch.
- Stride is configurable: set below seq_len for overlapping context at chunk
  boundaries if desired.

### Embedding / Unembedding Weights

W_e (token embeddings) and W_unembed (output projection) are **independent**.
`Embedding.token_emb` is (vocab × d_model), `OutputHead.proj` is a separate
`Linear` with its own (d_model × vocab) weight matrix. No weight tying.

### W_o and Heterogeneous Heads

`MultiHeadAttention` supports heads of different dimensions. Head outputs are
concatenated column-wise into a (seq_len × d_model) matrix — head dims must sum
to d_model. W_o is a plain (d_model × d_model) linear projection over the full
concatenated vector. It has no awareness of head boundaries; it treats the
concatenation uniformly.

### Feed-Forward Dimension

`d_ff = d_model` (1:1 ratio). The FF block is two linear layers:
d_model → d_ff → d_model. At d_model=64 this keeps the parameter count low
(~61k total) and training fast (~116 steps/sec on OpenBLAS).

### Memory Protection

- `Mat` tracks global allocated bytes with a configurable cap (default 3 GiB).
  Raises with a detailed message before exceeding the limit.
- `GC.collect` runs every 10 training steps to reclaim intermediate matrices.
- `just run` wraps execution with `ulimit -v` (default 8 GiB) as an OS-level
  safety net.

## Usage

The CLI uses a JSON schema for arguments. All flags use `--flag value` syntax.

### Window Training (standard)

```sh
# Quick training run
./bin/microgpt data/input.txt --steps 2000 --d-model 64 --n-layers 2 --seq-len 32

# With held-out validation
./bin/microgpt data/input.txt --steps 5000 --d-model 64 --n-layers 2 --seq-len 32 \
  --val-tokens 5000 --val-interval 30

# GPU accelerated
./bin/microgpt data/input.txt --steps 5000 --d-model 64 --n-layers 2 --seq-len 32 \
  --backend cublas

# Other options
--seed 42              # reproducible init
--lr 0.0003            # learning rate (default 0.0003)
--no-save              # don't save checkpoint
--model path.model     # save/load checkpoint path
```

### Backends

- `crystal` — pure Crystal CPU (default, no dependencies)
- `openblas` — CPU with OpenBLAS acceleration
- `cublas` — GPU via cuBLAS + custom CUDA kernels (requires NVIDIA GPU + CUDA toolkit)

## Building

```sh
just build           # debug build (CPU only)
just build-release   # release build (CPU only)
just build-cuda      # release build with GPU support
```

## Contributors

- [Thomas Sawyer](https://github.com/trans) — creator and maintainer

## License

This repository is released under the [PolyForm Noncommercial License
1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/) —
see [`LICENSE`](LICENSE).

Academic research, personal study, and use by educational or
research institutions are permitted and encouraged.

### Commercial licensing available

If you want to use this code in a commercial product, service, or
for-profit internal workflow, a separate commercial license is
required. Terms are flexible and sized to the deployment —
startup-friendly arrangements are welcome.

See [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) for details, or
contact **`transfire@gmail.com`** directly.
