# Cloud GPU Guide

Run microgpt experiments on rented GPUs via vast.ai.

## Prerequisites

1. **vast.ai account** with funds loaded
2. **API key** — get from https://cloud.vast.ai/account
3. Store the key in one of:
   - `~/.config/vastai/api_key` (just the key, no newline)
   - `VAST_API_KEY` environment variable
4. **Build the binaries**:
   ```sh
   just build-cuda    # GPU-enabled microgpt binary
   just build-cloud   # cloud CLI tool
   ```

## Quick Start

```sh
# Search for available GPUs
bin/cloud search --gpu RTX_4090 --max-price 1.5

# Run experiments
bin/cloud run --data data/input.txt --config models.yml --runs coop-4e-equal-b64,coop-4e-equal-b32

# Check active instances
bin/cloud list

# Clean up
bin/cloud destroy-all
```

## Commands

### `search`

Find available GPU offers on vast.ai.

```sh
bin/cloud search [--gpu NAME] [--max-price N]
```

- `--gpu NAME` — filter by GPU model (e.g. `RTX_4090`, `RTX_3090`, `RTX_5090`)
- `--max-price N` — maximum $/hr (default: 2.0)

### `run`

Rent a GPU, deploy code, run experiments, download results, destroy instance.

```sh
bin/cloud run --runs ID,ID [OPTIONS]
```

Options:
- `--runs ID,ID` — **required** comma-separated model IDs from config file
- `--config FILE` — model config YAML (default: `models.yml`)
- `--data FILE` — training data (default: `data/addition.txt`)
- `--eval FILE` — eval prompts file for accuracy testing
- `--gpu NAME` — GPU model filter
- `--max-price N` — max $/hr (default: 1.5)

The runner will:
1. Search for cheapest GPU matching constraints
2. Rent it and wait for SSH to be ready
3. Upload binary, config, and data files
4. Run each experiment sequentially, streaming output
5. Download `results.tsv`
6. Destroy the instance (always, even on error)

### `list`

Show active instances.

```sh
bin/cloud list
```

### `destroy`

Terminate a specific instance.

```sh
bin/cloud destroy <instance_id>
```

### `destroy-all`

Terminate all active instances. Use this to avoid unexpected charges.

```sh
bin/cloud destroy-all
```

## Model Configuration

Experiments are defined in `models.yml`. Each entry specifies mode, expert
architecture, stream dimensions, and training parameters:

```yaml
coop-4e-equal-b64:
  mode: cooperative
  experts: "d64n3,d64n3,d64n3,d64n3"
  stream_dim: 64
  counter: false
  router: global       # global | context | gated
  steps: 10000
  backend: cublas       # must be cublas for GPU
```

Key fields:
- `mode` — `cooperative` or `single`
- `experts` — comma-separated expert specs (e.g. `d64n3` = d_model=64, n_layers=3)
- `stream_dim` — shared communication stream width
- `counter` — enable/disable counter expert at E0
- `bigram` / `trigram` / `calculator` — algorithmic expert at E0
- `router` — router type: `global`, `context`, or `gated`
- `steps` — training steps
- `backend` — must be `cublas` for GPU runs

## GPU Selection Tips

- **Small models** (d=64, <1M params): any GPU works, cheapest is best
  - RTX 3060 at $0.06/hr is 5x more cost-effective than an RTX 5090 at $0.38/hr
  - Deploy overhead dominates for small models
- **Medium models** (d=256, ~10M params): 24GB+ VRAM (RTX 3090, 4090)
  - RTX 3060 (12GB) will OOM on d=256 models
- **CUDA compatibility**: binary is compiled with CUDA 13.1, so the GPU driver
  must support CUDA >= 13.1 (the search filter handles this)
- **Reliability**: search filters for >= 95% reliability by default

## Troubleshooting

**Instance won't start**: vast.ai sometimes has connectivity issues with specific
hosts. The runner will retry, but if it consistently fails, try a different GPU
or region.

**SSH permission denied**: your vast.ai SSH key may not be registered. Add it at
https://cloud.vast.ai/account — look for the SSH keys section.

**Binary crashes on instance**: make sure you built with `just build-cuda` (not
`just build`, which uses CPU stubs). The CUDA binary requires Ubuntu 24.04+
instances with CUDA 13.1+ drivers.

**Results not downloading**: results.tsv is written to the same directory as the
training data on the remote instance. The runner downloads it to the local
working directory.

**Charges still running**: always run `bin/cloud destroy-all` when done. The
runner auto-destroys on completion, but if it's interrupted (Ctrl+C, network
drop), the instance will keep running and charging.
