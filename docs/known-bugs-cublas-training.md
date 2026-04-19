# Known bug: `bin/microgpt` cublas backend produces broken window-trained models

## Symptom

Training via `bin/microgpt --backend cublas --file data/input.txt --steps N
--seq-len L ...` runs to completion with a plausible-looking loss curve
(training loss decreases monotonically from random to ~2.0 on Shakespeare at
seq_len=128, 10k steps). The saved model passes file-format checks and loads.
But:

- Generation from the trained model produces gibberish (mostly uppercase
  consonants and punctuation, no spaces in natural positions, no real English
  words).
- Held-out perplexity on the training corpus is ~170 — **worse than a
  random-init model (~164)**. Random baseline on a 65-character vocab is
  ~65 PPL.
- The training loss number the trainer reports is not reflective of the
  model's actual predictive quality.

## Reproduction

```
rm -f /tmp/bug_cublas.model
bin/microgpt --file data/input.txt --steps 500 --seq-len 128 \
    --backend cublas --model /tmp/bug_cublas.model --seed 42
# → "Final avg loss: 2.7677"

bin/perplexity --model /tmp/bug_cublas.model --file data/input.txt \
    --max-positions 4096 --backend openblas
# → Perplexity: ~170
```

## Openblas is not affected

Same command with `--backend openblas` (or `--backend crystal`) produces a
coherent-looking model:

```
bin/microgpt --file data/input.txt --steps 500 --seq-len 128 \
    --backend openblas --model /tmp/fine.model --seed 42
# → "Final avg loss: 2.7675"   (same loss)
bin/perplexity --model /tmp/fine.model --file data/input.txt \
    --max-positions 4096 --backend openblas
# → Perplexity: ~13.5            (coherent, reasonable)
```

The two trainers report nearly identical training losses for identical seed +
config, but the resulting models are very different.

## Scope

- `bin/microgpt` (the Crystal-side trainer, window-based training) when built
  with the cublas kernels (`just build-cuda`) and run with `--backend cublas`.
- Does **not** affect `bin/agpt_train` (the standalone CUDA AGPT trainer). That
  binary has been independently verified — d=16 AGPT trained with it hits
  PPL 15.28, a 50-super-epoch d=32 run plays out as expected, etc.
- Does **not** affect `bin/perplexity --backend openblas` (eval path is fine).

## Hypothesis

Most likely a cublas codepath in `src/microgpt/micro_gpt.cr` `train_step` or
`backward` computes the loss scalar correctly but applies gradients to a
buffer that is not the one saved by `model.save`. Something like:

- A weight-layout mismatch between the cublas matmul path and the Crystal save
  path.
- A stale cached view into the weights that gets updated in-place during
  forward but isn't written back to the canonical buffer.
- An in-place op that produces the illusion of training progress while
  actually corrupting the held-out distribution.

An earlier incident (now in the memory notes as "mode collapse to 'z' output")
was described as a **generation-only** cublas bug where inference on trained
models produced a degenerate distribution. This might be the same root cause
— the training path exercises the same kernels on the write-side.

## Workaround

Use `--backend openblas` for window training. Slower (CPU BLAS) but correct.
`bin/agpt_train` still runs full-speed on GPU — the AGPT path is unaffected.

## Impact on prior results

The paper's window-training baselines (`Window seq={16, 32, 128}`) need to be
confirmed to have been produced with the Crystal or openblas backend, not
cublas, before being cited. Re-running with `openblas` to be safe is cheap.
AGPT results are unaffected — they came from `bin/agpt_train` which does not
go through the suspect codepath.

## Fix priority

Medium. Workaround is clean. Fix would unblock faster window-baseline runs
and restore trust in cublas for future work.
