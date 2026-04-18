# Gutenberg 5M reproduction

The 5,000,000-character multi-author corpus used in the convergence and
fingerprint experiments (`rnd/convergence/gutenberg_*`) is a concatenation
of five Project Gutenberg books, UTF-8 stripped to the same 65-character
vocabulary used for Shakespeare, truncated to 5M characters.

## Source texts (Project Gutenberg)

| File                       | Book                    | PG ID | URL                                      |
|----------------------------|-------------------------|------:|------------------------------------------|
| pride.txt                  | Pride and Prejudice     |  1342 | https://www.gutenberg.org/files/1342/1342-0.txt |
| moby.txt                   | Moby Dick; Or, The Whale|  2701 | https://www.gutenberg.org/files/2701/2701-0.txt |
| tale2cities.txt            | A Tale of Two Cities    |    98 | https://www.gutenberg.org/files/98/98-0.txt     |
| great_expectations.txt     | Great Expectations      |  1400 | https://www.gutenberg.org/files/1400/1400-0.txt |
| war_peace.txt              | War and Peace           |  2600 | https://www.gutenberg.org/files/2600/2600-0.txt |

Pull date: 2026-04-17. Project Gutenberg occasionally re-issues texts, so
the exact bytes below are definitive if you need an identical repro.

## SHA-256 hashes of the pulled files (as-is, including PG header/footer)

```
212c4047137af6855be612024988dc8fe82d4720e7ac7e2ed4311427a57eabdb  pride.txt
3db7f02828083f96ecae140604d44c0d4e9bae3d91ef2f3a22e47360c5d04e5a  moby.txt
4e15bcce2f5a4992ca4bbf926905f260fb8ab708c0dd064c041430bdb8c709ae  tale2cities.txt
097f25ad09700bd3bd24dda227e8d74752abf2fb8c336669c385944a7d31a315  great_expectations.txt
8da24ee5a42954be84f7694dedfbb87d0cb2383eb9ed3f719bfb17ef91884b47  war_peace.txt
554bf64582b8716335991c28ccbbba15d55f6f2192a895f132a6354e124bf1fb  combined_raw.txt      # concatenation in the order above
bebb2d90b3a6486aa3c49ec78f11e95668f509530a3f97513c38a5876f6a5ced  gutenberg_5m.txt      # final 5M-char corpus (in data/)
```

## Preprocessing pipeline

`combined_raw.txt` is the five books concatenated in the order:
`pride → moby → tale2cities → great_expectations → war_peace`, with each
file's Project Gutenberg header and footer stripped (the
`*** START OF THE PROJECT GUTENBERG EBOOK ...` and `*** END OF ... ***`
markers and everything outside them).

`data/gutenberg_5m.txt` is produced from `combined_raw.txt` by:

1. Unicode → ASCII-ish: curly quotes → straight (`" "` → `"`, `' '` → `'`),
   em/en dashes → `-`, ellipsis → `...`, non-breaking space → space.
2. Drop any character not in the 65-char Shakespeare vocabulary
   (alphanumerics + basic punctuation + newline + space — same vocab as
   `data/input.txt`).
3. Truncate to the first 5,000,000 characters.

The raw book files total 7.27 MB; step 2+3 trims to exactly 5M chars.

## Experiment artefacts

`rnd/convergence/gutenberg_b20_mc1/` and `gutenberg_b20_mc5/` contain
path-convergence results for this corpus at D∈{4,8,16}, split strategy
b=20 (20-block interleave), mincount∈{1,5}. The CSV row files are huge
(~0.5 GB total) and are **not tracked in git** — regenerate via:

```
bin/microgpt --agpt --agpt-build-index --agpt-max-depth 16 \
    --file data/gutenberg_5m.txt --no-save --steps 0
# + the convergence analysis tool (see rnd/convergence_analysis.py)
```

Rendered plots (`*.png`) and coverage summaries (`coverage_*.txt`) in those
directories **are** tracked — they're the small condensed results.
