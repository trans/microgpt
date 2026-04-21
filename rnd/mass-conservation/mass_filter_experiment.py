#!/usr/bin/env python3
"""
Experiment: do mass=1 tails dominate the noise in path-probability convergence?

For each results CSV, compute Spearman ρ between log_pi_a and log_pi_b
under three filters:
  - all rows (baseline, matches the paper's count ≥ 1 numbers)
  - min(count_a, count_b) ≥ 2 (drop paths with mass 1 in either trie)
  - min(count_a, count_b) ≥ 5 (matches the paper's count ≥ 5 numbers)

And then the key new slice:
  - mass ≥ 2 AND path is at the max depth for its dataset
    (i.e., specifically the depth-capped tails, not all mass-1 paths)

If ρ improves disproportionately on the last slice vs. the mass≥2 filter,
the depth-cap truncation is dominating the noise independent of the
underlying mass effect. If it's about the same improvement as mass≥2
generally, the noise is just "mass-1 paths are noisy" and the cap isn't
a special source of artifact.

Usage:
  python3 rnd/mass_filter_experiment.py [--results-dir rnd/convergence]
"""
import argparse
import os
import glob
import pandas as pd
from scipy.stats import spearmanr


def summarize(df: pd.DataFrame, label: str) -> dict:
    med = df["log_abs_err"].median() if ("log_abs_err" in df and len(df) > 0) else None
    if len(df) < 2:
        return {"label": label, "n": len(df), "rho": None, "median_abs_err": med}
    rho, _ = spearmanr(df["log_pi_a"], df["log_pi_b"])
    return {"label": label, "n": len(df), "rho": rho, "median_abs_err": med}


def analyze_csv(path: str) -> None:
    df = pd.read_csv(path)
    if df.empty:
        print(f"{path}: empty")
        return

    # What max depth does this file cover? (guess from filename D{N}.)
    base = os.path.basename(path)
    max_depth = int(base.replace("results_D", "").replace(".csv", ""))

    # Per-depth overall stats (informational)
    print(f"\n=== {path} (max depth = {max_depth}) ===")
    print(f"  total rows: {len(df)}")
    print(f"  depth distribution: "
          f"{df['depth'].value_counts().sort_index().to_dict()}")

    results = []

    # All rows (baseline)
    results.append(summarize(df, "all rows"))

    # Mass >= 2 everywhere
    df_m2 = df[(df["count_a"] >= 2) & (df["count_b"] >= 2)]
    results.append(summarize(df_m2, "min(count) >= 2"))

    # Mass >= 5 everywhere (matches paper's count >= 5 baseline)
    df_m5 = df[(df["count_a"] >= 5) & (df["count_b"] >= 5)]
    results.append(summarize(df_m5, "min(count) >= 5"))

    # Depth-cap-only mass filter: only paths at the max depth
    df_cap = df[df["depth"] == max_depth]
    results.append(summarize(df_cap, f"at max depth (d={max_depth}) all"))
    df_cap_m2 = df_cap[(df_cap["count_a"] >= 2) & (df_cap["count_b"] >= 2)]
    results.append(summarize(df_cap_m2, f"at max depth (d={max_depth}) mass >= 2"))
    df_cap_m5 = df_cap[(df_cap["count_a"] >= 5) & (df_cap["count_b"] >= 5)]
    results.append(summarize(df_cap_m5, f"at max depth (d={max_depth}) mass >= 5"))

    # Shallow paths for comparison
    df_shallow = df[df["depth"] <= max_depth // 2]
    results.append(summarize(df_shallow, f"shallow (d<={max_depth//2}) all"))
    df_shallow_m2 = df_shallow[(df_shallow["count_a"] >= 2) & (df_shallow["count_b"] >= 2)]
    results.append(summarize(df_shallow_m2, f"shallow (d<={max_depth//2}) mass >= 2"))

    # Print table
    print(f"  {'filter':<42} {'n':>10} {'rho':>8} {'med|err|':>10}")
    for r in results:
        rho_str = f"{r['rho']:.4f}" if r["rho"] is not None else "  n/a "
        med_str = f"{r['median_abs_err']:.4f}" if r["median_abs_err"] is not None else "  n/a "
        print(f"  {r['label']:<42} {r['n']:>10} {rho_str:>8} {med_str:>10}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="rnd/convergence")
    p.add_argument("--glob", default="**/results_D*.csv")
    args = p.parse_args()

    pattern = os.path.join(args.results_dir, args.glob)
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        print(f"No CSVs matched: {pattern}")
        return

    for path in paths:
        analyze_csv(path)

    print("\n--- Interpretation hints ---")
    print("  If rho(all) << rho(mass>=2): mass-1 tails dominate the noise.")
    print("  If rho(at-max-depth) << rho(shallow) even after mass filter:")
    print("    depth-cap truncation is a distinct noise source beyond mass.")
    print("  If rho(at-max-depth mass>=2) ≈ rho(shallow mass>=2):")
    print("    cap does not cause noise beyond the mass effect.")


if __name__ == "__main__":
    main()
