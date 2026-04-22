#!/usr/bin/env python3
"""Analyze the mass-weight mode sweep results.

Reads best PPL values from rnd/sgd-sanity-check/logs/massweight_sweep.txt,
groups them by (depth, mode), and prints a summary table with mean/min/max.

Usage: python3 rnd/sgd-sanity-check/analyze.py
"""
from collections import defaultdict
from pathlib import Path
import re
import statistics
import sys


def parse_sweep_log(path):
    """Return dict mapping label -> list of best PPLs across runs.

    Labels look like agpt-d16-off, agpt-d32-linear, etc. (suffix _r1/_r2/_r3
    gets stripped so runs are grouped per (depth, mode)).
    """
    results = defaultdict(list)
    line_re = re.compile(r"(agpt-d\d+-\w+)_r\d+\s+best PPL = ([\d.]+)")
    with open(path) as f:
        for line in f:
            m = line_re.search(line)
            if m:
                label, ppl = m.group(1), float(m.group(2))
                results[label].append(ppl)
    return results


def main():
    log = Path(__file__).parent / "logs" / "massweight_sweep.txt"
    if not log.exists():
        print(f"{log} not found", file=sys.stderr)
        sys.exit(1)
    results = parse_sweep_log(log)
    if not results:
        print("No completed runs found in the sweep log.", file=sys.stderr)
        sys.exit(1)

    # Order: d=16 first, modes in a deliberate order
    mode_order = ["off", "log", "sqrt", "linear"]

    def sort_key(label):
        # label e.g. "agpt-d16-linear"
        parts = label.split("-")
        depth = int(parts[1][1:])  # after 'd'
        mode = parts[2]
        return (depth, mode_order.index(mode) if mode in mode_order else 99)

    print(f"{'config':<24} {'n':>3} {'mean':>7} {'min':>7} {'max':>7} {'range':>6} {'sd':>6}")
    print("-" * 67)
    for label in sorted(results, key=sort_key):
        ppls = results[label]
        n = len(ppls)
        mean = statistics.mean(ppls)
        mn = min(ppls)
        mx = max(ppls)
        rng = mx - mn
        sd = statistics.stdev(ppls) if n > 1 else 0.0
        print(f"{label:<24} {n:>3} {mean:>7.4f} {mn:>7.4f} {mx:>7.4f} {rng:>6.3f} {sd:>6.3f}")


if __name__ == "__main__":
    main()
