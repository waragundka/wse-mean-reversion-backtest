#!/usr/bin/env python3
"""Split-sample check and parameter surface.

The window is split into two halves. The variant with the highest excess
return in the first half is selected, as it would be by a user with
access to that half only, and its excess return in the second half is
reported. The Spearman rank correlation of excess returns between the
halves is also reported.

Variants overlap heavily in the securities and sessions traded, so a
positive rank correlation between the halves does not by itself indicate
predictive content.

The second part prints the parameter surface: mean annual excess return
by entry thresholds over the full window.

Usage: python3 code/04_robustness.py [LABEL]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
LABEL = sys.argv[1] if len(sys.argv) > 1 else "2021-2026"
RESULTS = BASE / "results" / LABEL
SESSIONS_PER_YEAR = 252


def annualised(s: np.ndarray) -> float:
    years = len(s) / SESSIONS_PER_YEAR
    return (1 + (1 + s).prod() - 1) ** (1 / years) - 1


def main() -> None:
    series = pd.read_parquet(RESULTS / "series.parquet")
    market = pd.read_parquet(RESULTS / "market.parquet")["market"]
    stat = pd.read_csv(RESULTS / "statistics.csv")
    net = stat[stat.costs == "net"]

    half = len(series) // 2
    boundary = series.index[half].date()
    print("split at %s: %d sessions in the first half, %d in the second"
          % (boundary, half, len(series) - half))

    rows = []
    for _, r in net.iterrows():
        s = series[r.series].to_numpy()
        m = market.to_numpy()
        rows.append({
            "variant": int(r.variant),
            "z_entry": r.z_entry, "rsi_entry": r.rsi_entry,
            "z_exit": r.z_exit, "rsi_exit": r.rsi_exit,
            "excess_h1": annualised(s[:half]) - annualised(m[:half]),
            "excess_h2": annualised(s[half:]) - annualised(m[half:]),
        })
    d = pd.DataFrame(rows)

    rho, p = stats.spearmanr(d.excess_h1, d.excess_h2)
    print("\nRANK CORRELATION between halves: rho = %.3f (p = %.3f)" % (rho, p))

    best_1 = d.loc[d.excess_h1.idxmax()]
    print("\nSELECTION ON THE FIRST HALF")
    print("  selected variant: no. %d "
          "(z_entry %.1f, rsi_entry %d, z_exit %.1f, rsi_exit %d)"
          % (best_1.variant, best_1.z_entry, best_1.rsi_entry,
             best_1.z_exit, best_1.rsi_exit))
    print("  excess return in the first half:   %+.1f pp per year" % (100 * best_1.excess_h1))
    print("  excess return in the second half:  %+.1f pp per year" % (100 * best_1.excess_h2))
    print("  median excess return of all variants in the second half: %+.1f pp"
          % (100 * d.excess_h2.median()))

    n_better = (d.excess_h2 > best_1.excess_h2).sum()
    print("  variants with a higher excess return in the second half: %d of %d"
          % (n_better, len(d)))
    print("  variants with a positive excess return in both halves: %d of %d"
          % (((d.excess_h1 > 0) & (d.excess_h2 > 0)).sum(), len(d)))

    print("\nPARAMETER SURFACE, annual excess return over the full window (pp)")
    full = net.copy()
    full["excess"] = 100 * (full.annual_return - annualised(market.to_numpy()))
    tab = full.pivot_table(index="z_entry", columns="rsi_entry", values="excess",
                           aggfunc="mean").round(1)
    print(tab.to_string())

    d.to_csv(RESULTS / "robustness.csv", index=False)
    print("\nsaved: robustness.csv")


if __name__ == "__main__":
    main()
