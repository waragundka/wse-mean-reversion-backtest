#!/usr/bin/env python3
"""Comparison of two disjoint windows.

The windows share no sessions, so the rank correlation of excess returns
between them is not induced by overlapping samples. The script reports
the excess return of each window's best variant in the other window and
the number of variants with a positive excess return in both.

Usage: python3 code/05_period_comparison.py [LABEL_A] [LABEL_B]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
PERIODS = sys.argv[1:3] if len(sys.argv) > 2 else ["2005-2020", "2021-2026"]
SESSIONS_PER_YEAR = 252


def load(period: str) -> pd.DataFrame:
    s = pd.read_csv(RESULTS / period / "statistics.csv")
    s = s[s.costs == "net"].copy()
    market = pd.read_parquet(RESULTS / period / "market.parquet")["market"].to_numpy()
    years = len(market) / SESSIONS_PER_YEAR
    r_market = (1 + ((1 + market).prod() - 1)) ** (1 / years) - 1
    s["excess"] = s.annual_return - r_market
    s["market_annual"] = r_market
    return s.set_index("variant")


def main() -> None:
    a, b = (load(o) for o in PERIODS)
    n = len(a)

    print("MARKET IN BOTH WINDOWS")
    for o, d in zip(PERIODS, (a, b)):
        print("  %-14s %.1f%% per year" % (o, 100 * d.market_annual.iloc[0]))

    print("\nSTRATEGY, FULL GRID (annual excess return over the market, pp)")
    print("  %-14s %8s %8s %8s %10s" % ("window", "median", "min", "max", "positive"))
    for o, d in zip(PERIODS, (a, b)):
        print("  %-14s %+8.1f %+8.1f %+8.1f %7d/%d"
              % (o, 100 * d.excess.median(), 100 * d.excess.min(),
                 100 * d.excess.max(), (d.excess > 0).sum(), n))

    common = a.index.intersection(b.index)
    rho, p = stats.spearmanr(a.loc[common, "excess"], b.loc[common, "excess"])
    print("\nRANK CORRELATION BETWEEN WINDOWS: rho = %+.3f (p = %.3f)" % (rho, p))

    print("\nBEST VARIANT OF EACH WINDOW IN THE OTHER WINDOW")
    for o1, o2, d1, d2 in ((PERIODS[0], PERIODS[1], a, b), (PERIODS[1], PERIODS[0], b, a)):
        w = d1.excess.idxmax()
        r = d1.loc[w]
        print("  best in %s: variant %d (z_entry %.1f, rsi_entry %d)"
              % (o1, w, r.z_entry, r.rsi_entry))
        print("      in %s: %+.1f pp per year" % (o1, 100 * r.excess))
        print("      in %s: %+.1f pp per year   (rank %d of %d)"
              % (o2, 100 * d2.loc[w, "excess"],
                 int(d2.excess.rank(ascending=False).loc[w]), n))

    both = ((a.excess > 0) & (b.excess > 0)).sum()
    print("\nvariants with a positive excess return in both windows: %d of %d" % (both, n))

    out = pd.DataFrame({
        "z_entry": a.z_entry, "rsi_entry": a.rsi_entry,
        "z_exit": a.z_exit, "rsi_exit": a.rsi_exit,
        "excess_%s" % PERIODS[0].replace("-", "_"): a.excess,
        "excess_%s" % PERIODS[1].replace("-", "_"): b.excess,
    })
    name = "period_comparison%s.csv" % ("_dl" if PERIODS[0].endswith("_dl") else "")
    out.to_csv(RESULTS / name)
    print("\nsaved: %s" % name)


if __name__ == "__main__":
    main()
