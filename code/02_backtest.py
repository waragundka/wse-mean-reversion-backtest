#!/usr/bin/env python3
"""Backtest of the threshold grid.

Rule: a security is bought when its price z-score and its RSI are both
below the entry thresholds. It is sold when either measure rises above
its exit threshold or when the maximum holding period is reached.

Design choices:

- Execution lag. A signal computed at the close of day t is executed at
  the close of day t+1.
- Costs. A commission of 0.39 per cent per side is applied. Each variant
  is computed net and gross of costs.
- Benchmark. The equal-weighted portfolio of the same liquid universe,
  not a market index. The comparison therefore concerns selection within
  the universe rather than market exposure.
- Full grid. Results are stored for every variant. Inference on the grid
  as a family is carried out in 03_tests.py.

Usage: python3 code/02_backtest.py [START] [END] [LABEL]
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

START = sys.argv[1] if len(sys.argv) > 1 else "2021-01-01"
END = sys.argv[2] if len(sys.argv) > 2 else "2026-12-31"
LABEL = sys.argv[3] if len(sys.argv) > 3 else "%s-%s" % (START[:4], END[:4])
SUFFIX = os.environ.get("DATA_SUFFIX", "")
RESULTS = BASE / "results" / LABEL
RESULTS.mkdir(parents=True, exist_ok=True)
COST = 0.0039            # commission per side, 0.39 per cent

GRID = {
    "z_entry":   [-3.0, -2.5, -2.0, -1.5],
    "rsi_entry": [20, 25, 30, 35],
    "z_exit":    [0.0, 0.5, 1.0],
    "rsi_exit":  [55, 65, 70],
    "max_days":  [60],
}


def run_variant(prices, returns, z, rsi, universe, par, cost):
    """Simulate one set of thresholds.

    Loop over sessions, vectorised over securities. State is kept in
    numpy arrays for speed.
    """
    n_days, n_sec = prices.shape
    c = prices.to_numpy()
    r = returns.to_numpy()
    zz = z.to_numpy()
    rr = rsi.to_numpy()
    uu = universe.to_numpy()

    held = np.zeros(n_sec, dtype=bool)
    entry_day = np.full(n_sec, -1)
    entry_price = np.full(n_sec, np.nan)

    pending_buy = np.zeros(n_sec, dtype=bool)
    pending_sell = np.zeros(n_sec, dtype=bool)

    daily = np.zeros(n_days)
    trades = []

    for i in range(n_days):
        # 1. the day's return accrues to positions open before the session
        if held.any():
            ret = r[i, held]
            ret = ret[~np.isnan(ret)]
            daily[i] = ret.mean() if ret.size else 0.0

        # 2. orders placed on the previous session execute at today's close
        exits = pending_sell & held & ~np.isnan(c[i])
        for k in np.flatnonzero(exits):
            result = c[i, k] / entry_price[k] - 1 - 2 * cost
            trades.append((entry_day[k], i, k, result))
        if exits.any():
            held[exits] = False
            # the exit cost is charged on the day it is paid
            daily[i] -= cost * exits.sum() / max(held.sum() + exits.sum(), 1)

        entries = pending_buy & ~held & ~np.isnan(c[i]) & uu[i]
        if entries.any():
            held[entries] = True
            entry_day[entries] = i
            entry_price[entries] = c[i, entries]

        # 3. signals from today's close, to be executed on the next session
        pending_buy = (~held & uu[i]
                       & (zz[i] <= par["z_entry"]) & (rr[i] <= par["rsi_entry"]))
        days_held = np.where(held, i - entry_day, 0)
        pending_sell = held & (
            (zz[i] >= par["z_exit"]) | (rr[i] >= par["rsi_exit"])
            | (days_held >= par["max_days"]))

    return daily, trades


def main() -> None:
    prices = pd.read_parquet(DATA / ("adjusted_prices%s.parquet" % SUFFIX)).loc[START:END]
    returns = pd.read_parquet(DATA / ("daily_returns%s.parquet" % SUFFIX)).reindex(
        index=prices.index, columns=prices.columns)
    z = pd.read_parquet(DATA / ("signal_z%s.parquet" % SUFFIX)).reindex(
        index=prices.index, columns=prices.columns)
    rsi = pd.read_parquet(DATA / ("signal_rsi%s.parquet" % SUFFIX)).reindex(
        index=prices.index, columns=prices.columns)
    universe = pd.read_parquet(DATA / ("signal_universe%s.parquet" % SUFFIX)).reindex(
        index=prices.index, columns=prices.columns).fillna(False)

    print("window %s: %s to %s, %d sessions"
          % (LABEL, prices.index.min().date(), prices.index.max().date(), len(prices)))

    # Benchmark: equal-weighted portfolio of the same liquid universe.
    mask = universe.to_numpy()
    r = returns.to_numpy()
    market = np.array([
        np.nanmean(r[i, mask[i]]) if mask[i].sum() >= 20 else 0.0
        for i in range(len(prices))])
    market = np.nan_to_num(market)
    pd.Series(market, index=prices.index, name="market").to_frame().to_parquet(
        RESULTS / "market.parquet")
    print("equal-weighted market: %.1f%% over the window"
          % (100 * ((1 + market).prod() - 1)))

    keys = list(GRID)
    variants = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    print("variants: %d" % len(variants))

    series, summary = {}, []
    for no, par in enumerate(variants, 1):
        for label, cost in (("net", COST), ("gross", 0.0)):
            daily, tr = run_variant(prices, returns, z, rsi, universe, par, cost)
            name = "v%03d_%s" % (no, label)
            series[name] = daily
            trade_returns = np.array([t[3] for t in tr]) if tr else np.array([])
            summary.append({
                "variant": no, "costs": label, **par,
                "n_trades": len(tr),
                "mean_trade_return": float(trade_returns.mean()) if tr else np.nan,
                "hit_rate": float((trade_returns > 0).mean()) if tr else np.nan,
                "mean_holding_days": float(np.mean([t[1] - t[0] for t in tr])) if tr else np.nan,
                "total_return": float((1 + daily).prod() - 1),
                "share_days_invested": float((daily != 0).mean()),
            })
        if no % 24 == 0:
            print("  %d/%d" % (no, len(variants)), flush=True)

    pd.DataFrame(series, index=prices.index).to_parquet(RESULTS / "series.parquet")
    s = pd.DataFrame(summary)
    s.to_csv(RESULTS / "summary.csv", index=False)
    print("\nsaved: series.parquet, summary.csv")

    net = s[s.costs == "net"]
    print("\nfive variants with the highest total return (net of costs):")
    print(net.nlargest(5, "total_return")[
        ["variant", "z_entry", "rsi_entry", "z_exit", "rsi_exit",
         "n_trades", "total_return"]].to_string(index=False))


if __name__ == "__main__":
    main()
