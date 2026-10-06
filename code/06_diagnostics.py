#!/usr/bin/env python3
"""Diagnostics: entry signal versus exit rule.

1. Fixed-horizon test. Securities are bought on the signal and held for
   exactly N sessions. No exit rule is involved. The portfolio is formed
   in calendar time: on each day it holds all cohorts entered in the last
   N sessions, which yields a daily return series to which the same
   alpha regression applies. If this portfolio underperforms the
   benchmark, the underperformance is attributable to the entry signal.

2. Exit reasons. Share, mean holding period and mean gross return of
   trades by the condition that closed them.

3. Post-exit returns. Return of the security in excess of the market over
   N sessions after the exit day.

The middle variant of the grid is used rather than the best one.

Usage: python3 code/06_diagnostics.py [START] [END]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = Path(__file__).resolve().parent.parent
DATA, RESULTS = BASE / "data", BASE / "results"
SESSIONS_PER_YEAR = 252
NW_LAGS = 10

START = sys.argv[1] if len(sys.argv) > 1 else "2005-01-01"
END = sys.argv[2] if len(sys.argv) > 2 else "2026-12-31"

Z_ENTRY, RSI_ENTRY = -2.0, 30
Z_EXIT, RSI_EXIT, MAX_DAYS = 0.5, 65, 60
HORIZONS = [5, 10, 21, 63, 126, 252]


def annualised(s):
    years = len(s) / SESSIONS_PER_YEAR
    return (1 + (1 + s).prod() - 1) ** (1 / years) - 1


def alpha(strategy, market):
    X = sm.add_constant(market)
    m = sm.OLS(strategy, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return m.params[0] * SESSIONS_PER_YEAR, m.tvalues[0], m.pvalues[0]


def main() -> None:
    prices = pd.read_parquet(DATA / "adjusted_prices.parquet").loc[START:END]
    returns = pd.read_parquet(DATA / "daily_returns.parquet").reindex(
        index=prices.index, columns=prices.columns)
    z = pd.read_parquet(DATA / "signal_z.parquet").reindex(
        index=prices.index, columns=prices.columns)
    rsi = pd.read_parquet(DATA / "signal_rsi.parquet").reindex(
        index=prices.index, columns=prices.columns)
    universe = pd.read_parquet(DATA / "signal_universe.parquet").reindex(
        index=prices.index, columns=prices.columns).fillna(False)

    r = returns.to_numpy()
    u = universe.to_numpy()
    zz, rr, c = z.to_numpy(), rsi.to_numpy(), prices.to_numpy()
    n_days, n_sec = c.shape

    market = np.array([np.nanmean(r[i, u[i]]) if u[i].sum() >= 20 else 0.0
                       for i in range(n_days)])
    market = np.nan_to_num(market)

    print("window %s to %s, %d sessions" % (prices.index.min().date(),
                                           prices.index.max().date(), n_days))
    print("market: %.1f%% per year" % (100 * annualised(market)))

    # entry signal, executed on the next session
    signal = u & (zz <= Z_ENTRY) & (rr <= RSI_ENTRY)
    entry = np.zeros_like(signal)
    entry[1:] = signal[:-1]
    print("entry events: %d" % entry.sum())

    # ---------------------------------------------------------------
    print("\n=== 1. BUY ON SIGNAL, HOLD N SESSIONS (no exit rule) ===")
    print("%8s %14s %14s %10s %8s" % ("N", "annual return", "market",
                                      "alpha", "p"))
    rows = []
    ent = pd.DataFrame(entry, index=prices.index, columns=prices.columns)
    for N in HORIZONS:
        held = ent.rolling(N, min_periods=1).max().astype(bool).to_numpy()
        series = np.array([
            np.nanmean(r[i, held[i]]) if held[i].sum() else 0.0
            for i in range(n_days)])
        series = np.nan_to_num(series)
        a, t, p = alpha(series, market)
        print("%8d %13.1f%% %13.1f%% %9.1f%% %8.3f"
              % (N, 100 * annualised(series), 100 * annualised(market), 100 * a, p))
        rows.append({"horizon": N, "annual_return": annualised(series),
                     "annual_alpha": a, "p": p})
    pd.DataFrame(rows).to_csv(RESULTS / "diagnostics_horizons.csv", index=False)

    # ---------------------------------------------------------------
    print("\n=== 2. EXIT REASONS ===")
    held = np.zeros(n_sec, dtype=bool)
    entry_day = np.full(n_sec, -1)
    entry_price = np.full(n_sec, np.nan)
    pending_buy = np.zeros(n_sec, dtype=bool)
    pending_sell = np.zeros(n_sec, dtype=bool)
    sell_reason = np.zeros(n_sec, dtype="<U12")
    trades = []

    for i in range(n_days):
        exits = pending_sell & held & ~np.isnan(c[i])
        for k in np.flatnonzero(exits):
            trades.append({
                "security": k, "entry": entry_day[k], "exit": i,
                "days": i - entry_day[k], "reason": sell_reason[k],
                "return": c[i, k] / entry_price[k] - 1,
            })
        held[exits] = False
        entries = pending_buy & ~held & ~np.isnan(c[i]) & u[i]
        held[entries] = True
        entry_day[entries] = i
        entry_price[entries] = c[i, entries]

        pending_buy = ~held & u[i] & (zz[i] <= Z_ENTRY) & (rr[i] <= RSI_ENTRY)
        days_held = np.where(held, i - entry_day, 0)
        by_z = held & (zz[i] >= Z_EXIT)
        by_rsi = held & (rr[i] >= RSI_EXIT) & ~by_z
        by_time = held & (days_held >= MAX_DAYS) & ~by_z & ~by_rsi
        pending_sell = by_z | by_rsi | by_time
        sell_reason = np.where(by_z, "z_exit",
                               np.where(by_rsi, "rsi_exit",
                                        np.where(by_time, "time_limit", "")))

    t = pd.DataFrame(trades)
    print("trades: %d, mean gross return %.2f%%, positive %.0f%%"
          % (len(t), 100 * t["return"].mean(), 100 * (t["return"] > 0).mean()))
    print("\n%-14s %8s %10s %12s %12s" % ("reason", "share", "mean days",
                                          "mean return", "positive"))
    for reason, g in t.groupby("reason"):
        print("%-14s %7.0f%% %10.0f %11.2f%% %11.0f%%"
              % (reason, 100 * len(g) / len(t), g.days.mean(),
                 100 * g["return"].mean(), 100 * (g["return"] > 0).mean()))

    # ---------------------------------------------------------------
    print("\n=== 3. RETURNS AFTER EXIT ===")
    print("(security return minus market return, from the exit day)")
    print("%10s %14s %10s" % ("N", "excess", "positive"))
    for N in (21, 63, 126):
        excess = []
        for _, x in t.iterrows():
            i0, k = int(x["exit"]), int(x["security"])
            i1 = i0 + N
            if i1 >= n_days or np.isnan(c[i0, k]) or np.isnan(c[i1, k]):
                continue
            sec = c[i1, k] / c[i0, k] - 1
            mkt = (1 + market[i0 + 1:i1 + 1]).prod() - 1
            excess.append(sec - mkt)
        excess = np.array(excess)
        print("%10d %13.2f%% %9.0f%%"
              % (N, 100 * excess.mean(), 100 * (excess > 0).mean()))

    # --- trade outcome by trailing one-year return at entry ---
    print("\n=== 4. TRADES BY ONE-YEAR RETURN BEFORE ENTRY ===")
    trend = (prices / prices.shift(252) - 1).to_numpy()
    t["trend_252"] = [trend[int(e), int(k)] for e, k in zip(t["entry"], t["security"])]
    v = t.dropna(subset=["trend_252"])
    print("median one-year return before entry: %.1f%%, median one-quarter return: %.1f%%"
          % (100 * v.trend_252.median(),
             100 * np.nanmedian([(prices.to_numpy()[int(e), int(k)]
                                  / prices.to_numpy()[max(int(e) - 63, 0), int(k)] - 1)
                                 for e, k in zip(t["entry"], t["security"])])))
    bins = [-np.inf, -0.5, -0.25, 0, 0.25, np.inf]
    labels = ["below -50%", "-50 to -25%", "-25 to 0%", "0 to +25%", "above +25%"]
    v = v.assign(bucket=pd.cut(v.trend_252, bins=bins, labels=labels))
    print("%-14s %8s %12s %10s %12s" % ("bucket", "share", "mean return",
                                        "positive", "time limit"))
    for b, g in v.groupby("bucket", observed=True):
        print("%-14s %7.0f%% %11.2f%% %9.0f%% %11.0f%%"
              % (b, 100 * len(g) / len(t), 100 * g["return"].mean(),
                 100 * (g["return"] > 0).mean(),
                 100 * (g.reason == "time_limit").mean()))

    t.to_csv(RESULTS / "diagnostics_trades.csv", index=False)
    print("\nsaved: diagnostics_horizons.csv, diagnostics_trades.csv")


if __name__ == "__main__":
    main()
