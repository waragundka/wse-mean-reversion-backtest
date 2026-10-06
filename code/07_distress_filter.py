#!/usr/bin/env python3
"""Can price-based features separate failing securities at the signal date?

At the signal date the security has fallen substantially. The script
asks whether features computed from price, turnover and daily range
distinguish securities that subsequently fail from those that do not,
and whether excluding the former changes the outcome of the signal.

Removal from trading is not treated as failure. A security is classified
as failed when its return series ends within one year of the signal and
its return over the last year of trading is below -50 per cent. Tender
offers and squeeze-outs, which often end with a gain, are thereby
excluded from the failure class.

All features use information available on the signal date.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA, RESULTS = BASE / "data", BASE / "results"

Z_ENTRY, RSI_ENTRY = -2.0, 30
HORIZON = 126            # sessions ahead over which the outcome is measured
FAILURE_WINDOW = 252     # sessions after the signal within which the series ends
FAILURE_THRESHOLD = -0.5 # return over the last year of trading
MARGIN = 40              # a series ending this close to the sample end is treated as live


def main() -> None:
    prices = pd.read_parquet(DATA / "adjusted_prices.parquet")
    returns = pd.read_parquet(DATA / "daily_returns.parquet")
    z = pd.read_parquet(DATA / "signal_z.parquet")
    rsi = pd.read_parquet(DATA / "signal_rsi.parquet")
    universe = pd.read_parquet(DATA / "signal_universe.parquet").fillna(False)
    panel = pd.read_parquet(DATA / "panel.parquet",
        columns=["data", "isin", "zamkniecie", "max", "min", "wolumen", "obrot_pln"])

    cols = prices.columns
    idx = prices.index

    def tab(name):
        return panel.pivot_table(index="data", columns="isin", values=name,
                                 aggfunc="last").reindex(index=idx, columns=cols)

    close, hi, lo = tab("zamkniecie"), tab("max"), tab("min")
    volume, turnover = tab("wolumen"), tab("obrot_pln")

    # --- features, all known on the signal date ---
    C = {}
    C["trend_252"] = prices / prices.shift(252) - 1
    C["drawdown_252"] = prices / prices.rolling(252, min_periods=60).max() - 1
    C["volatility"] = returns.rolling(63).std() * np.sqrt(252)
    C["price"] = close
    # recent turnover relative to its one-year median
    C["relative_turnover"] = (turnover.rolling(21).median()
                              / turnover.rolling(252, min_periods=60).median())
    # Amihud (2002): absolute return per million PLN of turnover
    C["amihud"] = (returns.abs() / turnover.replace(0, np.nan) * 1e6).rolling(63).median()
    C["no_trade_share"] = (volume.fillna(0) == 0).rolling(63).mean()
    # daily high-low range relative to the close
    C["daily_range"] = ((hi - lo) / close.replace(0, np.nan)).rolling(63).median()

    # --- events ---
    signal = (universe & (z <= Z_ENTRY) & (rsi <= RSI_ENTRY)).to_numpy()
    n_days, n_sec = signal.shape

    # last session with an observed return for each security
    rn = returns.to_numpy()
    last = np.array([np.max(np.flatnonzero(~np.isnan(rn[:, k])))
                     if (~np.isnan(rn[:, k])).any() else -1
                     for k in range(n_sec)])
    sample_end = n_days - 1
    c = prices.to_numpy()

    ended = (last >= 0) & (last < sample_end - MARGIN)
    last_year = np.full(n_sec, np.nan)
    for k in np.flatnonzero(ended):
        o = last[k]
        start = max(o - 252, 0)
        if np.isnan(c[start, k]) or np.isnan(c[o, k]):
            continue
        last_year[k] = c[o, k] / c[start, k] - 1
    failed = last_year < FAILURE_THRESHOLD

    print("securities: %d, series ended before the sample end: %d"
          % (n_sec, ended.sum()))
    known = ~np.isnan(last_year)
    print("  last-year return above +20%%: %.0f%%, below -20%%: %.0f%%, below -50%% (failure): %d"
          % (100 * (last_year[known] > 0.2).mean(), 100 * (last_year[known] < -0.2).mean(),
             failed.sum()))

    um = universe.to_numpy()
    market = np.array([np.nanmean(rn[i, um[i]]) if um[i].sum() >= 20 else 0.0
                       for i in range(n_days)])
    market = np.nan_to_num(market)
    market_cum = np.concatenate([[1.0], (1 + market).cumprod()])

    features = {k: v.to_numpy() for k, v in C.items()}
    data = []
    for i in range(252, n_days - HORIZON):
        for k in np.flatnonzero(signal[i]):
            o = last[k]
            j = min(i + HORIZON, o)       # to the horizon or to the end of trading
            if j <= i or np.isnan(c[i, k]) or np.isnan(c[j, k]):
                continue
            result = c[j, k] / c[i, k] - 1
            mkt = market_cum[j + 1] / market_cum[i + 1] - 1
            w = {"day": i, "security": k, "excess": result - mkt,
                 "failed": bool(failed[k] and o - i <= FAILURE_WINDOW)}
            for name, m in features.items():
                w[name] = m[i, k]
            data.append(w)

    d = pd.DataFrame(data).dropna()
    print("\nevents with complete features: %d" % len(d))
    print("followed by failure within one year: %d (%.2f%%)"
          % (d.failed.sum(), 100 * d.failed.mean()))
    print("mean excess return after %d sessions: %+.2f%%" % (HORIZON, 100 * d.excess.mean()))
    print("mean excess return excluding failures: %+.2f%% (%d events)"
          % (100 * d.loc[~d.failed, "excess"].mean(), (~d.failed).sum()))

    print("\n=== DISTRIBUTION OF OUTCOMES ===")
    bins = [-np.inf, -0.5, -0.25, -0.10, 0, np.inf]
    labels = ["below -50%", "-50 to -25%", "-25 to -10%", "-10 to 0%", "above 0%"]
    bucket = pd.cut(d.excess, bins=bins, labels=labels)
    for b, g in d.groupby(bucket, observed=True):
        print("%-14s share %5.1f%%   contribution to the mean %+6.2f pp"
              % (b, 100 * len(g) / len(d), 100 * g.excess.sum() / len(d)))

    print("\n=== FEATURE QUINTILES ===")
    print("mean excess return after %d sessions and failure rate, by quintile\n" % HORIZON)
    summary = []
    for name in features:
        # A feature with too few distinct values cannot be split into
        # quintiles and is skipped.
        try:
            d["q"] = pd.qcut(d[name], 5, labels=False, duplicates="drop") + 1
        except ValueError:
            print("%-18s skipped, too few distinct values" % name)
            continue
        if d["q"].nunique() < 3:
            print("%-18s skipped, too few distinct values" % name)
            continue
        g = d.groupby("q", observed=True).agg(
            excess=("excess", "mean"), failures=("failed", "mean"),
            n=("excess", "size"))
        spread = 100 * (g.excess.iloc[-1] - g.excess.iloc[0])
        print("%-18s %s   spread Q5-Q1: %+6.1f pp" % (
            name, "  ".join("%+5.1f%%" % (100 * x) for x in g.excess), spread))
        print("%-18s %s   failure rate" % ("", "  ".join("%5.1f%%" % (100 * x)
                                                        for x in g.failures)))
        summary.append({"feature": name, "spread_pp": spread,
                        "failure_rate_q1": g.failures.iloc[0],
                        "failure_rate_q5": g.failures.iloc[-1]})

    # --- combined filter: drop the highest-risk quintile of three features ---
    keep = ((d.volatility <= d.volatility.quantile(0.8))
            & (d.price >= d.price.quantile(0.2))
            & (d.drawdown_252 >= d.drawdown_252.quantile(0.2)))
    print("\n=== COMBINED FILTER ===")
    print("(excludes the top volatility quintile, the bottom price quintile")
    print(" and the bottom drawdown quintile)")
    print("events retained: %d of %d (%.0f%%)" % (keep.sum(), len(d), 100 * keep.mean()))
    print("failure rate:       %.2f%% -> %.2f%%"
          % (100 * d.failed.mean(), 100 * d.loc[keep, "failed"].mean()))
    print("mean excess return: %+.2f%% -> %+.2f%%"
          % (100 * d.excess.mean(), 100 * d.loc[keep, "excess"].mean()))

    p = pd.DataFrame(summary).sort_values("spread_pp", key=abs, ascending=False)
    d.drop(columns=["q"]).to_csv(RESULTS / "filter_events.csv", index=False)
    p.to_csv(RESULTS / "filter_features.csv", index=False)
    print("\nsaved: filter_events.csv, filter_features.csv")


if __name__ == "__main__":
    main()
