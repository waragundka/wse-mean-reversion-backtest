#!/usr/bin/env python3
"""Returns after distress-related current reports (ESPI).

Current reports are classified by subject into five categories:
bankruptcy, restructuring, delayed publication of financial statements,
audit issues and financial distress. Excess return over the market is
measured from the first session after the report date, because reports
are published during the day.

Two estimates are reported:

- Cross-sectional: mean excess return over all events. Events cluster in
  time, so the p-value is obtained by resampling event dates rather than
  individual events.
- Calendar-time: a portfolio holding every security for N sessions after
  an event, with alpha from a regression on the market and Newey-West
  standard errors. Reported for the largest category.

The price index is flat after a security's last session, so post-event
returns of securities that stop trading within the horizon are recorded
as zero from that point. Excess returns are therefore likely overstated
for the categories in which trading ends soon after the report.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = Path(__file__).resolve().parent.parent
DATA, RESULTS = BASE / "data", BASE / "results"
SESSIONS_PER_YEAR = 252
HORIZONS = [5, 21, 63, 126]
B = 2000
SEED = 20260929

# Sessions on which every held security has a missing return.
warnings.filterwarnings("ignore", message="Mean of empty slice")

# Category labels in the input file are Polish.
CATEGORIES = {
    "upadlosc": "bankruptcy",
    "restrukturyzacja": "restructuring",
    "opoznienie": "reporting_delay",
    "audyt": "audit",
    "finanse": "financial_distress",
}


def main() -> None:
    prices = pd.read_parquet(DATA / "adjusted_prices.parquet")
    returns = pd.read_parquet(DATA / "daily_returns.parquet")
    universe = pd.read_parquet(DATA / "signal_universe.parquet").fillna(False)
    ev = pd.read_csv(DATA / "espi_events.csv", parse_dates=["data"])

    idx = prices.index
    cols = list(prices.columns)
    col_no = {k: i for i, k in enumerate(cols)}
    c = prices.to_numpy()
    r = returns.reindex(index=idx, columns=cols).to_numpy()
    u = universe.reindex(index=idx, columns=cols).fillna(False).to_numpy()

    market = np.array([np.nanmean(r[i, u[i]]) if u[i].sum() >= 20 else 0.0
                       for i in range(len(idx))])
    market = np.nan_to_num(market)
    market_cum = np.concatenate([[1.0], (1 + market).cumprod()])

    # event -> first session strictly after the report date
    position = idx.searchsorted(ev["data"].to_numpy(), side="right")
    ev = ev.assign(i=position)
    ev = ev[ev["isin"].isin(col_no)]
    ev = ev.assign(k=ev["isin"].map(col_no))
    ev = ev[(ev.i > 0) & (ev.i < len(idx) - max(HORIZONS))]
    print("events matched to quotations: %d, securities: %d"
          % (len(ev), ev["isin"].nunique()))

    gen = np.random.default_rng(SEED)
    rows = []

    print("\n%-20s %8s %9s %9s %9s %8s" %
          ("category", "N", "horizon", "excess", "positive", "p"))
    for name, g in ev.groupby("zdarzenie"):
        label = CATEGORIES.get(name, name)
        for H in HORIZONS:
            excess, dates = [], []
            for _, x in g.iterrows():
                i0, k = int(x.i), int(x.k)
                i1 = i0 + H
                if np.isnan(c[i0, k]) or np.isnan(c[i1, k]):
                    continue
                sec = c[i1, k] / c[i0, k] - 1
                mkt = market_cum[i1 + 1] / market_cum[i0 + 1] - 1
                excess.append(sec - mkt)
                dates.append(i0)
            if len(excess) < 20:
                continue
            excess = np.array(excess)
            dates = np.array(dates)

            # Resampling by date: events from the same session enter or
            # leave the bootstrap sample together.
            unique = np.unique(dates)
            mean = excess.mean()
            draws = np.empty(B)
            for b in range(B):
                chosen = gen.choice(unique, len(unique), replace=True)
                mask = np.concatenate([np.flatnonzero(dates == d) for d in chosen])
                draws[b] = excess[mask].mean()
            p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())

            print("%-20s %8d %9d %8.2f%% %8.0f%% %8.3f"
                  % (label, len(excess), H, 100 * mean,
                     100 * (excess > 0).mean(), p))
            rows.append({"category": label, "n": len(excess), "horizon": H,
                         "mean_excess": mean, "share_positive": (excess > 0).mean(),
                         "p": p})
        print()

    pd.DataFrame(rows).to_csv(RESULTS / "espi_events.csv", index=False)

    # --- calendar-time check for the largest category ---
    largest = ev["zdarzenie"].value_counts().index[0]
    print("CALENDAR-TIME PORTFOLIO: %s" % CATEGORIES.get(largest, largest))
    g = ev[ev["zdarzenie"] == largest]
    for H in HORIZONS:
        entries = np.zeros((len(idx), len(cols)), dtype=bool)
        for _, x in g.iterrows():
            entries[int(x.i), int(x.k)] = True
        held = (pd.DataFrame(entries).rolling(H, min_periods=1).max()
                .astype(bool).to_numpy())
        series = np.array([np.nanmean(r[i, held[i]]) if held[i].sum() else 0.0
                           for i in range(len(idx))])
        series = np.nan_to_num(series)
        m = sm.OLS(series, sm.add_constant(market)).fit(
            cov_type="HAC", cov_kwds={"maxlags": 10})
        print("  %3d sessions: annual alpha %+6.1f%%, p = %.3f"
              % (H, 100 * m.params[0] * SESSIONS_PER_YEAR, m.pvalues[0]))

    print("\nsaved: espi_events.csv")


if __name__ == "__main__":
    main()
