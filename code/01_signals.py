#!/usr/bin/env python3
"""Signals: price z-score, Wilder RSI and the liquid universe.

Computed once for all securities and stored, because the same matrices
are reused by every variant in the threshold grid.

Three choices affect the validity of the results:

1. Adjusted prices are used. On raw prices a stock split appears as a
   price decline and would trigger the entry rule.
2. Securities later removed from trading remain in the panel, so the
   sample is not conditioned on survival.
3. The liquidity filter is a rolling median of turnover lagged by one
   session. Membership on day t uses information up to day t-1.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
# Empty suffix: inputs without delisting returns. "_dl": inputs produced
# by 09_delisting_returns.py. Both sets can be kept side by side.
SUFFIX = os.environ.get("DATA_SUFFIX", "")

MEAN_WINDOW = 63         # sessions, approximately one quarter
RSI_WINDOW = 14
LIQUIDITY_WINDOW = 63
TURNOVER_THRESHOLD = 100_000    # median turnover per session, PLN


def wilder_rsi(prices: pd.DataFrame, window: int = RSI_WINDOW) -> pd.DataFrame:
    """RSI for the whole price matrix, with Wilder's smoothing (alpha = 1/n)."""
    diff = prices.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    mean_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    mean_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = mean_gain / mean_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def main() -> None:
    prices = pd.read_parquet(DATA / ("adjusted_prices%s.parquet" % SUFFIX))
    returns = pd.read_parquet(DATA / ("daily_returns%s.parquet" % SUFFIX))
    panel = pd.read_parquet(DATA / "panel.parquet",
                            columns=["data", "isin", "obrot_pln"])

    print("prices: %d sessions x %d securities" % prices.shape)

    mean = prices.rolling(MEAN_WINDOW).mean()
    std = prices.rolling(MEAN_WINDOW).std()
    z = (prices - mean) / std.replace(0, np.nan)
    rsi = wilder_rsi(prices)

    turnover = panel.pivot_table(index="data", columns="isin", values="obrot_pln",
                                 aggfunc="last").reindex(index=prices.index,
                                                         columns=prices.columns)
    # Rolling median lagged by one session: the decision on day t uses
    # turnover up to and including day t-1.
    liquid = (turnover.rolling(LIQUIDITY_WINDOW, min_periods=20).median()
              .shift(1) > TURNOVER_THRESHOLD)
    listed = returns.reindex(index=prices.index, columns=prices.columns).notna()
    universe = liquid & listed

    print("mean number of securities in the universe per session (from 2021): %.0f"
          % universe.loc["2021-01-01":].sum(axis=1).mean())

    for name, frame in (("z", z), ("rsi", rsi), ("universe", universe)):
        frame.to_parquet(DATA / ("signal_%s%s.parquet" % (name, SUFFIX)))
    print("saved: signal_z, signal_rsi, signal_universe")


if __name__ == "__main__":
    main()
