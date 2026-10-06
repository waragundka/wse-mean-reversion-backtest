#!/usr/bin/env python3
"""Inference on the grid: alpha, multiple-testing adjustment, Reality Check.

With 144 variants evaluated on the same data, the best variant can
exceed the benchmark in the absence of any true effect. The ranking is
therefore not used as evidence. Three procedures are applied:

- Alpha from a regression on the benchmark. The strategy is invested
  only part of the time, so its return is regressed on the market return
  and the intercept is reported.
- Newey-West standard errors, to allow for serial correlation in daily
  returns.
- White's (2000) Reality Check with the stationary bootstrap of Politis
  and Romano (1994). The null hypothesis is that no variant in the grid
  has a positive expected excess return over the benchmark. Resampling
  is done in blocks to preserve serial dependence.

Holm (1979) adjusted p-values are reported for the family of 144 alphas.

Usage: python3 code/03_tests.py [LABEL]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = Path(__file__).resolve().parent.parent
LABEL = sys.argv[1] if len(sys.argv) > 1 else "2021-2026"
RESULTS = BASE / "results" / LABEL

SESSIONS_PER_YEAR = 252
NW_LAGS = 10             # Newey-West lags
MEAN_BLOCK = 10          # mean block length in the bootstrap
B = 2000                 # bootstrap replications
SEED = 20260929


def stationary_bootstrap(n: int, b: int, mean_block: float, gen) -> np.ndarray:
    """Index matrix for the stationary bootstrap of Politis and Romano (1994).

    Blocks have geometrically distributed lengths and wrap around the end
    of the sample.
    """
    p = 1.0 / mean_block
    idx = np.empty((b, n), dtype=np.int64)
    for i in range(b):
        positions = np.empty(n, dtype=np.int64)
        j = 0
        while j < n:
            start = gen.integers(0, n)
            length = min(gen.geometric(p), n - j)
            positions[j:j + length] = (start + np.arange(length)) % n
            j += length
        idx[i] = positions
    return idx


def measures(series: np.ndarray) -> dict:
    years = len(series) / SESSIONS_PER_YEAR
    total = (1 + series).prod() - 1
    curve = (1 + series).cumprod()
    drawdown = float((curve / np.maximum.accumulate(curve) - 1).min())
    vol = series.std() * np.sqrt(SESSIONS_PER_YEAR)
    return {
        "annual_return": (1 + total) ** (1 / years) - 1,
        "volatility": vol,
        "sharpe": (series.mean() * SESSIONS_PER_YEAR) / vol if vol > 0 else np.nan,
        "max_drawdown": drawdown,
    }


def main() -> None:
    series = pd.read_parquet(RESULTS / "series.parquet")
    market = pd.read_parquet(RESULTS / "market.parquet")["market"].to_numpy()
    summary = pd.read_csv(RESULTS / "summary.csv")

    n = len(market)
    print("sessions: %d, series in file: %d" % (n, series.shape[1]))

    m_market = measures(market)
    print("\nMARKET (equal-weighted liquid universe)")
    print("  annual return %.1f%%, volatility %.1f%%, Sharpe %.2f, max drawdown %.1f%%"
          % (100 * m_market["annual_return"], 100 * m_market["volatility"],
             m_market["sharpe"], 100 * m_market["max_drawdown"]))

    X = sm.add_constant(market)
    rows = []
    for col in series.columns:
        s = series[col].to_numpy()
        excess = s - market
        model = sm.OLS(s, X).fit(cov_type="HAC",
                                 cov_kwds={"maxlags": NW_LAGS})
        alpha_d, beta = model.params
        rows.append({
            "series": col,
            "variant": int(col[1:4]),
            "costs": "net" if col.endswith("_net") else "gross",
            **measures(s),
            "annual_alpha": alpha_d * SESSIONS_PER_YEAR,
            "beta": beta,
            "alpha_t": model.tvalues[0],
            "alpha_p": model.pvalues[0],
            "mean_daily_excess": excess.mean(),
        })
    t = pd.DataFrame(rows)
    t = t.merge(summary.drop(columns=["costs"]).drop_duplicates("variant"),
                on="variant", how="left")
    t.to_csv(RESULTS / "statistics.csv", index=False)

    net = t[t.costs == "net"].copy()

    print("\nFULL GRID, net of costs (%d variants)" % len(net))
    print("  annual return: median %.1f%%, max %.1f%%; Sharpe: median %.2f"
          % (100 * net.annual_return.median(), 100 * net.annual_return.max(),
             net.sharpe.median()))
    print("  variants with annual return above the market: %d of %d"
          % ((net.annual_return > m_market["annual_return"]).sum(), len(net)))
    print("  annual alpha: median %.1f%%, min %.1f%%, max %.1f%%"
          % (100 * net.annual_alpha.median(), 100 * net.annual_alpha.min(),
             100 * net.annual_alpha.max()))
    print("  positive alpha: %d of %d variants"
          % ((net.annual_alpha > 0).sum(), len(net)))

    # --- multiple testing ---
    n_significant = (net.alpha_p < 0.05).sum()
    print("\nUNADJUSTED (two-sided p < 0.05)")
    print("  significant: %d of %d (positive alpha: %d, negative alpha: %d)"
          % (n_significant, len(net),
             ((net.alpha_p < 0.05) & (net.annual_alpha > 0)).sum(),
             ((net.alpha_p < 0.05) & (net.annual_alpha < 0)).sum()))

    ordered = net.sort_values("alpha_p").reset_index(drop=True)
    m = len(ordered)
    holm = ordered.alpha_p.to_numpy() * (m - np.arange(m))
    holm = np.maximum.accumulate(holm)
    ordered["p_holm"] = np.clip(holm, 0, 1)
    sig = ordered[ordered.p_holm < 0.05]
    print("\nHOLM-ADJUSTED (familywise error rate over the grid)")
    print("  significant: %d of %d (positive alpha: %d, negative alpha: %d)"
          % (len(sig), m, (sig.annual_alpha > 0).sum(), (sig.annual_alpha < 0).sum()))

    # --- White's Reality Check ---
    print("\nREALITY CHECK (White, 2000), %d stationary-bootstrap replications" % B)
    columns = net.series.tolist()
    F = np.column_stack([series[k].to_numpy() - market for k in columns])
    means = F.mean(axis=0)
    V = np.sqrt(n) * means.max()

    gen = np.random.default_rng(SEED)
    idx = stationary_bootstrap(n, B, MEAN_BLOCK, gen)
    Vb = np.empty(B)
    for i in range(B):
        mb = F[idx[i]].mean(axis=0)
        Vb[i] = np.sqrt(n) * (mb - means).max()
    p_rc = float((Vb >= V).mean())

    best = net.loc[net.mean_daily_excess.idxmax()]
    print("  best variant: no. %d (z_entry %.1f, rsi_entry %d, z_exit %.1f, rsi_exit %d)"
          % (best.variant, best.z_entry, best.rsi_entry, best.z_exit, best.rsi_exit))
    print("  its annual return %.1f%%, Sharpe %.2f, max drawdown %.1f%%, alpha %.1f%% (p = %.3f)"
          % (100 * best.annual_return, best.sharpe, 100 * best.max_drawdown,
             100 * best.annual_alpha, best.alpha_p))
    print("  its mean excess return over the market: %.1f pp per year"
          % (100 * best.mean_daily_excess * SESSIONS_PER_YEAR))
    print("  statistic V = %.3f" % V)
    print("  bootstrap 95%% quantile = %.3f" % np.quantile(Vb, 0.95))
    print("  Reality Check p-value = %.4f" % p_rc)

    # --- transaction costs ---
    gross = t[t.costs == "gross"].set_index("variant")["annual_return"]
    net_r = net.set_index("variant")["annual_return"]
    diff = (gross - net_r).dropna()
    print("\nTRANSACTION COSTS")
    print("  median number of trades: %.0f" % net.n_trades.median())
    print("  median difference in annual return, gross minus net: %.1f pp"
          % (100 * diff.median()))

    ordered.to_csv(RESULTS / "significance.csv", index=False)
    print("\nsaved: statistics.csv, significance.csv")


if __name__ == "__main__":
    main()
