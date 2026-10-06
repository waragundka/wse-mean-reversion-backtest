#!/usr/bin/env python3
"""Delisting returns.

The adjusted price index is flat after a security's last session, so a
position in a security removed from trading is closed at the last quoted
price. For bankruptcies this understates the loss. Following the approach
of Shumway (1997), a delisting return is assigned on the first session
after the last quotation, according to the reason for removal recorded by
the exchange:

    bankruptcy, liquidation                         -100 per cent
    exclusion, regulatory request, rule violation    -30 per cent
    tender offer, merger, dematerialisation            0 per cent

These values are assumptions. For removals without a recorded reason an
assumed value is applied, with three calibrations for sensitivity
analysis (DELISTING_CALIBRATION = A, B or C):

    A  -20.9 per cent
    B  -30.0 per cent  (default)
    C  -36.3 per cent

The adjustment can only lower the returns of a rule that buys securities
after declines, since bankruptcies are concentrated among them.

Outputs: daily_returns_dl.parquet, adjusted_prices_dl.parquet. To use
them, run the pipeline with DATA_SUFFIX=_dl.

Reason texts in the exchange files are in Polish; the patterns below
match the Polish wording.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

CALIBRATIONS = {"A": -0.209, "B": -0.300, "C": -0.363}
CALIBRATION = os.environ.get("DELISTING_CALIBRATION", "B")
UNKNOWN = CALIBRATIONS[CALIBRATION]

# A series is treated as ended if its last return is more than 40
# sessions before the end of the sample.
MARGIN = 40


def return_from_reason(p: str) -> float:
    t = (p or "").lower() if isinstance(p, str) else ""
    if re.search(r"upadło|likwidac", t):
        return -1.00
    if re.search(r"wykluczen|żądanie knf|narusza przepisy|interes i bezpieczeń", t):
        return -0.30
    if re.search(r"dematerializ|połączeni|fuzj|wezwani|wniosek emitenta|zezwolenie knf", t):
        return 0.00
    return UNKNOWN


def main() -> None:
    returns = pd.read_parquet(DATA / "daily_returns.parquet")
    idx, cols = returns.index, list(returns.columns)
    rn = returns.to_numpy()

    last = np.array([
        np.max(np.flatnonzero(~np.isnan(rn[:, k])))
        if (~np.isnan(rn[:, k])).any() else -1
        for k in range(len(cols))])
    ended = [(k, last[k]) for k in range(len(cols))
             if 0 <= last[k] < len(idx) - MARGIN]
    print("securities in the panel: %d, series ended: %d" % (len(cols), len(ended)))

    # reasons from both exchange files; the first one found is kept
    reasons = {}
    for file in ("wse_delistings.csv", "wse_suspensions_exclusions.csv"):
        d = pd.read_csv(DATA / file)
        d = d[d.get("mic", "XWAR").eq("XWAR") if "mic" in d else slice(None)]
        if "rodzaj" in d:
            d = d[d["rodzaj"].str.contains("wycofanie", case=False, na=False)]
        for _, x in d.iterrows():
            i, p = x.get("isin"), x.get("przyczyna")
            if isinstance(i, str) and i not in reasons:
                reasons[i] = p

    known = sum(1 for k, _ in ended if cols[k] in reasons)
    print("with a recorded reason: %d (%.0f%%); others under calibration %s = %.1f%%"
          % (known, 100 * known / len(ended), CALIBRATION, 100 * UNKNOWN))

    counts = {}
    new = rn.copy()
    for k, o in ended:
        z = return_from_reason(reasons.get(cols[k]))
        counts[z] = counts.get(z, 0) + 1
        if o + 1 < len(idx):
            # the delisting return is placed on the first session after the last quotation
            new[o + 1, k] = z
    print("\nassigned delisting returns:")
    for z in sorted(counts):
        print("   %+6.1f%%  %3d securities" % (100 * z, counts[z]))

    r2 = pd.DataFrame(new, index=idx, columns=cols)
    r2.to_parquet(DATA / "daily_returns_dl.parquet")

    prices = 100 * (1 + r2.fillna(0)).cumprod()
    prices = prices.where(r2.notna().cumsum() > 0)
    prices.to_parquet(DATA / "adjusted_prices_dl.parquet")
    print("\nsaved: daily_returns_dl.parquet, adjusted_prices_dl.parquet")


if __name__ == "__main__":
    main()
