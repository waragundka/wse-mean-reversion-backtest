#!/usr/bin/env python3
"""Figure 1: parameter surface in both windows.

Mean annual excess return over the equal-weighted market by entry
thresholds, averaged over the nine exit-threshold combinations, net of
costs. A diverging scale centred at zero is shared by both panels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

BASE = Path(__file__).resolve().parent.parent
RESULTS, FIGURES = BASE / "results", BASE / "figures"
WINDOWS = [("2005-2020", "2005 to 2020"), ("2021-2026", "2021 to 2026")]
SESSIONS_PER_YEAR = 252

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
CMAP = LinearSegmentedColormap.from_list(
    "red_grey_blue", ["#b3261e", "#e58a7f", "#f0efec", "#7fa3e0", "#1f4fb3"])


def surface(label: str) -> pd.DataFrame:
    s = pd.read_csv(RESULTS / label / "statistics.csv")
    s = s[s.costs == "net"]
    m = pd.read_parquet(RESULTS / label / "market.parquet")["market"].to_numpy()
    r_market = (1 + m).prod() ** (SESSIONS_PER_YEAR / len(m)) - 1
    s = s.assign(excess=100 * (s.annual_return - r_market))
    return s.pivot_table(index="z_entry", columns="rsi_entry", values="excess",
                         aggfunc="mean")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    tabs = [surface(label) for label, _ in WINDOWS]
    limit = max(abs(t.to_numpy()).max() for t in tabs)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), facecolor=SURFACE,
                             constrained_layout=True)
    for ax, tab, (_, title) in zip(axes, tabs, WINDOWS):
        im = ax.imshow(tab.to_numpy(), cmap=CMAP, norm=norm, aspect="auto")
        ax.set_xticks(range(tab.shape[1]), [str(c) for c in tab.columns])
        ax.set_yticks(range(tab.shape[0]), ["%.1f" % v for v in tab.index])
        ax.set_xlabel("RSI entry threshold")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        # white gaps between cells
        ax.set_xticks(np.arange(-0.5, tab.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, tab.shape[0]), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                v = tab.iat[i, j]
                ax.text(j, i, "%+.1f" % v, ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 0.6 * limit else INK)
    axes[0].set_ylabel("z-score entry threshold")
    bar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    bar.set_label("annual excess return, pp", color=MUTED)
    bar.outline.set_visible(False)
    fig.savefig(FIGURES / "parameter_surface.png", dpi=200, facecolor=SURFACE)
    print("saved: figures/parameter_surface.png")


if __name__ == "__main__":
    main()
