#!/usr/bin/env bash
# Reproduces all results. Requires the input files described in the README
# to be present in data/. Runtime: several minutes.
set -euo pipefail
cd "$(dirname "$0")"

run_windows() {            # $1 = label suffix ("" or "_dl")
    python3 code/01_signals.py
    python3 code/02_backtest.py 2021-01-01 2026-12-31 "2021-2026$1"
    python3 code/03_tests.py "2021-2026$1"
    python3 code/04_robustness.py "2021-2026$1"
    python3 code/02_backtest.py 2005-01-01 2020-12-31 "2005-2020$1"
    python3 code/03_tests.py "2005-2020$1"
    python3 code/04_robustness.py "2005-2020$1"
    python3 code/05_period_comparison.py "2005-2020$1" "2021-2026$1"
}

# 1. Baseline: no delisting returns
run_windows ""
python3 code/06_diagnostics.py
python3 code/07_distress_filter.py
python3 code/08_espi_events.py

# 2. Sensitivity: delisting returns assigned
python3 code/09_delisting_returns.py
DATA_SUFFIX=_dl run_windows "_dl"

python3 code/10_figures.py
