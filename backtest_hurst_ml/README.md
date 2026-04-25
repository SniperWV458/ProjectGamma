
# Hurst Diagnostics v4 Update

This update replaces only:

```text
hurst_diagnostics.py
```

It improves two things:

## 1. Main plots now include H10 / H30 / H60 + MLExit variants

The previous main plot included H30 + MLExit but not H10 + MLExit. This was incomplete because H10-filter-short-MLExit can be one of the strongest combined variants.

The updated main plot includes:

```text
v2-A-beta-5d-LS
v2-A-beta-5d-LS-MLExit
v2-A-beta-5d-LS-H10-filter-short
v2-A-beta-5d-LS-H30-filter-short
v2-A-beta-5d-LS-H60-filter-short
v2-A-beta-5d-LS-H10-filter-short-MLExit
v2-A-beta-5d-LS-H30-filter-short-MLExit
v2-A-beta-5d-LS-H60-filter-short-MLExit
v2-A-beta-5d-LS-H30-filter-long
v2-A-beta-5d-LS-H30-filter-both
```

So the plot is a pre-specified diagnostic comparison, not cherry-picking the best line.

## 2. Kept-vs-filtered analysis is now trade-level

The old figure compared same-day row-level short payoff, which is too weak because Hurst is about persistence, not same-day payoff.

The new analysis converts baseline short positions into trade entries and compares:

```text
realized holding-period short payoff
forward 5-day short payoff
forward 10-day short payoff
```

for:

```text
baseline short trades kept by Hurst
baseline short trades filtered by Hurst
```

New outputs:

```text
04_short_trade_kept_vs_filtered_distribution.png
short_trade_kept_vs_filtered_<hurst_label>.csv
short_trade_kept_vs_filtered_summary_<hurst_label>.csv
```

## How to use

Copy the updated `hurst_diagnostics.py` into:

```text
D:\HKU资料\Capstone Project\backtest\backtest\
```

Then rerun:

```bash
cd /d "D:\HKU资料\Capstone Project\backtest\backtest"
D:\python\python.exe run_sentiment_gex.py --primary-only --rebuild-slim
```

or full run:

```bash
D:\python\python.exe run_sentiment_gex.py --rebuild-slim
```
