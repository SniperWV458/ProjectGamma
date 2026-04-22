from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from backtest_framework import (
    BacktestConfig,
    BaseSignal,
    BaseStrategy,
    equal_weight_daily_returns,
    load_panel,
    run_backtest,
)

PANEL_DEFAULT = Path("data/backtest_panel_main.parquet")
N_PER_LEG = 25
START_DATE = "2018-06-01"
END_DATE = None


class SentimentAvgSignal(BaseSignal):
    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        out = panel.copy()
        out["signal"] = pd.to_numeric(out["sent_avg"], errors="coerce")
        return out


class LongShortRankStrategy(BaseStrategy):
    def __init__(self, n_per_leg: int = 25) -> None:
        self.n_per_leg = int(n_per_leg)

    def target_weights(self, panel: pd.DataFrame) -> pd.DataFrame:
        sig_col = "signal"
        min_names = 2 * self.n_per_leg + 5
        rows = []
        for dt, g in panel.groupby("date", sort=True):
            g = g.loc[g[sig_col].notna()].copy()
            if len(g) < min_names:
                continue
            g = g.sort_values(sig_col, ascending=False, kind="mergesort")
            n = min(self.n_per_leg, len(g) // 2)
            if n < 1:
                continue
            longs = g.head(n)
            shorts = g.tail(n)
            w_long = 0.5 / n
            w_short = -0.5 / n
            for _, r in longs.iterrows():
                rows.append({"permno": r["permno"], "date": dt, "weight": w_long})
            for _, r in shorts.iterrows():
                rows.append({"permno": r["permno"], "date": dt, "weight": w_short})
        if not rows:
            return pd.DataFrame(columns=["permno", "date", "weight"])
        out = pd.DataFrame(rows)
        out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        return out


class ShortOnlyRankStrategy(BaseStrategy):
    """Equal-weight short the lowest-N names by (lagged) sentiment each day."""
    def __init__(self, n_short: int = 25) -> None:
        self.n_short = int(n_short)

    def target_weights(self, panel: pd.DataFrame) -> pd.DataFrame:
        sig_col = "signal"
        min_names = self.n_short + 5
        rows = []
        for dt, g in panel.groupby("date", sort=True):
            g = g.loc[g[sig_col].notna()].copy()
            if len(g) < min_names:
                continue
            g = g.sort_values(sig_col, ascending=True, kind="mergesort")
            n = min(self.n_short, len(g))
            if n < 1:
                continue
            w = -1.0 / n
            for _, r in g.head(n).iterrows():
                rows.append({"permno": r["permno"], "date": dt, "weight": w})
        if not rows:
            return pd.DataFrame(columns=["permno", "date", "weight"])
        out = pd.DataFrame(rows)
        out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        return out


def _print_block(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}\n")
    print(table.to_string())


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PANEL_DEFAULT
    if not path.exists():
        raise SystemExit(f"Panel not found: {path}")

    cfg = BacktestConfig(
        panel_path=path,
        start_date=pd.Timestamp(START_DATE) if START_DATE else None,
        end_date=pd.Timestamp(END_DATE) if END_DATE else None,
        signal_col="signal",
        signal_lag_days=1,
        missing_return_policy="zero_contribution",
        long_only=False,
        short_only=False,
        transaction_cost_bps=5.0,
    )

    panel_full = load_panel(path)
    if cfg.start_date is not None:
        panel_full = panel_full.loc[panel_full["date"] >= cfg.start_date]
    if cfg.end_date is not None:
        panel_full = panel_full.loc[panel_full["date"] <= cfg.end_date]
    benchmark = equal_weight_daily_returns(panel_full, cfg.return_col)

    res_ls = run_backtest(
        cfg,
        SentimentAvgSignal(),
        LongShortRankStrategy(n_per_leg=N_PER_LEG),
        benchmark_daily_returns=benchmark,
    )

    cfg_short = replace(cfg, short_only=True)
    res_short = run_backtest(
        cfg_short,
        SentimentAvgSignal(),
        ShortOnlyRankStrategy(n_short=N_PER_LEG),
        benchmark_daily_returns=benchmark,
    )

    _print_block("Long–short (top/bottom %d)" % N_PER_LEG, res_ls.performance.to_pandas().T)
    _print_block("Short-only (lowest %d sentiment)" % N_PER_LEG, res_short.performance.to_pandas().T)
    if res_ls.benchmark_performance is not None:
        _print_block("Benchmark (EW all names)", res_ls.benchmark_performance.to_pandas().T)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    res_ls.nav.plot(ax=axes[0], label="Long–short")
    res_short.nav.plot(ax=axes[0], label="Short-only (low sent)")
    if res_ls.benchmark_nav is not None:
        res_ls.benchmark_nav.plot(ax=axes[0], label="EW benchmark", alpha=0.85)
    axes[0].set_title("Net asset value (1 = start)")
    axes[0].set_ylabel("NAV")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    for name, series in [
        ("Long–short", res_ls.nav),
        ("Short-only", res_short.nav),
    ]:
        peak = series.cummax()
        dd = (peak - series) / peak
        dd.plot(ax=axes[1], label=name)
    axes[1].set_title("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Sentiment portfolios (signal = sent_avg, lag = 1 day)", y=1.02, fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
