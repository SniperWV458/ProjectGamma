"""
Run and compare GEX-Fragility + Sentiment strategy variants.

Gate flavours
    simple  : D1_neg & D4_top_tercile                (all ~232 stocks, rarely fires)
    v2      : D5_below_flip & D4_top_tercile          (19 stocks, ~14% hit-rate)
    v3      : D5_below_flip                           (19 stocks, ~43% hit-rate, broadest)

Borrow cost stressed at 50 bps/yr and 200 bps/yr for short legs.

Usage:
    python backtest/run_sentiment_gex.py [--rebuild-factors]
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_framework import (
    BacktestConfig,
    equal_weight_daily_returns,
    load_panel,
    performance_metrics,
    run_backtest,
)
from gex_factors_builder import build_contract_factors, CONTRACT_PATH, OUTPUT_PATH
from sentiment_gex_strategy import FragilityGEXSignal, GEXSentimentStrategy

ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "data" / "backtest_panel_main.parquet"
SLIM_PANEL_PATH = ROOT / "data" / "backtest_panel_slim.parquet"
START_DATE = "2018-06-01"

SLIM_COLS = [
    "permno", "secid", "ticker", "date",
    "ret",
    "gex_net_gex_1pct", "gex_call_gex_1pct", "gex_put_gex_1pct", "gex_spot",
    "sent_avg", "sent_posts",
    "bs_b_mkt",
    "market_equity", "dollar_volume",
]


def _ensure_slim_panel() -> Path:
    if SLIM_PANEL_PATH.exists():
        return SLIM_PANEL_PATH
    print("Creating slim panel ...")
    try:
        slim = pd.read_parquet(PANEL_PATH, columns=SLIM_COLS)
    except Exception:
        import pyarrow.parquet as pq
        available = [f.name for f in pq.ParquetFile(PANEL_PATH).schema_arrow]
        slim = pd.read_parquet(PANEL_PATH, columns=[c for c in SLIM_COLS if c in available])
    slim.to_parquet(SLIM_PANEL_PATH, index=False)
    print(f"Slim panel saved: {SLIM_PANEL_PATH}  shape={slim.shape}")
    return SLIM_PANEL_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def adjust_for_borrow(
    daily_returns: pd.Series,
    weights: pd.DataFrame,
    borrow_bps_annual: float,
) -> pd.Series:
    if borrow_bps_annual == 0:
        return daily_returns
    daily_rate = borrow_bps_annual / 1e4 / 252
    gross_short = (
        weights[weights["weight"] < 0]
        .groupby("date")["weight"]
        .sum()
        .abs()
        .reindex(daily_returns.index)
        .fillna(0.0)
    )
    return daily_returns - gross_short * daily_rate


def _turnover(weights: pd.DataFrame) -> float:
    if weights.empty:
        return 0.0
    w = weights.sort_values(["permno", "date"]).copy()
    prev = w.groupby("permno", sort=False)["weight"].shift(1).fillna(0.0)
    return float(np.abs(w["weight"].values - prev.values).sum() / max(weights["date"].nunique(), 1))


def _avg_positions(weights: pd.DataFrame) -> float:
    if weights.empty:
        return 0.0
    return float(weights.groupby("date")["permno"].count().mean())


def _summary_row(label: str, result, borrow_bps: float) -> dict:
    dr = adjust_for_borrow(result.daily_returns, result.weights_long, borrow_bps)
    perf = performance_metrics(dr, annualization=252)
    return {
        "config": label,
        "sharpe": round(perf.sharpe_ratio, 3),
        "sortino": round(perf.sortino_ratio, 3),
        "ann_ret%": round(perf.annualized_return * 100, 2),
        "cum_ret%": round(perf.cumulative_return * 100, 1),
        "ann_vol%": round(perf.annualized_volatility * 100, 2),
        "max_dd%": round(perf.max_drawdown * 100, 2),
        "hit_rate%": round((dr > 0).mean() * 100, 1),
        "avg_pos": round(_avg_positions(result.weights_long), 1),
        "daily_to": round(_turnover(result.weights_long), 4),
        "n_days": perf.n_days,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(rebuild_factors: bool = False) -> None:
    if rebuild_factors or not OUTPUT_PATH.exists():
        print("Building contract-level GEX factors ...")
        build_contract_factors(CONTRACT_PATH, OUTPUT_PATH)
    else:
        print(f"Contract factors: {OUTPUT_PATH}")

    slim_path = _ensure_slim_panel()

    base_cfg = BacktestConfig(
        panel_path=slim_path,
        start_date=pd.Timestamp(START_DATE),
        end_date=None,
        signal_col="signal",
        signal_lag_days=0,
        missing_return_policy="zero_contribution",
        transaction_cost_bps=5.0,
        long_only=False,
        short_only=False,
    )

    panel_full = load_panel(slim_path)
    panel_bench = panel_full.loc[panel_full["date"] >= pd.Timestamp(START_DATE)]
    benchmark = equal_weight_daily_returns(panel_bench, "ret")
    del panel_full, panel_bench
    gc.collect()

    # Configurations: (label, variant, gate, neutrality, n_leg, tstop, long_leg)
    # Optimised for 231-stock contract universe (sweep: n_per_leg x time_stop x gate x variant).
    # Best Sharpe: v2-A-beta-5d, n_per_leg=5, time_stop=5d.
    configs = [
        # --- Top configs from parameter sweep (full 231-stock universe) ---
        ("v2-A-beta-5d-LS",     "A", "v2", "beta",   5,  5, "momentum"),  # best Sharpe ~1.00
        ("v2-A-dollar-5d-LS",   "A", "v2", "dollar", 5,  5, "momentum"),  # Sharpe ~0.96
        ("v2-A-beta-10d-LS",    "A", "v2", "beta",   5, 10, "momentum"),  # Sharpe ~0.90
        ("v2-A-beta-21d-LS",    "A", "v2", "beta",   5, 21, "momentum"),  # Sharpe ~0.90
        ("v2-B-dollar-5d-LS",   "B", "v2", "dollar", 5,  5, "momentum"),  # Sharpe ~0.91
        # --- Short-only variants (must use dollar neutrality — beta-neutral requires both legs) ---
        ("v2-A-dollar-5d-Short","A", "v2", "dollar", 5,  5, "none"),
        ("v2-A-dollar-10d-Short","A","v2", "dollar", 5, 10, "none"),
        # --- Broader gate (v3) ---
        ("v3-A-dollar-5d-LS",   "A", "v3", "dollar", 5,  5, "momentum"),
    ]

    signal = FragilityGEXSignal()
    results = {}

    for label, variant, gate, neutral, n_leg, tstop, ll in configs:
        print(f"\n{'='*55}\nRunning: {label}")
        strategy = GEXSentimentStrategy(
            variant=variant,
            gate=gate,
            neutrality=neutral,
            n_per_leg=n_leg,
            time_stop_days=tstop,
            long_leg=ll,
        )
        try:
            res = run_backtest(
                base_cfg,
                signal,
                strategy,
                benchmark_daily_returns=benchmark,
            )
            results[label] = res
            p = res.performance
            avg_pos = _avg_positions(res.weights_long)
            print(
                f"  Sharpe={p.sharpe_ratio:.3f}  AnnRet={p.annualized_return*100:.1f}%"
                f"  MaxDD={p.max_drawdown*100:.1f}%  AvgPos={avg_pos:.1f}  Days={p.n_days}"
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
        gc.collect()

    if not results:
        print("No successful results.")
        return

    # Performance tables
    print("\n" + "=" * 90)
    print("PERFORMANCE SUMMARY  (borrow cost = 50 bps/yr)")
    print("=" * 90)
    rows_50 = [_summary_row(lbl, r, 50) for lbl, r in results.items()]
    df50 = pd.DataFrame(rows_50).set_index("config")
    print(df50.to_string())

    print("\n" + "=" * 90)
    print("PERFORMANCE SUMMARY  (borrow cost = 200 bps/yr)")
    print("=" * 90)
    rows_200 = [_summary_row(lbl, r, 200) for lbl, r in results.items()]
    df200 = pd.DataFrame(rows_200).set_index("config")
    print(df200.to_string())

    best = df50["sharpe"].idxmax()
    print(f"\nBest config by Sharpe (50 bps borrow): {best}")

    bench_perf = list(results.values())[0].benchmark_performance
    if bench_perf:
        print(
            f"Benchmark (EW all names): Sharpe={bench_perf.sharpe_ratio:.3f}"
            f"  AnnRet={bench_perf.annualized_return*100:.1f}%"
            f"  MaxDD={bench_perf.max_drawdown*100:.1f}%"
        )

    # NAV plot
    n_plot = min(6, len(results))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for lbl in list(results.keys())[:n_plot]:
        r = results[lbl]
        dr = adjust_for_borrow(r.daily_returns, r.weights_long, 50)
        (1 + dr).cumprod().plot(ax=axes[0], label=lbl, alpha=0.85)

    if list(results.values())[0].benchmark_nav is not None:
        list(results.values())[0].benchmark_nav.plot(
            ax=axes[0], label="EW benchmark", color="black",
            linewidth=2, linestyle="--", alpha=0.6,
        )
    axes[0].set_title("NAV (borrow 50 bps/yr, txcost 5 bps)")
    axes[0].set_ylabel("NAV")
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    best_res = results[best]
    best_dr = adjust_for_borrow(best_res.daily_returns, best_res.weights_long, 50)
    best_nav = (1 + best_dr).cumprod()
    dd = (best_nav.cummax() - best_nav) / best_nav.cummax()
    dd.plot(ax=axes[1], color="darkred", label=f"Drawdown - {best}")
    axes[1].set_title("Drawdown (best config)")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "GEX-Fragility x Sentiment Strategy  |  signal lag=1d, txcost=5bps",
        fontsize=10,
    )
    plt.tight_layout()
    out_png = ROOT / "backtest" / "sentiment_gex_nav.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_png}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-factors", action="store_true")
    args = parser.parse_args()
    main(rebuild_factors=args.rebuild_factors)
