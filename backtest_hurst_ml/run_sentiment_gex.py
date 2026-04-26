
"""
Run GEX-Fragility x Sentiment strategy with original baseline, ML exit, Hurst overlays,
and Hurst diagnostics.

This runner is designed for the ML-version codebase.

Core comparison:
1. Original baseline:
   FragilityGEXSignal + GEXSentimentStrategy

2. Original baseline + MLExit:
   Same original strategy + MLExitRule

3. Hurst short-leg overlays:
   Original strategy logic + Hurst filter/score only on short leg

4. Robustness checks:
   Hurst on long leg only, and Hurst on both legs

Additional outputs:
- long/short return attribution
- entry count and holding-period diagnostics
- drawdown episode panels
- kept-vs-filtered short trade diagnostics
- Hurst-bin analysis inside baseline short setup
- subperiod robustness
"""
from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_hurst_ml.backtest_framework import (
    BacktestConfig,
    BacktestResult,
    equal_weight_daily_returns,
    load_panel,
    performance_metrics,
    run_backtest,
)
from backtest_hurst_ml.gex_factors_builder import build_contract_factors, CONTRACT_PATH, OUTPUT_PATH
from backtest_hurst_ml.sentiment_gex_strategy import FragilityGEXSignal, GEXSentimentStrategy
from backtest_hurst_ml.hurst_extension import FragilityGEXWithHurstSignal, GEXSentimentHurstStrategy
from backtest_hurst_ml.hurst_diagnostics import (
    adjust_for_borrow,
    _avg_positions,
    make_all_diagnostic_outputs,
    position_diagnostics,
    summary_row,
)

try:
    from backtest_hurst_ml.ml_exit import MLExitRule
except Exception:
    MLExitRule = None

try:
    from backtest_hurst_ml.ml_exit_lgbm import LGBMExitRule
except Exception:
    LGBMExitRule = None


ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = Path(r"E:\Pythonfiles\ProjectGamma\data\backtest_panel_main.parquet")
SLIM_PANEL_PATH = Path(r"E:\Pythonfiles\ProjectGamma\data\backtest_panel_slim.parquet")
START_DATE = "2018-06-01"
OUTPUT_DIR = ROOT / "backtest_hurst_ml" / "output_new_no_skip"

SLIM_COLS = [
    "permno", "secid", "ticker", "date",
    "ret", "price_abs",
    "gex_net_gex_1pct", "gex_call_gex_1pct", "gex_put_gex_1pct", "gex_spot",
    "sent_avg", "sent_posts",
    "bs_b_mkt",
    "market_equity", "dollar_volume",
]


@dataclass
class StrategySpec:
    label: str
    variant: str = "A"
    gate: str = "v2"
    neutrality: str = "beta"
    n_per_leg: int = 5
    time_stop_days: int = 5
    long_leg: str = "momentum"
    family: str = "base"              # base / hurst
    hurst_mode: str = "none"          # none / filter / score
    hurst_window: Optional[int] = None
    use_ml_exit: bool = False
    use_lgbm_exit: bool = False
    apply_hurst_to_shorts: bool = True
    apply_hurst_to_longs: bool = False


def _parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq
    return [f.name for f in pq.ParquetFile(path).schema_arrow]


def _ensure_slim_panel(force_rebuild: bool = False) -> Path:
    """
    Create / refresh slim panel. Hurst requires price_abs.
    """
    if SLIM_PANEL_PATH.exists() and not force_rebuild:
        try:
            cols = _parquet_columns(SLIM_PANEL_PATH)
            missing_required = [c for c in ["ret", "sent_avg", "bs_b_mkt"] if c not in cols]
            missing_hurst = ["price_abs"] if "price_abs" not in cols else []
            if not missing_required and not missing_hurst:
                return SLIM_PANEL_PATH
            print(f"Existing slim panel missing {missing_required + missing_hurst}; rebuilding slim panel ...")
        except Exception:
            print("Could not inspect slim panel schema; rebuilding slim panel ...")

    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Panel not found: {PANEL_PATH}. Place backtest_panel_main.parquet under data/."
        )

    print("Creating slim panel ...")
    available = _parquet_columns(PANEL_PATH)
    use_cols = [c for c in SLIM_COLS if c in available]
    missing = [c for c in SLIM_COLS if c not in available]
    if missing:
        print(f"Warning: source panel missing columns: {missing}")

    slim = pd.read_parquet(PANEL_PATH, columns=use_cols)
    SLIM_PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    slim.to_parquet(SLIM_PANEL_PATH, index=False)
    print(f"Slim panel saved: {SLIM_PANEL_PATH}  shape={slim.shape}")
    return SLIM_PANEL_PATH


def make_config(slim_path: Path) -> BacktestConfig:
    return BacktestConfig(
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


def build_original_strategy(spec: StrategySpec) -> GEXSentimentStrategy:
    return GEXSentimentStrategy(
        variant=spec.variant,
        gate=spec.gate,
        neutrality=spec.neutrality,
        n_per_leg=spec.n_per_leg,
        time_stop_days=spec.time_stop_days,
        long_leg=spec.long_leg,
    )


def build_hurst_strategy(spec: StrategySpec) -> GEXSentimentHurstStrategy:
    hurst_col = f"hurst_{spec.hurst_window}" if spec.hurst_window is not None else "hurst_30"
    return GEXSentimentHurstStrategy(
        variant=spec.variant,
        gate=spec.gate,
        neutrality=spec.neutrality,
        n_per_leg=spec.n_per_leg,
        time_stop_days=spec.time_stop_days,
        long_leg=spec.long_leg,
        hurst_mode=spec.hurst_mode,
        hurst_col=hurst_col,
        hurst_threshold=0.55,
        hurst_strength=1.0,
        apply_hurst_to_shorts=spec.apply_hurst_to_shorts,
        apply_hurst_to_longs=spec.apply_hurst_to_longs,
    )


def _resolve_eval_start(
    dates: pd.Index,
    *,
    eval_start_date: Optional[str] = None,
    eval_after_train_days: Optional[int] = None,
) -> Optional[pd.Timestamp]:
    if eval_start_date:
        return pd.Timestamp(eval_start_date).normalize()
    if eval_after_train_days is None:
        return None

    ordered = pd.Index(pd.to_datetime(dates).normalize()).drop_duplicates().sort_values()
    if len(ordered) == 0:
        return None
    idx = min(max(int(eval_after_train_days), 0), len(ordered) - 1)
    return pd.Timestamp(ordered[idx]).normalize()


def _slice_result_for_eval(result: BacktestResult, eval_start: Optional[pd.Timestamp]) -> BacktestResult:
    if eval_start is None:
        return result

    eval_start = pd.Timestamp(eval_start).normalize()
    daily = result.daily_returns[result.daily_returns.index >= eval_start].copy()
    if daily.empty:
        return result

    nav = (1.0 + daily).cumprod()
    weights = result.weights_long.copy()
    if not weights.empty:
        weights["date"] = pd.to_datetime(weights["date"]).dt.normalize()
        weights = weights[weights["date"] >= eval_start].copy()

    bench_ret = None
    bench_nav = None
    bench_perf = None
    if result.benchmark_daily_returns is not None:
        bench_ret = result.benchmark_daily_returns[result.benchmark_daily_returns.index >= eval_start].copy()
        bench_ret = bench_ret.reindex(daily.index).fillna(0.0)
        bench_nav = (1.0 + bench_ret).cumprod()
        bench_perf = performance_metrics(bench_ret, annualization=252)

    return BacktestResult(
        daily_returns=daily,
        nav=nav,
        weights_long=weights,
        panel_with_signal=result.panel_with_signal,
        performance=performance_metrics(daily, annualization=252),
        benchmark_daily_returns=bench_ret,
        benchmark_nav=bench_nav,
        benchmark_performance=bench_perf,
        panel_pre_signal_lag=result.panel_pre_signal_lag,
    )


def build_specs(
    hurst_windows: list[int],
    primary_only: bool,
    no_ml_exit: bool,
    key_h10_exits_only: bool = False,
) -> list[StrategySpec]:
    if key_h10_exits_only:
        specs = [
            StrategySpec("v2-A-beta-5d-LS-H10-filter-short", "A", "v2", "beta", 5, 5, "momentum",
                         family="hurst", hurst_mode="filter", hurst_window=10,
                         apply_hurst_to_shorts=True, apply_hurst_to_longs=False),
        ]
        if not no_ml_exit:
            specs.extend([
                StrategySpec("v2-A-beta-5d-LS-H10-filter-short-MLExit", "A", "v2", "beta", 5, 5, "momentum",
                             family="hurst", hurst_mode="filter", hurst_window=10, use_ml_exit=True,
                             apply_hurst_to_shorts=True, apply_hurst_to_longs=False),
                StrategySpec("v2-A-beta-5d-LS-H10-filter-short-LGBMExit", "A", "v2", "beta", 5, 5, "momentum",
                             family="hurst", hurst_mode="filter", hurst_window=10, use_lgbm_exit=True,
                             apply_hurst_to_shorts=True, apply_hurst_to_longs=False),
            ])
        return specs

    specs: list[StrategySpec] = []

    if not primary_only:
        specs.extend([
            StrategySpec("v2-A-beta-5d-LS", "A", "v2", "beta", 5, 5, "momentum", family="base"),
            StrategySpec("v2-A-dollar-5d-LS", "A", "v2", "dollar", 5, 5, "momentum", family="base"),
            StrategySpec("v2-A-beta-10d-LS", "A", "v2", "beta", 5, 10, "momentum", family="base"),
            StrategySpec("v2-A-beta-21d-LS", "A", "v2", "beta", 5, 21, "momentum", family="base"),
            StrategySpec("v2-B-dollar-5d-LS", "B", "v2", "dollar", 5, 5, "momentum", family="base"),
            StrategySpec("v2-A-dollar-5d-Short", "A", "v2", "dollar", 5, 5, "none", family="base"),
            StrategySpec("v2-A-dollar-10d-Short", "A", "v2", "dollar", 5, 10, "none", family="base"),
            StrategySpec("v3-A-dollar-5d-LS", "A", "v3", "dollar", 5, 5, "momentum", family="base"),
        ])
    else:
        specs.append(StrategySpec("v2-A-beta-5d-LS", "A", "v2", "beta", 5, 5, "momentum", family="base"))

    if not no_ml_exit:
        specs.append(
            StrategySpec("v2-A-beta-5d-LS-MLExit", "A", "v2", "beta", 5, 5, "momentum", family="base", use_ml_exit=True)
        )
        specs.append(
            StrategySpec("v2-A-beta-5d-LS-LGBMExit", "A", "v2", "beta", 5, 5, "momentum", family="base", use_lgbm_exit=True)
        )

    for w in hurst_windows:
        # Main Hurst tests: Hurst only on short leg.
        specs.append(
            StrategySpec(f"v2-A-beta-5d-LS-H{w}-filter-short", "A", "v2", "beta", 5, 5, "momentum",
                         family="hurst", hurst_mode="filter", hurst_window=w,
                         apply_hurst_to_shorts=True, apply_hurst_to_longs=False)
        )
        specs.append(
            StrategySpec(f"v2-A-beta-5d-LS-H{w}-score-short", "A", "v2", "beta", 5, 5, "momentum",
                         family="hurst", hurst_mode="score", hurst_window=w,
                         apply_hurst_to_shorts=True, apply_hurst_to_longs=False)
        )

        # Placebo / robustness: Hurst only on long leg.
        specs.append(
            StrategySpec(f"v2-A-beta-5d-LS-H{w}-filter-long", "A", "v2", "beta", 5, 5, "momentum",
                         family="hurst", hurst_mode="filter", hurst_window=w,
                         apply_hurst_to_shorts=False, apply_hurst_to_longs=True)
        )

        # Robustness: Hurst on both legs.
        specs.append(
            StrategySpec(f"v2-A-beta-5d-LS-H{w}-filter-both", "A", "v2", "beta", 5, 5, "momentum",
                         family="hurst", hurst_mode="filter", hurst_window=w,
                         apply_hurst_to_shorts=True, apply_hurst_to_longs=True)
        )

        if not no_ml_exit:
            specs.append(
                StrategySpec(f"v2-A-beta-5d-LS-H{w}-filter-short-MLExit", "A", "v2", "beta", 5, 5, "momentum",
                             family="hurst", hurst_mode="filter", hurst_window=w, use_ml_exit=True,
                             apply_hurst_to_shorts=True, apply_hurst_to_longs=False)
            )
            specs.append(
                StrategySpec(f"v2-A-beta-5d-LS-H{w}-filter-short-LGBMExit", "A", "v2", "beta", 5, 5, "momentum",
                             family="hurst", hurst_mode="filter", hurst_window=w, use_lgbm_exit=True,
                             apply_hurst_to_shorts=True, apply_hurst_to_longs=False)
            )

    return specs


def main(
    rebuild_factors: bool = False,
    rebuild_slim: bool = False,
    hurst_windows: Optional[list[int]] = None,
    primary_only: bool = False,
    no_ml_exit: bool = False,
    key_h10_exits_only: bool = False,
    eval_start_date: Optional[str] = None,
    eval_after_train_days: Optional[int] = None,
) -> None:
    if hurst_windows is None:
        hurst_windows = [10, 30, 60]
    if key_h10_exits_only:
        hurst_windows = [10]

    if rebuild_factors or not OUTPUT_PATH.exists():
        print("Building contract-level GEX factors ...")
        build_contract_factors(CONTRACT_PATH, OUTPUT_PATH)
    else:
        print(f"Contract factors: {OUTPUT_PATH}")

    slim_path = _ensure_slim_panel(force_rebuild=rebuild_slim)
    base_cfg = make_config(slim_path)

    panel_full = load_panel(slim_path)
    print(f"Panel date range: {panel_full['date'].min()} to {panel_full['date'].max()}  rows={len(panel_full):,}")
    eval_start = _resolve_eval_start(
        pd.Index(sorted(panel_full.loc[panel_full["date"] >= pd.Timestamp(START_DATE), "date"].unique())),
        eval_start_date=eval_start_date,
        eval_after_train_days=eval_after_train_days,
    )
    if eval_start is not None:
        print(f"Performance evaluation window starts at: {eval_start.date()} (post-training slice)")
    panel_bench = panel_full.loc[panel_full["date"] >= pd.Timestamp(START_DATE)]
    benchmark = equal_weight_daily_returns(panel_bench, "ret")
    del panel_full, panel_bench
    gc.collect()

    original_signal = FragilityGEXSignal()
    hurst_signal = FragilityGEXWithHurstSignal(
        sentiment_col="sent_avg",
        price_col="price_abs",
        regime_lookback_days=5,
        hurst_windows=hurst_windows,
    )

    specs = build_specs(
        hurst_windows,
        primary_only=primary_only,
        no_ml_exit=no_ml_exit,
        key_h10_exits_only=key_h10_exits_only,
    )
    results = {}

    for spec in specs:
        print(f"\n{'=' * 72}\nRunning: {spec.label}")

        signal = original_signal if spec.family == "base" else hurst_signal
        strategy = build_original_strategy(spec) if spec.family == "base" else build_hurst_strategy(spec)

        entry_exit = None
        if spec.use_ml_exit:
            if MLExitRule is None:
                print("  MLExitRule unavailable; skipping.")
                continue
            entry_exit = MLExitRule(forward_days=5, train_window_days=504, retrain_freq_days=63, exit_threshold=0.55)
        elif spec.use_lgbm_exit:
            if LGBMExitRule is None:
                print("  LGBMExitRule unavailable; skipping.")
                continue
            entry_exit = LGBMExitRule(
                feature_panel_path=PANEL_PATH,
                hazard_horizon_days=5,
                adverse_return_threshold=-0.005,
                train_window_days=756,
                retrain_freq_days=63,
                min_train_samples=500,
                policy_mode="top_quantile",
                policy_quantile=0.10,
                policy_sides=["short"],
                use_expanded_training_candidates=True,
            )

        try:
            res = run_backtest(
                base_cfg,
                signal,
                strategy,
                entry_exit=entry_exit,
                benchmark_daily_returns=benchmark,
            )
            eval_res = _slice_result_for_eval(res, eval_start)
            results[spec.label] = eval_res

            p = eval_res.performance
            diag = position_diagnostics(eval_res.weights_long, eval_res.daily_returns.index)
            first_dt = eval_res.weights_long["date"].min() if not eval_res.weights_long.empty else None
            print(
                f"  Sharpe={p.sharpe_ratio:.3f}  AnnRet={p.annualized_return * 100:.1f}%"
                f"  MaxDD={p.max_drawdown * 100:.1f}%  AvgPos={_avg_positions(eval_res.weights_long):.1f}"
                f"  FirstActive={first_dt}  LongEntries={diag['long_entries']}  ShortEntries={diag['short_entries']}"
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")

        gc.collect()

    if not results:
        print("No successful results.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("PERFORMANCE SUMMARY  (borrow cost = 50 bps/yr)")
    print("=" * 100)
    df50 = pd.DataFrame([summary_row(lbl, r, 50) for lbl, r in results.items()]).set_index("config")
    df50.to_csv(OUTPUT_DIR / "performance_50bps.csv")
    cols_show = [
        "first_active_date", "sharpe", "ann_ret%", "ann_vol%", "max_dd%",
        "long_gross_sharpe", "short_after_borrow_sharpe",
        "long_entries", "short_entries", "avg_long_holding_days", "avg_short_holding_days"
    ]
    print(df50[[c for c in cols_show if c in df50.columns]].to_string())

    print("\n" + "=" * 100)
    print("PERFORMANCE SUMMARY  (borrow cost = 200 bps/yr)")
    print("=" * 100)
    df200 = pd.DataFrame([summary_row(lbl, r, 200) for lbl, r in results.items()]).set_index("config")
    df200.to_csv(OUTPUT_DIR / "performance_200bps.csv")
    print(df200[[c for c in cols_show if c in df200.columns]].to_string())

    best = df50["sharpe"].idxmax()
    print(f"\nBest config by Sharpe (50 bps borrow): {best}")

    bench_perf = list(results.values())[0].benchmark_performance
    if bench_perf:
        print(
            f"Benchmark (EW all names): Sharpe={bench_perf.sharpe_ratio:.3f}"
            f"  AnnRet={bench_perf.annualized_return * 100:.1f}%"
            f"  MaxDD={bench_perf.max_drawdown * 100:.1f}%"
        )

    if key_h10_exits_only:
        baseline_label = "v2-A-beta-5d-LS-H10-filter-short"
        main_hurst_label = "v2-A-beta-5d-LS-H10-filter-short"
        hurst_col = "hurst_10"
    else:
        baseline_label = "v2-A-beta-5d-LS"
        main_hurst_label = "v2-A-beta-5d-LS-H30-filter-short"
        hurst_col = "hurst_30"

    make_all_diagnostic_outputs(
        results,
        OUTPUT_DIR,
        borrow_bps=50,
        baseline_label=baseline_label,
        main_hurst_label=main_hurst_label,
        hurst_col=hurst_col,
    )

    print(f"\nDiagnostic outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-factors", action="store_true")
    parser.add_argument("--rebuild-slim", action="store_true", help="Recreate data/backtest_panel_slim.parquet with price_abs included.")
    parser.add_argument("--hurst-windows", nargs="+", type=int, default=[10, 30, 60])
    parser.add_argument("--primary-only", action="store_true", default=True, help="Run only primary baseline + ML + Hurst variants.")
    parser.add_argument("--no-ml-exit", action="store_true", help="Skip MLExit variants.")
    parser.add_argument(
        "--key-h10-exits",
        action="store_true",
        default=False,
        help="Run only v2-A-beta-5d-LS-H10-filter-short, MLExit, and LGBMExit.",
    )
    parser.add_argument(
        "--eval-start-date",
        default=None,
        help="Report portfolio performance only from this date onward (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--eval-after-train-days",
        type=int,
        default=0,
        help="Report performance from START_DATE plus this many trading dates (e.g. 756 for LGBM train window).",
    )
    args = parser.parse_args()

    main(
        rebuild_factors=args.rebuild_factors,
        rebuild_slim=args.rebuild_slim,
        hurst_windows=args.hurst_windows,
        primary_only=args.primary_only,
        no_ml_exit=args.no_ml_exit,
        key_h10_exits_only=args.key_h10_exits,
        eval_start_date=args.eval_start_date,
        eval_after_train_days=args.eval_after_train_days,
    )
