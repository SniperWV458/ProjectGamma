"""
ML-based exit timing for GEX-Fragility positions.

Replaces the fixed time-stop with a gradient-boosting classifier that decides,
each day a position is held, whether to exit or keep holding.

Training signal: "exit" = the position loses money over the next `forward_days`.

Walk-forward scheme:
    - Initial training window: first `train_window_days` active trading days
    - Retrain every `retrain_freq_days` thereafter (rolling window)

Features per (permno, date) while in position:
    days_held, sentiment_z, net_gex_z, d4_put_dom, d5_below_flip,
    stock_ret_1d, stock_ret_5d, pos_return_sofar, market_ret_5d, market_ret_20d

Usage:
    from ml_exit import MLExitRule, build_ml_exit_features
    rule = MLExitRule(forward_days=5, train_window_days=504, retrain_freq_days=63)
    res  = run_backtest(cfg, signal, strategy, entry_exit=rule, ...)

    # Or train and evaluate standalone:
    python backtest/ml_exit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_framework import (
    BacktestConfig,
    BaseEntryExitRule,
    equal_weight_daily_returns,
    load_panel,
    performance_metrics,
    run_backtest,
)
from sentiment_gex_strategy import FragilityGEXSignal, GEXSentimentStrategy

ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "days_held",
    "sentiment_z",
    "net_gex_z",
    "d4_put_dom",
    "d5_below_flip",
    "stock_ret_1d",
    "stock_ret_5d",
    "pos_return_sofar",
    "market_ret_5d",
    "market_ret_20d",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_position_features(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    forward_days: int = 5,
) -> pd.DataFrame:
    """
    For each (permno, date) row in weights, compute FEATURE_COLS and a binary label:
        label = 1 (exit) if the stock's cumulative return over the next forward_days is < 0
                  adjusted for position side (short: negative forward return = good, so
                  label=1 means future return > 0 for shorts = should exit the short).

    Returns a DataFrame with FEATURE_COLS + ['label', 'permno', 'date', 'side'].
    """
    w = weights.copy()
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    w["side"] = np.where(w["weight"] > 0, "long", "short")

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    # Market return (equal-weight all names) — date-level features
    panel = panel.sort_values(["permno", "date"])
    panel["ret"] = pd.to_numeric(panel["ret"], errors="coerce")

    mkt = panel.groupby("date")["ret"].mean()
    mkt_df = pd.DataFrame({
        "date": mkt.index,
        "market_ret_5d":  mkt.rolling(5,  min_periods=1).sum().values,
        "market_ret_20d": mkt.rolling(20, min_periods=5).sum().values,
    })

    # Per-stock rolling returns and forward return — stock-level features
    grp = panel.groupby("permno", sort=False)["ret"]
    panel["stock_ret_1d"] = grp.shift(1)
    panel["stock_ret_5d"] = grp.transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    )
    panel["fwd_ret"] = grp.transform(
        lambda x: x.shift(-1).rolling(forward_days, min_periods=1).sum()
    )

    # Merge panel features
    feat_panel = panel[["permno", "date", "ret",
                         "sentiment_z", "net_gex_z",
                         "d4_put_dom", "d5_below_flip",
                         "stock_ret_1d", "stock_ret_5d", "fwd_ret"]].copy()
    feat_panel = feat_panel.merge(mkt_df, on="date", how="left")

    # Add days_held — resets to 1 whenever there is a date gap > 5 calendar days
    w_sorted = w.sort_values(["permno", "date"]).copy()
    prev_date = w_sorted.groupby("permno", sort=False)["date"].shift(1)
    gap = (w_sorted["date"] - prev_date).dt.days.fillna(0)
    new_pos = (gap > 5) | (prev_date.isna())
    w_sorted["_pos_id"] = new_pos.groupby(w_sorted["permno"]).cumsum()
    w_sorted["days_held"] = (
        w_sorted.groupby(["permno", "_pos_id"], sort=False).cumcount() + 1
    )
    w_sorted = w_sorted.drop(columns=["_pos_id"])

    # Cumulative position return so far (simplified: rolling sum of past ret * side)
    pos_rows = w_sorted.merge(
        feat_panel[["permno", "date", "ret"]], on=["permno", "date"], how="left"
    )
    pos_rows["signed_ret"] = pos_rows["ret"] * np.where(pos_rows["weight"] > 0, 1, -1)
    pos_rows["pos_return_sofar"] = (
        pos_rows.groupby("permno", sort=False)["signed_ret"]
        .transform(lambda x: x.expanding().sum().shift(1).fillna(0))
    )

    # Merge all features
    out = w_sorted.merge(
        feat_panel[["permno", "date", "sentiment_z", "net_gex_z", "d4_put_dom",
                    "d5_below_flip", "stock_ret_1d", "stock_ret_5d",
                    "market_ret_5d", "market_ret_20d", "fwd_ret"]],
        on=["permno", "date"], how="left",
    )
    out["pos_return_sofar"] = pos_rows["pos_return_sofar"].values

    # Label: exit signal
    # For longs: exit if forward return < 0
    # For shorts: exit if forward return > 0  (position loses money)
    out["label"] = np.where(
        out["side"] == "long",
        (out["fwd_ret"] < 0).astype(int),
        (out["fwd_ret"] > 0).astype(int),
    )

    return out[["permno", "date", "side", "weight"] + FEATURE_COLS + ["label", "fwd_ret"]]


# ---------------------------------------------------------------------------
# MLExitRule — plugs into run_backtest as entry_exit=
# ---------------------------------------------------------------------------

class MLExitRule(BaseEntryExitRule):
    """
    Walk-forward ML exit: trains a GradientBoostingClassifier on the first
    `train_window_days` days, then retrains every `retrain_freq_days`.

    The model predicts P(exit) for each active position. Positions with
    P(exit) > `exit_threshold` are dropped from the weights DataFrame.
    """

    def __init__(
        self,
        forward_days: int = 5,
        train_window_days: int = 504,
        retrain_freq_days: int = 63,
        exit_threshold: float = 0.55,
        min_train_samples: int = 200,
    ):
        self.forward_days = forward_days
        self.train_window_days = train_window_days
        self.retrain_freq_days = retrain_freq_days
        self.exit_threshold = exit_threshold
        self.min_train_samples = min_train_samples
        self._model = None
        self._last_train_date = None

    def apply(self, weights: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
        except ImportError:
            print("  [MLExitRule] scikit-learn not available — skipping ML exit.")
            return weights

        if weights.empty:
            return weights

        feats = build_position_features(weights, panel, self.forward_days)
        feats = feats.dropna(subset=FEATURE_COLS + ["label"]).reset_index(drop=True)
        if feats.empty:
            return weights

        dates = sorted(feats["date"].unique())
        cutoff_idx = self.train_window_days
        if cutoff_idx >= len(dates):
            return weights

        exit_mask = pd.Series(False, index=feats.index)

        for i, dt in enumerate(dates):
            if i < cutoff_idx:
                continue

            # Retrain if needed
            if (
                self._model is None
                or (i - cutoff_idx) % self.retrain_freq_days == 0
            ):
                train_dates = dates[max(0, i - self.train_window_days): i]
                train = feats[feats["date"].isin(train_dates)]
                X_tr = train[FEATURE_COLS].values
                y_tr = train["label"].values
                if len(y_tr) < self.min_train_samples or len(np.unique(y_tr)) < 2:
                    continue
                self._model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", GradientBoostingClassifier(
                        n_estimators=100, max_depth=3,
                        learning_rate=0.05, subsample=0.8,
                        random_state=42,
                    )),
                ])
                self._model.fit(X_tr, y_tr)

            if self._model is None:
                continue

            day_rows = feats[feats["date"] == dt]
            if day_rows.empty:
                continue
            X_pred = day_rows[FEATURE_COLS].values
            proba = self._model.predict_proba(X_pred)
            exit_class_idx = list(self._model.named_steps["clf"].classes_).index(1)
            p_exit = proba[:, exit_class_idx]

            should_exit = p_exit > self.exit_threshold
            exit_mask.loc[day_rows.index[should_exit]] = True

        # Remove rows flagged for exit
        keep_idx = feats.index[~exit_mask]
        keep = feats.loc[keep_idx, ["permno", "date", "weight"]]
        result = weights.merge(keep[["permno", "date"]], on=["permno", "date"], how="inner")
        return result


# ---------------------------------------------------------------------------
# Standalone evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    from run_sentiment_gex import adjust_for_borrow, _avg_positions

    SLIM = ROOT / "data" / "backtest_panel_slim.parquet"
    START = pd.Timestamp("2018-06-01")

    cfg = BacktestConfig(
        panel_path=SLIM,
        start_date=START,
        signal_col="signal",
        signal_lag_days=0,
        missing_return_policy="zero_contribution",
        transaction_cost_bps=5.0,
    )

    panel = load_panel(SLIM)
    bench = equal_weight_daily_returns(panel[panel["date"] >= START], "ret")
    del panel

    signal = FragilityGEXSignal()
    strat  = GEXSentimentStrategy(
        variant="A", gate="v2", neutrality="beta",
        n_per_leg=5, time_stop_days=5, long_leg="momentum",
    )

    print("Running base strategy ...")
    res_base = run_backtest(cfg, signal, strat, benchmark_daily_returns=bench)
    p_base = res_base.performance
    print(f"  Base  Sharpe={p_base.sharpe_ratio:.3f}  AnnRet={p_base.annualized_return*100:.1f}%"
          f"  AnnVol={p_base.annualized_volatility*100:.1f}%  MaxDD={p_base.max_drawdown*100:.1f}%")

    print("\nRunning with ML exit ...")
    ml_rule = MLExitRule(forward_days=5, train_window_days=504,
                         retrain_freq_days=63, exit_threshold=0.55)
    res_ml = run_backtest(cfg, signal, strat, entry_exit=ml_rule,
                          benchmark_daily_returns=bench)
    p_ml = res_ml.performance
    print(f"  ML    Sharpe={p_ml.sharpe_ratio:.3f}  AnnRet={p_ml.annualized_return*100:.1f}%"
          f"  AnnVol={p_ml.annualized_volatility*100:.1f}%  MaxDD={p_ml.max_drawdown*100:.1f}%")

    # Feature importance
    if ml_rule._model is not None:
        imp = ml_rule._model.named_steps["clf"].feature_importances_
        print("\nFeature importances (final model):")
        for name, val in sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1]):
            print(f"  {name:<22} {val:.4f}")

    # Comparison plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    (1 + res_base.daily_returns).cumprod().plot(ax=ax, label="Base (time-stop 5d)", alpha=0.9)
    (1 + res_ml.daily_returns).cumprod().plot(ax=ax, label="ML exit", alpha=0.9)
    if res_base.benchmark_nav is not None:
        res_base.benchmark_nav.plot(ax=ax, label="EW benchmark", color="black",
                                    linestyle="--", linewidth=1.5, alpha=0.6)
    ax.set_title("Base vs ML-exit NAV  |  v2-A-beta-5d, txcost=5bps")
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = ROOT / "backtest" / "output" / "ml_exit_nav.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
