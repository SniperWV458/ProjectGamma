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
from backtest_hurst_ml.backtest_framework import (
    BacktestConfig,
    BaseEntryExitRule,
    equal_weight_daily_returns,
    load_panel,
    performance_metrics,
    run_backtest,
)
from backtest_hurst_ml.sentiment_gex_strategy import FragilityGEXSignal, GEXSentimentStrategy

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

    # Per-stock returns — align with post–close t: same day t is known, enter on t+1.
    grp = panel.groupby("permno", sort=False)["ret"]
    panel["stock_ret_1d"] = panel["ret"]
    panel["stock_ret_5d"] = grp.transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )
    fwd = grp.transform(lambda s, k=1: s.shift(-k))
    for k in range(2, forward_days + 1):
        fwd = fwd + grp.transform(lambda s, kk=k: s.shift(-kk))
    panel["fwd_ret"] = fwd

    # Merge panel features
    feat_panel = panel[["permno", "date", "ret",
                         "sentiment_z", "net_gex_z",
                         "d4_put_dom", "d5_below_flip",
                         "stock_ret_1d", "stock_ret_5d", "fwd_ret"]].copy()
    feat_panel = feat_panel.merge(mkt_df, on="date", how="left")

    # Add position spell id + days_held.
    # A new spell starts on a long date gap or when side flips.
    w_sorted = w.sort_values(["permno", "date"]).copy()
    prev_date = w_sorted.groupby("permno", sort=False)["date"].shift(1)
    prev_side = w_sorted.groupby("permno", sort=False)["side"].shift(1)
    gap = (w_sorted["date"] - prev_date).dt.days.fillna(0)
    side_flip = prev_side.ne(w_sorted["side"]) & prev_side.notna()
    new_pos = (gap > 5) | (prev_date.isna()) | side_flip
    w_sorted["_pos_id"] = new_pos.groupby(w_sorted["permno"]).cumsum()
    w_sorted["days_held"] = (
        w_sorted.groupby(["permno", "_pos_id"], sort=False).cumcount() + 1
    )

    # Cumulative position return so far (simplified: rolling sum of past ret * side)
    pos_rows = w_sorted.merge(
        feat_panel[["permno", "date", "ret"]],
        on=["permno", "date"],
        how="left",
    )
    pos_rows["signed_ret"] = pos_rows["ret"] * np.where(pos_rows["weight"] > 0, 1, -1)
    pos_rows["pos_return_sofar"] = (
        pos_rows.groupby(["permno", "_pos_id"], sort=False)["signed_ret"]
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

    return out[["permno", "date", "side", "weight", "_pos_id"] + FEATURE_COLS + ["label", "fwd_ret"]]


# ---------------------------------------------------------------------------
# Training diagnostics (split/eval/plots)
# ---------------------------------------------------------------------------

def _make_training_pipeline():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])


def chronological_split_with_purge(
    feats: pd.DataFrame,
    *,
    train_frac: float = 0.50,
    val_frac: float = 0.25,
    test_frac: float = 0.25,
    purge_days: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Chronological split on unique dates with a date-level purge between boundaries.
    """
    total = float(train_frac + val_frac + test_frac)
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"train/val/test fractions must sum to 1.0 (got {total}).")
    if feats.empty:
        raise ValueError("Cannot split an empty feature table.")

    dates = np.array(sorted(pd.to_datetime(feats["date"]).dropna().unique()))
    n = len(dates)
    if n < 20:
        raise ValueError(f"Not enough unique dates for split (need >=20, got {n}).")

    train_end = max(1, int(np.floor(n * train_frac)))
    val_end_nominal = train_end + max(1, int(np.floor(n * val_frac)))
    purge = max(0, int(purge_days))

    val_start = min(train_end + purge, n)
    val_end = min(val_end_nominal, n)
    test_start = min(val_end + purge, n)

    if val_start >= val_end:
        raise ValueError("Validation split is empty after applying purge.")
    if test_start >= n:
        raise ValueError("Test split is empty after applying purge.")

    train_dates = set(dates[:train_end])
    val_dates = set(dates[val_start:val_end])
    test_dates = set(dates[test_start:])

    split = {
        "train": feats[feats["date"].isin(train_dates)].copy(),
        "validation": feats[feats["date"].isin(val_dates)].copy(),
        "test": feats[feats["date"].isin(test_dates)].copy(),
        "meta": pd.DataFrame({
            "split": ["train", "validation", "test"],
            "start_date": [min(train_dates), min(val_dates), min(test_dates)],
            "end_date": [max(train_dates), max(val_dates), max(test_dates)],
            "rows": [
                int(feats["date"].isin(train_dates).sum()),
                int(feats["date"].isin(val_dates).sum()),
                int(feats["date"].isin(test_dates).sum()),
            ],
        }),
    }
    return split


def _class1_proba(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.named_steps["clf"].classes_)
    if 1 not in classes:
        raise ValueError("Trained classifier does not contain class=1.")
    return proba[:, classes.index(1)]


def _signed_forward_return(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["side"].values == "long", df["fwd_ret"].values, -df["fwd_ret"].values)


def _classification_metrics(
    y_true: np.ndarray,
    p_exit: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (p_exit >= threshold).astype(int)
    out: dict[str, float] = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(np.mean((p_exit - y_true) ** 2)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, p_exit))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, p_exit))
    except ValueError:
        out["pr_auc"] = float("nan")
    try:
        out["log_loss"] = float(log_loss(y_true, p_exit, labels=[0, 1]))
    except ValueError:
        out["log_loss"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["tn"] = float(tn)
    out["fp"] = float(fp)
    out["fn"] = float(fn)
    out["tp"] = float(tp)
    return out


def _proxy_pnl_metrics(df: pd.DataFrame, p_exit: np.ndarray, *, threshold: float) -> dict[str, float]:
    signed_fwd = _signed_forward_return(df)
    should_exit = p_exit >= threshold
    keep_all = signed_fwd
    model_keep = np.where(should_exit, 0.0, signed_fwd)
    oracle_keep = np.where(df["label"].values == 1, 0.0, signed_fwd)

    return {
        "proxy_keep_all_mean_return": float(np.mean(keep_all)),
        "proxy_model_mean_return": float(np.mean(model_keep)),
        "proxy_oracle_mean_return": float(np.mean(oracle_keep)),
        "proxy_model_uplift_vs_keep_all": float(np.mean(model_keep) - np.mean(keep_all)),
        "proxy_model_efficiency_vs_oracle": float(
            np.mean(model_keep) / np.mean(oracle_keep)
        ) if np.mean(oracle_keep) != 0 else float("nan"),
        "pred_exit_rate": float(np.mean(should_exit)),
    }


def _calibration_table(y_true: np.ndarray, p_exit: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binned = pd.cut(p_exit, bins=bins, include_lowest=True, duplicates="drop")
    cdf = pd.DataFrame({"y": y_true, "p": p_exit, "bin": binned})
    out = (
        cdf.groupby("bin", observed=True)
        .agg(count=("y", "size"), event_rate=("y", "mean"), mean_pred=("p", "mean"))
        .reset_index()
    )
    out["bin"] = out["bin"].astype(str)
    return out


def evaluate_training_efficacy(
    train_df: pd.DataFrame,
    *,
    exit_threshold: float = 0.55,
) -> dict[str, object]:
    """
    Train on the training split only and report training-only efficacy diagnostics.
    """
    if train_df.empty:
        raise ValueError("Training split is empty.")

    fit_df = train_df.dropna(subset=FEATURE_COLS + ["label", "fwd_ret", "side"]).copy()
    if fit_df.empty:
        raise ValueError("Training split has no usable rows after dropping NA.")

    y_train = fit_df["label"].astype(int).values
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training split needs both classes to fit classifier.")

    model = _make_training_pipeline()
    X_train = fit_df[FEATURE_COLS].values
    model.fit(X_train, y_train)
    p_exit = _class1_proba(model, X_train)

    metrics = _classification_metrics(y_train, p_exit, threshold=exit_threshold)
    metrics.update(_proxy_pnl_metrics(fit_df, p_exit, threshold=exit_threshold))
    metrics.update({
        "split": "train",
        "n_rows": float(len(fit_df)),
        "label_mean": float(np.mean(y_train)),
    })

    pred_df = fit_df[["permno", "date", "side", "label", "fwd_ret"]].copy()
    pred_df["p_exit"] = p_exit
    pred_df["pred_exit"] = (p_exit >= exit_threshold).astype(int)
    cal_df = _calibration_table(y_train, p_exit)
    cal_df["split"] = "train"

    return {
        "model": model,
        "metrics": metrics,
        "predictions": pred_df,
        "calibration": cal_df,
    }


def evaluate_model_on_split(
    model,
    split_df: pd.DataFrame,
    *,
    split_name: str,
    exit_threshold: float = 0.55,
) -> dict[str, object]:
    if split_df.empty:
        raise ValueError(f"{split_name} split is empty.")

    eval_df = split_df.dropna(subset=FEATURE_COLS + ["label", "fwd_ret", "side"]).copy()
    if eval_df.empty:
        raise ValueError(f"{split_name} split has no usable rows after dropping NA.")

    X = eval_df[FEATURE_COLS].values
    y = eval_df["label"].astype(int).values
    p_exit = _class1_proba(model, X)

    metrics = _classification_metrics(y, p_exit, threshold=exit_threshold)
    metrics.update(_proxy_pnl_metrics(eval_df, p_exit, threshold=exit_threshold))
    metrics.update({
        "split": split_name,
        "n_rows": float(len(eval_df)),
        "label_mean": float(np.mean(y)),
    })

    pred_df = eval_df[["permno", "date", "side", "label", "fwd_ret"]].copy()
    pred_df["p_exit"] = p_exit
    pred_df["pred_exit"] = (p_exit >= exit_threshold).astype(int)

    cal_df = _calibration_table(y, p_exit)
    cal_df["split"] = split_name
    return {"metrics": metrics, "predictions": pred_df, "calibration": cal_df}


def _threshold_sweep(
    df: pd.DataFrame,
    p_exit: np.ndarray,
    *,
    split_name: str,
) -> pd.DataFrame:
    thresholds = np.linspace(0.10, 0.90, 17)
    rows = []
    y_true = df["label"].astype(int).values
    for th in thresholds:
        row = {"split": split_name}
        row.update(_classification_metrics(y_true, p_exit, threshold=float(th)))
        row.update(_proxy_pnl_metrics(df, p_exit, threshold=float(th)))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_model_training_performance(
    model,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    output_dir: Path,
    exit_threshold: float = 0.55,
) -> dict[str, pd.DataFrame]:
    """
    Create validation/test performance charts and persist metric artifacts.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    output_dir.mkdir(parents=True, exist_ok=True)

    eval_val = evaluate_model_on_split(model, validation_df, split_name="validation", exit_threshold=exit_threshold)
    eval_test = evaluate_model_on_split(model, test_df, split_name="test", exit_threshold=exit_threshold)

    val_pred = eval_val["predictions"]
    test_pred = eval_test["predictions"]
    val_y = val_pred["label"].astype(int).values
    test_y = test_pred["label"].astype(int).values
    val_p = val_pred["p_exit"].values
    test_p = test_pred["p_exit"].values

    metrics_df = pd.DataFrame([eval_val["metrics"], eval_test["metrics"]])
    predictions_df = pd.concat([val_pred.assign(split="validation"), test_pred.assign(split="test")], ignore_index=True)
    calibration_df = pd.concat([eval_val["calibration"], eval_test["calibration"]], ignore_index=True)
    sweep_df = pd.concat(
        [
            _threshold_sweep(val_pred, val_p, split_name="validation"),
            _threshold_sweep(test_pred, test_p, split_name="test"),
        ],
        ignore_index=True,
    )

    metrics_df.to_csv(output_dir / "ml_exit_val_test_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "ml_exit_val_test_predictions.csv", index=False)
    calibration_df.to_csv(output_dir / "ml_exit_val_test_calibration.csv", index=False)
    sweep_df.to_csv(output_dir / "ml_exit_threshold_sweep.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_roc, ax_pr, ax_cal, ax_sweep = axes.ravel()

    for label, y_true, p in [("Validation", val_y, val_p), ("Test", test_y, test_p)]:
        if np.unique(y_true).size >= 2:
            fpr, tpr, _ = roc_curve(y_true, p)
            ax_roc.plot(fpr, tpr, label=label, linewidth=2)
            prec_curve, rec_curve, _ = precision_recall_curve(y_true, p)
            ax_pr.plot(rec_curve, prec_curve, label=label, linewidth=2)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)
    ax_roc.set_title("ROC Curve")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    ax_roc.grid(alpha=0.3)

    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend()
    ax_pr.grid(alpha=0.3)

    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6, label="Perfect calibration")
    for split_name, g in calibration_df.groupby("split"):
        ax_cal.plot(g["mean_pred"], g["event_rate"], marker="o", label=split_name.capitalize())
    ax_cal.set_title("Calibration")
    ax_cal.set_xlabel("Mean Predicted Exit Probability")
    ax_cal.set_ylabel("Observed Exit Rate")
    ax_cal.legend()
    ax_cal.grid(alpha=0.3)

    for split_name, g in sweep_df.groupby("split"):
        ax_sweep.plot(g["threshold"], g["proxy_model_mean_return"], linewidth=2, label=f"{split_name} proxy_return")
        ax_sweep.plot(g["threshold"], g["f1"], linestyle="--", linewidth=1.8, label=f"{split_name} f1")
    ax_sweep.axvline(exit_threshold, linestyle=":", color="black", alpha=0.6, label="configured threshold")
    ax_sweep.set_title("Threshold Sweep (Proxy Return and F1)")
    ax_sweep.set_xlabel("Exit Threshold")
    ax_sweep.set_ylabel("Value")
    ax_sweep.legend(ncol=2)
    ax_sweep.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "ml_exit_val_test_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "metrics": metrics_df,
        "predictions": predictions_df,
        "calibration": calibration_df,
        "threshold_sweep": sweep_df,
    }


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
                # Embargo the last `forward_days` labels to avoid overlap with day `dt`.
                train_end = i - self.forward_days
                if train_end > 0:
                    train_dates = dates[max(0, train_end - self.train_window_days): train_end]
                    train = feats[feats["date"].isin(train_dates)]
                    X_tr = train[FEATURE_COLS].values
                    y_tr = train["label"].values
                    if len(y_tr) >= self.min_train_samples and len(np.unique(y_tr)) >= 2:
                        model = Pipeline([
                            ("scaler", StandardScaler()),
                            ("clf", GradientBoostingClassifier(
                                n_estimators=100, max_depth=3,
                                learning_rate=0.05, subsample=0.8,
                                random_state=42,
                            )),
                        ])
                        model.fit(X_tr, y_tr)
                        self._model = model

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

        # Convert any exit on a spell into a full spell termination.
        if exit_mask.any():
            exited_spells = feats.loc[exit_mask, ["permno", "_pos_id"]].drop_duplicates()
            keep = feats.merge(
                exited_spells.assign(_drop=1),
                on=["permno", "_pos_id"],
                how="left",
            )
            keep = keep[keep["_drop"].isna()]
        else:
            keep = feats

        result = weights.merge(
            keep[["permno", "date"]].drop_duplicates(),
            on=["permno", "date"],
            how="inner",
        )
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
    panel_eval = panel[panel["date"] >= START].copy()
    bench = equal_weight_daily_returns(panel_eval, "ret")
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

    # Train/validation/test diagnostics from the same feature universe.
    print("\nBuilding ML training diagnostics (train/validation/test) ...")
    diag_out = ROOT / "backtest_hurst_ml" / "output_ml" / "ml_exit_training"
    try:
        diag_out.mkdir(parents=True, exist_ok=True)
        panel_diag = res_base.panel_with_signal.copy()
        required_diag_cols = ["sentiment_z", "net_gex_z", "d4_put_dom", "d5_below_flip"]
        missing_diag_cols = [c for c in required_diag_cols if c not in panel_diag.columns]
        if missing_diag_cols:
            # Keep diagnostics robust even if upstream signal columns are unavailable.
            for col in missing_diag_cols:
                panel_diag[col] = 0.0
            print(f"  [Diagnostics] warning: missing signal columns filled with 0.0: {missing_diag_cols}")

        feats = build_position_features(res_base.weights_long, panel_diag, forward_days=ml_rule.forward_days)
        feats = feats.dropna(subset=FEATURE_COLS + ["label", "fwd_ret", "side"]).reset_index(drop=True)
        splits = chronological_split_with_purge(
            feats,
            train_frac=0.50,
            val_frac=0.25,
            test_frac=0.25,
            purge_days=ml_rule.forward_days,
        )

        train_eval = evaluate_training_efficacy(splits["train"], exit_threshold=ml_rule.exit_threshold)
        train_metrics_df = pd.DataFrame([train_eval["metrics"]])
        train_metrics_df.to_csv(diag_out / "ml_exit_train_metrics.csv", index=False)
        train_eval["calibration"].to_csv(diag_out / "ml_exit_train_calibration.csv", index=False)
        train_eval["predictions"].to_csv(diag_out / "ml_exit_train_predictions.csv", index=False)

        print("  Train efficacy:")
        for key in [
            "roc_auc", "pr_auc", "log_loss", "f1", "precision", "recall",
            "proxy_model_mean_return", "proxy_model_uplift_vs_keep_all", "pred_exit_rate",
        ]:
            val = train_eval["metrics"].get(key, np.nan)
            print(f"    {key:<34} {val:.6f}")

        report = plot_model_training_performance(
            train_eval["model"],
            splits["validation"],
            splits["test"],
            output_dir=diag_out,
            exit_threshold=ml_rule.exit_threshold,
        )
        splits["meta"].to_csv(diag_out / "ml_exit_split_meta.csv", index=False)
        report["metrics"].to_csv(diag_out / "ml_exit_val_test_metrics.csv", index=False)
        print(f"  Diagnostics saved under: {diag_out}")
    except ImportError:
        print("  [Diagnostics] scikit-learn/matplotlib unavailable; skipping train/val/test diagnostics.")
    except Exception as exc:
        print(f"  [Diagnostics] failed: {exc}")

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
