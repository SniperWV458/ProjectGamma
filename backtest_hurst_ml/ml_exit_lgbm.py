from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_hurst_ml.backtest_framework import (
    BacktestConfig,
    BaseEntryExitRule,
    equal_weight_daily_returns,
    load_panel,
    run_backtest,
)
from backtest_hurst_ml.sentiment_gex_strategy import FragilityGEXSignal, GEXSentimentStrategy

ROOT = Path(__file__).resolve().parent.parent

CONFIG: dict[str, Any] = {
    "panel_path": str(ROOT / "data" / "backtest_panel_main.parquet"),
    "start_date": "2018-06-01",
    "run_name": "ml_exit_lgbm",
    "output_root": str(ROOT / "backtest_hurst_ml" / "output_ml" / "ml_exit_lgbm"),
    "hazard_horizon_days": 5,
    "adverse_return_threshold": -0.005,
    "train_frac": 0.50,
    "val_frac": 0.25,
    "test_frac": 0.25,
    "purge_days": 5,
    "policy_mode": "top_quantile",  # threshold | top_quantile
    "policy_threshold_default": 0.55,
    "policy_threshold_grid": [round(x, 2) for x in np.linspace(0.25, 0.80, 12)],
    "policy_quantile_default": 0.20,
    "policy_quantile_grid": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
    "policy_sides": ["short"],
    "benchmark_n_per_leg": 5,
    "training_n_per_leg": 50,
    "training_variants": ["A", "B"],
    "training_gates": ["v2", "v3"],
    "training_neutralities": ["beta", "dollar"],
    "side_experiments": ["long", "short"],
    "use_early_stopping": False,
    "early_stopping_rounds": 300,
    "plot_dpi": 150,
    "random_state": 42,
    "min_train_samples": 500,
    "verbose_eval": 0,
}

LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 20,
    "min_child_weight": 1e-3,
    "min_split_gain": 0.0,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.2,
    "reg_lambda": 2.0,
    "class_weight": None,
    "verbosity": -1,
    "verbose": -1,
}

BASE_FEATURE_COLS = [
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

OPTIONAL_FEATURE_COLS = [
    "bs_b_umd",
    "bs_b_mkt",
    "bs_b_smb",
    "bs_b_hml",
    "bs_ivol",
    "bs_tvol",
    "bs_R2",
    "bs_alpha",
    "gex_net_gex_1pct",
    "gex_call_gex_1pct",
    "gex_put_gex_1pct",
    "gex_total_open_interest",
    "gex_total_option_volume",
    "gex_n_contracts",
    "gex_n_expiries",
    "sent_avg",
    "sent_median",
    "sent_pos_prob",
    "sent_neg_prob",
    "sent_neu_prob",
    "sent_posts_filled",
    "market_equity",
    "dollar_volume",
    "price_abs",
]

CS_FEATURE_SOURCE_COLS = [
    "sentiment_z",
    "net_gex_z",
    "d4_put_dom",
    "stock_ret_1d",
    "stock_ret_5d",
    "bs_b_umd",
    "bs_b_mkt",
    "bs_b_smb",
    "bs_b_hml",
    "bs_ivol",
    "bs_tvol",
    "bs_R2",
    "bs_alpha",
    "gex_net_gex_1pct",
    "gex_call_gex_1pct",
    "gex_put_gex_1pct",
    "gex_total_open_interest",
    "gex_total_option_volume",
    "sent_avg",
    "sent_pos_prob",
    "sent_neg_prob",
    "sent_posts_filled",
    "market_equity",
    "dollar_volume",
    "price_abs",
]


def _future_sum(series: pd.Series, horizon: int) -> pd.Series:
    out = series.shift(-1)
    for j in range(2, horizon + 1):
        out = out + series.shift(-j)
    return out


def add_cross_sectional_features(panel: pd.DataFrame, source_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Add same-date rank/z features. This is cross-sectional at date t only, so it
    does not leak information across time.
    """
    out = panel.copy()
    created: list[str] = []
    for col in source_cols:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        grp = s.groupby(out["date"])
        med = grp.transform("median")
        q75 = grp.transform(lambda x: x.quantile(0.75))
        q25 = grp.transform(lambda x: x.quantile(0.25))
        scale = ((q75 - q25) / 1.349).replace(0.0, np.nan)
        z_col = f"{col}_cs_z"
        rank_col = f"{col}_cs_rank"
        out[z_col] = ((s - med) / scale).clip(-8, 8)
        out[rank_col] = grp.rank(pct=True)
        created.extend([z_col, rank_col])
    return out, created


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if np.unique(y_true).size < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if np.unique(y_true).size < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        p = np.clip(y_prob, 1e-7, 1 - 1e-7)
        return float(log_loss(y_true, p, labels=[0, 1]))
    except Exception:
        return float("nan")


def chronological_split_with_purge(
    df: pd.DataFrame,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    purge_days: int,
) -> dict[str, pd.DataFrame]:
    total = float(train_frac + val_frac + test_frac)
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"train+val+test fractions must sum to 1.0 (got {total}).")

    unique_dates = np.array(sorted(pd.to_datetime(df["date"]).dropna().unique()))
    n = len(unique_dates)
    if n < 30:
        raise ValueError(f"Insufficient unique dates ({n}) for robust split.")

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    train_idx_end = n_train
    val_idx_start = train_idx_end + purge_days
    val_idx_end = val_idx_start + n_val
    test_idx_start = val_idx_end + purge_days
    if test_idx_start >= n:
        raise ValueError("Not enough dates after purge for non-empty test split.")

    train_dates = set(unique_dates[:train_idx_end])
    val_dates = set(unique_dates[val_idx_start:val_idx_end])
    test_dates = set(unique_dates[test_idx_start:])
    if not train_dates or not val_dates or not test_dates:
        raise ValueError("One split is empty; adjust fractions/purge.")

    out = {
        "train": df[df["date"].isin(train_dates)].copy(),
        "validation": df[df["date"].isin(val_dates)].copy(),
        "test": df[df["date"].isin(test_dates)].copy(),
        "meta": pd.DataFrame(
            {
                "split": ["train", "validation", "test"],
                "rows": [
                    int(df["date"].isin(train_dates).sum()),
                    int(df["date"].isin(val_dates).sum()),
                    int(df["date"].isin(test_dates).sum()),
                ],
                "start_date": [min(train_dates), min(val_dates), min(test_dates)],
                "end_date": [max(train_dates), max(val_dates), max(test_dates)],
            }
        ),
    }
    return out


def build_lgbm_position_features(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    hazard_horizon_days: int,
    adverse_return_threshold: float = 0.0,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build model features and discrete-time hazard label.
    """
    w = weights.copy()
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    w["side"] = np.where(w["weight"] > 0, "long", "short")

    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p = p.sort_values(["permno", "date"]).reset_index(drop=True)
    p["ret"] = pd.to_numeric(p["ret"], errors="coerce")

    required_panel_cols = ["sentiment_z", "net_gex_z", "d4_put_dom", "d5_below_flip"]
    for col in required_panel_cols:
        if col not in p.columns:
            p[col] = 0.0
            print(f"[ml_exit_lgbm] warning: missing {col}; filled with 0.0")

    mkt = p.groupby("date")["ret"].mean()
    mkt_df = pd.DataFrame(
        {
            "date": mkt.index,
            "market_ret_5d": mkt.rolling(5, min_periods=1).sum().values,
            "market_ret_20d": mkt.rolling(20, min_periods=5).sum().values,
        }
    )

    grp_ret = p.groupby("permno", sort=False)["ret"]
    p["stock_ret_1d"] = p["ret"]
    p["stock_ret_5d"] = grp_ret.transform(lambda x: x.rolling(5, min_periods=1).sum())
    p["ret_t1"] = grp_ret.transform(lambda s: s.shift(-1))
    for h in range(1, hazard_horizon_days + 1):
        p[f"fwd_ret_{h}d"] = grp_ret.transform(lambda s, hh=h: _future_sum(s, hh))
    p, cs_feature_cols = add_cross_sectional_features(p, CS_FEATURE_SOURCE_COLS)

    use_optional = [c for c in OPTIONAL_FEATURE_COLS if c in p.columns]
    missing_optional = [c for c in OPTIONAL_FEATURE_COLS if c not in p.columns]
    if missing_optional:
        print(f"[ml_exit_lgbm] optional feature columns not found: {missing_optional}")

    feat_cols_panel = [
        "permno",
        "date",
        "ret",
        "sentiment_z",
        "net_gex_z",
        "d4_put_dom",
        "d5_below_flip",
        "stock_ret_1d",
        "stock_ret_5d",
        "ret_t1",
    ] + use_optional + cs_feature_cols
    feat_cols_panel += [f"fwd_ret_{h}d" for h in range(1, hazard_horizon_days + 1)]
    feat_panel = p[feat_cols_panel].copy().merge(mkt_df, on="date", how="left")

    w_sorted = w.sort_values(["permno", "date"]).copy()
    prev_date = w_sorted.groupby("permno", sort=False)["date"].shift(1)
    prev_side = w_sorted.groupby("permno", sort=False)["side"].shift(1)
    gap = (w_sorted["date"] - prev_date).dt.days.fillna(0)
    side_flip = prev_side.ne(w_sorted["side"]) & prev_side.notna()
    new_pos = (gap > 5) | (prev_date.isna()) | side_flip
    w_sorted["_pos_id"] = new_pos.groupby(w_sorted["permno"]).cumsum()
    w_sorted["days_held"] = w_sorted.groupby(["permno", "_pos_id"], sort=False).cumcount() + 1

    out = w_sorted.merge(feat_panel, on=["permno", "date"], how="left")
    out["side_sign"] = np.where(out["side"] == "long", 1.0, -1.0)
    out["signed_ret"] = out["ret"] * out["side_sign"]
    out["pos_return_sofar"] = (
        out.groupby(["permno", "_pos_id"], sort=False)["signed_ret"]
        .transform(lambda x: x.expanding().sum().shift(1).fillna(0.0))
    )

    for h in range(1, hazard_horizon_days + 1):
        out[f"signed_fwd_{h}d"] = out[f"fwd_ret_{h}d"] * out["side_sign"]
    out["signed_ret_t1"] = out["ret_t1"] * out["side_sign"]

    hazard_parts = []
    for h in range(1, hazard_horizon_days + 1):
        hazard_parts.append(out[f"signed_fwd_{h}d"] <= adverse_return_threshold)
    hazard_mat = np.column_stack([hp.to_numpy(dtype=bool) for hp in hazard_parts])
    out["hazard_label"] = hazard_mat.any(axis=1).astype(int)
    out["signed_fwd_k"] = out[f"signed_fwd_{hazard_horizon_days}d"]

    model_feature_cols = BASE_FEATURE_COLS + use_optional + cs_feature_cols
    keep_cols = [
        "permno",
        "date",
        "side",
        "weight",
        "_pos_id",
        "hazard_label",
        "signed_ret",
        "signed_ret_t1",
        "signed_fwd_k",
    ] + model_feature_cols
    return out[keep_cols].copy(), model_feature_cols


def feature_coverage_table(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        s = df[col]
        rows.append(
            {
                "feature": col,
                "non_null_rate": float(s.notna().mean()),
                "zero_rate": float((s == 0).mean()) if pd.api.types.is_numeric_dtype(s) else float("nan"),
                "mean": float(s.mean()) if pd.api.types.is_numeric_dtype(s) else float("nan"),
                "std": float(s.std()) if pd.api.types.is_numeric_dtype(s) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def prediction_distribution_table(pred_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    p = pred_df["y_pred_proba"].astype(float)
    return pd.DataFrame(
        [
            {
                "split": split_name,
                "count": int(p.count()),
                "mean": float(p.mean()),
                "std": float(p.std()),
                "min": float(p.min()),
                "p01": float(p.quantile(0.01)),
                "p05": float(p.quantile(0.05)),
                "p10": float(p.quantile(0.10)),
                "p25": float(p.quantile(0.25)),
                "p50": float(p.quantile(0.50)),
                "p75": float(p.quantile(0.75)),
                "p90": float(p.quantile(0.90)),
                "p95": float(p.quantile(0.95)),
                "p99": float(p.quantile(0.99)),
                "max": float(p.max()),
            }
        ]
    )


def collect_training_candidate_weights(
    backtest_cfg: BacktestConfig,
    signal: FragilityGEXSignal,
    benchmark_daily_returns: pd.Series,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for variant in cfg["training_variants"]:
        for gate in cfg["training_gates"]:
            for neutrality in cfg["training_neutralities"]:
                strategy = GEXSentimentStrategy(
                    variant=str(variant),
                    gate=str(gate),
                    neutrality=str(neutrality),
                    n_per_leg=int(cfg["training_n_per_leg"]),
                    time_stop_days=int(cfg["hazard_horizon_days"]),
                    long_leg="momentum",
                )
                try:
                    result = run_backtest(
                        backtest_cfg,
                        signal,
                        strategy,
                        benchmark_daily_returns=benchmark_daily_returns,
                    )
                except Exception as exc:
                    print(f"[ml_exit_lgbm] skipped training candidates {variant}/{gate}/{neutrality}: {exc}")
                    continue
                w = result.weights_long[["permno", "date", "weight"]].copy()
                w["source_variant"] = str(variant)
                w["source_gate"] = str(gate)
                w["source_neutrality"] = str(neutrality)
                frames.append(w)

    if not frames:
        raise ValueError("No training candidate weights were generated.")

    weights = pd.concat(frames, ignore_index=True)
    weights["date"] = pd.to_datetime(weights["date"]).dt.normalize()
    weights["side"] = np.where(weights["weight"] > 0, "long", "short")
    weights = (
        weights.sort_values(["date", "permno", "side", "source_variant", "source_gate", "source_neutrality"])
        .drop_duplicates(["permno", "date", "side"], keep="first")
        .drop(columns=["side", "source_variant", "source_gate", "source_neutrality"])
        .reset_index(drop=True)
    )
    print(f"[ml_exit_lgbm] training candidates after dedupe: {len(weights)} rows")
    return weights


def export_prefit_diagnostics(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    out_dir: Path,
) -> None:
    frames = []
    for split_name in ["train", "validation", "test"]:
        frames.append(splits[split_name].assign(split=split_name))
    all_df = pd.concat(frames, ignore_index=True)
    all_df["year_month"] = pd.to_datetime(all_df["date"]).dt.to_period("M").astype(str)

    label_by_split_side = (
        all_df.groupby(["split", "side", "hazard_label"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    label_by_split_side.to_csv(out_dir / "prefit_label_counts_by_split_side.csv", index=False)

    label_by_month = (
        all_df.groupby(["split", "year_month"])
        .agg(rows=("hazard_label", "size"), label_rate=("hazard_label", "mean"))
        .reset_index()
    )
    label_by_month.to_csv(out_dir / "prefit_label_rate_by_month.csv", index=False)

    return_dist = (
        all_df.groupby(["split", "hazard_label"])["signed_fwd_k"]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .reset_index()
    )
    return_dist.to_csv(out_dir / "prefit_signed_fwd_k_by_label.csv", index=False)

    date_counts = (
        all_df.groupby(["split", "date"])
        .agg(rows=("hazard_label", "size"), label_rate=("hazard_label", "mean"))
        .reset_index()
    )
    date_counts.to_csv(out_dir / "prefit_rows_by_date.csv", index=False)

    feature_coverage_table(all_df, feature_cols).to_csv(out_dir / "feature_coverage_after_split.csv", index=False)


def model_fit_summary(model, feature_cols: list[str], pred_frames: list[pd.DataFrame], experiment: str) -> pd.DataFrame:
    importances = model.booster_.feature_importance(importance_type="gain")
    pred_all = pd.concat(pred_frames, ignore_index=True)
    return pd.DataFrame(
        [
            {
                "experiment": experiment,
                "n_features": len(feature_cols),
                "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
                "n_trees": int(model.booster_.num_trees()),
                "total_gain": float(np.sum(importances)),
                "prediction_std": float(pred_all["y_pred_proba"].std()),
                "unique_prediction_count": int(pred_all["y_pred_proba"].nunique()),
            }
        ]
    )


def _fit_lgbm_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    random_state: int,
    verbose_eval: int,
):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required for ml_exit_lgbm.py. Install with: pip install lightgbm"
        ) from exc

    params = dict(LGBM_PARAMS)
    params["random_state"] = random_state

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if n_pos > 0:
        params["scale_pos_weight"] = float(n_neg / max(n_pos, 1))

    model = lgb.LGBMClassifier(**params)
    callbacks = []
    if int(verbose_eval) > 0:
        callbacks.append(lgb.log_evaluation(period=max(0, int(verbose_eval))))
    if bool(CONFIG.get("use_early_stopping", False)):
        callbacks.insert(
            0,
            lgb.early_stopping(
                stopping_rounds=int(CONFIG.get("early_stopping_rounds", 300)),
                verbose=False,
            ),
        )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc", "binary_logloss"],
        callbacks=callbacks,
    )
    return model


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n_rows": float(len(y_true)),
        "label_rate": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "roc_auc": _safe_auc(y_true, y_prob),
        "pr_auc": _safe_pr_auc(y_true, y_prob),
        "log_loss": _safe_log_loss(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)) if len(y_true) else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _eligible_policy_index(df_pred: pd.DataFrame, policy_sides: list[str] | None) -> pd.Index:
    if not policy_sides:
        return df_pred.index
    return df_pred[df_pred["side"].isin(policy_sides)].index


def _policy_exit_mask(
    df_pred: pd.DataFrame,
    *,
    mode: str,
    threshold: float,
    quantile: float,
    policy_sides: list[str] | None = None,
) -> np.ndarray:
    eligible_idx = _eligible_policy_index(df_pred, policy_sides)
    eligible = df_pred.loc[eligible_idx]
    exit_mask = pd.Series(False, index=df_pred.index)
    if mode == "threshold":
        exit_mask.loc[eligible.index] = eligible["y_pred_proba"].values >= float(threshold)
        return exit_mask.values.astype(bool)
    if mode == "top_quantile":
        for _, g in eligible.groupby("date", sort=False):
            n_exit = max(1, int(np.ceil(len(g) * float(quantile))))
            ranked_idx = g.sort_values(["y_pred_proba", "permno"], ascending=[False, True]).head(n_exit).index
            exit_mask.loc[ranked_idx] = True
        return exit_mask.values.astype(bool)
    raise ValueError(f"Unknown policy mode: {mode}")


def _same_count_baseline_returns(
    df_pred: pd.DataFrame,
    exit_mask: np.ndarray,
    policy_sides: list[str] | None = None,
) -> dict[str, float]:
    oracle_contrib = []
    random_contrib = []
    rng = np.random.default_rng(42)
    for _, g in df_pred.groupby("date", sort=False):
        idx = g.index.to_numpy()
        n_exit = int(exit_mask[df_pred.index.get_indexer(idx)].sum())
        eligible = g.loc[_eligible_policy_index(g, policy_sides)]
        if n_exit <= 0:
            oracle_exit = set()
            random_exit = set()
        else:
            oracle_exit = set(eligible.sort_values("signed_fwd_k", ascending=True).head(n_exit).index)
            random_pool = eligible.index.to_numpy()
            random_exit = set(rng.choice(random_pool, size=min(n_exit, len(random_pool)), replace=False))
        oracle_contrib.extend(np.where(g.index.isin(oracle_exit), 0.0, g["signed_fwd_k"].values))
        random_contrib.extend(np.where(g.index.isin(random_exit), 0.0, g["signed_fwd_k"].values))

    return {
        "proxy_oracle_same_count_mean_fwd_k": float(np.nanmean(oracle_contrib)),
        "proxy_random_same_count_mean_fwd_k": float(np.nanmean(random_contrib)),
    }


def _policy_proxy_metrics(
    df_pred: pd.DataFrame,
    *,
    mode: str,
    threshold: float,
    quantile: float,
    policy_sides: list[str] | None = None,
) -> dict[str, float]:
    exit_mask = _policy_exit_mask(
        df_pred,
        mode=mode,
        threshold=threshold,
        quantile=quantile,
        policy_sides=policy_sides,
    )
    keep_all = df_pred["signed_fwd_k"].values
    model_keep = np.where(exit_mask, 0.0, keep_all)
    baselines = _same_count_baseline_returns(df_pred, exit_mask, policy_sides=policy_sides)
    eligible_count = len(_eligible_policy_index(df_pred, policy_sides))
    return {
        "policy_exit_rate": float(np.mean(exit_mask)),
        "policy_eligible_rate": float(eligible_count / len(df_pred)) if len(df_pred) else float("nan"),
        "policy_unique_prediction_count": float(df_pred["y_pred_proba"].nunique()),
        "proxy_keep_all_mean_fwd_k": float(np.nanmean(keep_all)),
        "proxy_model_mean_fwd_k": float(np.nanmean(model_keep)),
        "proxy_uplift_vs_keep_all": float(np.nanmean(model_keep) - np.nanmean(keep_all)),
        "proxy_oracle_same_count_mean_fwd_k": baselines["proxy_oracle_same_count_mean_fwd_k"],
        "proxy_random_same_count_mean_fwd_k": baselines["proxy_random_same_count_mean_fwd_k"],
    }


def tune_policy_on_validation(df_val_pred: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, float], pd.DataFrame]:
    mode = str(cfg["policy_mode"])
    rows: list[dict[str, float]] = []
    if mode == "threshold":
        for th in cfg["policy_threshold_grid"]:
            met = _policy_proxy_metrics(
                df_val_pred,
                mode=mode,
                threshold=float(th),
                quantile=float(cfg["policy_quantile_default"]),
                policy_sides=list(cfg.get("policy_sides") or []),
            )
            rows.append({"policy_value": float(th), **met})
    elif mode == "top_quantile":
        for q in cfg["policy_quantile_grid"]:
            met = _policy_proxy_metrics(
                df_val_pred,
                mode=mode,
                threshold=float(cfg["policy_threshold_default"]),
                quantile=float(q),
                policy_sides=list(cfg.get("policy_sides") or []),
            )
            rows.append({"policy_value": float(q), **met})
    else:
        raise ValueError(f"Unsupported policy_mode={mode}")

    sweep_df = pd.DataFrame(rows).sort_values("policy_value").reset_index(drop=True)
    best_ix = int(sweep_df["proxy_uplift_vs_keep_all"].astype(float).values.argmax())
    best_val = float(sweep_df.loc[best_ix, "policy_value"])
    if mode == "threshold":
        best_policy = {
            "mode": mode,
            "threshold": best_val,
            "quantile": float(cfg["policy_quantile_default"]),
            "policy_sides": list(cfg.get("policy_sides") or []),
        }
    else:
        best_policy = {
            "mode": mode,
            "threshold": float(cfg["policy_threshold_default"]),
            "quantile": best_val,
            "policy_sides": list(cfg.get("policy_sides") or []),
        }
    return best_policy, sweep_df


def evaluate_split(
    model,
    split_df: pd.DataFrame,
    feature_cols: list[str],
    split_name: str,
    policy: dict[str, float],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    eval_df = split_df.dropna(subset=["hazard_label", "signed_ret_t1", "signed_fwd_k"]).copy()
    X = eval_df[feature_cols]
    y_true = eval_df["hazard_label"].astype(int).values
    y_prob = model.predict_proba(X)[:, 1]

    y_pred_50 = (y_prob >= 0.50).astype(int)
    cls = _classification_metrics(y_true, y_prob, y_pred_50)

    pred = eval_df[["permno", "date", "side", "_pos_id", "hazard_label", "signed_ret_t1", "signed_fwd_k"]].copy()
    pred["y_pred_proba"] = y_prob

    pol = _policy_proxy_metrics(
        pred,
        mode=str(policy["mode"]),
        threshold=float(policy["threshold"]),
        quantile=float(policy["quantile"]),
        policy_sides=list(policy.get("policy_sides") or []),
    )
    pred["policy_exit"] = _policy_exit_mask(
        pred,
        mode=str(policy["mode"]),
        threshold=float(policy["threshold"]),
        quantile=float(policy["quantile"]),
        policy_sides=list(policy.get("policy_sides") or []),
    ).astype(int)

    if np.unique(y_true).size >= 2:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        cal_df = pd.DataFrame(
            {"split": split_name, "mean_pred": mean_pred, "event_rate": frac_pos}
        )
    else:
        cal_df = pd.DataFrame(columns=["split", "mean_pred", "event_rate"])

    metrics = {"split": split_name, **cls, **pol}
    return metrics, pred, cal_df


def run_model_experiment(
    experiment: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    X_train, y_train = train_df[feature_cols], train_df["hazard_label"].astype(int)
    X_val, y_val = val_df[feature_cols], val_df["hazard_label"].astype(int)

    model = _fit_lgbm_classifier(
        X_train,
        y_train,
        X_val,
        y_val,
        random_state=int(cfg["random_state"]),
        verbose_eval=int(cfg["verbose_eval"]),
    )

    train_pred_for_summary = train_df[["permno", "date", "side", "_pos_id", "hazard_label", "signed_ret_t1", "signed_fwd_k"]].copy()
    train_pred_for_summary["y_pred_proba"] = model.predict_proba(X_train)[:, 1]
    val_pred_for_tuning = val_df[["permno", "date", "side", "_pos_id", "hazard_label", "signed_ret_t1", "signed_fwd_k"]].copy()
    val_pred_for_tuning["y_pred_proba"] = model.predict_proba(X_val)[:, 1]

    best_policy, sweep_df = tune_policy_on_validation(val_pred_for_tuning, cfg)
    print(f"[ml_exit_lgbm] {experiment} selected policy: {best_policy}")

    metrics_val, pred_val, cal_val = evaluate_split(model, val_df, feature_cols, "validation", best_policy)
    metrics_test, pred_test, cal_test = evaluate_split(model, test_df, feature_cols, "test", best_policy)
    metrics_df = pd.DataFrame([metrics_val, metrics_test])
    pred_dist = pd.concat(
        [
            prediction_distribution_table(train_pred_for_summary, "train"),
            prediction_distribution_table(pred_val, "validation"),
            prediction_distribution_table(pred_test, "test"),
        ],
        ignore_index=True,
    )
    summary_df = model_fit_summary(model, feature_cols, [train_pred_for_summary, pred_val, pred_test], experiment)
    total_gain = float(summary_df["total_gain"].iloc[0])
    pred_std = float(summary_df["prediction_std"].iloc[0])
    if total_gain <= 0.0 or pred_std <= 0.0:
        print(
            f"[ml_exit_lgbm] WARNING: {experiment} model produced weak/constant fit: "
            f"total_gain={total_gain:.6g}, prediction_std={pred_std:.6g}"
        )

    prefix = f"{experiment}_"
    metrics_df.to_csv(run_dir / f"{prefix}model_metrics_validation_test.csv", index=False)
    sweep_df.to_csv(run_dir / f"{prefix}validation_policy_sweep.csv", index=False)
    pd.concat([cal_val, cal_test], ignore_index=True).to_csv(
        run_dir / f"{prefix}calibration_validation_test.csv", index=False
    )
    pd.concat([pred_val.assign(split="validation"), pred_test.assign(split="test")], ignore_index=True).to_csv(
        run_dir / f"{prefix}predictions_validation_test.csv", index=False
    )
    pred_dist.to_csv(run_dir / f"{prefix}prediction_distribution.csv", index=False)
    summary_df.to_csv(run_dir / f"{prefix}model_fit_summary.csv", index=False)
    pd.DataFrame(
        {"feature": feature_cols, "importance_gain": model.booster_.feature_importance(importance_type="gain")}
    ).sort_values("importance_gain", ascending=False).to_csv(
        run_dir / f"{prefix}feature_importance_gain.csv", index=False
    )

    return {
        "experiment": experiment,
        "model": model,
        "best_policy": best_policy,
        "sweep": sweep_df,
        "metrics": metrics_df,
        "pred_val": pred_val,
        "pred_test": pred_test,
        "cal_val": cal_val,
        "cal_test": cal_test,
        "pred_dist": pred_dist,
        "summary": summary_df,
    }


class LGBMExitRule(BaseEntryExitRule):
    """
    Walk-forward LightGBM exit rule for run_backtest(entry_exit=...).

    The rule trains only on dates before each prediction date with a forward-horizon
    embargo. Rich optional features are merged from `feature_panel_path` at date t.
    """

    def __init__(
        self,
        *,
        feature_panel_path: str | Path | None = None,
        hazard_horizon_days: int = 5,
        adverse_return_threshold: float = -0.005,
        train_window_days: int = 756,
        retrain_freq_days: int = 63,
        min_train_samples: int = 500,
        policy_mode: str = "top_quantile",
        policy_quantile: float = 0.10,
        policy_threshold: float = 0.55,
        policy_sides: list[str] | None = None,
        use_expanded_training_candidates: bool = True,
    ):
        self.feature_panel_path = Path(feature_panel_path) if feature_panel_path is not None else None
        self.hazard_horizon_days = int(hazard_horizon_days)
        self.adverse_return_threshold = float(adverse_return_threshold)
        self.train_window_days = int(train_window_days)
        self.retrain_freq_days = int(retrain_freq_days)
        self.min_train_samples = int(min_train_samples)
        self.policy_mode = policy_mode
        self.policy_quantile = float(policy_quantile)
        self.policy_threshold = float(policy_threshold)
        self.policy_sides = policy_sides if policy_sides is not None else ["short"]
        self.use_expanded_training_candidates = bool(use_expanded_training_candidates)
        self._model = None
        self._feature_cols: list[str] | None = None

    def _enrich_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        if self.feature_panel_path is None or not self.feature_panel_path.exists():
            return panel
        try:
            import pyarrow.parquet as pq

            available = {f.name for f in pq.ParquetFile(self.feature_panel_path).schema_arrow}
            wanted = ["permno", "date"] + [
                c for c in OPTIONAL_FEATURE_COLS if c in available and c not in panel.columns
            ]
            if len(wanted) <= 2:
                return panel
            extra = pd.read_parquet(self.feature_panel_path, columns=wanted)
            extra["date"] = pd.to_datetime(extra["date"]).dt.normalize()
            out = panel.copy()
            out["date"] = pd.to_datetime(out["date"]).dt.normalize()
            return out.merge(extra, on=["permno", "date"], how="left")
        except Exception as exc:
            print(f"  [LGBMExitRule] feature panel enrichment failed; using provided panel only: {exc}")
            return panel

    def _training_weights(self, weights: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
        if not self.use_expanded_training_candidates:
            return weights
        frames: list[pd.DataFrame] = []
        for variant in CONFIG["training_variants"]:
            for gate in CONFIG["training_gates"]:
                for neutrality in CONFIG["training_neutralities"]:
                    try:
                        strat = GEXSentimentStrategy(
                            variant=str(variant),
                            gate=str(gate),
                            neutrality=str(neutrality),
                            n_per_leg=int(CONFIG["training_n_per_leg"]),
                            time_stop_days=self.hazard_horizon_days,
                            long_leg="momentum",
                        )
                        frames.append(strat.target_weights(panel)[["permno", "date", "weight"]])
                    except Exception as exc:
                        print(f"  [LGBMExitRule] skipped candidate {variant}/{gate}/{neutrality}: {exc}")
        if not frames:
            return weights
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out["side"] = np.where(out["weight"] > 0, "long", "short")
        out = (
            out.sort_values(["date", "permno", "side"])
            .drop_duplicates(["permno", "date", "side"], keep="first")
            .drop(columns=["side"])
            .reset_index(drop=True)
        )
        return out

    def apply(self, weights: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
        if weights.empty:
            return weights

        panel_rich = self._enrich_panel(panel)
        train_weights = self._training_weights(weights, panel_rich)
        train_feats, feature_cols = build_lgbm_position_features(
            train_weights,
            panel_rich,
            hazard_horizon_days=self.hazard_horizon_days,
            adverse_return_threshold=self.adverse_return_threshold,
        )
        pred_feats, _ = build_lgbm_position_features(
            weights,
            panel_rich,
            hazard_horizon_days=self.hazard_horizon_days,
            adverse_return_threshold=self.adverse_return_threshold,
        )
        train_feats = train_feats.dropna(subset=["hazard_label", "signed_ret_t1", "signed_fwd_k"]).reset_index(drop=True)
        pred_feats = pred_feats.dropna(subset=["hazard_label", "signed_ret_t1", "signed_fwd_k"]).reset_index(drop=True)
        if train_feats.empty or pred_feats.empty:
            return weights

        dates = sorted(pd.to_datetime(pred_feats["date"]).unique())
        if len(dates) <= self.train_window_days:
            print("  [LGBMExitRule] insufficient dates for walk-forward training; skipping.")
            return weights

        exit_rows: list[pd.DataFrame] = []
        model = None
        for i, dt in enumerate(dates):
            if i < self.train_window_days:
                continue
            if model is None or (i - self.train_window_days) % self.retrain_freq_days == 0:
                train_end = i - self.hazard_horizon_days
                if train_end <= 0:
                    continue
                train_dates = dates[max(0, train_end - self.train_window_days): train_end]
                tr = train_feats[train_feats["date"].isin(train_dates)].copy()
                if len(tr) < self.min_train_samples or tr["hazard_label"].nunique() < 2:
                    continue
                model = _fit_lgbm_classifier(
                    tr[feature_cols],
                    tr["hazard_label"].astype(int),
                    tr[feature_cols],
                    tr["hazard_label"].astype(int),
                    random_state=int(CONFIG["random_state"]),
                    verbose_eval=0,
                )
                self._model = model
                self._feature_cols = feature_cols

            if model is None:
                continue
            day = pred_feats[pred_feats["date"] == dt].copy()
            if day.empty:
                continue
            day["y_pred_proba"] = model.predict_proba(day[feature_cols])[:, 1]
            day["policy_exit"] = _policy_exit_mask(
                day,
                mode=self.policy_mode,
                threshold=self.policy_threshold,
                quantile=self.policy_quantile,
                policy_sides=self.policy_sides,
            )
            exits = day[day["policy_exit"]]
            if not exits.empty:
                exit_rows.append(exits[["permno", "_pos_id", "date"]])

        if not exit_rows:
            return weights

        exits = pd.concat(exit_rows, ignore_index=True)
        first_exit = (
            exits.groupby(["permno", "_pos_id"], as_index=False)["date"]
            .min()
            .rename(columns={"date": "_first_exit_date"})
        )
        keep = pred_feats.merge(first_exit, on=["permno", "_pos_id"], how="left")
        keep = keep[keep["_first_exit_date"].isna() | (keep["date"] < keep["_first_exit_date"])]
        result = weights.merge(
            keep[["permno", "date"]].drop_duplicates(),
            on=["permno", "date"],
            how="inner",
        )
        print(f"  [LGBMExitRule] kept {len(result):,}/{len(weights):,} rows after short-only LGBM exits.")
        return result


def make_naive_proxy_nav(pred_df: pd.DataFrame) -> pd.DataFrame:
    # Uses the same forward horizon as the hazard label for a lightweight proxy benchmark.
    daily = (
        pred_df.groupby("date")
        .agg(
            no_exit_ret=("signed_fwd_k", "mean"),
            model_ret=("signed_fwd_k", lambda s: np.nan),
        )
        .reset_index()
    )
    model_ret = (
        pred_df.assign(model_contrib=np.where(pred_df["policy_exit"] == 1, 0.0, pred_df["signed_fwd_k"]))
        .groupby("date")["model_contrib"]
        .mean()
        .rename("model_ret")
        .reset_index()
    )
    daily = daily.drop(columns=["model_ret"]).merge(model_ret, on="date", how="left")
    daily["no_exit_nav"] = (1.0 + daily["no_exit_ret"].fillna(0.0)).cumprod()
    daily["model_nav"] = (1.0 + daily["model_ret"].fillna(0.0)).cumprod()
    return daily


def plot_diagnostics(
    pred_val: pd.DataFrame,
    pred_test: pd.DataFrame,
    cal_val: pd.DataFrame,
    cal_test: pd.DataFrame,
    sweep_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if plt is None:
        print("[ml_exit_lgbm] matplotlib not installed; skipping PNG plots (CSVs still written).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, dfp in [("Validation", pred_val), ("Test", pred_test)]:
        y = dfp["hazard_label"].astype(int).values
        p = dfp["y_pred_proba"].values
        if np.unique(y).size >= 2:
            fpr, tpr, _ = roc_curve(y, p)
            axes[0].plot(fpr, tpr, label=name, linewidth=2)
            prec, rec, _ = precision_recall_curve(y, p)
            axes[1].plot(rec, prec, label=name, linewidth=2)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    axes[0].set_title("ROC (Validation/Test)")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_title("PR (Validation/Test)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(out_dir / "roc_pr_validation_test.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    cal = pd.concat([cal_val, cal_test], ignore_index=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6, label="perfect")
    for split, g in cal.groupby("split"):
        ax.plot(g["mean_pred"], g["event_rate"], marker="o", linewidth=2, label=split)
    ax.set_title("Calibration")
    ax.set_xlabel("Mean predicted hazard")
    ax.set_ylabel("Observed hazard rate")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "calibration_validation_test.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_df["policy_value"], sweep_df["proxy_uplift_vs_keep_all"], marker="o", linewidth=2)
    ax.set_title("Validation policy sweep (proxy uplift)")
    ax.set_xlabel("policy_value (threshold or top-quantile)")
    ax.set_ylabel("proxy uplift vs no-exit")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "validation_policy_sweep.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    nav_df = nav_df.sort_values("date")
    ax.plot(nav_df["date"], nav_df["no_exit_nav"], label="No-exit baseline", linewidth=2)
    ax.plot(nav_df["date"], nav_df["model_nav"], label="Model policy", linewidth=2)
    ax.set_title("Naive proxy NAV (test)")
    ax.set_ylabel("NAV")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "naive_proxy_nav_test.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = dict(CONFIG)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["output_root"]) / f"{cfg['run_name']}_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    panel_path = Path(cfg["panel_path"])
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel not found: {panel_path}")
    start = pd.Timestamp(cfg["start_date"])

    backtest_cfg = BacktestConfig(
        panel_path=panel_path,
        start_date=start,
        signal_col="signal",
        signal_lag_days=0,
        missing_return_policy="zero_contribution",
        transaction_cost_bps=5.0,
    )

    panel = load_panel(panel_path)
    panel_eval = panel[panel["date"] >= start].copy()
    bench = equal_weight_daily_returns(panel_eval, "ret")

    signal = FragilityGEXSignal()
    benchmark_strategy = GEXSentimentStrategy(
        variant="A",
        gate="v2",
        neutrality="beta",
        n_per_leg=int(cfg["benchmark_n_per_leg"]),
        time_stop_days=int(cfg["hazard_horizon_days"]),
        long_leg="momentum",
    )

    print("[ml_exit_lgbm] running benchmark backtest for unchanged narrow strategy...")
    res_benchmark = run_backtest(backtest_cfg, signal, benchmark_strategy, benchmark_daily_returns=bench)

    print("[ml_exit_lgbm] collecting expanded training candidates...")
    training_weights = collect_training_candidate_weights(backtest_cfg, signal, bench, cfg)
    feats, model_feature_cols = build_lgbm_position_features(
        training_weights,
        res_benchmark.panel_with_signal,
        hazard_horizon_days=int(cfg["hazard_horizon_days"]),
        adverse_return_threshold=float(cfg["adverse_return_threshold"]),
    )
    benchmark_feats, _ = build_lgbm_position_features(
        res_benchmark.weights_long,
        res_benchmark.panel_with_signal,
        hazard_horizon_days=int(cfg["hazard_horizon_days"]),
        adverse_return_threshold=float(cfg["adverse_return_threshold"]),
    )

    feature_coverage_table(feats, model_feature_cols).to_csv(run_dir / "feature_coverage.csv", index=False)

    # LightGBM handles missing feature values; only require target/proxy fields.
    feats = feats.dropna(subset=["hazard_label", "signed_ret_t1", "signed_fwd_k"]).reset_index(drop=True)
    benchmark_feats = benchmark_feats.dropna(subset=["hazard_label", "signed_ret_t1", "signed_fwd_k"]).reset_index(drop=True)
    splits = chronological_split_with_purge(
        feats,
        train_frac=float(cfg["train_frac"]),
        val_frac=float(cfg["val_frac"]),
        test_frac=float(cfg["test_frac"]),
        purge_days=int(cfg["purge_days"]),
    )
    export_prefit_diagnostics(splits, model_feature_cols, run_dir)

    train_df = splits["train"].dropna(subset=["hazard_label"])
    val_df = splits["validation"].dropna(subset=["hazard_label"])
    test_df = splits["test"].dropna(subset=["hazard_label"])
    if len(train_df) < int(cfg["min_train_samples"]):
        raise ValueError(f"Insufficient train samples: {len(train_df)} < {cfg['min_train_samples']}")
    if train_df["hazard_label"].nunique() < 2:
        raise ValueError("Training split has only one hazard_label class; adjust label threshold or split window.")

    print(
        f"[ml_exit_lgbm] split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"| features={len(model_feature_cols)}"
    )

    core_feature_cols = [c for c in BASE_FEATURE_COLS if c in model_feature_cols]
    experiments = [
        run_model_experiment("core", train_df, val_df, test_df, core_feature_cols, cfg, run_dir),
        run_model_experiment("rich", train_df, val_df, test_df, model_feature_cols, cfg, run_dir),
    ]
    for side in cfg["side_experiments"]:
        side_train = train_df[train_df["side"] == side].copy()
        side_val = val_df[val_df["side"] == side].copy()
        side_test = test_df[test_df["side"] == side].copy()
        if (
            len(side_train) < int(cfg["min_train_samples"])
            or side_train["hazard_label"].nunique() < 2
            or side_val["hazard_label"].nunique() < 2
            or side_test.empty
        ):
            print(
                f"[ml_exit_lgbm] skipping {side}_rich side experiment: "
                f"train={len(side_train)}, val={len(side_val)}, test={len(side_test)}, "
                f"train_classes={side_train['hazard_label'].nunique() if not side_train.empty else 0}"
            )
            continue
        experiments.append(
            run_model_experiment(
                f"{side}_rich",
                side_train,
                side_val,
                side_test,
                model_feature_cols,
                cfg,
                run_dir,
            )
        )
    ablation_summary = pd.concat(
        [
            pd.concat(
                [
                    exp["summary"].reset_index(drop=True),
                    exp["metrics"][exp["metrics"]["split"] == "test"].add_prefix("test_").reset_index(drop=True),
                ],
                axis=1,
            )
            for exp in experiments
        ],
        ignore_index=True,
    )
    ablation_summary.to_csv(run_dir / "ablation_summary.csv", index=False)

    selected = next(exp for exp in experiments if exp["experiment"] == "rich")
    model = selected["model"]
    best_policy = selected["best_policy"]
    sweep_df = selected["sweep"]
    metrics_df = selected["metrics"]
    pred_val = selected["pred_val"]
    pred_test = selected["pred_test"]
    cal_val = selected["cal_val"]
    cal_test = selected["cal_test"]
    pred_dist = selected["pred_dist"]

    nav_test = make_naive_proxy_nav(pred_test)

    benchmark_test = benchmark_feats[benchmark_feats["date"].isin(test_df["date"].unique())].copy()
    benchmark_metrics, benchmark_pred, _ = evaluate_split(
        model,
        benchmark_test,
        model_feature_cols,
        "benchmark_test",
        best_policy,
    )
    benchmark_nav = make_naive_proxy_nav(benchmark_pred)

    cal_df = pd.concat([cal_val, cal_test], ignore_index=True)
    pred_all = pd.concat([pred_val.assign(split="validation"), pred_test.assign(split="test")], ignore_index=True)

    splits["meta"].to_csv(run_dir / "split_meta.csv", index=False)
    metrics_df.to_csv(run_dir / "model_metrics_validation_test.csv", index=False)
    pd.DataFrame([benchmark_metrics]).to_csv(run_dir / "benchmark_strategy_test_metrics.csv", index=False)
    sweep_df.to_csv(run_dir / "validation_policy_sweep.csv", index=False)
    cal_df.to_csv(run_dir / "calibration_validation_test.csv", index=False)
    pred_all.to_csv(run_dir / "predictions_validation_test.csv", index=False)
    benchmark_pred.to_csv(run_dir / "benchmark_strategy_test_predictions.csv", index=False)
    nav_test.to_csv(run_dir / "naive_proxy_nav_test.csv", index=False)
    benchmark_nav.to_csv(run_dir / "benchmark_strategy_proxy_nav_test.csv", index=False)
    pred_dist.to_csv(run_dir / "prediction_distribution.csv", index=False)

    selected["summary"].to_csv(run_dir / "model_fit_summary.csv", index=False)

    plot_diagnostics(
        pred_val,
        pred_test,
        cal_val,
        cal_test,
        sweep_df,
        nav_test,
        run_dir,
        int(cfg["plot_dpi"]),
    )

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "CONFIG": cfg,
                "LGBM_PARAMS": LGBM_PARAMS,
                "best_policy": best_policy,
                "selected_experiment": selected["experiment"],
            },
            f,
            indent=2,
            default=str,
        )

    print("[ml_exit_lgbm] done")
    print(f"[ml_exit_lgbm] outputs: {run_dir}")
    print("[ml_exit_lgbm] key test metrics:")
    print(metrics_df[metrics_df["split"] == "test"].to_string(index=False))
    print("[ml_exit_lgbm] unchanged benchmark strategy test metrics:")
    print(pd.DataFrame([benchmark_metrics]).to_string(index=False))
    print(
        "[ml_exit_lgbm] naive test proxy uplift:",
        float(metrics_df.loc[metrics_df["split"] == "test", "proxy_uplift_vs_keep_all"].iloc[0]),
    )


if __name__ == "__main__":
    main()
