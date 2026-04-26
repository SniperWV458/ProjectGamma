
"""
Diagnostics and visualizations for Hurst-enhanced GEX x Sentiment backtests.

v4 updates:
1. Main plots now include H10/H30/H60 short filters AND H10/H30/H60 short+MLExit variants.
2. The kept-vs-filtered diagnostic is upgraded from row-level same-day payoff
   to trade-entry-level analysis:
   - baseline short entries
   - whether each baseline short trade is kept by the Hurst-filter strategy
   - realized holding-period short payoff
   - forward 5-day short payoff
   - forward 10-day short payoff
3. Outputs clearer CSV files for Hurst trade quality analysis.

Outputs:
1. performance_50bps.csv / performance_200bps.csv with long/short attribution.
2. 01_nav_drawdown.png
3. 02_leg_cumulative_pnl.png
4. 03_active_positions.png
5. 04_short_trade_kept_vs_filtered_distribution.png
6. 05_hurst_bins_short_setup.png
7. 06_drawdown_episode_1/2/3.png
8. 07_robustness_short_long_both.csv
9. 08_subperiod_robustness_50bps.csv
10. short_trade_kept_vs_filtered_<hurst_label>.csv
11. short_trade_kept_vs_filtered_summary_<hurst_label>.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtest_hurst_ml.backtest_framework import performance_metrics


def adjust_for_borrow(daily_returns: pd.Series, weights: pd.DataFrame, borrow_bps_annual: float) -> pd.Series:
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


def _side_diagnostics(weights: pd.DataFrame, dates: pd.Index, side: str) -> dict:
    prefix = side
    empty = {
        f"{prefix}_signal_days": 0,
        f"{prefix}_entries": 0,
        f"{prefix}_position_rows": 0,
        f"avg_active_{prefix}s": 0.0,
        f"avg_{prefix}_holding_days": np.nan,
        f"median_{prefix}_holding_days": np.nan,
        f"max_{prefix}_holding_days": np.nan,
    }
    if weights.empty:
        return empty

    w = weights.copy()
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    w = w.loc[w["weight"] > 0].copy() if side == "long" else w.loc[w["weight"] < 0].copy()
    if w.empty:
        return empty

    all_dates = pd.Index(pd.to_datetime(dates).normalize()).sort_values()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    durations: list[int] = []

    for _, g in w.groupby("permno", sort=False):
        active_dates = sorted(g["date"].unique())
        if not active_dates:
            continue
        current_len = 1
        prev_idx = date_to_idx.get(active_dates[0], None)

        for d in active_dates[1:]:
            cur_idx = date_to_idx.get(d, None)
            if prev_idx is not None and cur_idx is not None and cur_idx == prev_idx + 1:
                current_len += 1
            else:
                durations.append(current_len)
                current_len = 1
            prev_idx = cur_idx
        durations.append(current_len)

    arr = np.asarray(durations, dtype=float)
    return {
        f"{prefix}_signal_days": int(w["date"].nunique()),
        f"{prefix}_entries": int(len(arr)),
        f"{prefix}_position_rows": int(len(w)),
        f"avg_active_{prefix}s": float(w.groupby("date")["permno"].nunique().mean()),
        f"avg_{prefix}_holding_days": float(np.mean(arr)) if len(arr) else np.nan,
        f"median_{prefix}_holding_days": float(np.median(arr)) if len(arr) else np.nan,
        f"max_{prefix}_holding_days": float(np.max(arr)) if len(arr) else np.nan,
    }


def position_diagnostics(weights: pd.DataFrame, dates: pd.Index) -> dict:
    if weights.empty:
        out = {
            "signal_days_any": 0,
            "positive_weight_rows": 0,
            "negative_weight_rows": 0,
            "dates_with_both_long_short": 0,
            "avg_gross_exposure": 0.0,
            "avg_net_exposure": 0.0,
        }
        out.update(_side_diagnostics(weights, dates, "long"))
        out.update(_side_diagnostics(weights, dates, "short"))
        return out

    w = weights.copy()
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    by_day = w.groupby("date")["weight"]
    both = by_day.apply(lambda x: (x > 0).any() and (x < 0).any()).sum()

    out = {
        "signal_days_any": int(w.loc[w["weight"] != 0, "date"].nunique()),
        "positive_weight_rows": int((w["weight"] > 0).sum()),
        "negative_weight_rows": int((w["weight"] < 0).sum()),
        "dates_with_both_long_short": int(both),
        "avg_gross_exposure": float(by_day.apply(lambda x: x.abs().sum()).mean()),
        "avg_net_exposure": float(by_day.sum().mean()),
    }
    out.update(_side_diagnostics(w, dates, "long"))
    out.update(_side_diagnostics(w, dates, "short"))
    return out


def _stats_from_daily(x: pd.Series, prefix: str) -> dict:
    x = x.fillna(0.0).astype(float)
    if len(x) == 0:
        return {
            f"{prefix}_cum_ret%": np.nan,
            f"{prefix}_ann_ret%": np.nan,
            f"{prefix}_ann_vol%": np.nan,
            f"{prefix}_sharpe": np.nan,
            f"{prefix}_hit_rate%": np.nan,
        }

    nav = (1 + x).cumprod()
    ann_ret = nav.iloc[-1] ** (252 / len(x)) - 1
    ann_vol = x.std(ddof=1) * np.sqrt(252) if len(x) > 1 else np.nan
    sharpe = x.mean() / x.std(ddof=1) * np.sqrt(252) if len(x) > 1 and x.std(ddof=1) > 0 else np.nan

    return {
        f"{prefix}_cum_ret%": round((nav.iloc[-1] - 1) * 100, 2),
        f"{prefix}_ann_ret%": round(ann_ret * 100, 2),
        f"{prefix}_ann_vol%": round(ann_vol * 100, 2) if np.isfinite(ann_vol) else np.nan,
        f"{prefix}_sharpe": round(sharpe, 3) if np.isfinite(sharpe) else np.nan,
        f"{prefix}_hit_rate%": round((x > 0).mean() * 100, 1),
    }


def leg_daily_returns(result, borrow_bps_annual: float = 50) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Return long_daily, short_daily_gross, short_daily_after_borrow.

    Transaction costs are not allocated by leg.
    Total result.daily_returns already includes transaction cost.
    """
    weights = result.weights_long.copy()
    panel = result.panel_with_signal.copy()
    dates = result.daily_returns.index

    if weights.empty:
        z = pd.Series(0.0, index=dates)
        return z, z, z

    weights["date"] = pd.to_datetime(weights["date"]).dt.normalize()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    m = weights.merge(panel[["permno", "date", "ret"]], on=["permno", "date"], how="left")
    m["ret"] = pd.to_numeric(m["ret"], errors="coerce").fillna(0.0)
    m["contrib"] = m["weight"] * m["ret"]

    long_daily = m.loc[m["weight"] > 0].groupby("date")["contrib"].sum().reindex(dates).fillna(0.0)
    short_daily = m.loc[m["weight"] < 0].groupby("date")["contrib"].sum().reindex(dates).fillna(0.0)

    daily_rate = borrow_bps_annual / 1e4 / 252
    gross_short = weights[weights["weight"] < 0].groupby("date")["weight"].sum().abs().reindex(dates).fillna(0.0)
    short_after_borrow = short_daily - gross_short * daily_rate

    return long_daily, short_daily, short_after_borrow


def leg_return_attribution(result, borrow_bps_annual: float = 50) -> dict:
    long_daily, short_daily, short_after_borrow = leg_daily_returns(result, borrow_bps_annual)
    out = {}
    out.update(_stats_from_daily(long_daily, "long_gross"))
    out.update(_stats_from_daily(short_daily, "short_gross"))
    out.update(_stats_from_daily(short_after_borrow, "short_after_borrow"))
    return out


def summary_row(label: str, result, borrow_bps: float) -> dict:
    dr = adjust_for_borrow(result.daily_returns, result.weights_long, borrow_bps)
    perf = performance_metrics(dr, annualization=252)
    diag = position_diagnostics(result.weights_long, result.daily_returns.index)
    active_dates = result.weights_long["date"] if not result.weights_long.empty else pd.Series(dtype="datetime64[ns]")

    row = {
        "config": label,
        "first_active_date": active_dates.min() if not active_dates.empty else pd.NaT,
        "last_active_date": active_dates.max() if not active_dates.empty else pd.NaT,
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
    row.update(diag)
    row.update(leg_return_attribution(result, borrow_bps_annual=borrow_bps))
    return row


def save_outputs(results: dict, out_dir: Path, borrow_bps: float = 50) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_list, returns_list = [], []
    for label, res in results.items():
        w = res.weights_long.copy()
        w["config"] = label
        weights_list.append(w)

        dr = adjust_for_borrow(res.daily_returns, res.weights_long, borrow_bps)
        dr.name = label
        returns_list.append(dr)

    if weights_list:
        pd.concat(weights_list, ignore_index=True).to_parquet(out_dir / "positions.parquet", index=False)
    if returns_list:
        pd.concat(returns_list, axis=1).to_csv(out_dir / "daily_returns.csv")

    rows = [summary_row(lbl, r, borrow_bps) for lbl, r in results.items()]
    pd.DataFrame(rows).set_index("config").to_csv(out_dir / f"performance_{borrow_bps}bps.csv")


def plot_nav_drawdown(results: dict, labels: list[str], out_dir: Path, borrow_bps: float = 50) -> None:
    labels = [x for x in labels if x in results]
    if not labels:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for lbl in labels:
        r = results[lbl]
        dr = adjust_for_borrow(r.daily_returns, r.weights_long, borrow_bps)
        nav = (1 + dr).cumprod()
        nav.plot(ax=axes[0], label=lbl, alpha=0.9)
        dd = (nav.cummax() - nav) / nav.cummax()
        dd.plot(ax=axes[1], label=lbl, alpha=0.8)

    axes[0].set_title(f"NAV: Baseline vs ML Exit vs Hurst variants (borrow {borrow_bps} bps/yr, txcost 5 bps)")
    axes[0].set_ylabel("NAV")
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(loc="upper left", fontsize=7)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "01_nav_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_leg_cumulative_pnl(results: dict, labels: list[str], out_dir: Path, borrow_bps: float = 50) -> None:
    labels = [x for x in labels if x in results]
    if not labels:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for lbl in labels:
        long_d, short_d, short_b = leg_daily_returns(results[lbl], borrow_bps_annual=borrow_bps)
        (1 + long_d).cumprod().plot(ax=axes[0], label=lbl, alpha=0.85)
        (1 + short_d).cumprod().plot(ax=axes[1], label=lbl, alpha=0.85)
        (1 + short_b).cumprod().plot(ax=axes[2], label=lbl, alpha=0.85)

    axes[0].set_title("Long-leg cumulative PnL contribution")
    axes[1].set_title("Short-leg cumulative PnL contribution, gross")
    axes[2].set_title(f"Short-leg cumulative PnL contribution, after {borrow_bps} bps/yr borrow")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7)
        ax.set_ylabel("NAV")
    plt.tight_layout()
    plt.savefig(out_dir / "02_leg_cumulative_pnl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_active_positions(results: dict, labels: list[str], out_dir: Path) -> None:
    labels = [x for x in labels if x in results]
    if not labels:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for lbl in labels:
        w = results[lbl].weights_long.copy()
        if w.empty:
            continue
        w["date"] = pd.to_datetime(w["date"]).dt.normalize()
        long_count = w[w["weight"] > 0].groupby("date")["permno"].nunique().reindex(results[lbl].daily_returns.index).fillna(0.0)
        short_count = w[w["weight"] < 0].groupby("date")["permno"].nunique().reindex(results[lbl].daily_returns.index).fillna(0.0)
        long_count.rolling(20, min_periods=1).mean().plot(ax=axes[0], label=lbl, alpha=0.85)
        short_count.rolling(20, min_periods=1).mean().plot(ax=axes[1], label=lbl, alpha=0.85)

    axes[0].set_title("20-day rolling average active long positions")
    axes[1].set_title("20-day rolling average active short positions")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7)
        ax.set_ylabel("# names")
    plt.tight_layout()
    plt.savefig(out_dir / "03_active_positions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_short_trade_entries(result) -> pd.DataFrame:
    """
    Convert short position rows into baseline short trade entries.

    A trade is defined as one consecutive short-position spell for the same permno.
    """
    w = result.weights_long.copy()
    if w.empty:
        return pd.DataFrame()

    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    w = w.loc[w["weight"] < 0, ["permno", "date", "weight"]].copy()
    if w.empty:
        return pd.DataFrame()

    all_dates = pd.Index(pd.to_datetime(result.daily_returns.index).normalize()).sort_values()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    trades = []

    for permno, g in w.groupby("permno", sort=False):
        g = g.sort_values("date").copy()
        dates = list(g["date"].unique())
        if not dates:
            continue

        start = dates[0]
        prev = dates[0]

        for d in dates[1:]:
            prev_i = date_to_idx.get(prev)
            cur_i = date_to_idx.get(d)
            if prev_i is not None and cur_i is not None and cur_i == prev_i + 1:
                prev = d
            else:
                trades.append({"permno": permno, "entry_date": start, "exit_date": prev})
                start = d
                prev = d

        trades.append({"permno": permno, "entry_date": start, "exit_date": prev})

    trades = pd.DataFrame(trades)
    if trades.empty:
        return trades

    trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.normalize()
    trades["exit_date"] = pd.to_datetime(trades["exit_date"]).dt.normalize()
    trades["holding_days"] = trades.apply(
        lambda r: date_to_idx.get(r["exit_date"], np.nan) - date_to_idx.get(r["entry_date"], np.nan) + 1,
        axis=1,
    )
    return trades


def _stock_path_short_payoff(panel: pd.DataFrame, permno, start_date, horizon: int) -> float:
    """
    Forward horizon short payoff from the day after start_date using stock returns:
        short payoff = -(prod(1 + ret) - 1)
    """
    p = panel.loc[panel["permno"] == permno].sort_values("date")
    idx = p.index[p["date"] == start_date]
    if len(idx) == 0:
        return np.nan

    # Use row positions within this permno.
    loc = p.index.get_loc(idx[0])
    sub = p.iloc[loc + 1: loc + 1 + horizon]
    if len(sub) == 0:
        return np.nan

    rets = pd.to_numeric(sub["ret"], errors="coerce").fillna(0.0).values
    long_ret = np.prod(1.0 + rets) - 1.0
    return -float(long_ret)


def _realized_trade_short_payoff(panel: pd.DataFrame, permno, entry_date, exit_date) -> float:
    p = panel.loc[
        (panel["permno"] == permno)
        & (panel["date"] >= entry_date)
        & (panel["date"] <= exit_date)
    ].sort_values("date")
    if p.empty:
        return np.nan
    rets = pd.to_numeric(p["ret"], errors="coerce").fillna(0.0).values
    long_ret = np.prod(1.0 + rets) - 1.0
    return -float(long_ret)


def kept_vs_filtered_short_analysis(
    baseline_label: str,
    hurst_label: str,
    results: dict,
    out_dir: Path,
) -> None:
    """
    Trade-entry-level comparison between baseline shorts that Hurst kept vs filtered.

    This replaces the weaker row-level same-day payoff diagnostic.

    For each baseline short trade entry:
    - check whether Hurst strategy shorts the same permno on entry date or during the baseline trade window
    - compute realized baseline holding-period short payoff
    - compute forward 5-day and 10-day short payoff from entry date
    """
    if baseline_label not in results or hurst_label not in results:
        return

    base = results[baseline_label]
    hur = results[hurst_label]

    trades = _make_short_trade_entries(base)
    if trades.empty:
        return

    # Prepare panel with returns.
    panel = base.panel_with_signal[["permno", "date", "ret"]].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    # Hurst short rows for kept/filtered classification.
    hw = hur.weights_long.copy()
    if hw.empty:
        hshort = pd.DataFrame(columns=["permno", "date"])
    else:
        hw["date"] = pd.to_datetime(hw["date"]).dt.normalize()
        hshort = hw.loc[hw["weight"] < 0, ["permno", "date"]].drop_duplicates().copy()

    hshort_keys = set(zip(hshort["permno"], hshort["date"])) if not hshort.empty else set()

    kept_flags = []
    overlap_days = []
    realized_payoff = []
    fwd5 = []
    fwd10 = []

    for _, r in trades.iterrows():
        permno = r["permno"]
        entry = r["entry_date"]
        exit_ = r["exit_date"]

        trade_dates = panel.loc[
            (panel["permno"] == permno)
            & (panel["date"] >= entry)
            & (panel["date"] <= exit_),
            "date",
        ].tolist()

        overlaps = sum((permno, d) in hshort_keys for d in trade_dates)
        kept_flags.append(overlaps > 0)
        overlap_days.append(overlaps)

        realized_payoff.append(_realized_trade_short_payoff(panel, permno, entry, exit_))
        fwd5.append(_stock_path_short_payoff(panel, permno, entry, 5))
        fwd10.append(_stock_path_short_payoff(panel, permno, entry, 10))

    trades["kept_by_hurst"] = kept_flags
    trades["hurst_overlap_days"] = overlap_days
    trades["realized_trade_short_payoff"] = realized_payoff
    trades["fwd5_short_payoff"] = fwd5
    trades["fwd10_short_payoff"] = fwd10

    out_csv = out_dir / f"short_trade_kept_vs_filtered_{hurst_label}.csv"
    trades.to_csv(out_csv, index=False)

    def _summ(x: pd.Series) -> pd.Series:
        return pd.Series({
            "n_trades": len(x),
            "mean": x.mean(),
            "median": x.median(),
            "hit_rate": (x > 0).mean(),
            "p25": x.quantile(0.25),
            "p75": x.quantile(0.75),
        })

    summary_parts = []
    for col in ["realized_trade_short_payoff", "fwd5_short_payoff", "fwd10_short_payoff"]:
        tmp = trades.groupby("kept_by_hurst")[col].apply(_summ).unstack()
        tmp["metric"] = col
        summary_parts.append(tmp.reset_index())

    summary = pd.concat(summary_parts, ignore_index=True)
    summary.to_csv(out_dir / f"short_trade_kept_vs_filtered_summary_{hurst_label}.csv", index=False)

    # Plot trade-level distributions.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    metrics = [
        ("realized_trade_short_payoff", "Realized trade short payoff"),
        ("fwd5_short_payoff", "Forward 5D short payoff"),
        ("fwd10_short_payoff", "Forward 10D short payoff"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        data = [
            trades.loc[~trades["kept_by_hurst"], col].dropna().values,
            trades.loc[trades["kept_by_hurst"], col].dropna().values,
        ]
        ax.boxplot(data, labels=["Filtered", "Kept"], showfliers=False)
        ax.axhline(0, color="black", linewidth=1, alpha=0.5)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Baseline short trades: filtered vs kept by {hurst_label}", y=1.03)
    plt.tight_layout()
    plt.savefig(out_dir / "04_short_trade_kept_vs_filtered_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def hurst_bins_short_setup(
    baseline_label: str,
    result,
    hurst_col: str,
    out_dir: Path,
) -> None:
    """
    Bin baseline short positions by Hurst and compare realized same-day short payoff.
    """
    w = result.weights_long.copy()
    if w.empty or hurst_col not in result.panel_with_signal.columns:
        return

    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    s = w[w["weight"] < 0].copy()
    if s.empty:
        return

    panel_cols = ["permno", "date", "ret", hurst_col, "sentiment_z", "d4_put_dom", "d5_below_flip"]
    panel_cols = [c for c in panel_cols if c in result.panel_with_signal.columns]
    p = result.panel_with_signal[panel_cols].copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()

    m = s.merge(p, on=["permno", "date"], how="left")
    m["ret"] = pd.to_numeric(m["ret"], errors="coerce")
    m["short_payoff"] = -m["ret"]

    bins = [-np.inf, 0.45, 0.50, 0.55, np.inf]
    labels = ["<0.45", "0.45-0.50", "0.50-0.55", ">0.55"]
    m["hurst_bin"] = pd.cut(m[hurst_col], bins=bins, labels=labels)

    agg_dict = {
        "n": ("short_payoff", "size"),
        "avg_short_payoff": ("short_payoff", "mean"),
        "median_short_payoff": ("short_payoff", "median"),
        "hit_rate": ("short_payoff", lambda x: float((x > 0).mean())),
    }
    if "sentiment_z" in m.columns:
        agg_dict["avg_sentiment_z"] = ("sentiment_z", "mean")
    if "d4_put_dom" in m.columns:
        agg_dict["avg_d4_put_dom"] = ("d4_put_dom", "mean")
    if "d5_below_flip" in m.columns:
        agg_dict["avg_d5_below_flip"] = ("d5_below_flip", "mean")

    summary = m.groupby("hurst_bin", observed=False).agg(**agg_dict)
    summary.to_csv(out_dir / f"hurst_bins_{baseline_label}_{hurst_col}.csv")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    summary["avg_short_payoff"].plot(kind="bar", ax=ax1)
    ax1.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax1.set_title(f"Baseline short setup: same-day payoff by {hurst_col} bin")
    ax1.set_ylabel("Average equal-weight short payoff")
    ax1.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "05_hurst_bins_short_setup.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _select_drawdown_troughs(nav: pd.Series, n: int = 3, min_gap: int = 40) -> list[pd.Timestamp]:
    dd = (nav.cummax() - nav) / nav.cummax()
    candidates = dd.sort_values(ascending=False).index.tolist()
    selected: list[pd.Timestamp] = []
    date_index = pd.Index(nav.index)
    for dt in candidates:
        pos = date_index.get_loc(dt)
        if all(abs(pos - date_index.get_loc(x)) >= min_gap for x in selected):
            selected.append(dt)
        if len(selected) >= n:
            break
    return selected


def plot_drawdown_episodes(
    baseline_label: str,
    hurst_label: str,
    results: dict,
    out_dir: Path,
    borrow_bps: float = 50,
    hurst_col: str = "hurst_30",
    window: int = 30,
) -> None:
    if baseline_label not in results or hurst_label not in results:
        return

    base = results[baseline_label]
    hur = results[hurst_label]

    base_dr = adjust_for_borrow(base.daily_returns, base.weights_long, borrow_bps)
    hur_dr = adjust_for_borrow(hur.daily_returns, hur.weights_long, borrow_bps)
    base_nav = (1 + base_dr).cumprod()
    hur_nav = (1 + hur_dr.reindex(base_nav.index).fillna(0.0)).cumprod()

    troughs = _select_drawdown_troughs(base_nav, n=3, min_gap=40)
    dates = pd.Index(base_nav.index)

    for k, trough in enumerate(troughs, start=1):
        loc = dates.get_loc(trough)
        lo = max(0, loc - window)
        hi = min(len(dates), loc + window + 1)
        win_dates = dates[lo:hi]

        bnav_w = base_nav.reindex(win_dates)
        hnav_w = hur_nav.reindex(win_dates)
        if len(bnav_w.dropna()) == 0:
            continue
        bnav_norm = bnav_w / bnav_w.iloc[0]
        hnav_norm = hnav_w / hnav_w.iloc[0]

        _, bshort_gross, _ = leg_daily_returns(base, borrow_bps_annual=borrow_bps)
        _, hshort_gross, _ = leg_daily_returns(hur, borrow_bps_annual=borrow_bps)

        def short_count(res):
            w = res.weights_long.copy()
            if w.empty:
                return pd.Series(0.0, index=win_dates)
            w["date"] = pd.to_datetime(w["date"]).dt.normalize()
            return w[w["weight"] < 0].groupby("date")["permno"].nunique().reindex(win_dates).fillna(0.0)

        def avg_hurst_short(res):
            if hurst_col not in res.panel_with_signal.columns:
                return pd.Series(np.nan, index=win_dates)
            w = res.weights_long.copy()
            if w.empty:
                return pd.Series(np.nan, index=win_dates)
            w["date"] = pd.to_datetime(w["date"]).dt.normalize()
            sw = w[w["weight"] < 0][["permno", "date"]].copy()
            p = res.panel_with_signal[["permno", "date", hurst_col]].copy()
            p["date"] = pd.to_datetime(p["date"]).dt.normalize()
            m = sw.merge(p, on=["permno", "date"], how="left")
            return m.groupby("date")[hurst_col].mean().reindex(win_dates)

        fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

        bnav_norm.plot(ax=axes[0], label=baseline_label)
        hnav_norm.plot(ax=axes[0], label=hurst_label)
        axes[0].axvline(trough, color="red", linestyle="--", alpha=0.6)
        axes[0].set_title(f"Drawdown episode {k}: NAV normalized around baseline trough {pd.Timestamp(trough).date()}")
        axes[0].set_ylabel("Norm NAV")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        bshort_gross.reindex(win_dates).fillna(0.0).plot(ax=axes[1], label=f"{baseline_label} short PnL")
        hshort_gross.reindex(win_dates).fillna(0.0).plot(ax=axes[1], label=f"{hurst_label} short PnL")
        axes[1].axhline(0, color="black", linewidth=1, alpha=0.4)
        axes[1].set_title("Daily short-leg gross PnL contribution")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        short_count(base).plot(ax=axes[2], label=f"{baseline_label} #shorts")
        short_count(hur).plot(ax=axes[2], label=f"{hurst_label} #shorts")
        axes[2].set_title("Active short positions")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

        avg_hurst_short(base).plot(ax=axes[3], label=f"{baseline_label} avg {hurst_col}")
        avg_hurst_short(hur).plot(ax=axes[3], label=f"{hurst_label} avg {hurst_col}")
        axes[3].axhline(0.55, color="red", linestyle="--", linewidth=1, alpha=0.5, label="H=0.55")
        axes[3].set_title("Average Hurst among active short positions")
        axes[3].legend(fontsize=8)
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / f"06_drawdown_episode_{k}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def robustness_tables(results: dict, out_dir: Path, borrow_bps: float = 50) -> None:
    rows = [summary_row(lbl, r, borrow_bps) for lbl, r in results.items()]
    df = pd.DataFrame(rows).set_index("config")
    focus = df.loc[[x for x in df.index if ("H30" in x or x == "v2-A-beta-5d-LS")]].copy()
    focus.to_csv(out_dir / "07_robustness_short_long_both.csv")


def subperiod_robustness(results: dict, labels: list[str], out_dir: Path, borrow_bps: float = 50) -> None:
    periods = [
        ("2018-2020", "2018-06-01", "2020-12-31"),
        ("2021-2022", "2021-01-01", "2022-12-31"),
        ("2023-2024", "2023-01-01", "2024-12-31"),
    ]

    rows = []
    for lbl in labels:
        if lbl not in results:
            continue
        r = results[lbl]
        dr = adjust_for_borrow(r.daily_returns, r.weights_long, borrow_bps)
        dr.index = pd.to_datetime(dr.index)
        for name, start, end in periods:
            x = dr.loc[(dr.index >= pd.Timestamp(start)) & (dr.index <= pd.Timestamp(end))]
            if len(x) < 20:
                continue
            p = performance_metrics(x, annualization=252)
            rows.append({
                "config": lbl,
                "period": name,
                "n_days": p.n_days,
                "sharpe": p.sharpe_ratio,
                "ann_ret%": p.annualized_return * 100,
                "ann_vol%": p.annualized_volatility * 100,
                "max_dd%": p.max_drawdown * 100,
                "hit_rate%": (x > 0).mean() * 100,
            })
    pd.DataFrame(rows).to_csv(out_dir / "08_subperiod_robustness_50bps.csv", index=False)


def make_all_diagnostic_outputs(
    results: dict,
    out_dir: Path,
    borrow_bps: float = 50,
    baseline_label: str = "v2-A-beta-5d-LS",
    main_hurst_label: str = "v2-A-beta-5d-LS-H30-filter-short",
    hurst_col: str = "hurst_30",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    save_outputs(results, out_dir, borrow_bps=borrow_bps)

    # Main plot labels are pre-specified diagnostics, not automatically best performers.
    # Include H10/H30/H60 short filters and ML/LGBM exit variants to avoid cherry-picking.
    key_labels = [
        baseline_label,
        "v2-A-beta-5d-LS-MLExit",
        "v2-A-beta-5d-LS-LGBMExit",
        "v2-A-beta-5d-LS-H10-filter-short",
        "v2-A-beta-5d-LS-H30-filter-short",
        "v2-A-beta-5d-LS-H60-filter-short",
        "v2-A-beta-5d-LS-H10-filter-short-MLExit",
        "v2-A-beta-5d-LS-H30-filter-short-MLExit",
        "v2-A-beta-5d-LS-H60-filter-short-MLExit",
        "v2-A-beta-5d-LS-H10-filter-short-LGBMExit",
        "v2-A-beta-5d-LS-H30-filter-short-LGBMExit",
        "v2-A-beta-5d-LS-H60-filter-short-LGBMExit",
        "v2-A-beta-5d-LS-H30-filter-long",
        "v2-A-beta-5d-LS-H30-filter-both",
    ]
    key_labels.extend([label for label in results if label.endswith("LGBMExit")])
    key_labels = [x for x in dict.fromkeys(key_labels) if x in results]

    plot_nav_drawdown(results, key_labels, out_dir, borrow_bps=borrow_bps)
    plot_leg_cumulative_pnl(results, key_labels, out_dir, borrow_bps=borrow_bps)
    plot_active_positions(results, key_labels, out_dir)

    # Trade-level, forward-return diagnostics.
    kept_vs_filtered_short_analysis(baseline_label, main_hurst_label, results, out_dir)

    if baseline_label in results:
        hurst_bins_short_setup(baseline_label, results[baseline_label], hurst_col, out_dir)

    plot_drawdown_episodes(baseline_label, main_hurst_label, results, out_dir, borrow_bps=borrow_bps, hurst_col=hurst_col)
    robustness_tables(results, out_dir, borrow_bps=borrow_bps)
    subperiod_robustness(results, key_labels, out_dir, borrow_bps=borrow_bps)
