'''
Author: Xiwei Wang; Date: 2026-04-20
YF Modification: 2026-04-21
'''
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """
    Settings for loading the panel and computing portfolio returns.

    ``missing_return_policy``:
      - ``zero_contribution``: NaN ``return_col`` counts as 0 P&L for that name-date.
      - ``drop_name``: exclude that name from the day's sum; renormalize weights among
        names with valid returns to preserve the day's gross exposure.

    ``risk_free_annual``: nominal annual risk-free rate. Default 0 gives daily carry 0
    (``(1+r)^{1/252}-1``), so Sharpe/Sortino excess returns equal raw daily returns.

    **Long / short:** P&L is ``sum(weight * ret)`` per day. Negative ``weight`` is a short.
    With ``long_only=False`` and ``short_only=False`` (default), strategies may mix longs
    and shorts (e.g. dollar-neutral). ``long_only=True`` zeros shorts; ``short_only=True``
    zeros longs for a pure short book (incompatible with ``long_only``).
    """
    panel_path: Union[str, Path]
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    return_col: str = "ret"
    signal_col: str = "signal"
    signal_lag_days: int = 1
    annualization: int = 252
    risk_free_annual: float = 0.0
    min_observations: int = 20
    assert_unique_permno_date: bool = True
    long_only: bool = False
    short_only: bool = False
    """If True, weights are clipped to <= 0 (long legs removed). Mutually exclusive with ``long_only``."""
    gross_exposure_cap: Optional[float] = None
    transaction_cost_bps: float = 0.0
    missing_return_policy: str = "zero_contribution"
    verify_signal_lag: bool = False
    """If True, assert post-lag ``signal_col`` matches ``groupby(permno).shift``."""
    store_panel_pre_signal_lag: bool = False
    """If True, ``BacktestResult.panel_pre_signal_lag`` holds the frame right after ``compute``."""


# ---------------------------------------------------------------------------
# Types for optional hooks
# ---------------------------------------------------------------------------


UniverseFilter = Callable[[pd.DataFrame], pd.DataFrame]


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass
class PerformanceReport:
    """
    ``risk_free_annual`` / ``risk_free_daily`` echo the config: daily rate is
    ``(1 + annual)^(1/252) - 1`` with ``annualization`` from ``BacktestConfig``.
    When both are 0, Sharpe and Sortino are based on raw daily returns (excess = return).

    ``win_rate`` is the fraction of days with strictly positive *excess* return
    (return minus ``risk_free_daily``), same basis as Sharpe.
    """

    n_days: int
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    risk_free_annual: float
    risk_free_daily: float
    win_rate: float

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame([self.__dict__])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    nav: pd.Series
    weights_long: pd.DataFrame
    panel_with_signal: pd.DataFrame
    performance: PerformanceReport
    benchmark_daily_returns: Optional[pd.Series] = None
    benchmark_nav: Optional[pd.Series] = None
    benchmark_performance: Optional[PerformanceReport] = None
    panel_pre_signal_lag: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Panel I/O
# ---------------------------------------------------------------------------


def load_panel(path: Union[str, Path], *, assert_unique_permno_date: bool = True) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path)
    if "date" not in df.columns or "permno" not in df.columns:
        raise ValueError("Panel must contain columns 'permno' and 'date'.")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    df = df.loc[df["permno"].notna() & df["date"].notna()].sort_values(["permno", "date"])
    df = df.reset_index(drop=True)
    if assert_unique_permno_date:
        dup = df.duplicated(subset=["permno", "date"], keep=False)
        if dup.any():
            raise ValueError(
                f"Duplicate (permno, date) rows: {int(dup.sum())} rows. "
                "Panel must be unique on (permno, date)."
            )
    return df


def slice_panel(
    panel: pd.DataFrame,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    out = panel.copy()
    if start is not None:
        start = pd.Timestamp(start).normalize()
        out = out.loc[out["date"] >= start]
    if end is not None:
        end = pd.Timestamp(end).normalize()
        out = out.loc[out["date"] <= end]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Signal / Strategy (user extends)
# ---------------------------------------------------------------------------


class BaseSignal(ABC):
    """
    Compute cross-sectional or time-series features on the panel.

    Implement ``compute`` to add columns (e.g. ``raw_signal``). Avoid lookahead:
    use only information available at or before each row's ``date``. The engine
    applies ``signal_lag_days`` shift on ``signal_col`` before mapping to returns.
    """

    @abstractmethod
    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``panel`` with new columns; preserve (permno, date)."""


class BaseStrategy(ABC):
    """
    Map lagged signal column(s) to target portfolio weights.

    Return a long DataFrame with columns ``permno``, ``date``, ``weight``.
    Weights are dollar weights for that calendar date's ``return_col`` in the
    panel (after signal lag is applied in the engine). Use **positive** weights for
    longs and **negative** for shorts; daily P&L is ``sum(weight * return)``. Typical
    patterns: long-only weights sum to 1; dollar-neutral long–short sum to ~0; short-only
    weights are non-positive and may sum to -1. ``BacktestConfig.long_only`` / ``short_only``
    can enforce sign constraints after your strategy runs.

    """

    @abstractmethod
    def target_weights(self, panel: pd.DataFrame) -> pd.DataFrame:
        ...


class ExampleSignal(BaseSignal):
    """Stub: sets ``signal`` to 0. Replace with your feature engineering."""

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        out = panel.copy()
        out["signal"] = 0.0
        return out


class ExampleStrategy(BaseStrategy):
    """Stub: zero weights everywhere. Replace with your allocation rule."""

    def target_weights(self, panel: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "permno": pd.Series(dtype="Int64"),
                "date": pd.Series(dtype="datetime64[ns]"),
                "weight": pd.Series(dtype="float64"),
            }
        )


class BaseEntryExitRule(ABC):
    """
    Extension point for min holding period, stops, etc.

    Override ``apply`` to modify ``weights`` (long format) before P&L.
    Default implementation is identity.
    """

    def apply(self, weights: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
        return weights


# ---------------------------------------------------------------------------
# Signal lag
# ---------------------------------------------------------------------------


def apply_signal_lag(
    panel: pd.DataFrame,
    signal_col: str,
    lag_days: int,
) -> pd.DataFrame:
    """
    Within each permno, replace ``signal_col`` with ``.shift(lag_days)`` along the
    trading calendar. So on date *t* the value is the one that appeared on the row
    *lag_days* steps earlier — i.e. an older observation is what drives positions on *t*.

    With ``lag_days == 1``, weights on *t* are driven by the pre-lag signal from the
    prior trading day, multiplied by ``ret`` on *t* (close-to-close). ``lag_days == 0``
    uses same-row signal (only valid if that signal is known before the return window).
    """
    if lag_days < 0:
        raise ValueError("signal_lag_days must be non-negative.")
    out = panel.copy()
    if signal_col not in out.columns:
        raise KeyError(f"Missing signal column '{signal_col}' after Signal.compute.")
    out[signal_col] = out.groupby("permno", sort=False)[signal_col].shift(lag_days)
    return out


def assert_signal_lag_consistent(
    panel_after_compute: pd.DataFrame,
    panel_after_lag: pd.DataFrame,
    *,
    signal_col: str,
    lag_days: int,
) -> None:
    """
    Verify ``panel_after_lag[signal_col]`` equals ``groupby(permno)[signal_col].shift(lag_days)``
    on the pre-lag frame. Same row count and order as produced by ``apply_signal_lag``.
    """
    if len(panel_after_compute) != len(panel_after_lag):
        raise AssertionError("Panel row counts differ before/after lag.")
    expected = panel_after_compute.groupby("permno", sort=False)[signal_col].shift(lag_days)
    actual = panel_after_lag[signal_col]
    try:
        pd.testing.assert_series_equal(
            expected.reset_index(drop=True),
            actual.reset_index(drop=True),
            check_names=False,
            atol=1e-9,
            rtol=0.0,
        )
    except AssertionError as e:
        raise AssertionError(f"Signal lag column '{signal_col}' does not match shift({lag_days}).") from e


# ---------------------------------------------------------------------------
# Weight normalization helpers
# ---------------------------------------------------------------------------


def _scale_weights_to_gross(w: np.ndarray, cap: float) -> np.ndarray:
    s = np.nansum(np.abs(w))
    if s <= 0 or not np.isfinite(s):
        return w
    if s > cap:
        w = w * (cap / s)
    return w


def _apply_long_only(w: np.ndarray) -> np.ndarray:
    w = np.where(np.isfinite(w) & (w < 0), 0.0, w)
    return w


def _apply_short_only(w: np.ndarray) -> np.ndarray:
    w = np.where(np.isfinite(w) & (w > 0), 0.0, w)
    return w


# ---------------------------------------------------------------------------
# Portfolio returns
# ---------------------------------------------------------------------------


def compute_portfolio_returns(
    panel: pd.DataFrame,
    weights_long: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    For each date, sum weight * return_col. Returns (daily_returns Series indexed by date,
    weights_long possibly reindexed for drop_name policy — caller uses original weights for turnover).
    """
    ret_col = config.return_col
    if ret_col not in panel.columns:
        raise KeyError(f"Return column '{ret_col}' not in panel.")

    m = weights_long.merge(
        panel[["permno", "date", ret_col]],
        on=["permno", "date"],
        how="left",
    )
    m = m.rename(columns={ret_col: "_ret"})

    if config.missing_return_policy == "zero_contribution":
        r = m["_ret"].to_numpy(dtype=float)
        r = np.where(np.isfinite(r), r, 0.0)
        m["_contrib"] = m["weight"].to_numpy(dtype=float) * r
        daily = m.groupby("date", sort=True)["_contrib"].sum()
    elif config.missing_return_policy == "drop_name":

        def _agg_day(g: pd.DataFrame) -> float:
            w = g["weight"].to_numpy(dtype=float)
            r = g["_ret"].to_numpy(dtype=float)
            valid = np.isfinite(r) & np.isfinite(w)
            if not valid.any():
                return 0.0
            w_all_abs = np.nansum(np.abs(w))
            w = w[valid]
            r = r[valid]
            s = np.sum(np.abs(w))
            if s <= 0:
                return 0.0
            w = w / s * w_all_abs
            return float(np.sum(w * r))

        daily = m.groupby("date", sort=True, group_keys=False).apply(_agg_day)
    else:
        raise ValueError(f"Unknown missing_return_policy: {config.missing_return_policy}")

    daily = daily.astype(float)
    daily.name = "portfolio_return"
    return daily, m.drop(columns=["_ret", "_contrib"], errors="ignore")


def apply_transaction_costs(
    weights_long: pd.DataFrame,
    daily_returns: pd.Series,
    cost_bps: float,
) -> pd.Series:
    """Subtract turnover * (cost_bps / 1e4) from each day's return."""
    if cost_bps == 0:
        return daily_returns
    w = weights_long.sort_values(["permno", "date"])
    prev = w.groupby("permno", sort=False)["weight"].shift(1).fillna(0.0)
    w = w.assign(_prev=prev)
    w["_turn_row"] = np.abs(w["weight"].to_numpy(dtype=float) - w["_prev"].to_numpy(dtype=float))
    turn = w.groupby("date", sort=True)["_turn_row"].sum()
    turn = turn.reindex(daily_returns.index).fillna(0.0)
    cost = turn * (cost_bps / 10_000.0)
    return daily_returns - cost


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def _daily_rf(annual_rf: float, ann: int) -> float:
    return (1.0 + float(annual_rf)) ** (1.0 / ann) - 1.0


def performance_metrics(
    daily_returns: pd.Series,
    *,
    annualization: int = 252,
    risk_free_annual: float = 0.0,
    min_observations: int = 20,
) -> PerformanceReport:
    """
    Excess return per day is ``r - rf_daily`` with
    ``rf_daily = (1 + risk_free_annual) ** (1/annualization) - 1``.
    """
    r = daily_returns.dropna().astype(float)
    n = len(r)
    rf_d = _daily_rf(risk_free_annual, annualization)
    if n < min_observations:
        return PerformanceReport(
            n_days=n,
            cumulative_return=float("nan"),
            annualized_return=float("nan"),
            annualized_volatility=float("nan"),
            sharpe_ratio=float("nan"),
            sortino_ratio=float("nan"),
            max_drawdown=float("nan"),
            risk_free_annual=risk_free_annual,
            risk_free_daily=rf_d,
            win_rate=float("nan"),
        )

    excess = r - rf_d
    nav = (1.0 + r).cumprod()
    cum_ret = float(nav.iloc[-1] - 1.0)
    ann_ret = float((1.0 + cum_ret) ** (annualization / n) - 1.0) if n > 0 else float("nan")
    vol = float(r.std(ddof=1) * np.sqrt(annualization)) if n > 1 else float("nan")
    ex_mean = float(excess.mean())
    ex_std = float(excess.std(ddof=1))
    sharpe = float(np.sqrt(annualization) * ex_mean / ex_std) if ex_std > 0 else float("nan")

    downside = excess[excess < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    sortino = float(np.sqrt(annualization) * ex_mean / dstd) if dstd and dstd > 0 else float("nan")

    peak = nav.cummax()
    dd = (peak - nav) / peak
    max_dd = float(dd.max())
    win_rate = float((excess > 0).mean())

    return PerformanceReport(
        n_days=n,
        cumulative_return=cum_ret,
        annualized_return=ann_ret,
        annualized_volatility=vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        risk_free_annual=risk_free_annual,
        risk_free_daily=rf_d,
        win_rate=win_rate,
    )


# ---------------------------------------------------------------------------
# Post-process weights — applied before P&L
# ---------------------------------------------------------------------------


def finalize_weights(
    weights_long: pd.DataFrame,
    config: BacktestConfig,
    dates_present: pd.Index,
) -> pd.DataFrame:
    if weights_long.empty:
        return weights_long

    wdf = weights_long.copy()
    wdf["weight"] = pd.to_numeric(wdf["weight"], errors="coerce").fillna(0.0)

    parts = []
    for dt, g in wdf.groupby("date", sort=True):
        if dt not in dates_present:
            continue
        w = g["weight"].to_numpy(dtype=float)
        if config.long_only:
            w = _apply_long_only(w)
        elif config.short_only:
            w = _apply_short_only(w)
        if config.gross_exposure_cap is not None:
            w = _scale_weights_to_gross(w, config.gross_exposure_cap)
        g = g.copy()
        g["weight"] = w
        parts.append(g)
    if not parts:
        return pd.DataFrame(columns=["permno", "date", "weight"])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_backtest(
    config: BacktestConfig,
    signal: BaseSignal,
    strategy: BaseStrategy,
    *,
    universe_filter: Optional[UniverseFilter] = None,
    entry_exit: Optional[BaseEntryExitRule] = None,
    benchmark_daily_returns: Optional[pd.Series] = None,
) -> BacktestResult:
    if config.long_only and config.short_only:
        raise ValueError("BacktestConfig.long_only and short_only cannot both be True.")

    panel = load_panel(
        config.panel_path,
        assert_unique_permno_date=config.assert_unique_permno_date,
    )
    panel = slice_panel(panel, config.start_date, config.end_date)

    if universe_filter is not None:
        panel = universe_filter(panel)

    if config.return_col not in panel.columns:
        raise KeyError(f"Return column '{config.return_col}' missing from panel.")

    panel_s = signal.compute(panel)
    panel_lag = apply_signal_lag(panel_s, config.signal_col, config.signal_lag_days)
    if config.verify_signal_lag:
        assert_signal_lag_consistent(
            panel_s,
            panel_lag,
            signal_col=config.signal_col,
            lag_days=config.signal_lag_days,
        )
    weights = strategy.target_weights(panel_lag)

    required_w = {"permno", "date", "weight"}
    if not required_w.issubset(weights.columns):
        raise ValueError(f"target_weights must return columns: {required_w}")

    weights = weights[list(required_w)].copy()
    weights["permno"] = pd.to_numeric(weights["permno"], errors="coerce").astype("Int64")
    weights["date"] = pd.to_datetime(weights["date"], errors="coerce").dt.normalize()

    if entry_exit is not None:
        weights = entry_exit.apply(weights, panel_lag)

    dates = pd.Index(sorted(panel_lag["date"].unique()))
    weights = finalize_weights(weights, config, dates)

    daily, _ = compute_portfolio_returns(panel_lag, weights, config)
    daily = apply_transaction_costs(weights, daily, config.transaction_cost_bps)
    daily = daily.reindex(dates).fillna(0.0)

    nav = (1.0 + daily).cumprod()
    perf = performance_metrics(
        daily,
        annualization=config.annualization,
        risk_free_annual=config.risk_free_annual,
        min_observations=config.min_observations,
    )

    bench_nav = None
    bench_perf = None
    bret = None
    if benchmark_daily_returns is not None:
        bret = benchmark_daily_returns.reindex(dates).fillna(0.0)
        bench_nav = (1.0 + bret).cumprod()
        bench_perf = performance_metrics(
            bret,
            annualization=config.annualization,
            risk_free_annual=config.risk_free_annual,
            min_observations=config.min_observations,
        )

    pre_lag = panel_s if config.store_panel_pre_signal_lag else None
    return BacktestResult(
        daily_returns=daily,
        nav=nav,
        weights_long=weights,
        panel_with_signal=panel_lag,
        performance=perf,
        benchmark_daily_returns=bret,
        benchmark_nav=bench_nav,
        benchmark_performance=bench_perf,
        panel_pre_signal_lag=pre_lag,
    )


# ---------------------------------------------------------------------------
# Visualization (matplotlib)
# ---------------------------------------------------------------------------


def plot_nav(
    result: BacktestResult,
    *,
    ax: Any = None,
    show_benchmark: bool = True,
    title: str = "Net asset value (1 = start)",
) -> Any:
    if plt is None:
        raise ImportError("matplotlib is required for plot_nav.")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    result.nav.plot(ax=ax, label="Strategy")
    if show_benchmark and result.benchmark_nav is not None:
        result.benchmark_nav.plot(ax=ax, label="Benchmark", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_cumulative_return(
    result: BacktestResult,
    *,
    ax: Any = None,
    show_benchmark: bool = True,
    title: str = "Cumulative simple return",
) -> Any:
    if plt is None:
        raise ImportError("matplotlib is required for plot_cumulative_return.")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    cum_s = result.nav - 1.0
    cum_s.plot(ax=ax, label="Strategy")
    if show_benchmark and result.benchmark_nav is not None:
        (result.benchmark_nav - 1.0).plot(ax=ax, label="Benchmark", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_drawdown(
    result: BacktestResult,
    *,
    ax: Any = None,
    title: str = "Drawdown",
) -> Any:
    if plt is None:
        raise ImportError("matplotlib is required for plot_drawdown.")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    nav = result.nav
    peak = nav.cummax()
    dd = (peak - nav) / peak
    dd.plot(ax=ax, color="darkred")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    return ax


def plot_equity_overview(result: BacktestResult) -> Any:
    if plt is None:
        raise ImportError("matplotlib is required for plot_equity_overview.")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    plot_nav(result, ax=axes[0])
    plot_cumulative_return(result, ax=axes[1])
    plot_drawdown(result, ax=axes[2])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Benchmark helper: equal-weight all names each day from panel
# ---------------------------------------------------------------------------


def equal_weight_daily_returns(
    panel: pd.DataFrame,
    return_col: str = "ret",
) -> pd.Series:
    """Cross-sectional equal-weight return by date (ignores NaN names)."""
    r = pd.to_numeric(panel[return_col], errors="coerce")
    return (
        panel.assign(_r=r)
        .groupby("date", sort=True)["_r"]
        .mean()
        .fillna(0.0)
        .astype(float)
    )
