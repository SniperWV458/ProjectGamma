
"""
Hurst extension for the ML version of the GEX-Fragility x Sentiment backtest.

Design:
- Keep teammate's original FragilityGEXSignal unchanged.
- Append lagged Hurst columns only.
- Keep original long-short strategy unchanged unless a Hurst overlay is explicitly requested.

Why this matters:
Baseline must remain the original strategy. Hurst variants must be incremental overlays,
otherwise NAV / Sharpe comparisons are not apples-to-apples.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_hurst_ml.backtest_framework import BaseSignal, BaseStrategy
from backtest_hurst_ml.sentiment_gex_strategy import FragilityGEXSignal, _lifecycle_position


def rolling_hurst_rs(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling Hurst exponent using classical rescaled-range R/S on log returns.

    The first valid value appears after `window` price observations. The signal
    class shifts Hurst by one day, matching the original factor convention.
    """
    prices = np.asarray(prices, dtype=float)
    out = np.full(len(prices), np.nan)

    valid = np.isfinite(prices) & (prices > 0)
    if valid.sum() < window:
        return out

    start = None
    for i, ok in enumerate(valid):
        if ok and start is None:
            start = i

        if ((not ok) or i == len(valid) - 1) and start is not None:
            end = i if not ok else i + 1
            arr = prices[start:end]

            if len(arr) >= window:
                logret = np.diff(np.log(arr))
                m = window - 1
                if m < 5 or len(logret) < m:
                    start = None
                    continue

                windows = np.lib.stride_tricks.sliding_window_view(logret, m)
                mean = windows.mean(axis=1, keepdims=True)
                dev = windows - mean
                cumdev = dev.cumsum(axis=1)

                r = cumdev.max(axis=1) - cumdev.min(axis=1)
                s = windows.std(axis=1, ddof=1)
                rs = np.where(s > 0, r / s, np.nan)

                h = np.log(rs) / np.log(m)
                out[start + window - 1: start + window - 1 + len(h)] = h

            start = None

    return out


class FragilityGEXWithHurstSignal(BaseSignal):
    """
    Original FragilityGEXSignal + lagged Hurst columns.

    This class calls the teammate's original FragilityGEXSignal.compute(panel)
    first, then appends Hurst columns. It does not reimplement D1-D6 or gates.

    Added columns, by default:
        hurst_10, hurst_30, hurst_60

    Use with BacktestConfig(signal_lag_days=0), same as the original strategy.
    """

    def __init__(
        self,
        contract_factors_path: Path | None = None,
        sentiment_col: str = "sent_avg",
        price_col: str = "price_abs",
        regime_lookback_days: int = 5,
        hurst_windows: Iterable[int] = (10, 30, 60),
    ):
        if contract_factors_path is None:
            self.base_signal = FragilityGEXSignal(
                sentiment_col=sentiment_col,
                regime_lookback_days=regime_lookback_days,
            )
        else:
            self.base_signal = FragilityGEXSignal(
                contract_factors_path=Path(contract_factors_path),
                sentiment_col=sentiment_col,
                regime_lookback_days=regime_lookback_days,
            )
        self.price_col = price_col
        self.hurst_windows = tuple(int(w) for w in hurst_windows)

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        out = self.base_signal.compute(panel.copy())

        if self.price_col not in out.columns:
            raise KeyError(
                f"Missing '{self.price_col}'. Rebuild slim panel with price_abs included "
                "or use a panel containing price_abs."
            )

        h = out[["permno", "date", self.price_col]].copy()
        h["date"] = pd.to_datetime(h["date"]).dt.normalize()
        h = h.sort_values(["permno", "date"]).reset_index(drop=True)

        h_cols = []
        for w in self.hurst_windows:
            col = f"hurst_{w}"
            h[col] = np.nan
            h_cols.append(col)

        for _, idx in h.groupby("permno", sort=False).indices.items():
            prices = h.iloc[idx][self.price_col].to_numpy(dtype=float)
            for w in self.hurst_windows:
                h.loc[h.index[idx], f"hurst_{w}"] = rolling_hurst_rs(prices, w)

        # Pre-lag exactly once, consistent with original FragilityGEXSignal.
        for col in h_cols:
            h[col] = h.groupby("permno", sort=False)[col].shift(1)

        drop_existing = [c for c in h_cols if c in out.columns]
        if drop_existing:
            out = out.drop(columns=drop_existing)

        out = out.merge(h[["permno", "date"] + h_cols], on=["permno", "date"], how="left")
        return out


class GEXSentimentHurstStrategy(BaseStrategy):
    """
    Original GEX x Sentiment long-short construction + optional Hurst overlay.

    Original logic preserved:
    - short leg = fragile regime + optimistic/crowded sentiment
    - long leg  = stable regime + positive sentiment momentum
    - lifecycle/time-stop logic preserved

    Hurst modes:
    - none   : original logic
    - filter : require Hurst > threshold on the selected leg(s)
    - score  : keep candidates unchanged, but boost ranking when Hurst is high

    Recommended research design:
    - apply_hurst_to_shorts=True, apply_hurst_to_longs=False
      because the thesis is about persistent downside amplification inside fragile regimes.
    - long-only and both-side Hurst variants are placebo / robustness checks.
    """

    def __init__(
        self,
        variant: Literal["A", "B"] = "A",
        gate: Literal["simple", "full", "v2", "v3"] = "v2",
        neutrality: Literal["dollar", "beta"] = "beta",
        n_per_leg: int = 5,
        time_stop_days: int = 5,
        sentiment_threshold: float = 0.5,
        long_sentiment_threshold: float = 0.3,
        long_leg: Literal["momentum", "none"] = "momentum",
        hurst_mode: Literal["none", "filter", "score"] = "none",
        hurst_col: str = "hurst_30",
        hurst_threshold: float = 0.55,
        hurst_strength: float = 1.0,
        apply_hurst_to_shorts: bool = True,
        apply_hurst_to_longs: bool = False,
    ):
        self.variant = variant
        gate_map = {
            "simple": "fragility_gate_simple",
            "full": "fragility_gate_full",
            "v2": "fragility_gate_v2",
            "v3": "fragility_gate_v3",
        }
        self.gate_col = gate_map.get(gate, "fragility_gate_simple")
        self.neutrality = neutrality
        self.n_per_leg = int(n_per_leg)
        self.time_stop_days = int(time_stop_days)
        self.sent_thr = float(sentiment_threshold)
        self.long_thr = float(long_sentiment_threshold)
        self.long_leg = long_leg
        self.hurst_mode = hurst_mode
        self.hurst_col = hurst_col
        self.hurst_threshold = float(hurst_threshold)
        self.hurst_strength = float(hurst_strength)
        self.apply_hurst_to_shorts = bool(apply_hurst_to_shorts)
        self.apply_hurst_to_longs = bool(apply_hurst_to_longs)

    def target_weights(self, panel: pd.DataFrame) -> pd.DataFrame:
        if self.gate_col not in panel.columns:
            raise KeyError(f"Missing gate column: {self.gate_col}")
        if "sentiment_z" not in panel.columns:
            raise KeyError("Missing sentiment_z. Use FragilityGEXWithHurstSignal.")
        if self.hurst_mode != "none" and self.hurst_col not in panel.columns:
            raise KeyError(f"Missing Hurst column: {self.hurst_col}")

        all_rows: list[dict] = []

        for permno, grp in panel.groupby("permno", sort=False):
            grp = grp.sort_values("date").copy().reset_index(drop=True)

            gate = grp[self.gate_col].fillna(0.0).astype(bool).values
            sz = grp["sentiment_z"].fillna(np.nan).values
            hz = (
                grp[self.hurst_col].to_numpy(dtype=float)
                if self.hurst_col in grp.columns
                else np.full(len(grp), np.nan)
            )
            h_ok = np.where(np.isfinite(hz), hz > self.hurst_threshold, False)

            # Short candidates: original setup.
            if self.variant == "A":
                short_cand = gate & np.where(np.isfinite(sz), sz > self.sent_thr, False)
            else:
                short_cand = gate & np.isfinite(sz)

            if self.hurst_mode == "filter" and self.apply_hurst_to_shorts:
                short_cand = short_cand & h_ok

            short_score = np.where(np.isfinite(sz), sz, -np.inf)
            if self.hurst_mode == "score" and self.apply_hurst_to_shorts:
                boost = 1.0 + self.hurst_strength * np.clip(
                    np.where(np.isfinite(hz), hz - self.hurst_threshold, 0.0),
                    0.0,
                    None,
                )
                short_score = short_score * boost

            # Long candidates: original setup.
            if self.long_leg == "none":
                long_cand = np.zeros(len(grp), dtype=bool)
            else:
                has_flip = (
                    grp["gamma_flip_level"].notna().values
                    if "gamma_flip_level" in grp.columns
                    else np.zeros(len(grp), dtype=bool)
                )
                above_flip = has_flip & ~grp["d5_below_flip"].fillna(0.0).astype(bool).values
                long_cand = (
                    above_flip
                    & (~gate)
                    & np.where(np.isfinite(sz), sz > self.long_thr, False)
                )
                if self.hurst_mode == "filter" and self.apply_hurst_to_longs:
                    long_cand = long_cand & h_ok

            long_score = np.where(np.isfinite(sz), sz, -np.inf)
            if self.hurst_mode == "score" and self.apply_hurst_to_longs:
                boost = 1.0 + self.hurst_strength * np.clip(
                    np.where(np.isfinite(hz), hz - self.hurst_threshold, 0.0),
                    0.0,
                    None,
                )
                long_score = long_score * boost

            short_in_pos = _lifecycle_position(short_cand, self.time_stop_days)
            long_in_pos = _lifecycle_position(long_cand, self.time_stop_days)

            for i in range(len(grp)):
                if short_in_pos[i] or long_in_pos[i]:
                    beta = float(grp["bs_b_mkt"].iloc[i]) if "bs_b_mkt" in grp.columns else 1.0
                    if not np.isfinite(beta):
                        beta = 1.0
                    is_short = bool(short_in_pos[i])
                    all_rows.append({
                        "permno": permno,
                        "date": grp["date"].iloc[i],
                        "side": "short" if is_short else "long",
                        "sentiment_z": float(sz[i]) if np.isfinite(sz[i]) else 0.0,
                        "score": float(short_score[i]) if is_short else float(long_score[i]),
                        "hurst": float(hz[i]) if np.isfinite(hz[i]) else np.nan,
                        "beta": beta,
                    })

        if not all_rows:
            return pd.DataFrame(columns=["permno", "date", "weight"])

        df = pd.DataFrame(all_rows)
        df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        weight_rows: list[dict] = []

        for dt, day in df.groupby("date", sort=True):
            shorts = day[day["side"] == "short"].copy()
            longs = day[day["side"] == "long"].copy()

            if len(shorts) > self.n_per_leg:
                rank_col = "score" if (self.hurst_mode == "score" and self.apply_hurst_to_shorts) else "sentiment_z"
                shorts = shorts.nlargest(self.n_per_leg, rank_col)

            if len(longs) > self.n_per_leg:
                # Preserve ML version's corrected long ranking: highest positive sentiment.
                rank_col = "score" if (self.hurst_mode == "score" and self.apply_hurst_to_longs) else "sentiment_z"
                longs = longs.nlargest(self.n_per_leg, rank_col)

            n_s, n_l = len(shorts), len(longs)
            if n_s == 0 and n_l == 0:
                continue

            if self.neutrality == "dollar":
                w_s = -1.0 / n_s if n_s else 0.0
                w_l = 1.0 / n_l if n_l else 0.0
            else:
                beta_l = longs["beta"].mean() if n_l else 1.0
                beta_s = shorts["beta"].mean() if n_s else 1.0

                if not (np.isfinite(beta_s) and beta_s != 0):
                    beta_s = 1.0
                if not np.isfinite(beta_l):
                    beta_l = 1.0

                w_l = 1.0 / n_l if n_l else 0.0
                beta_l_total = n_l * w_l * beta_l if n_l else 0.0
                w_s = -beta_l_total / (n_s * beta_s) if n_s else 0.0

            for _, r in shorts.iterrows():
                if w_s != 0:
                    weight_rows.append({"permno": r["permno"], "date": dt, "weight": w_s})
            for _, r in longs.iterrows():
                if w_l != 0:
                    weight_rows.append({"permno": r["permno"], "date": dt, "weight": w_l})

        if not weight_rows:
            return pd.DataFrame(columns=["permno", "date", "weight"])

        out = pd.DataFrame(weight_rows)
        out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        return out
