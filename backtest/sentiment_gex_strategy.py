"""
Signal and Strategy classes for the GEX-Fragility + Sentiment long-short strategy.

Signal:    FragilityGEXSignal  - computes D1-D6, sentiment_z, fragility gates (pre-lagged)
Strategy:  GEXSentimentStrategy - implements Variant A / B with entry-exit lifecycle

Use with BacktestConfig(signal_lag_days=0) since the signal already pre-lags by 1 day.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_framework import BaseSignal, BaseStrategy

CONTRACT_FACTORS_DEFAULT = (
    Path(__file__).resolve().parent.parent / "data" / "gex_contract_factors.parquet"
)


# ---------------------------------------------------------------------------
# Lifecycle helper
# ---------------------------------------------------------------------------

def _lifecycle_position(candidate: np.ndarray, time_stop: int) -> np.ndarray:
    """
    Enter on the first True in `candidate`; hold while True and days_held < time_stop.
    After a time-stop exit, require a False gap before re-entering.
    """
    n = len(candidate)
    in_pos = np.zeros(n, dtype=bool)
    holding = False
    days_held = 0
    force_gap = False

    for i in range(n):
        c = bool(candidate[i])
        if not c:
            holding = False
            days_held = 0
            force_gap = False
        elif force_gap:
            pass
        elif holding:
            if days_held >= time_stop:
                holding = False
                days_held = 0
                force_gap = True
            else:
                in_pos[i] = True
                days_held += 1
        else:
            holding = True
            days_held = 1
            in_pos[i] = True

    return in_pos


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

class FragilityGEXSignal(BaseSignal):
    """
    Computes D1-D6 GEX fragility factors + sentiment z-score.

    All factor columns are manually shifted 1 trading day per permno to prevent
    lookahead.  Pair with BacktestConfig(signal_lag_days=0).

    Gate flavours added to panel:
        fragility_gate_simple : D1_neg & D4_top_tercile          (any stock with panel GEX)
        fragility_gate_full   : D1_neg & D5_below_flip &         (requires contract data)
                                (D3_top_tercile | D4_top_tercile)
        fragility_gate_v2     : D5_below_flip & D4_top_tercile   (requires contract data)
        fragility_gate_v3     : D5_below_flip                    (requires contract data)

    D2/D3/D5 are populated only for stocks present in gex_contract_factors.parquet;
    all other stocks receive NaN for these columns and are excluded from v2/v3/full gates.
    """

    def __init__(
        self,
        contract_factors_path: Path = CONTRACT_FACTORS_DEFAULT,
        sentiment_col: str = "sent_avg",
        regime_lookback_days: int = 5,
    ):
        self._cfp = Path(contract_factors_path)
        self._sent_col = sentiment_col
        self._lookback = regime_lookback_days
        self._cf: Optional[pd.DataFrame] = None

    def _contract_factors(self) -> pd.DataFrame:
        if self._cf is None:
            if not self._cfp.exists():
                raise FileNotFoundError(
                    f"Contract factors not found: {self._cfp}\n"
                    "Run:  python backtest/gex_factors_builder.py"
                )
            self._cf = pd.read_parquet(self._cfp)
            self._cf["date"] = pd.to_datetime(self._cf["date"])
        return self._cf

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        # Work on a slim copy to avoid duplicating the full ~110-column panel.
        slim_cols = [
            "permno", "secid", "date",
            "gex_net_gex_1pct", "gex_call_gex_1pct", "gex_put_gex_1pct", "gex_spot",
            self._sent_col,
        ]
        slim_cols = [c for c in slim_cols if c in panel.columns]
        s = panel[slim_cols].copy()

        # Merge contract-level factors (D2/D3/D5 available only for stocks in contract_gex)
        cf = self._contract_factors()
        s = s.merge(
            cf[["secid", "date", "d2_hhi", "d3_term_ratio_short", "gamma_flip_level"]],
            on=["secid", "date"],
            how="left",
        )

        # D1: net GEX sign
        s["d1_neg"] = (s["gex_net_gex_1pct"] < 0).astype(float)
        s["d1_pos"] = (s["gex_net_gex_1pct"] > 0).astype(float)

        # D4: put dominance (computable from panel aggregate GEX for all stocks)
        total_abs = s["gex_call_gex_1pct"].abs() + s["gex_put_gex_1pct"].abs()
        s["d4_put_dom"] = s["gex_put_gex_1pct"].abs() / total_abs.replace(0.0, np.nan)

        # D5: below gamma flip (NaN for stocks without contract data)
        s["d5_below_flip"] = (s["gex_spot"] < s["gamma_flip_level"]).astype(float)

        # D6: any flip (positive->negative) in last N days
        flip_event = (
            (s.groupby("permno", sort=False)["gex_net_gex_1pct"].shift(1) > 0)
            & (s["gex_net_gex_1pct"] < 0)
        ).astype(float)
        s["d6_switch"] = (
            flip_event.groupby(s["permno"], sort=False)
            .transform(lambda x: x.rolling(self._lookback, min_periods=1).max())
            .fillna(0.0)
        )

        # Cross-sectional tercile ranks (per date) using built-in rank pct
        def _cs_tercile(col: str) -> pd.Series:
            pct = s.groupby("date")[col].transform("rank", pct=True)
            return pd.cut(
                pct,
                bins=[0.0, 1 / 3, 2 / 3, 1.0],
                labels=[1, 2, 3],
                include_lowest=True,
            ).astype(float)

        s["d2_tercile"] = _cs_tercile("d2_hhi")
        s["d3_tercile"] = _cs_tercile("d3_term_ratio_short")
        s["d4_tercile"] = _cs_tercile("d4_put_dom")

        # Sentiment z-score (cross-sectional per date)
        g_mean = s.groupby("date")[self._sent_col].transform("mean")
        g_std = s.groupby("date")[self._sent_col].transform("std")
        s["sentiment_z"] = (s[self._sent_col] - g_mean) / g_std.replace(0.0, np.nan)

        # Net GEX z-score
        gex_mean = s.groupby("date")["gex_net_gex_1pct"].transform("mean")
        gex_std = s.groupby("date")["gex_net_gex_1pct"].transform("std")
        s["net_gex_z"] = (s["gex_net_gex_1pct"] - gex_mean) / gex_std.replace(0.0, np.nan)

        # Fragility gates
        d4_top = s["d4_tercile"] == 3
        d5_below = s["d5_below_flip"] == 1

        # simple: D1_neg & D4_top  (all 232 stocks)
        s["fragility_gate_simple"] = (s["d1_neg"] == 1) & d4_top

        # full: D1_neg & D5 & (D3 or D4)  (original — rarely fires for large-caps)
        d3_top = s["d3_tercile"] == 3
        s["fragility_gate_full"] = (s["d1_neg"] == 1) & d5_below & (d3_top | d4_top)

        # v2: D5_below_flip & D4_top  (fires ~14% for 19 stocks — recommended)
        s["fragility_gate_v2"] = d5_below & d4_top

        # v3: D5_below_flip only  (broadest — fires ~43% for 19 stocks)
        s["fragility_gate_v3"] = d5_below.astype(float)

        # Lag all factor columns 1 day per permno (prevents lookahead)
        lag_cols = [
            "d1_neg", "d1_pos",
            "d2_hhi", "d2_tercile",
            "d3_term_ratio_short", "d3_tercile",
            "d4_put_dom", "d4_tercile",
            "d5_below_flip", "d6_switch",
            "sentiment_z", "net_gex_z",
            "fragility_gate_simple", "fragility_gate_full",
            "fragility_gate_v2", "fragility_gate_v3",
            "gamma_flip_level",
        ]
        for col in lag_cols:
            if col in s.columns:
                s[col] = s.groupby("permno", sort=False)[col].shift(1)

        # Write computed factor columns back into the original panel in-place
        # (avoids copying the full ~110-column panel)
        for col in s.columns:
            if col not in slim_cols:
                panel[col] = s[col].values

        # Required signal column (not lagged again since signal_lag_days=0)
        panel["signal"] = panel.get("sentiment_z", pd.Series(0.0, index=panel.index)).fillna(0.0)

        return panel


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class GEXSentimentStrategy(BaseStrategy):
    """
    Long-short strategy driven by GEX fragility gate + sentiment signal.

    Parameters
    ----------
    variant : 'A' or 'B'
        A - gate AND sentiment_z > threshold.  Enter on first qualifying day.
        B - within gate==True, short the top-N by sentiment_z each day.
    gate : 'simple' or 'full'
        simple - D1_neg & D4_top_tercile  (all ~232 stocks)
        full   - D1_neg & D5_below_flip & (D3_top | D4_top)  (19 stocks)
    neutrality : 'dollar' or 'beta'
    n_per_leg : max simultaneous positions per leg
    time_stop_days : exit position after this many consecutive holding days
    sentiment_threshold : short-leg Variant A entry threshold (z-score units)
    long_sentiment_threshold : long-leg sentiment threshold (z > this → long; positive follows momentum)
    long_leg : 'momentum' (positive sentiment + positive GEX) or 'none' (pure short book)
    """

    def __init__(
        self,
        variant: Literal["A", "B"] = "A",
        gate: Literal["simple", "full"] = "simple",
        neutrality: Literal["dollar", "beta"] = "dollar",
        n_per_leg: int = 5,
        time_stop_days: int = 21,
        sentiment_threshold: float = 0.5,
        long_sentiment_threshold: float = 0.3,
        long_leg: Literal["momentum", "none"] = "momentum",
    ):
        self.variant = variant
        _gate_map = {
            "simple": "fragility_gate_simple",
            "full":   "fragility_gate_full",
            "v2":     "fragility_gate_v2",   # D5_below_flip & D4_top (recommended for 19 stocks)
            "v3":     "fragility_gate_v3",   # D5_below_flip only (broadest)
        }
        self.gate_col = _gate_map.get(gate, "fragility_gate_simple")
        self.neutrality = neutrality
        self.n_per_leg = n_per_leg
        self.time_stop_days = time_stop_days
        self.sent_thr = sentiment_threshold
        self.long_thr = long_sentiment_threshold
        self.long_leg = long_leg

    def target_weights(self, panel: pd.DataFrame) -> pd.DataFrame:
        gate_col = self.gate_col
        if gate_col not in panel.columns:
            raise KeyError(
                f"Gate column '{gate_col}' not in panel. "
                "Ensure FragilityGEXSignal.compute() ran first."
            )

        all_rows: list[dict] = []

        for permno, grp in panel.groupby("permno", sort=False):
            grp = grp.sort_values("date").copy().reset_index(drop=True)

            gate = grp[gate_col].fillna(0.0).astype(bool).values
            sz = grp["sentiment_z"].fillna(np.nan).values
            d1_pos = grp["d1_pos"].fillna(0.0).astype(bool).values

            # Short candidates
            if self.variant == "A":
                short_cand = gate & np.where(np.isfinite(sz), sz > self.sent_thr, False)
            else:
                # Variant B: per-day ranking handled below; flag all gate+finite-sz stocks
                short_cand = gate & np.isfinite(sz)

            # Long candidates: above gamma flip (stable/dampening regime) + positive sentiment
            # Requires gamma_flip_level to be non-NaN to avoid including non-contract stocks.
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

            short_in_pos = _lifecycle_position(short_cand, self.time_stop_days)
            long_in_pos = _lifecycle_position(long_cand, self.time_stop_days)

            for i in range(len(grp)):
                if short_in_pos[i] or long_in_pos[i]:
                    beta = float(grp["bs_b_mkt"].iloc[i]) if "bs_b_mkt" in grp.columns else 1.0
                    if not np.isfinite(beta):
                        beta = 1.0
                    all_rows.append({
                        "permno": permno,
                        "date": grp["date"].iloc[i],
                        "side": "short" if short_in_pos[i] else "long",
                        "sentiment_z": float(sz[i]) if np.isfinite(sz[i]) else 0.0,
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

            # Variant B: keep only top-N by sentiment_z within the day's short pool
            if self.variant == "B" and len(shorts) > self.n_per_leg:
                shorts = shorts.nlargest(self.n_per_leg, "sentiment_z")
            elif len(shorts) > self.n_per_leg:
                shorts = shorts.head(self.n_per_leg)

            if len(longs) > self.n_per_leg:
                longs = longs.nsmallest(self.n_per_leg, "sentiment_z")

            n_s, n_l = len(shorts), len(longs)
            if n_s == 0 and n_l == 0:
                continue

            if self.neutrality == "dollar":
                w_s = -1.0 / n_s if n_s else 0.0
                w_l = 1.0 / n_l if n_l else 0.0
            else:
                # Beta-neutral: scale short leg so portfolio beta ~ 0
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
                weight_rows.append({"permno": r["permno"], "date": dt, "weight": w_s})
            for _, r in longs.iterrows():
                weight_rows.append({"permno": r["permno"], "date": dt, "weight": w_l})

        if not weight_rows:
            return pd.DataFrame(columns=["permno", "date", "weight"])

        out = pd.DataFrame(weight_rows)
        out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        return out
