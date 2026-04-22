from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Millisecond intraday (MII) factors: only these names are pulled from the optional MII file.
MII_FACTOR_COLS: tuple[str, ...] = (
    "QuotedSpread_Dollar_tw",
    "QuotedSpread_Percent_tw",
    "BestOfrDepth_Dollar_tw",
    "BestBidDepth_Dollar_tw",
    "BestOfrDepth_Share_tw",
    "BestBidDepth_Share_tw",
    "EffectiveSpread_Dollar_Ave",
    "EffectiveSpread_Percent_Ave",
    "EffectiveSpread_Dollar_DW",
    "EffectiveSpread_Dollar_SW",
    "EffectiveSpread_Percent_DW",
    "EffectiveSpread_Percent_SW",
    "DollarRealizedSpread_LR_Ave",
    "PercentRealizedSpread_LR_Ave",
    "DollarRealizedSpread_LR_SW",
    "DollarRealizedSpread_LR_DW",
    "PercentRealizedSpread_LR_SW",
    "PercentRealizedSpread_LR_DW",
    "DollarPriceImpact_LR_Ave",
    "PercentPriceImpact_LR_Ave",
    "DollarPriceImpact_LR_SW",
    "DollarPriceImpact_LR_DW",
    "PercentPriceImpact_LR_SW",
    "PercentPriceImpact_LR_DW",
    "ivol_t",
    "ivol_q",
    "bs_ratio_num",
    "bs_ratio_vol",
    "TSignSqrtDVol1",
    "TSignSqrtDVol2",
    "HIndex",
    "var_ratio1",
    "var_ratio2",
    "var_ratio3",
    "var_ratio4",
    "var_ratio5",
    "bs_ratio_retail_num",
    "bs_ratio_retail_vol",
    "bs_ratio_Inst20k_num",
    "bs_ratio_Inst20k_vol",
    "bs_ratio_Inst50k_num",
    "bs_ratio_Inst50k_vol",
)


def build_backtest_panel_from_inventory_files(
    data_dir: Path,
    output_file: Optional[Path] = None,
    crsp_file: str = "crsp_dsf_final.csv",
    bs_file: str = "BS.csv",
    sentiment_file: str = "stocktwits_daily_sentiment_panel.csv",
    gex_file: str = "underlying_gex_daily_with_permno.parquet",
    linked_file: Optional[str] = "linked_secids_top_liquid_2012_2024.parquet",
    sentiment_trading_date_alignment: str = "backward",
    sentiment_max_alignment_days: int = 7,
    mii_file: Optional[str] = "MII.csv",
    compress_output: str = "zstd",
) -> Path:
    """
    Build a single daily backtest panel using CRSP daily stock data as backbone.

    Final canonical identifier block
    --------------------------------
    - permno: primary key
    - permco: CRSP company id
    - secid: canonical secid from link table, fallback to GEX secid
    - ticker: canonical ticker from link table
    - crsp_ticker: raw daily CRSP ticker kept separately for reference
    - date: trading date
    - link_method: mapping provenance if available

    Join policy
    -----------
    - CRSP backbone: (permno, date)
    - BS: (permno, date)
    - GEX: (permno, date), with secid retained as side identifier
    - link file: (permno)
    - sentiment: raw ticker/date is first aligned to CRSP trading dates, then merged on (permno, date)
    - millisecond-intraday (MII): optional ``MII.csv``-style file with ``DATE``, ``SYM_ROOT``
      (same as CRSP/ link root ticker) and only factor columns in ``MII_FACTOR_COLS``. Join is
      ``(date, sent_align_ticker)`` = ``(DATE, SYM_ROOT standardized)`` — same string as
      map_ticker + crsp_ticker for sentiment.

    Sentiment flags
    ---------------
    - sentiment_source_available_flag
    - sentiment_aligned_flag
    - sentiment_missing_flag

    sentiment_trading_date_alignment: ``"backward"`` | ``"forward"`` | ``"exact"``
        ``"backward"`` (default): map each raw calendar row to the **last** trading
        day on or before that date (weekend / holiday collection rolls to the prior
        session, e.g. Sat-Sun to Friday) so features line up with an end-of-prior session
        information set. ``"forward"``: first trading day on or after the raw date.
        ``"exact"``: only rows where raw date is already a valid (permno, trade) day.
    """

    data_dir = Path(data_dir)
    output_file = output_file or data_dir / "backtest_panel_main.parquet"

    crsp_path = data_dir / crsp_file
    bs_path = data_dir / bs_file
    sentiment_path = data_dir / sentiment_file
    gex_path = data_dir / gex_file
    linked_path = data_dir / linked_file if linked_file else None
    mii_path: Optional[Path] = None
    if mii_file:
        mp = Path(mii_file)
        mii_path = mp if mp.is_absolute() else (data_dir / mii_file)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_exists(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    def _to_datetime(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, errors="coerce").dt.normalize()

    def _pct_string_to_decimal(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        s2 = s.astype(str).str.strip()
        is_pct = s2.str.endswith("%")
        out = pd.to_numeric(s2.str.replace("%", "", regex=False), errors="coerce")
        out.loc[is_pct] = out.loc[is_pct] / 100.0
        return out

    def _safe_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _standardize_ticker(s: pd.Series) -> pd.Series:
        out = s.astype(str).str.upper().str.strip()
        out = out.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA})
        return out

    def _coalesce_columns(df: pd.DataFrame, new_col: str, candidates: list[str]) -> pd.DataFrame:
        existing = [c for c in candidates if c in df.columns]
        if not existing:
            df[new_col] = pd.Series([pd.NA] * len(df), index=df.index)
            return df
        out = df[existing[0]].copy()
        for c in existing[1:]:
            out = out.combine_first(df[c])
        df[new_col] = out
        return df

    def _align_sentiment_one_ticker(
        sent_grp: pd.DataFrame,
        trade_grp: pd.DataFrame,
        alignment: str = "backward",
        max_days: int = 7,
    ) -> pd.DataFrame:
        """
        Align sentiment calendar dates to actual trading dates for one ticker.

        - ``direction="backward"``: raw date maps to the **last** trading day on or
          before that date (weekend / holiday non-trades roll to the **previous** session).
        - ``direction="forward"``: raw date maps to the **first** trading day on or after
          that date (non-trades roll to the **next** session).
        - ``"exact"``: keep only (ticker, day) that already exist on the trade calendar.

        Effect on backtest timing
        -------------------------
        ``sent_avg`` on CRSP row date *T* aggregates rows whose ``sentiment_date_raw``
        mapped to trade_date *T*. CRSP ``ret`` on that row is close *T-1* → close *T*.
        If sentiment includes posts after the close for *T* (or, with backward
        alignment, after the close of the last session *T*), that row is not strictly
        concurrent with the printed daily return; use a signal lag in the strategy when
        required by your data cutoff.
        """
        if alignment not in ("backward", "forward", "exact"):
            raise ValueError(
                f"sentiment_trading_date_alignment must be 'backward', 'forward', or 'exact'; got {alignment!r}"
            )

        sent_grp = sent_grp.sort_values("date").reset_index(drop=True).copy()
        trade_grp = trade_grp.sort_values("date").reset_index(drop=True).copy()

        if sent_grp.empty or trade_grp.empty:
            return pd.DataFrame()

        sent_grp = sent_grp.rename(columns={"date": "sentiment_date_raw"})
        trade_grp = trade_grp.rename(columns={"date": "trade_date"})

        if alignment in ("forward", "backward"):
            out = pd.merge_asof(
                sent_grp,
                trade_grp,
                left_on="sentiment_date_raw",
                right_on="trade_date",
                direction=alignment,
                allow_exact_matches=True,
                tolerance=pd.Timedelta(days=max_days),
            )
        else:
            out = sent_grp.merge(
                trade_grp,
                left_on=["ticker", "sentiment_date_raw"],
                right_on=["ticker", "trade_date"],
                how="left",
            )
        return out

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------
    for p in [crsp_path, bs_path, sentiment_path, gex_path]:
        _ensure_exists(p)
    if linked_path is not None:
        _ensure_exists(linked_path)

    # ------------------------------------------------------------------
    # 1) Load link file first: canonical permno-secid-ticker map
    # ------------------------------------------------------------------
    link_df = None
    if linked_path is not None and linked_path.exists():
        link_df = pd.read_parquet(linked_path)
        link_df.columns = [c.strip() for c in link_df.columns]

        required_link = {"permno", "secid"}
        if not required_link.issubset(link_df.columns):
            raise ValueError(f"Link file must contain columns: {required_link}")

        link_df["permno"] = pd.to_numeric(link_df["permno"], errors="coerce").astype("Int64")
        link_df["secid"] = pd.to_numeric(link_df["secid"], errors="coerce").astype("Int64")

        if "ticker" in link_df.columns:
            link_df["ticker"] = _standardize_ticker(link_df["ticker"])

        link_keep = [c for c in ["permno", "secid", "ticker", "link_method"] if c in link_df.columns]
        link_df = (
            link_df[link_keep]
            .drop_duplicates(subset=["permno"], keep="last")
            .rename(columns={
                "secid": "map_secid",
                "ticker": "map_ticker",
                "link_method": "map_link_method",
            })
        )

    # ------------------------------------------------------------------
    # 2) Load CRSP as backbone
    # ------------------------------------------------------------------
    crsp = pd.read_csv(crsp_path, low_memory=False)
    crsp.columns = [c.strip() for c in crsp.columns]

    required_crsp = {"PERMNO", "DlyCalDt"}
    if not required_crsp.issubset(crsp.columns):
        raise ValueError(f"CRSP file must contain columns: {required_crsp}")

    crsp["permno"] = pd.to_numeric(crsp["PERMNO"], errors="coerce").astype("Int64")
    crsp["date"] = _to_datetime(crsp["DlyCalDt"])

    if "Ticker" in crsp.columns:
        crsp["crsp_ticker"] = _standardize_ticker(crsp["Ticker"])
    else:
        crsp["crsp_ticker"] = pd.NA

    if "PERMCO" in crsp.columns:
        crsp["PERMCO"] = pd.to_numeric(crsp["PERMCO"], errors="coerce").astype("Int64")

    crsp = crsp.loc[crsp["permno"].notna() & crsp["date"].notna()].copy()
    crsp = crsp.sort_values(["permno", "date"]).reset_index(drop=True)
    crsp = crsp.drop_duplicates(subset=["permno", "date"], keep="last").copy()

    # attach canonical link info to CRSP early
    if link_df is not None:
        crsp = crsp.merge(link_df, on="permno", how="left", validate="m:1")

    # convenience columns
    if "TradingStatusFlg" in crsp.columns:
        crsp["is_active_status"] = (crsp["TradingStatusFlg"] == "A").astype("int8")
    else:
        crsp["is_active_status"] = pd.Series(1, index=crsp.index, dtype="int8")

    if "DlyDelFlg" in crsp.columns:
        crsp["is_not_delisted_today"] = (crsp["DlyDelFlg"] == "N").astype("int8")
    else:
        crsp["is_not_delisted_today"] = pd.Series(1, index=crsp.index, dtype="int8")

    if "DlyPrc" in crsp.columns:
        crsp["DlyPrc"] = pd.to_numeric(crsp["DlyPrc"], errors="coerce")
        crsp["price_abs"] = crsp["DlyPrc"].abs()
    else:
        crsp["price_abs"] = np.nan

    if "DlyVol" in crsp.columns:
        crsp["DlyVol"] = pd.to_numeric(crsp["DlyVol"], errors="coerce")
    if "ShrOut" in crsp.columns:
        crsp["ShrOut"] = pd.to_numeric(crsp["ShrOut"], errors="coerce")
    if "DlyRet" in crsp.columns:
        crsp["DlyRet"] = pd.to_numeric(crsp["DlyRet"], errors="coerce")

    crsp["dollar_volume"] = crsp["price_abs"] * crsp["DlyVol"] if "DlyVol" in crsp.columns else np.nan
    crsp["market_equity"] = crsp["price_abs"] * crsp["ShrOut"] if "ShrOut" in crsp.columns else np.nan
    crsp["has_valid_return"] = crsp["DlyRet"].notna().astype("int8") if "DlyRet" in crsp.columns else 0

    # ------------------------------------------------------------------
    # 3) Load Beta Suite
    # ------------------------------------------------------------------
    bs = pd.read_csv(bs_path, low_memory=False)
    bs.columns = [c.strip() for c in bs.columns]

    required_bs = {"PERMNO", "date"}
    if not required_bs.issubset(bs.columns):
        raise ValueError(f"BS file must contain columns: {required_bs}")

    bs["permno"] = pd.to_numeric(bs["PERMNO"], errors="coerce").astype("Int64")
    bs["date"] = _to_datetime(bs["date"])

    pct_like_cols = ["ret", "ivol", "tvol", "R2", "exret"]
    for c in pct_like_cols:
        if c in bs.columns:
            bs[c] = _pct_string_to_decimal(bs[c])

    numeric_bs_cols = ["n", "alpha", "b_mkt", "b_smb", "b_hml", "b_umd"]
    bs = _safe_numeric(bs, numeric_bs_cols)

    bs_keep_map = {
        "n": "bs_n",
        "ret": "bs_ret",
        "alpha": "bs_alpha",
        "b_mkt": "bs_b_mkt",
        "b_smb": "bs_b_smb",
        "b_hml": "bs_b_hml",
        "b_umd": "bs_b_umd",
        "ivol": "bs_ivol",
        "tvol": "bs_tvol",
        "R2": "bs_R2",
        "exret": "bs_exret",
    }

    bs_keep = ["permno", "date"] + [c for c in bs_keep_map if c in bs.columns]
    bs = bs[bs_keep].copy().rename(columns=bs_keep_map)
    bs = bs.drop_duplicates(subset=["permno", "date"], keep="last")

    # ------------------------------------------------------------------
    # 4) Load GEX
    # ------------------------------------------------------------------
    gex = pd.read_parquet(gex_path)
    gex.columns = [c.strip() for c in gex.columns]

    if "permno" not in gex.columns or "date" not in gex.columns:
        raise ValueError("GEX file must contain permno and date.")

    gex["permno"] = pd.to_numeric(gex["permno"], errors="coerce").astype("Int64")
    gex["date"] = _to_datetime(gex["date"])

    if "secid" in gex.columns:
        gex["secid"] = pd.to_numeric(gex["secid"], errors="coerce").astype("Int64")

    gex_cols = [
        "permno", "date", "secid",
        "spot", "spot_close", "spot_return", "spot_cfadj", "spot_shrout",
        "n_contracts", "n_optionids", "n_expiries",
        "call_gex_1pct", "put_gex_1pct", "net_gex_1pct",
        "call_gex_1pt", "put_gex_1pt", "net_gex_1pt",
        "total_open_interest", "total_option_volume",
    ]
    gex = gex[[c for c in gex_cols if c in gex.columns]].copy()
    gex = gex.drop_duplicates(subset=["permno", "date"], keep="last")

    gex_rename = {c: f"gex_{c}" for c in gex.columns if c not in {"permno", "date"}}
    gex = gex.rename(columns=gex_rename)

    # ------------------------------------------------------------------
    # 5) Load sentiment and align to CRSP trading dates
    #
    # sent_daily keys (permno, date): date is CRSP trade_date. sent_* fields aggregate
    # raw sentiment rows via merge_asof calendar→trade alignment (see _align_sentiment_one_ticker).
    # Pair sent_* with returns in the backtest with a rule that matches your collection cutoff
    # (e.g. signal lag) so the sentiment window does not span the same return you predict.
    # ------------------------------------------------------------------
    sent = pd.read_csv(sentiment_path, low_memory=False)
    sent.columns = [c.strip() for c in sent.columns]

    required_sent = {"ticker", "date"}
    if not required_sent.issubset(sent.columns):
        raise ValueError(f"Sentiment file must contain columns: {required_sent}")

    sent["ticker"] = _standardize_ticker(sent["ticker"])
    sent["date"] = _to_datetime(sent["date"])

    sent_numeric = [
        "Posts", "AvgSentiment", "MedianSentiment",
        "PositiveProbAvg", "NegativeProbAvg", "NeutralProbAvg",
    ]
    sent = _safe_numeric(sent, [c for c in sent_numeric if c in sent.columns])

    sent = sent.loc[sent["ticker"].notna() & sent["date"].notna()].copy()
    sent = sent.drop_duplicates().sort_values(["ticker", "date"]).reset_index(drop=True)

    # Prefer canonical ticker from link table for alignment.
    # Fall back to raw CRSP ticker only when canonical ticker is absent.
    crsp["sent_align_ticker"] = crsp["map_ticker"] if "map_ticker" in crsp.columns else pd.NA
    if "crsp_ticker" in crsp.columns:
        crsp["sent_align_ticker"] = crsp["sent_align_ticker"].combine_first(crsp["crsp_ticker"])
    crsp["sent_align_ticker"] = _standardize_ticker(crsp["sent_align_ticker"])

    trade_calendar = crsp[["permno", "sent_align_ticker", "date"]].copy()
    trade_calendar = trade_calendar.rename(columns={"sent_align_ticker": "ticker"})
    trade_calendar = trade_calendar.loc[
        trade_calendar["ticker"].notna()
        & trade_calendar["permno"].notna()
        & trade_calendar["date"].notna()
    ].copy()

    trade_calendar = (
        trade_calendar
        .drop_duplicates(subset=["ticker", "date", "permno"], keep="last")
        .sort_values(["ticker", "date", "permno"])
        .reset_index(drop=True)
    )

    aligned_parts = []
    common_tickers = sorted(set(sent["ticker"]).intersection(set(trade_calendar["ticker"])))

    for ticker in common_tickers:
        sent_grp = sent.loc[sent["ticker"] == ticker].copy()
        trade_grp = trade_calendar.loc[trade_calendar["ticker"] == ticker].copy()

        aligned = _align_sentiment_one_ticker(
            sent_grp=sent_grp,
            trade_grp=trade_grp,
            alignment=sentiment_trading_date_alignment,
            max_days=sentiment_max_alignment_days,
        )
        if not aligned.empty:
            aligned_parts.append(aligned)

    if aligned_parts:
        sent_aligned = pd.concat(aligned_parts, ignore_index=True)
    else:
        sent_aligned = pd.DataFrame(columns=[
            "ticker", "sentiment_date_raw", "Posts", "AvgSentiment", "MedianSentiment",
            "PositiveProbAvg", "NegativeProbAvg", "NeutralProbAvg", "permno", "trade_date"
        ])

    # Keep only rows that found a nearby valid trading date
    sent_aligned = sent_aligned.loc[
        sent_aligned["permno"].notna() & sent_aligned["trade_date"].notna()
    ].copy()

    sent_aligned["permno"] = pd.to_numeric(sent_aligned["permno"], errors="coerce").astype("Int64")
    sent_aligned["date"] = _to_datetime(sent_aligned["trade_date"])
    # trade date minus collection date: <=0 for backward (Fri for Sat), >=0 for forward (Mon for Sat)
    sent_aligned["sentiment_calendar_to_trade_lag_days"] = (
        sent_aligned["date"] - sent_aligned["sentiment_date_raw"]
    ).dt.days.astype("Int64")

    lag = sent_aligned["sentiment_calendar_to_trade_lag_days"]
    maxd = sentiment_max_alignment_days
    if sentiment_trading_date_alignment == "forward":
        keep = lag.between(0, maxd, inclusive="both")
    elif sentiment_trading_date_alignment == "backward":
        keep = lag.between(-maxd, 0, inclusive="both")
    else:
        keep = (lag == 0) & lag.notna()
    sent_aligned = sent_aligned.loc[keep].copy()

    agg_map = {
        "Posts": "sum",
        "AvgSentiment": "mean",
        "MedianSentiment": "mean",
        "PositiveProbAvg": "mean",
        "NegativeProbAvg": "mean",
        "NeutralProbAvg": "mean",
        "sentiment_calendar_to_trade_lag_days": "min",
        "sentiment_date_raw": "min",
    }
    existing_agg = {k: v for k, v in agg_map.items() if k in sent_aligned.columns}

    sent_daily = (
        sent_aligned
        .groupby(["permno", "date"], as_index=False)
        .agg(existing_agg)
    )

    sent_daily = sent_daily.rename(columns={
        "Posts": "sent_posts",
        "AvgSentiment": "sent_avg",
        "MedianSentiment": "sent_median",
        "PositiveProbAvg": "sent_pos_prob",
        "NegativeProbAvg": "sent_neg_prob",
        "NeutralProbAvg": "sent_neu_prob",
    })
    sent_daily["sentiment_aligned_flag"] = 1

    # coverage start by raw sentiment ticker, mapped using same alignment ticker logic
    sent_first_date_by_ticker = (
        sent.groupby("ticker", as_index=False)["date"]
        .min()
        .rename(columns={"date": "sentiment_first_raw_date"})
    )

    crsp = crsp.merge(
        sent_first_date_by_ticker,
        left_on="sent_align_ticker",
        right_on="ticker",
        how="left",
        suffixes=("", "_sentcov"),
    )

    if "ticker_sentcov" in crsp.columns:
        crsp = crsp.drop(columns=["ticker_sentcov"])

    crsp["sentiment_source_available_flag"] = (
        crsp["sentiment_first_raw_date"].notna()
        & (crsp["date"] >= crsp["sentiment_first_raw_date"])
    ).astype("int8")

    # ------------------------------------------------------------------
    # 5.5) Optional millisecond intraday (MII): ``csv_inventory_log`` / ``MII.csv`` uses
    #      ``DATE`` + ``SYM_ROOT``; root ticker is standardized to match ``sent_align_ticker`` on CRSP.
    #      Only columns in ``MII_FACTOR_COLS`` are merged.
    # ------------------------------------------------------------------
    mii_for_merge: Optional[pd.DataFrame] = None
    if mii_path is not None:
        _ensure_exists(mii_path)
        mii_want = list(MII_FACTOR_COLS)
        suf = mii_path.suffix.lower()
        if suf == ".parquet":
            mii_raw = pd.read_parquet(mii_path)
        elif suf == ".csv":
            mii_raw = pd.read_csv(mii_path, low_memory=False)
        else:
            raise ValueError(
                f"MII file must be .parquet or .csv, got {mii_path} (suffix {suf!r})"
            )
        mii_raw.columns = [c.strip() for c in mii_raw.columns]
        dcol = "DATE" if "DATE" in mii_raw.columns else ("date" if "date" in mii_raw.columns else None)
        if dcol is None:
            raise ValueError(f"MII file {mii_path} must contain a DATE or date column")
        if "SYM_ROOT" not in mii_raw.columns:
            raise ValueError(f"MII file {mii_path} must contain SYM_ROOT (ticker root)")

        mii_raw["date"] = _to_datetime(mii_raw[dcol])
        mii_raw["sent_align_ticker"] = _standardize_ticker(mii_raw["SYM_ROOT"])
        mii_keep = [c for c in mii_want if c in mii_raw.columns]
        if not mii_keep:
            raise ValueError(
                f"None of the MII factor columns in MII_FACTOR_COLS are present in {mii_path}. "
                f"Expected a subset of: {MII_FACTOR_COLS[:6]}..."
            )
        mii_for_merge = mii_raw[["date", "sent_align_ticker"] + mii_keep].copy()
        mii_for_merge = mii_for_merge.loc[
            mii_for_merge["date"].notna() & mii_for_merge["sent_align_ticker"].notna()
        ].copy()
        mii_for_merge = mii_for_merge.sort_values(["date", "sent_align_ticker"]).drop_duplicates(
            subset=["date", "sent_align_ticker"], keep="last"
        )
        mii_for_merge = _safe_numeric(mii_for_merge, mii_keep)

    # ------------------------------------------------------------------
    # 6) Merge side panels onto CRSP backbone
    # ------------------------------------------------------------------
    panel = crsp.merge(bs, on=["permno", "date"], how="left", validate="1:1")
    panel = panel.merge(gex, on=["permno", "date"], how="left", validate="1:1")
    panel = panel.merge(sent_daily, on=["permno", "date"], how="left", validate="1:1")
    if mii_for_merge is not None and not mii_for_merge.empty:
        panel = panel.merge(
            mii_for_merge,
            on=["date", "sent_align_ticker"],
            how="left",
            validate="m:1",
        )

    # ------------------------------------------------------------------
    # 7) Canonical identifier block
    # ------------------------------------------------------------------
    panel["permno"] = pd.to_numeric(panel["permno"], errors="coerce").astype("Int64")
    panel["date"] = _to_datetime(panel["date"])

    if "PERMCO" in panel.columns:
        panel["PERMCO"] = pd.to_numeric(panel["PERMCO"], errors="coerce").astype("Int64")
    if "map_secid" in panel.columns:
        panel["map_secid"] = pd.to_numeric(panel["map_secid"], errors="coerce").astype("Int64")
    if "gex_secid" in panel.columns:
        panel["gex_secid"] = pd.to_numeric(panel["gex_secid"], errors="coerce").astype("Int64")
    if "map_ticker" in panel.columns:
        panel["map_ticker"] = _standardize_ticker(panel["map_ticker"])
    if "crsp_ticker" in panel.columns:
        panel["crsp_ticker"] = _standardize_ticker(panel["crsp_ticker"])

    panel = _coalesce_columns(panel, "secid", ["map_secid", "gex_secid"])
    panel = _coalesce_columns(panel, "ticker", ["map_ticker"])
    panel = _coalesce_columns(panel, "permco", ["PERMCO"])
    panel = _coalesce_columns(panel, "link_method", ["map_link_method"])

    # ------------------------------------------------------------------
    # 8) Sentiment flags
    # ------------------------------------------------------------------
    if "sentiment_aligned_flag" not in panel.columns:
        panel["sentiment_aligned_flag"] = 0
    panel["sentiment_aligned_flag"] = panel["sentiment_aligned_flag"].fillna(0).astype("int8")

    panel["sentiment_missing_flag"] = (
        (panel["sentiment_source_available_flag"] == 1)
        & (panel["sentiment_aligned_flag"] == 0)
    ).astype("int8")

    if "sent_posts" in panel.columns:
        panel["sent_posts"] = pd.to_numeric(panel["sent_posts"], errors="coerce")
        panel["sent_posts_filled"] = panel["sent_posts"]
        mask = panel["sentiment_missing_flag"] == 1
        panel.loc[mask, "sent_posts_filled"] = 0.0

    # ------------------------------------------------------------------
    # 9) Alias common CRSP fields
    # ------------------------------------------------------------------
    alias_map = {
        "DlyRet": "ret",
        "DlyRetx": "retx",
        "DlyPrc": "prc",
        "DlyOpen": "open",
        "DlyHigh": "high",
        "DlyLow": "low",
        "DlyClose": "close",
        "DlyVol": "vol",
        "ShrOut": "shrout",
    }
    for src, dst in alias_map.items():
        if src in panel.columns and dst not in panel.columns:
            panel = panel.rename(columns={src: dst})

    panel["panel_row_id"] = np.arange(len(panel), dtype=np.int64)

    # ------------------------------------------------------------------
    # 10) Strict duplicate check
    # ------------------------------------------------------------------
    dup_mask = panel.duplicated(subset=["permno", "date"], keep=False)
    n_dup_rows = int(dup_mask.sum())

    if n_dup_rows > 0:
        show_cols = [c for c in ["permno", "date", "ticker", "crsp_ticker", "secid", "link_method"] if c in panel.columns]
        dup_sample = panel.loc[dup_mask, show_cols].sort_values(["permno", "date"]).head(20)
        raise ValueError(
            "Duplicate (permno, date) rows found after merges. "
            f"Duplicate row count: {n_dup_rows}. "
            f"Sample:\n{dup_sample.to_string(index=False)}"
        )

    # ------------------------------------------------------------------
    # 11) Drop redundant identifier and intermediate columns
    # ------------------------------------------------------------------
    redundant_cols = [
        "PERMNO",
        "PERMCO",
        "Ticker",
        "map_secid",
        "map_ticker",
        "map_link_method",
        "gex_secid",
        "trade_date",
        "sentiment_first_raw_date",
        "sent_align_ticker",
    ]
    drop_cols = [c for c in redundant_cols if c in panel.columns]
    if drop_cols:
        panel = panel.drop(columns=drop_cols)

    # ------------------------------------------------------------------
    # 12) Reorder columns
    # ------------------------------------------------------------------
    id_first_cols = [
        "panel_row_id",
        "permno",
        "permco",
        "secid",
        "ticker",
        "crsp_ticker",
        "date",
        "link_method",
    ]
    id_first_cols = [c for c in id_first_cols if c in panel.columns]
    other_cols = [c for c in panel.columns if c not in id_first_cols]
    panel = panel[id_first_cols + other_cols]

    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 13) Save
    # ------------------------------------------------------------------
    output_file.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_file, index=False, compression=compress_output)

    # ------------------------------------------------------------------
    # 14) Build log
    # ------------------------------------------------------------------
    log_path = output_file.with_suffix(".build_log.txt")

    bs_matched_rows = int(panel["bs_alpha"].notna().sum()) if "bs_alpha" in panel.columns else 0
    gex_matched_rows = int(panel["gex_net_gex_1pct"].notna().sum()) if "gex_net_gex_1pct" in panel.columns else 0
    sent_aligned_rows = int(panel["sentiment_aligned_flag"].sum()) if "sentiment_aligned_flag" in panel.columns else 0
    sent_available_rows = int(panel["sentiment_source_available_flag"].sum()) if "sentiment_source_available_flag" in panel.columns else 0
    sent_missing_rows = int(panel["sentiment_missing_flag"].sum()) if "sentiment_missing_flag" in panel.columns else 0
    sent_posts_max = float(panel["sent_posts"].max()) if "sent_posts" in panel.columns and panel["sent_posts"].notna().any() else np.nan

    lines = [
        f"Output file: {output_file}",
        f"Rows: {len(panel):,}",
        f"Columns: {panel.shape[1]:,}",
        "",
        "Coverage summary:",
        f"  Unique permnos: {panel['permno'].nunique(dropna=True):,}",
        f"  Unique secids: {panel['secid'].nunique(dropna=True) if 'secid' in panel.columns else 0:,}",
        f"  Unique canonical tickers: {panel['ticker'].nunique(dropna=True) if 'ticker' in panel.columns else 0:,}",
        f"  Unique CRSP daily tickers: {panel['crsp_ticker'].nunique(dropna=True) if 'crsp_ticker' in panel.columns else 0:,}",
        f"  Date range: {panel['date'].min()} -> {panel['date'].max()}",
        f"  BS matched rows: {bs_matched_rows:,}",
        f"  GEX matched rows: {gex_matched_rows:,}",
        f"  Sentiment aligned rows: {sent_aligned_rows:,}",
        f"  Sentiment source-available rows: {sent_available_rows:,}",
        f"  Sentiment missing rows: {sent_missing_rows:,}",
        f"  Max sent_posts: {sent_posts_max}",
        "",
        "Canonical identifier policy:",
        "  permno = main panel key",
        "  secid = link-file secid preferred, GEX secid fallback",
        "  ticker = canonical ticker from link file",
        "  crsp_ticker = raw daily CRSP ticker retained separately",
        "  joins use permno/date wherever available",
        "",
        "Sentiment alignment policy:",
        f"  trading_date alignment = {sentiment_trading_date_alignment}",
        f"  max calendar/trade step days = {sentiment_max_alignment_days}",
    ]
    if mii_for_merge is not None and not mii_for_merge.empty:
        mii_factor_cols = [c for c in mii_for_merge.columns if c not in ("date", "sent_align_ticker")]
        mii_matched = int(panel[mii_factor_cols].notna().any(axis=1).sum()) if mii_factor_cols else 0
        lines.extend(
            [
                "",
                f"MII file: {mii_path} (join on date + sent_align_ticker, SYM_ROOT from MII = ticker)",
                f"  factor columns merged: {mii_factor_cols}",
                f"  rows with any MII factor non-null: {mii_matched:,}",
            ]
        )
    log_path.write_text("\n".join(lines), encoding="utf-8")

    return output_file


if __name__ == "__main__":
    out = build_backtest_panel_from_inventory_files(
        data_dir=Path(r"E:\Pythonfiles\ProjectGamma\data"),
        output_file=Path(r"E:\Pythonfiles\ProjectGamma\data\backtest_panel_main.parquet"),
        crsp_file="crsp_dsf_final.csv",
        bs_file="BS.csv",
        sentiment_file="stocktwits_daily_sentiment_panel.csv",
        gex_file="underlying_gex_daily_with_permno.parquet",
        linked_file="linked_secids_top_liquid_2012_2024.parquet",
        sentiment_trading_date_alignment="backward",
        sentiment_max_alignment_days=7,
        mii_file="MII.csv",  # or None to skip
        compress_output="zstd",
    )
    print(f"Saved to: {out}")