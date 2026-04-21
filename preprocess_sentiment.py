"""
Daily StockTwits FinBERT aggregates per ticker.

Timing note for backtests
---------------------------
Each row's ``date`` comes from the source CSV ``Date`` field (calendar day). The pipeline
does **not** encode whether that label means "posts through US equity close", "UTC
midnight–midnight", or "available before next open". If ``date`` effectively includes
after-hours posts, same-day sentiment can embed information from after the close that
CRSP ``DlyRet`` for that calendar date is measured against.

**Recommendation:** When merging into CRSP via ``panel_builder`` (forward calendar →
trade date), treat ``sent_avg`` as knowable only with at least **one trading-day lag**
relative to the return you forecast (see ``backtest_framework`` ``signal_lag_days``).
Use **two** lags for a more conservative stance if daily files are end-of-calendar-day.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_COLUMNS = [
    "ticker",
    "date",
    "Posts",
    "AvgSentiment",
    "MedianSentiment",
    "PositiveProbAvg",
    "NegativeProbAvg",
    "NeutralProbAvg",
]

FILENAME_SUFFIX = "_Full_201801_202412_FinBERT_Daily.csv"
INPUT_DIR = Path("data/factors/stocktwits_output_daily")
OUTPUT_FILE = Path("data/factors/stocktwits_daily_sentiment_panel.csv")


def combine_daily_sentiment(input_dir: Path) -> pd.DataFrame:
    """Combine all per-ticker daily FinBERT files into one table."""
    csv_files = sorted(input_dir.glob(f"*{FILENAME_SUFFIX}"))
    if not csv_files:
        raise FileNotFoundError(f"No files matching *{FILENAME_SUFFIX} under: {input_dir}")

    combined_parts: list[pd.DataFrame] = []

    for file_path in csv_files:
        ticker_from_filename = file_path.name[: -len(FILENAME_SUFFIX)]
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:  # pragma: no cover
            print(f"Skipping {file_path.name}: failed to read ({exc})")
            continue

        if df.empty:
            continue

        df = df.rename(columns={"Symbol": "ticker", "Date": "date"})
        if "ticker" not in df.columns:
            df["ticker"] = ticker_from_filename
        else:
            df["ticker"] = df["ticker"].fillna(ticker_from_filename).astype(str).str.strip()

        if "date" not in df.columns:
            print(f"Skipping {file_path.name}: missing column ['date'/'Date']")
            continue

        df["ticker"] = df["ticker"].str.upper()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df[df["date"].notna()]

        for col in TARGET_COLUMNS[2:]:
            if col not in df.columns:
                df[col] = pd.NA

        df = df[TARGET_COLUMNS]
        combined_parts.append(df)

    if not combined_parts:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    combined = pd.concat(combined_parts, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    return combined


def main() -> None:
    combined = combine_daily_sentiment(INPUT_DIR)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"Input directory: {INPUT_DIR.resolve()}")
    print(f"Matched files: {len(sorted(INPUT_DIR.glob(f'*{FILENAME_SUFFIX}')))}")
    print(f"Output rows: {len(combined)}")
    print(f"Saved: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
