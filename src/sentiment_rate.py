from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import pipeline


INPUT_CSV = r"data/factors/stocktwits_output_daily/APP_History_Backup.csv"
OUTPUT_CSV = r"data/factors/stocktwits_output_daily/APP_Full_201801_202412_FinBERT_Daily_comparison.csv"

SYMBOL = "APP"
TEXT_COL = "Text"
TIME_COL = "Time"
ID_COL = "ID"


MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 128
MAX_LENGTH = 256


def build_finbert_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        top_k=None,
        device=device,
    )


def scores_to_row(scores: List[Dict[str, Any]]) -> Dict[str, float]:
    prob_map = {"positive": np.nan, "negative": np.nan, "neutral": np.nan}
    for item in scores:
        prob_map[str(item["label"]).strip().lower()] = float(item["score"])

    pos = prob_map["positive"]
    neg = prob_map["negative"]
    neu = prob_map["neutral"]

    return {
        "PositiveProb": pos,
        "NegativeProb": neg,
        "NeutralProb": neu,
        "Sentiment": pos - neg,
    }


def batched_finbert_inference(
    texts: List[str],
    clf,
    batch_size: int = 128,
    max_length: int = 256,
    show_progress: bool = True,
) -> pd.DataFrame:
    rows = []
    iterator = range(0, len(texts), batch_size)

    if show_progress:
        iterator = tqdm(
            iterator,
            total=(len(texts) + batch_size - 1) // batch_size,
            desc="FinBERT inference",
        )

    for start in iterator:
        batch = texts[start:start + batch_size]
        outputs = clf(
            batch,
            batch_size=batch_size,
            truncation=True,
            max_length=max_length,
        )
        rows.extend(scores_to_row(scores) for scores in outputs)

    return pd.DataFrame(rows)


def run_finbert_daily_single_symbol(
    input_csv: str | Path,
    output_csv: str | Path,
    symbol: str,
    text_col: str = "Text",
    time_col: str = "Time",
    id_col: str = "ID",
    batch_size: int = 128,
    max_length: int = 256,
) -> pd.DataFrame:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required_cols = [text_col, time_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if id_col not in df.columns:
        df[id_col] = np.arange(len(df), dtype=np.int64)

    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    df = df[df[time_col].notna()].copy()
    df = df[df[text_col] != ""].copy()

    df["DateDT"] = df[time_col].dt.floor("D").dt.tz_localize(None)
    texts = df[text_col].tolist()

    clf = build_finbert_pipeline()
    score_df = batched_finbert_inference(
        texts=texts,
        clf=clf,
        batch_size=batch_size,
        max_length=max_length,
        show_progress=True,
    )

    df = pd.concat([df.reset_index(drop=True), score_df], axis=1)
    df["Symbol"] = str(symbol).upper()

    daily = (
        df.groupby(["Symbol", "DateDT"], as_index=False)
        .agg(
            Posts=(id_col, "nunique"),
            AvgSentiment=("Sentiment", "mean"),
            MedianSentiment=("Sentiment", "median"),
            PositiveProbAvg=("PositiveProb", "mean"),
            NegativeProbAvg=("NegativeProb", "mean"),
            NeutralProbAvg=("NeutralProb", "mean"),
        )
        .sort_values(["Symbol", "DateDT"])
        .reset_index(drop=True)
    )

    daily["Date"] = daily["DateDT"].map(lambda x: f"{x.month}/{x.day}/{x.year}")
    daily = daily[
        [
            "Symbol",
            "Date",
            "Posts",
            "AvgSentiment",
            "MedianSentiment",
            "PositiveProbAvg",
            "NegativeProbAvg",
            "NeutralProbAvg",
        ]
    ]

    daily.to_csv(output_csv, index=False)
    return daily


if __name__ == "__main__":
    result = run_finbert_daily_single_symbol(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        symbol=SYMBOL,
        text_col=TEXT_COL,
        time_col=TIME_COL,
        id_col=ID_COL,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    print(result.head())