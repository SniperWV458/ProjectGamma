"""
Pre-compute contract-level GEX factors from contract_gex.parquet.

Outputs data/gex_contract_factors.parquet with columns:
    secid, date, d2_hhi, d3_term_ratio_short, gamma_flip_level

Run once before the backtest:
    python backtest/gex_factors_builder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT / "data" / "contract_gex.parquet"
OUTPUT_PATH = ROOT / "data" / "gex_contract_factors_19.parquet"


# ---------------------------------------------------------------------------
# D2 and D3 — pure SQL aggregation via DuckDB
# ---------------------------------------------------------------------------

def _compute_d2_d3(con: duckdb.DuckDBPyConnection, contract: str) -> pd.DataFrame:
    """
    D2: GEX Herfindahl-Hirschman Index across strikes per (secid, date).
    D3: Short-term gamma ratio — |GEX from 1/2_5/6_21 DTE| / |total GEX|.
    """
    query = f"""
    WITH per_strike AS (
        SELECT secid, date, strike,
               SUM(ABS(gex_dollar_1pct)) AS abs_gex,
               SUM(CASE WHEN dte_bucket IN ('1DTE','2_5DTE','6_21DTE')
                        THEN ABS(gex_dollar_1pct) ELSE 0 END) AS short_abs_gex
        FROM read_parquet('{contract}')
        GROUP BY secid, date, strike
    ),
    totals AS (
        SELECT secid, date,
               SUM(abs_gex)       AS total_abs_gex,
               SUM(short_abs_gex) AS total_short_gex
        FROM per_strike
        GROUP BY secid, date
    )
    SELECT
        p.secid,
        p.date,
        SUM((p.abs_gex / t.total_abs_gex) * (p.abs_gex / t.total_abs_gex)) AS d2_hhi,
        t.total_short_gex / NULLIF(t.total_abs_gex, 0)                      AS d3_term_ratio_short
    FROM per_strike  p
    JOIN totals      t ON p.secid = t.secid AND p.date = t.date
    WHERE t.total_abs_gex > 0
    GROUP BY p.secid, p.date, t.total_short_gex, t.total_abs_gex
    """
    df = con.execute(query).df()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# D5 — gamma flip level (vectorised, no Python loops)
# ---------------------------------------------------------------------------

def _compute_gamma_flip(con: duckdb.DuckDBPyConnection, contract: str) -> pd.DataFrame:
    """
    Compute the gamma flip level (the strike where cumulative dealer GEX changes sign)
    entirely inside DuckDB — no per-strike data is loaded into Python memory.

    Uses window-function cumulative sum over strikes, then LAG() to detect sign changes,
    then linear interpolation.  Scales to arbitrarily large contract_gex files.
    """
    print("  computing gamma flip via DuckDB window functions ...", flush=True)
    query = f"""
    WITH per_strike AS (
        SELECT secid, date, strike,
               SUM(gex_dollar_1pct) AS net_gex_strike
        FROM read_parquet('{contract}')
        GROUP BY secid, date, strike
    ),
    cumulative AS (
        SELECT secid, date, strike,
               SUM(net_gex_strike) OVER (
                   PARTITION BY secid, date
                   ORDER BY strike
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               ) AS cumgex
        FROM per_strike
    ),
    with_lag AS (
        SELECT secid, date, strike,
               cumgex,
               LAG(cumgex) OVER (PARTITION BY secid, date ORDER BY strike) AS prev_cumgex,
               LAG(strike)  OVER (PARTITION BY secid, date ORDER BY strike) AS prev_strike
        FROM cumulative
    ),
    sign_changes AS (
        SELECT secid, date,
               prev_strike
               + (-prev_cumgex / NULLIF(cumgex - prev_cumgex, 0))
               * (strike - prev_strike)                         AS gamma_flip_level,
               ROW_NUMBER() OVER (PARTITION BY secid, date ORDER BY strike) AS rn
        FROM with_lag
        WHERE prev_cumgex IS NOT NULL
          AND cumgex * prev_cumgex < 0
    )
    SELECT secid, date, gamma_flip_level
    FROM sign_changes
    WHERE rn = 1
    """
    flip = con.execute(query).df()
    flip["date"] = pd.to_datetime(flip["date"])
    return flip


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_contract_factors(
    contract_path: Path = CONTRACT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    contract = str(contract_path)
    print(f"Building GEX contract factors from {contract_path.name} ...")

    con = duckdb.connect()

    print("  computing D2 (HHI) and D3 (term ratio) ...")
    df_d2d3 = _compute_d2_d3(con, contract)

    print("  computing D5 (gamma flip level) ...")
    df_flip = _compute_gamma_flip(con, contract)

    con.close()

    result = df_d2d3.merge(df_flip, on=["secid", "date"], how="outer")
    result = result.sort_values(["secid", "date"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}  shape={result.shape}")
    print(result.head(3).to_string())
    return result


if __name__ == "__main__":
    build_contract_factors()
