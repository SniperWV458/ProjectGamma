from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Sequence
import pandas as pd

import duckdb


import gc

import pyarrow as pa
import pyarrow.parquet as pq



def export_parquet_sample_to_csv(
        parquet_file: Path | str,
        output_file: Optional[Path | str] = None,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        order_by: Optional[str] = None,
) -> Path:
    parquet_file = Path(parquet_file)

    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_file}")

    if (n is None and frac is None) or (n is not None and frac is not None):
        raise ValueError("Provide exactly one of n or frac.")

    if frac is not None and not (0 < frac <= 1):
        raise ValueError("frac must be in (0, 1].")

    if output_file is None:
        if n is not None:
            output_file = parquet_file.with_name(f"{parquet_file.stem}_head_{n}.csv")
        else:
            pct = int(frac * 100)
            output_file = parquet_file.with_name(f"{parquet_file.stem}_sample_{pct}pct.csv")
    output_file = Path(output_file)

    con = duckdb.connect()
    try:
        if n is not None:
            order_sql = f"ORDER BY {order_by}" if order_by else ""
            con.execute(f"""
                COPY (
                    SELECT *
                    FROM read_parquet('{parquet_file.as_posix()}')
                    {order_sql}
                    LIMIT {int(n)}
                )
                TO '{output_file.as_posix()}'
                (HEADER, DELIMITER ',')
            """)
        else:
            total_df = con.execute(f"""
                SELECT COUNT(*) AS n_rows
                FROM read_parquet('{parquet_file.as_posix()}')
            """).fetchdf()
            total_rows = int(total_df.loc[0, "n_rows"])
            sample_n = max(1, int(total_rows * frac))

            order_sql = f"ORDER BY {order_by}" if order_by else ""
            con.execute(f"""
                COPY (
                    SELECT *
                    FROM read_parquet('{parquet_file.as_posix()}')
                    {order_sql}
                    LIMIT {sample_n}
                )
                TO '{output_file.as_posix()}'
                (HEADER, DELIMITER ',')
            """)

    finally:
        con.close()

    return output_file


def log_parquet_inventory(
        directory: Path,
        output_file: Optional[Path] = None,
        recursive: bool = True,
) -> Path:
    output_file = output_file or Path(f"{directory}/parquet_inventory_log.txt")
    pattern = "**/*.parquet" if recursive else "*.parquet"
    parquet_files = sorted(directory.glob(pattern))

    con = duckdb.connect()

    def _fmt_value(x):
        if x is None:
            return "NULL"
        if isinstance(x, float):
            return f"{x:.6g}"
        return str(x)

    lines = []
    lines.append(f"Data directory: {directory.resolve()}")
    lines.append(f"Parquet files found: {len(parquet_files)}")
    lines.append("")

    if not parquet_files:
        lines.append("No parquet files found.")
        output_file.write_text("\n".join(lines), encoding="utf-8")
        con.close()
        return output_file

    try:
        for fp in parquet_files:
            rel = fp.relative_to(directory)

            lines.append("=" * 100)
            lines.append(f"FILE: {rel.as_posix()}")
            lines.append(f"PATH: {fp.resolve()}")

            try:
                file_size_mb = fp.stat().st_size / (1024 * 1024)
                lines.append(f"SIZE_MB: {file_size_mb:.3f}")
            except Exception as e:
                lines.append(f"SIZE_MB: ERROR ({e})")

            try:
                schema_df = con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{fp.as_posix()}')"
                ).fetchdf()

                count_df = con.execute(
                    f"SELECT COUNT(*) AS n_rows FROM read_parquet('{fp.as_posix()}')"
                ).fetchdf()
                n_rows = int(count_df.loc[0, "n_rows"])

                lines.append(f"ROWS: {n_rows}")
                lines.append("")
                lines.append("SCHEMA:")
                for _, row in schema_df.iterrows():
                    col_name = row["column_name"]
                    col_type = row["column_type"]
                    null_flag = row["null"]
                    lines.append(f"  - {col_name}: {col_type} | null={null_flag}")

                lines.append("")
                lines.append("COLUMN STATISTICS:")

                for _, row in schema_df.iterrows():
                    col = row["column_name"]
                    col_type = str(row["column_type"]).upper()

                    safe_col = f'"{col}"'

                    try:
                        if any(t in col_type for t in
                               ["INT", "DOUBLE", "FLOAT", "DECIMAL", "BIGINT", "SMALLINT", "HUGEINT"]):
                            stat_df = con.execute(f"""
                                SELECT
                                    COUNT(*) AS n,
                                    COUNT({safe_col}) AS non_null,
                                    COUNT(DISTINCT {safe_col}) AS n_distinct,
                                    MIN({safe_col}) AS min_val,
                                    MAX({safe_col}) AS max_val,
                                    AVG({safe_col}) AS mean_val
                                FROM read_parquet('{fp.as_posix()}')
                            """).fetchdf()

                            r = stat_df.iloc[0]
                            lines.append(
                                f"  - {col}: n={_fmt_value(r['n'])}, "
                                f"non_null={_fmt_value(r['non_null'])}, "
                                f"distinct={_fmt_value(r['n_distinct'])}, "
                                f"min={_fmt_value(r['min_val'])}, "
                                f"max={_fmt_value(r['max_val'])}, "
                                f"mean={_fmt_value(r['mean_val'])}"
                            )

                        elif "DATE" in col_type or "TIMESTAMP" in col_type:
                            stat_df = con.execute(f"""
                                SELECT
                                    COUNT(*) AS n,
                                    COUNT({safe_col}) AS non_null,
                                    COUNT(DISTINCT {safe_col}) AS n_distinct,
                                    MIN({safe_col}) AS min_val,
                                    MAX({safe_col}) AS max_val
                                FROM read_parquet('{fp.as_posix()}')
                            """).fetchdf()

                            r = stat_df.iloc[0]
                            lines.append(
                                f"  - {col}: n={_fmt_value(r['n'])}, "
                                f"non_null={_fmt_value(r['non_null'])}, "
                                f"distinct={_fmt_value(r['n_distinct'])}, "
                                f"min={_fmt_value(r['min_val'])}, "
                                f"max={_fmt_value(r['max_val'])}"
                            )

                        elif "BOOL" in col_type:
                            stat_df = con.execute(f"""
                                SELECT
                                    COUNT(*) AS n,
                                    COUNT({safe_col}) AS non_null,
                                    SUM(CASE WHEN {safe_col} THEN 1 ELSE 0 END) AS n_true,
                                    SUM(CASE WHEN NOT {safe_col} THEN 1 ELSE 0 END) AS n_false
                                FROM read_parquet('{fp.as_posix()}')
                            """).fetchdf()

                            r = stat_df.iloc[0]
                            lines.append(
                                f"  - {col}: n={_fmt_value(r['n'])}, "
                                f"non_null={_fmt_value(r['non_null'])}, "
                                f"true={_fmt_value(r['n_true'])}, "
                                f"false={_fmt_value(r['n_false'])}"
                            )

                        else:
                            stat_df = con.execute(f"""
                                SELECT
                                    COUNT(*) AS n,
                                    COUNT({safe_col}) AS non_null,
                                    COUNT(DISTINCT {safe_col}) AS n_distinct
                                FROM read_parquet('{fp.as_posix()}')
                            """).fetchdf()

                            sample_df = con.execute(f"""
                                SELECT DISTINCT {safe_col} AS val
                                FROM read_parquet('{fp.as_posix()}')
                                WHERE {safe_col} IS NOT NULL
                                LIMIT 5
                            """).fetchdf()

                            r = stat_df.iloc[0]
                            samples = ", ".join(_fmt_value(v) for v in sample_df["val"].tolist())
                            lines.append(
                                f"  - {col}: n={_fmt_value(r['n'])}, "
                                f"non_null={_fmt_value(r['non_null'])}, "
                                f"distinct={_fmt_value(r['n_distinct'])}, "
                                f"samples=[{samples}]"
                            )

                    except Exception as e:
                        lines.append(f"  - {col}: STAT ERROR ({e})")

                lines.append("")

            except Exception as e:
                lines.append(f"ERROR READING FILE: {e}")
                lines.append("")

    finally:
        con.close()

    output_file.write_text("\n".join(lines), encoding="utf-8")
    return output_file


def log_csv_inventory(
    directory: Path,
    output_file: Optional[Path] = None,
    recursive: bool = True,
    encoding: str = "utf-8",
) -> Path:
    output_file = output_file or Path(directory) / "csv_inventory_log.txt"
    pattern = "**/*.csv" if recursive else "*.csv"
    csv_files = sorted(Path(directory).glob(pattern))

    def _fmt_value(x):
        if x is None:
            return "NULL"
        if pd.isna(x):
            return "NULL"
        if isinstance(x, float):
            return f"{x:.6g}"
        return str(x)

    lines = []
    lines.append(f"Data directory: {Path(directory).resolve()}")
    lines.append(f"CSV files found: {len(csv_files)}")
    lines.append("")

    if not csv_files:
        lines.append("No csv files found.")
        output_file.write_text("\n".join(lines), encoding=encoding)
        return output_file

    for fp in csv_files:
        rel = fp.relative_to(directory)

        lines.append("=" * 100)
        lines.append(f"FILE: {rel.as_posix()}")
        lines.append(f"PATH: {fp.resolve()}")

        try:
            file_size_mb = fp.stat().st_size / (1024 * 1024)
            lines.append(f"SIZE_MB: {file_size_mb:.3f}")
        except Exception as e:
            lines.append(f"SIZE_MB: ERROR ({e})")

        try:
            df = pd.read_csv(fp, low_memory=False)

            n_rows, n_cols = df.shape
            lines.append(f"ROWS: {n_rows}")
            lines.append(f"COLUMNS: {n_cols}")
            lines.append("")
            lines.append("SCHEMA:")

            for col in df.columns:
                dtype = df[col].dtype
                null_count = int(df[col].isna().sum())
                lines.append(f"  - {col}: {dtype} | null_count={null_count}")

            lines.append("")
            lines.append("COLUMN STATISTICS:")

            for col in df.columns:
                s = df[col]
                dtype_str = str(s.dtype).upper()

                try:
                    n = len(s)
                    non_null = int(s.notna().sum())

                    if pd.api.types.is_numeric_dtype(s):
                        distinct = int(s.nunique(dropna=True))
                        min_val = s.min(skipna=True)
                        max_val = s.max(skipna=True)
                        mean_val = s.mean(skipna=True)

                        lines.append(
                            f"  - {col}: n={_fmt_value(n)}, "
                            f"non_null={_fmt_value(non_null)}, "
                            f"distinct={_fmt_value(distinct)}, "
                            f"min={_fmt_value(min_val)}, "
                            f"max={_fmt_value(max_val)}, "
                            f"mean={_fmt_value(mean_val)}"
                        )

                    elif pd.api.types.is_datetime64_any_dtype(s):
                        distinct = int(s.nunique(dropna=True))
                        min_val = s.min(skipna=True)
                        max_val = s.max(skipna=True)

                        lines.append(
                            f"  - {col}: n={_fmt_value(n)}, "
                            f"non_null={_fmt_value(non_null)}, "
                            f"distinct={_fmt_value(distinct)}, "
                            f"min={_fmt_value(min_val)}, "
                            f"max={_fmt_value(max_val)}"
                        )

                    elif pd.api.types.is_bool_dtype(s):
                        n_true = int((s == True).sum(skipna=True))
                        n_false = int((s == False).sum(skipna=True))

                        lines.append(
                            f"  - {col}: n={_fmt_value(n)}, "
                            f"non_null={_fmt_value(non_null)}, "
                            f"true={_fmt_value(n_true)}, "
                            f"false={_fmt_value(n_false)}"
                        )

                    else:
                        distinct = int(s.nunique(dropna=True))
                        samples = s.dropna().astype(str).drop_duplicates().head(5).tolist()
                        sample_str = ", ".join(_fmt_value(v) for v in samples)

                        lines.append(
                            f"  - {col}: n={_fmt_value(n)}, "
                            f"non_null={_fmt_value(non_null)}, "
                            f"distinct={_fmt_value(distinct)}, "
                            f"samples=[{sample_str}]"
                        )

                except Exception as e:
                    lines.append(f"  - {col}: STAT ERROR ({e})")

            lines.append("")

        except Exception as e:
            lines.append(f"ERROR READING FILE: {e}")
            lines.append("")

    output_file.write_text("\n".join(lines), encoding=encoding)
    return output_file

def export_secid_crsp_name_mapping(
    secid_file: str | Path,
    stocknames_file: str | Path,
    output_file: str | Path,
    replace: bool = False,
    keep_all_name_rows: bool = False,
) -> Path:
    secid_file = Path(secid_file)
    stocknames_file = Path(stocknames_file)
    output_file = Path(output_file)

    if output_file.exists() and not replace:
        return output_file

    secid_df = pd.read_parquet(secid_file).copy()
    stock_df = pd.read_parquet(stocknames_file).copy()

    if secid_df.empty:
        raise ValueError(f"No rows found in secid file: {secid_file}")
    if stock_df.empty:
        raise ValueError(f"No rows found in stocknames file: {stocknames_file}")

    secid_df["permno"] = pd.to_numeric(secid_df["permno"], errors="coerce").astype("Int64")
    secid_df["secid"] = pd.to_numeric(secid_df["secid"], errors="coerce").astype("Int64")
    stock_df["permno"] = pd.to_numeric(stock_df["permno"], errors="coerce").astype("Int64")

    for c in ["namedt", "nameenddt", "st_date", "end_date"]:
        if c in stock_df.columns:
            stock_df[c] = pd.to_datetime(stock_df[c], errors="coerce")

    stock_cols = [
        "permno",
        "permco",
        "ticker",
        "ncusip",
        "cusip",
        "comnam",
        "siccd",
        "shrcd",
        "exchcd",
        "hexcd",
        "shrcls",
        "namedt",
        "nameenddt",
        "st_date",
        "end_date",
    ]
    stock_cols = [c for c in stock_cols if c in stock_df.columns]
    stock_df = stock_df[stock_cols].copy()

    if not keep_all_name_rows:
        sort_cols = [c for c in ["end_date", "nameenddt", "st_date", "namedt"] if c in stock_df.columns]

        if sort_cols:
            stock_df = (
                stock_df.sort_values(
                    by=["permno"] + sort_cols,
                    ascending=[True] + [False] * len(sort_cols),
                    na_position="last",
                )
                .drop_duplicates(subset=["permno"], keep="first")
                .reset_index(drop=True)
            )
        else:
            stock_df = stock_df.drop_duplicates(subset=["permno"], keep="first").reset_index(drop=True)

    merged = secid_df.merge(
        stock_df,
        on="permno",
        how="left",
        validate="many_to_one" if not keep_all_name_rows else "many_to_many",
    )

    preferred_order = [
        "secid",
        "permno",
        "link_method",
        "permco",
        "ticker",
        "comnam",
        "ncusip",
        "cusip",
        "siccd",
        "shrcd",
        "exchcd",
        "hexcd",
        "shrcls",
        "namedt",
        "nameenddt",
        "st_date",
        "end_date",
    ]
    ordered_cols = [c for c in preferred_order if c in merged.columns] + [c for c in merged.columns if c not in preferred_order]
    merged = merged[ordered_cols].sort_values(["secid", "permno"], na_position="last").reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)

    return output_file

def export_ids_to_txt(
    parquet_file: Path | str,
    output_file: Optional[Path | str] = None,
    id_col: str = "secid",
    replace: bool = False,
    sort_values: bool = True,
    drop_duplicates: bool = True,
) -> Path:
    parquet_file = Path(parquet_file)

    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_file}")

    allowed = {"secid", "permno", "ticker"}
    if id_col not in allowed:
        raise ValueError(f"id_col must be one of {sorted(allowed)}, got: {id_col}")

    if output_file is None:
        output_file = parquet_file.with_name(f"{parquet_file.stem}_{id_col}.txt")
    output_file = Path(output_file)

    if output_file.exists() and not replace:
        return output_file

    quoted_col = '"' + id_col.replace('"', '""') + '"'

    con = duckdb.connect()
    try:
        df = con.execute(f"""
            SELECT {quoted_col} AS id_value
            FROM read_parquet('{parquet_file.as_posix()}')
            WHERE {quoted_col} IS NOT NULL
        """).fetchdf()
    finally:
        con.close()

    s = df["id_value"]

    if id_col in {"secid", "permno"}:
        s = pd.to_numeric(s, errors="coerce").dropna().astype("Int64")
        s = s.dropna().astype(int)
    else:
        s = (
            s.astype("string")
             .str.strip()
             .replace("", pd.NA)
             .dropna()
        )

    if drop_duplicates:
        s = pd.Series(s.unique())

    if sort_values:
        s = s.sort_values(ignore_index=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(s.astype(str).tolist()), encoding="utf-8")

    return output_file


def optionmetrics_csv_to_parquet(
    csv_path: Union[str, Path],
    parquet_path: Optional[Union[str, Path]] = None,
    *,
    compression: str = "zstd",
    parse_dates: bool = True,
    low_memory: bool = False,
) -> Path:
    csv_path = Path(csv_path)
    if parquet_path is None:
        parquet_path = csv_path.with_suffix(".parquet")
    parquet_path = Path(parquet_path)

    expected_columns = [
        "secid",
        "date",
        "symbol",
        "symbol_flag",
        "exdate",
        "last_date",
        "cp_flag",
        "strike_price",
        "best_bid",
        "best_offer",
        "volume",
        "open_interest",
        "impl_volatility",
        "delta",
        "gamma",
        "vega",
        "theta",
        "optionid",
        "cfadj",
        "am_settlement",
        "contract_size",
        "ss_flag",
        "forward_price",
        "expiry_indicator",
        "root",
        "suffix",
        "cusip",
        "ticker",
        "sic",
        "index_flag",
        "exchange_d",
        "class",
        "issue_type",
        "industry_group",
        "issuer",
        "div_convention",
        "exercise_style",
        "am_set_flag",
    ]

    dtype_map = {
        "secid": "Int64",
        "symbol": "string",
        "symbol_flag": "string",
        "cp_flag": "string",
        "strike_price": "float64",
        "best_bid": "float64",
        "best_offer": "float64",
        "volume": "float64",
        "open_interest": "float64",
        "impl_volatility": "float64",
        "delta": "float64",
        "gamma": "float64",
        "vega": "float64",
        "theta": "float64",
        "optionid": "Int64",
        "cfadj": "float64",
        "am_settlement": "float64",
        "contract_size": "float64",
        "ss_flag": "string",
        "forward_price": "float64",
        "expiry_indicator": "string",
        "root": "string",
        "suffix": "string",
        "cusip": "string",
        "ticker": "string",
        "sic": "Int64",
        "index_flag": "string",
        "exchange_d": "string",
        "class": "string",
        "issue_type": "string",
        "industry_group": "string",
        "issuer": "string",
        "div_convention": "string",
        "exercise_style": "string",
        "am_set_flag": "string",
    }

    date_cols = ["date", "exdate", "last_date"]

    df = pd.read_csv(
        csv_path,
        dtype=dtype_map,
        usecols=expected_columns,
        low_memory=low_memory,
    )

    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    if parse_dates:
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False, compression=compression)

    return parquet_path


def attach_permno_to_underlying_gex(
    link_file: str | Path,
    underlying_file: str | Path,
    output_file: str | Path | None = None,
    how: str = "left",
    validate: bool = True,
) -> pd.DataFrame:
    link_file = Path(link_file)
    underlying_file = Path(underlying_file)

    if output_file is None:
        output_file = underlying_file.with_name(
            underlying_file.stem + "_with_permno.parquet"
        )
    else:
        output_file = Path(output_file)

    # Read files
    link_df = pd.read_parquet(link_file)
    underlying_df = pd.read_parquet(underlying_file)

    # Keep only the mapping columns needed
    mapping = link_df[["secid", "permno"]].copy()

    # Standardize dtype a bit to reduce merge surprises
    mapping["secid"] = pd.to_numeric(mapping["secid"], errors="coerce").astype("Int64")
    mapping["permno"] = pd.to_numeric(mapping["permno"], errors="coerce").astype("Int64")
    underlying_df["secid"] = pd.to_numeric(underlying_df["secid"], errors="coerce").astype("Int64")

    if validate:
        # Check whether one secid maps to multiple permnos
        dup_map = (
            mapping.dropna(subset=["secid", "permno"])
            .groupby("secid")["permno"]
            .nunique()
        )
        bad = dup_map[dup_map > 1]
        if not bad.empty:
            raise ValueError(
                f"Found secid values mapping to multiple permnos. "
                f"Example bad secids: {bad.index.tolist()[:10]}"
            )

    # Deduplicate mapping in case repeated identical secid-permno rows exist
    mapping = mapping.drop_duplicates(subset=["secid"])

    # Merge
    merged = underlying_df.merge(mapping, on="secid", how=how)

    # Put permno near secid for readability
    cols = merged.columns.tolist()
    if "permno" in cols:
        cols.insert(cols.index("secid") + 1, cols.pop(cols.index("permno")))
        merged = merged[cols]

    # Save
    merged.to_parquet(output_file, index=False)

    # Diagnostics
    matched = merged["permno"].notna().sum()
    total = len(merged)
    unique_secid_in_underlying = merged["secid"].nunique(dropna=True)
    unique_secid_matched = merged.loc[merged["permno"].notna(), "secid"].nunique(dropna=True)

    print(f"Saved merged file to: {output_file}")
    print(f"Rows in underlying file: {total}")
    print(f"Rows with matched permno: {matched} / {total} ({matched / total:.2%})")
    print(f"Unique secid in underlying: {unique_secid_in_underlying}")
    print(f"Unique secid matched: {unique_secid_matched}")

    return merged


def extract_crsp_rows_for_linked_permnos(
    link_file: str | Path,
    crsp_file: str | Path,
    output_file: str | Path | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    link_file = Path(link_file)
    crsp_file = Path(crsp_file)

    if output_file is None:
        output_file = crsp_file.with_name(
            crsp_file.stem + "_linked_permnos.parquet"
        )
    else:
        output_file = Path(output_file)

    # Read linked permnos
    link_df = pd.read_parquet(link_file, columns=["permno"])
    link_df["permno"] = pd.to_numeric(link_df["permno"], errors="coerce").astype("Int64")

    permno_set = set(link_df["permno"].dropna().unique().tolist())
    if not permno_set:
        raise ValueError("No valid permno values found in link file.")

    # Read CRSP and filter
    crsp_df = pd.read_parquet(crsp_file)
    crsp_df["permno"] = pd.to_numeric(crsp_df["permno"], errors="coerce").astype("Int64")

    filtered = crsp_df[crsp_df["permno"].isin(permno_set)].copy()

    # Save
    filtered.to_parquet(output_file, index=False)

    if validate:
        print(f"Saved filtered CRSP file to: {output_file}")
        print(f"Input linked permnos: {len(permno_set)}")
        print(f"Filtered CRSP rows: {len(filtered)}")
        print(f"Matched CRSP permnos: {filtered['permno'].nunique(dropna=True)}")
        print(
            f"Date range: {filtered['date'].min()} to {filtered['date'].max()}"
            if not filtered.empty else "Filtered result is empty."
        )

    return filtered


def convert_optionmetrics_csvs_to_parquet(
    csv_paths: Sequence[Union[str, Path]],
    output_dir: Union[str, Path],
    *,
    chunksize: int = 250_000,
    compression: str = "zstd",
    compression_level: int | None = 6,
) -> list[Path]:
    """
    Convert multiple OptionMetrics CSV files into Parquet files with a fixed schema.

    Preserved schema:
        secid             -> BIGINT
        date              -> DATE
        exdate            -> DATE
        cp_flag           -> string
        strike_price      -> DOUBLE
        best_bid          -> DOUBLE
        best_offer        -> DOUBLE
        volume            -> DOUBLE
        open_interest     -> DOUBLE
        impl_volatility   -> DOUBLE
        delta             -> DOUBLE
        gamma             -> DOUBLE
        vega              -> DOUBLE
        theta             -> DOUBLE
        optionid          -> BIGINT
        cfadj             -> DOUBLE
        contract_size     -> DOUBLE
        ss_flag           -> INTEGER
        forward_price     -> DOUBLE
        expiry_indicator  -> string
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = [Path(p) for p in csv_paths]
    parquet_paths: list[Path] = []

    arrow_schema = pa.schema([
        ("secid", pa.int64()),
        ("date", pa.date32()),
        ("exdate", pa.date32()),
        ("cp_flag", pa.string()),
        ("strike_price", pa.float64()),
        ("best_bid", pa.float64()),
        ("best_offer", pa.float64()),
        ("volume", pa.float64()),
        ("open_interest", pa.float64()),
        ("impl_volatility", pa.float64()),
        ("delta", pa.float64()),
        ("gamma", pa.float64()),
        ("vega", pa.float64()),
        ("theta", pa.float64()),
        ("optionid", pa.int64()),
        ("cfadj", pa.float64()),
        ("contract_size", pa.float64()),
        ("ss_flag", pa.int32()),
        ("forward_price", pa.float64()),
        ("expiry_indicator", pa.string()),
    ])

    expected_columns = [field.name for field in arrow_schema]

    for csv_path in csv_paths:
        parquet_path = output_dir / f"{csv_path.stem}.parquet"
        if parquet_path.exists():
            parquet_path.unlink()

        writer = None
        try:
            for chunk in pd.read_csv(
                csv_path,
                chunksize=chunksize,
                low_memory=False,
            ):
                chunk.columns = [str(c).strip() for c in chunk.columns]

                missing = [c for c in expected_columns if c not in chunk.columns]
                if missing:
                    raise ValueError(
                        f"CSV {csv_path} is missing required columns: {missing}"
                    )

                chunk = chunk[expected_columns].copy()
                chunk = _cast_optionmetrics_schema(chunk)

                table = pa.Table.from_pandas(
                    chunk,
                    schema=arrow_schema,
                    preserve_index=False,
                    safe=False,
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        where=str(parquet_path),
                        schema=arrow_schema,
                        compression=compression,
                        compression_level=compression_level,
                        use_dictionary=True,
                        write_statistics=True,
                    )

                writer.write_table(table)

                del chunk, table
                gc.collect()

            if writer is not None:
                writer.close()
                writer = None

            parquet_paths.append(parquet_path)

        except Exception:
            if writer is not None:
                writer.close()
            if parquet_path.exists():
                parquet_path.unlink(missing_ok=True)
            raise

    return parquet_paths


def combine_optionmetrics_parquets(
    parquet_paths: Sequence[Union[str, Path]],
    output_parquet_path: Union[str, Path],
    *,
    compression: str = "zstd",
    compression_level: int | None = 6,
    row_group_size: int = 250_000,
    delete_input_parquets_after_success: bool = False,
) -> Path:
    """
    Combine multiple intermediate Parquet files into one final Parquet file
    by streaming row groups, preserving schema and minimizing peak memory usage.
    """
    parquet_paths = [Path(p) for p in parquet_paths]
    output_parquet_path = Path(output_parquet_path)
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if output_parquet_path.exists():
        output_parquet_path.unlink()

    expected_schema = pa.schema([
        ("secid", pa.int64()),
        ("date", pa.date32()),
        ("exdate", pa.date32()),
        ("cp_flag", pa.string()),
        ("strike_price", pa.float64()),
        ("best_bid", pa.float64()),
        ("best_offer", pa.float64()),
        ("volume", pa.float64()),
        ("open_interest", pa.float64()),
        ("impl_volatility", pa.float64()),
        ("delta", pa.float64()),
        ("gamma", pa.float64()),
        ("vega", pa.float64()),
        ("theta", pa.float64()),
        ("optionid", pa.int64()),
        ("cfadj", pa.float64()),
        ("contract_size", pa.float64()),
        ("ss_flag", pa.int32()),
        ("forward_price", pa.float64()),
        ("expiry_indicator", pa.string()),
    ])

    writer = None
    try:
        for parquet_path in parquet_paths:
            pf = pq.ParquetFile(parquet_path)

            if pf.schema_arrow != expected_schema:
                raise ValueError(
                    f"Schema mismatch in {parquet_path}.\n"
                    f"Expected:\n{expected_schema}\n\n"
                    f"Found:\n{pf.schema_arrow}"
                )

            for rg_idx in range(pf.num_row_groups):
                table = pf.read_row_group(rg_idx)

                if writer is None:
                    writer = pq.ParquetWriter(
                        where=str(output_parquet_path),
                        schema=expected_schema,
                        compression=compression,
                        compression_level=compression_level,
                        use_dictionary=True,
                        write_statistics=True,
                    )

                writer.write_table(table, row_group_size=row_group_size)

                del table
                gc.collect()

        if writer is not None:
            writer.close()
            writer = None

        if delete_input_parquets_after_success:
            for p in parquet_paths:
                p.unlink(missing_ok=True)

        return output_parquet_path

    except Exception:
        if writer is not None:
            writer.close()
        if output_parquet_path.exists():
            output_parquet_path.unlink(missing_ok=True)
        raise


def _cast_optionmetrics_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast DataFrame columns to the exact target schema.
    """
    out = df.copy()

    # BIGINT
    out["secid"] = pd.to_numeric(out["secid"], errors="coerce").astype("Int64")
    out["optionid"] = pd.to_numeric(out["optionid"], errors="coerce").astype("Int64")

    # DATE
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["exdate"] = pd.to_datetime(out["exdate"], errors="coerce").dt.date

    # STRING
    out["cp_flag"] = out["cp_flag"].astype("string")
    out["expiry_indicator"] = out["expiry_indicator"].astype("string")

    # DOUBLE
    double_cols = [
        "strike_price",
        "best_bid",
        "best_offer",
        "volume",
        "open_interest",
        "impl_volatility",
        "delta",
        "gamma",
        "vega",
        "theta",
        "cfadj",
        "contract_size",
        "forward_price",
    ]
    for col in double_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    # INTEGER
    out["ss_flag"] = pd.to_numeric(out["ss_flag"], errors="coerce").astype("Int32")

    return out
