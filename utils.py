from pathlib import Path
from typing import Optional

import duckdb


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
