from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Literal

import duckdb
import pandas as pd
import wrds
from tqdm.auto import tqdm

FileType = Literal["csv", "parquet"]


@dataclass
class DataFetcherConfig:
    wrds_username: str = "sniperw"
    data_dir: str = "data"
    start_date: str = "2000-01-01"
    end_date: str = "2025-12-31"
    start_year: int = 2024
    end_year: int = 2024
    file_type: FileType = "parquet"
    compression: str = "zstd"
    replace: bool = False
    crsp_permno_chunk_size: int = 500
    optionm_secid_chunk_size: int = 25
    min_abs_prc: Optional[float] = 5.0
    include_ticker_fallback: bool = False
    keep_intermediate_csv: bool = False


class DataFetcher:
    def __init__(self, config: Optional[DataFetcherConfig] = None):
        self.cfg = config or DataFetcherConfig()
        self.data_dir = Path(self.cfg.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db: Optional[wrds.Connection] = None
        self.connect()

    def connect(self) -> wrds.Connection:
        if self.db is None:
            self.db = wrds.Connection(wrds_username=self.cfg.wrds_username)
        return self.db

    def close(self) -> None:
        if self.db is not None:
            try:
                self.db.close()
            except Exception:
                pass
            self.db = None

    def path(self, *parts: str) -> Path:
        p = self.data_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def exists_and_skip(self, filepath: Path, replace: Optional[bool] = None) -> bool:
        replace = self.cfg.replace if replace is None else replace
        return filepath.exists() and not replace

    def output_ext(self) -> str:
        return "parquet" if self.cfg.file_type == "parquet" else "csv"

    def monthly_dir(self, name: str) -> Path:
        d = self.path(name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def yearly_dir(self, name: str) -> Path:
        d = self.path(name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def chunk_list(values: Sequence[int], chunk_size: int) -> list[list[int]]:
        return [list(values[i:i + chunk_size]) for i in range(0, len(values), chunk_size)]

    @staticmethod
    def sql_in_clause(values: Sequence[int | float | str]) -> str:
        out = []
        for v in values:
            if isinstance(v, str):
                out.append(f"'{v}'")
            elif isinstance(v, float):
                if math.isnan(v):
                    continue
                out.append(str(int(v)) if float(v).is_integer() else str(v))
            else:
                out.append(str(v))
        return ",".join(out)

    def iter_years(self) -> range:
        return range(self.cfg.start_year, self.cfg.end_year + 1)

    def iter_months(self) -> list[tuple[int, int]]:
        return [(y, m) for y in self.iter_years() for m in range(1, 13)]

    def write_df(self, df: pd.DataFrame, filepath: Path, file_type: Optional[FileType] = None) -> Path:
        file_type = file_type or self.cfg.file_type
        filepath.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        try:
            rel = con.from_df(df)
            if file_type == "csv":
                rel.write_csv(str(filepath))
            elif file_type == "parquet":
                rel.to_parquet(str(filepath), compression=self.cfg.compression)
            else:
                raise ValueError(f"Unsupported file_type: {file_type}")
        finally:
            con.close()
        return filepath

    def read_df(self, filepath: Path, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
        if filepath.suffix.lower() == ".csv":
            return pd.read_csv(filepath, parse_dates=parse_dates)
        if filepath.suffix.lower() == ".parquet":
            return pd.read_parquet(filepath)
        raise ValueError(f"Unsupported file: {filepath}")

    def combine_csvs_to_parquet_with_duckdb(
        self,
        output_path: Path,
        cast_sql: str,
        compression: Optional[str] = None,
    ) -> Path:
        compression = compression or self.cfg.compression
        con = duckdb.connect()
        try:
            con.execute(f"""
                COPY (
                    {cast_sql}
                )
                TO '{output_path.as_posix()}'
                (FORMAT PARQUET, COMPRESSION '{compression}')
            """)
        finally:
            con.close()
        return output_path

    def raw_sql(self, sql: str, date_cols: Optional[list[str]] = None) -> pd.DataFrame:
        if self.db is None:
            self.connect()
        return self.db.raw_sql(sql, date_cols=date_cols or [])

    def preview_table(self, library: str, table: str, n: int = 5) -> pd.DataFrame:
        q = f"select * from {library}.{table} limit {n}"
        df = self.raw_sql(q)
        print(f"\n=== {library}.{table} ===")
        print(df.columns.tolist())
        print(df.head())
        return df

    def list_tables(self, library: str) -> list[str]:
        return self.connect().list_tables(library=library)

    def log_parquet_inventory(
            self,
            output_file: Optional[Path] = None,
            recursive: bool = True,
    ) -> Path:
        output_file = output_file or self.path("parquet_inventory_log.txt")
        pattern = "**/*.parquet" if recursive else "*.parquet"
        parquet_files = sorted(self.data_dir.glob(pattern))

        con = duckdb.connect()

        def _fmt_value(x):
            if x is None:
                return "NULL"
            if isinstance(x, float):
                return f"{x:.6g}"
            return str(x)

        lines = []
        lines.append(f"Data directory: {self.data_dir.resolve()}")
        lines.append(f"Parquet files found: {len(parquet_files)}")
        lines.append("")

        if not parquet_files:
            lines.append("No parquet files found.")
            output_file.write_text("\n".join(lines), encoding="utf-8")
            con.close()
            return output_file

        try:
            for fp in parquet_files:
                rel = fp.relative_to(self.data_dir)

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

    def export_parquet_sample_to_csv(
            self,
            parquet_file: Path | str,
            output_file: Optional[Path | str] = None,
            n: Optional[int] = None,
            frac: Optional[float] = None,
            order_by: Optional[str] = None,
            replace: Optional[bool] = None,
    ) -> Path:
        replace = self.cfg.replace if replace is None else replace
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

        if self.exists_and_skip(output_file, replace):
            return output_file

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

    @property
    def crsp_stocknames_file(self) -> Path:
        return self.path(f"crsp_common_stocknames_{self.cfg.start_year}_{self.cfg.end_year}.parquet")

    @property
    def crsp_id_master_file(self) -> Path:
        return self.path(f"crsp_id_master_{self.cfg.start_year}_{self.cfg.end_year}.parquet")

    @property
    def crsp_daily_file(self) -> Path:
        return self.path(f"crsp_daily_common_{self.cfg.start_year}_{self.cfg.end_year}.parquet")

    @property
    def crsp_liquidity_file(self) -> Path:
        return self.path(f"crsp_liquidity_panel_{self.cfg.start_year}_{self.cfg.end_year}.parquet")

    @property
    def crsp_top_liquid_file(self) -> Path:
        return self.path(f"crsp_top_liquid_universe_{self.cfg.start_year}_{self.cfg.end_year}.parquet")

    @property
    def optionm_secnmd_file(self) -> Path:
        return self.path(f"optionm_secnmd_{self.cfg.start_year}_{self.cfg.end_year}.parquet")

    @property
    def crsp_optionm_link_file(self) -> Path:
        return self.path(f"crsp_optionm_link_{self.cfg.start_year}_{self.cfg.end_year}.parquet")

    @property
    def crsp_optionm_link_dominant_file(self) -> Path:
        # return self.path(f"crsp_optionm_link_dominant_{self.cfg.start_year}_{self.cfg.end_year}.parquet")
        # temporary misalignment
        # todo
        return self.path(f"crsp_optionm_link_dominant_2000_2025.parquet")

    @property
    def linked_secids_file(self) -> Path:
        return self.path(f"linked_secids_top_liquid_{self.cfg.start_year}_{self.cfg.end_year}.parquet")

    @property
    def opprcd_final_file(self) -> Path:
        return self.path(f"opprcd_linked_{self.cfg.start_year}_{self.cfg.end_year}.parquet")

    @property
    def daily_net_gamma_file(self) -> Path:
        return self.path(f"daily_net_gamma_{self.cfg.start_year}_{self.cfg.end_year}.parquet")

    def get_common_stock_permnos(self) -> list[int]:
        sql = f"""
            select distinct permno
            from crsp.stocknames
            where shrcd in (10, 11)
              and exchcd in (1, 2, 3)
              and nameenddt >= '{self.cfg.start_date}'
              and namedt <= '{self.cfg.end_date}'
            order by permno
        """
        df = self.raw_sql(sql)
        return df["permno"].dropna().astype(int).tolist()

    def fetch_crsp_stocknames(self, replace: Optional[bool] = None) -> Path:
        output = self.crsp_stocknames_file
        if self.exists_and_skip(output, replace):
            return output
        sql = f"""
            select
                permno, permco, ticker, ncusip, cusip, comnam, siccd,
                shrcd, exchcd, hexcd, shrcls, namedt, nameenddt, st_date, end_date
            from crsp.stocknames
            where shrcd in (10, 11)
              and exchcd in (1, 2, 3)
              and nameenddt >= '{self.cfg.start_date}'
              and namedt <= '{self.cfg.end_date}'
        """
        df = self.raw_sql(sql, date_cols=["namedt", "nameenddt", "st_date", "end_date"])
        return self.write_df(df, output, file_type="parquet")

    def build_crsp_id_master(self, replace: Optional[bool] = None) -> Path:
        output = self.crsp_id_master_file
        if self.exists_and_skip(output, replace):
            return output
        df = pd.read_parquet(self.crsp_stocknames_file)
        for col in ["ticker", "ncusip", "cusip", "comnam"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        df["cusip_hist"] = df["ncusip"].replace({"NAN": pd.NA, "NONE": pd.NA})
        if "cusip" in df.columns:
            df["cusip_hist"] = df["cusip_hist"].fillna(df["cusip"])
        df["cusip8"] = df["cusip_hist"].astype(str).str[:8]
        df["cusip6"] = df["cusip_hist"].astype(str).str[:6]
        keep_cols = [
            "permno", "permco", "ticker", "ncusip", "cusip", "cusip_hist",
            "cusip8", "cusip6", "comnam", "siccd", "shrcd", "exchcd",
            "hexcd", "shrcls", "namedt", "nameenddt",
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]
        out = (
            df[keep_cols]
            .drop_duplicates()
            .sort_values(["permno", "namedt", "nameenddt"])
            .reset_index(drop=True)
        )
        return self.write_df(out, output, file_type="parquet")

    def fetch_crsp_dsf(
            self,
            permno_list: Optional[Sequence[int]] = None,
            replace: Optional[bool] = None,
            output_name: str = "crsp_dsf_monthly",
            combine_final: bool = True,
    ) -> Path:
        replace = self.cfg.replace if replace is None else replace
        out_dir = self.monthly_dir(output_name)

        if permno_list is None:
            permno_list = self.get_common_stock_permnos()

        permno_list = [int(x) for x in permno_list if pd.notna(x)]
        permno_chunks = self.chunk_list(permno_list, self.cfg.crsp_permno_chunk_size)

        month_pbar = tqdm(self.iter_months(), desc="CRSP DSF months", unit="month")

        for year, month in month_pbar:
            ym = f"{year}_{month:02d}"
            out_file = out_dir / f"crsp_dsf_{ym}.parquet"

            if self.exists_and_skip(out_file, replace):
                month_pbar.set_postfix_str(f"skip {ym}")
                continue

            start_date = pd.Timestamp(year=year, month=month, day=1)
            end_date = start_date + pd.offsets.MonthEnd(1)

            monthly_parts = []

            chunk_pbar = tqdm(
                permno_chunks,
                desc=f"{year}-{month:02d}",
                unit="chunk",
                leave=False,
            )

            for sub_permnos in chunk_pbar:
                in_clause = self.sql_in_clause(sub_permnos)

                prc_filter = ""
                if self.cfg.min_abs_prc is not None:
                    prc_filter = f" and abs(prc) >= {float(self.cfg.min_abs_prc)}"

                sql = f"""
                    select
                        permno,
                        date,
                        prc,
                        ret,
                        retx,
                        shrout,
                        vol,
                        bidlo,
                        askhi
                    from crsp.dsf
                    where date between '{start_date.date()}' and '{end_date.date()}'
                      and permno in ({in_clause})
                      {prc_filter}
                """

                df_part = self.raw_sql(sql, date_cols=["date"])
                monthly_parts.append(df_part)

            if monthly_parts:
                df_month = pd.concat(monthly_parts, ignore_index=True)
            else:
                df_month = pd.DataFrame(
                    columns=["permno", "date", "prc", "ret", "retx", "shrout", "vol", "bidlo", "askhi"]
                )

            self.write_df(df_month, out_file, file_type="parquet")
            month_pbar.set_postfix_str(f"saved {ym} ({len(df_month):,} rows)")

        if combine_final:
            return self.combine_monthly_crsp_files(monthly_subdir=output_name, replace=replace)
        return out_dir

    def combine_monthly_crsp_files(
            self,
            monthly_subdir: str = "crsp_dsf_monthly",
            replace: Optional[bool] = None,
    ) -> Path:
        replace = self.cfg.replace if replace is None else replace
        output_path = self.crsp_daily_file

        if self.exists_and_skip(output_path, replace):
            return output_path

        input_dir = self.path(monthly_subdir)
        pq_glob = input_dir / "crsp_dsf_*.parquet"
        files = sorted(input_dir.glob("crsp_dsf_*.parquet"))
        if not files:
            raise FileNotFoundError(f"No monthly parquet files found in {input_dir}")

        cast_sql = f"""
            SELECT
                CAST(permno AS BIGINT) AS permno,
                CAST(date AS DATE) AS date,
                CAST(prc AS DOUBLE) AS prc,
                CAST(ret AS DOUBLE) AS ret,
                CAST(retx AS DOUBLE) AS retx,
                CAST(shrout AS DOUBLE) AS shrout,
                CAST(vol AS DOUBLE) AS vol,
                CAST(bidlo AS DOUBLE) AS bidlo,
                CAST(askhi AS DOUBLE) AS askhi
            FROM read_parquet('{pq_glob.as_posix()}')
            ORDER BY date, permno
        """

        return self.combine_csvs_to_parquet_with_duckdb(
            output_path=output_path,
            cast_sql=cast_sql,
            compression=self.cfg.compression,
        )

    def build_crsp_liquidity_panel(
            self,
            crsp_daily_file: Optional[Path] = None,
            output_file: Optional[Path] = None,
            trailing_days: int = 21,
            replace: Optional[bool] = None,
    ) -> Path:
        replace = self.cfg.replace if replace is None else replace
        crsp_daily_file = crsp_daily_file or self.crsp_daily_file
        output_file = output_file or self.crsp_liquidity_file

        if self.exists_and_skip(output_file, replace):
            return output_file

        con = duckdb.connect()
        try:
            source = crsp_daily_file.as_posix()
            source_sql = f"read_parquet('{source}')"

            con.execute(f"""
                COPY (
                    WITH base AS (
                        SELECT
                            CAST(permno AS BIGINT) AS permno,
                            CAST(date AS DATE) AS date,
                            CAST(prc AS DOUBLE) AS prc,
                            CAST(vol AS DOUBLE) AS vol,
                            CAST(shrout AS DOUBLE) AS shrout,
                            CAST(bidlo AS DOUBLE) AS bidlo,
                            CAST(askhi AS DOUBLE) AS askhi
                        FROM {source_sql}
                    ),
                    feats AS (
                        SELECT
                            permno,
                            date,
                            abs(prc) * vol AS dollar_vol,
                            CASE
                                WHEN shrout IS NOT NULL AND shrout > 0
                                THEN vol / (shrout * 1000.0)
                                ELSE NULL
                            END AS turnover,
                            CASE
                                WHEN askhi IS NOT NULL AND bidlo IS NOT NULL
                                THEN askhi - bidlo
                                ELSE NULL
                            END AS spread,
                            CASE
                                WHEN prc IS NOT NULL AND abs(prc) > 0
                                     AND askhi IS NOT NULL AND bidlo IS NOT NULL
                                THEN (askhi - bidlo) / abs(prc)
                                ELSE NULL
                            END AS rel_spread
                        FROM base
                    ),
                    roll AS (
                        SELECT
                            permno,
                            date,
                            dollar_vol,
                            turnover,
                            spread,
                            rel_spread,
                            AVG(dollar_vol) OVER (
                                PARTITION BY permno
                                ORDER BY date
                                ROWS BETWEEN {trailing_days - 1} PRECEDING AND CURRENT ROW
                            ) AS avg_dollar_vol_{trailing_days}d,
                            AVG(turnover) OVER (
                                PARTITION BY permno
                                ORDER BY date
                                ROWS BETWEEN {trailing_days - 1} PRECEDING AND CURRENT ROW
                            ) AS avg_turnover_{trailing_days}d,
                            AVG(rel_spread) OVER (
                                PARTITION BY permno
                                ORDER BY date
                                ROWS BETWEEN {trailing_days - 1} PRECEDING AND CURRENT ROW
                            ) AS avg_rel_spread_{trailing_days}d,
                            date_trunc('month', date) AS month
                        FROM feats
                    )
                    SELECT *
                    FROM roll
                    ORDER BY date, permno
                )
                TO '{output_file.as_posix()}'
                (FORMAT PARQUET, COMPRESSION '{self.cfg.compression}')
            """)
        finally:
            con.close()

        return output_file

    def build_top_liquid_stock_universe(
            self,
            liquidity_file: Optional[Path] = None,
            output_file: Optional[Path] = None,
            top_pct: float = 0.30,
            ranking_col: str = "avg_dollar_vol_21d",
            replace: Optional[bool] = None,
            monthly_snapshot: str = "last",
    ) -> Path:
        replace = self.cfg.replace if replace is None else replace
        liquidity_file = liquidity_file or self.crsp_liquidity_file

        if not (0 < top_pct <= 1):
            raise ValueError("top_pct must be in (0, 1].")

        if monthly_snapshot not in {"last", "first"}:
            raise ValueError("monthly_snapshot must be 'last' or 'first'.")

        if output_file is None:
            pct_str = str(top_pct).replace(".", "p")
            output_file = self.path(
                f"crsp_top_liquid_universe_{self.cfg.start_year}_{self.cfg.end_year}.parquet"
            )

        if self.exists_and_skip(output_file, replace):
            return output_file

        order_direction = "DESC" if monthly_snapshot == "last" else "ASC"

        con = duckdb.connect()
        try:
            con.execute(f"""
                COPY (
                    WITH base AS (
                        SELECT *
                        FROM read_parquet('{liquidity_file.as_posix()}')
                        WHERE {ranking_col} IS NOT NULL
                    ),
                    one_row_per_permno_month AS (
                        SELECT *
                        FROM (
                            SELECT
                                *,
                                ROW_NUMBER() OVER (
                                    PARTITION BY month, permno
                                    ORDER BY date {order_direction}
                                ) AS month_pick
                            FROM base
                        )
                        WHERE month_pick = 1
                    ),
                    ranked AS (
                        SELECT
                            *,
                            ROW_NUMBER() OVER (
                                PARTITION BY month
                                ORDER BY {ranking_col} DESC, permno
                            ) AS rn,
                            COUNT(*) OVER (
                                PARTITION BY month
                            ) AS n_in_month
                        FROM one_row_per_permno_month
                    )
                    SELECT *
                    FROM ranked
                    WHERE rn <= CEIL(n_in_month * {top_pct})
                    ORDER BY month, rn
                )
                TO '{output_file.as_posix()}'
                (FORMAT PARQUET, COMPRESSION '{self.cfg.compression}')
            """)
        finally:
            con.close()

        return output_file

    def export_secids_to_txt(
            self,
            parquet_file: Path | str,
            output_file: Optional[Path | str] = None,
            secid_col: str = "secid",
            replace: Optional[bool] = None,
            sort_values: bool = True,
            drop_duplicates: bool = True,
    ) -> Path:
        replace = self.cfg.replace if replace is None else replace
        parquet_file = Path(parquet_file)

        if not parquet_file.exists():
            raise FileNotFoundError(f"Parquet file does not exist: {parquet_file}")

        if output_file is None:
            output_file = parquet_file.with_suffix(".txt")
        output_file = Path(output_file)

        if self.exists_and_skip(output_file, replace):
            return output_file

        con = duckdb.connect()
        try:
            df = con.execute(f"""
                SELECT {secid_col} AS secid
                FROM read_parquet('{parquet_file.as_posix()}')
                WHERE {secid_col} IS NOT NULL
            """).fetchdf()
        finally:
            con.close()

        secids = df["secid"].astype("Int64").dropna().astype(int)

        if drop_duplicates:
            secids = pd.Series(secids.unique())

        if sort_values:
            secids = secids.sort_values(ignore_index=True)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("\n".join(secids.astype(str).tolist()), encoding="utf-8")

        return output_file

    def extract_optionm_secnmd(self, replace: Optional[bool] = None) -> Path:
        output = self.optionm_secnmd_file
        if self.exists_and_skip(output, replace):
            return output
        sql = f"""
            select
                secid,
                effect_date,
                cusip,
                ticker,
                class,
                issuer,
                issue,
                sic
            from optionm_all.secnmd
            where effect_date >= '{self.cfg.start_date}'
        """
        df = self.raw_sql(sql, date_cols=["effect_date"])
        for col in ["cusip", "ticker", "class", "issuer", "issue"]:
            if col in df.columns:
                df[col] = df[col].astype("string").str.strip().str.upper()
        df["cusip8"] = df["cusip"].str[:8]
        df["cusip6"] = df["cusip"].str[:6]
        return self.write_df(df, output, file_type="parquet")

    def build_crsp_optionm_link(self, replace: Optional[bool] = None) -> tuple[Path, Path]:
        output_all = self.crsp_optionm_link_file
        output_dom = self.crsp_optionm_link_dominant_file
        if output_all.exists() and output_dom.exists() and not (self.cfg.replace if replace is None else replace):
            return output_all, output_dom
        crsp = pd.read_parquet(self.crsp_id_master_file)
        opt = pd.read_parquet(self.optionm_secnmd_file)
        for df in (crsp, opt):
            for col in df.columns:
                if df[col].dtype == object or str(df[col].dtype).startswith("string"):
                    df[col] = df[col].astype("string").str.strip().str.upper()
        if "cusip_hist" not in crsp.columns:
            crsp["cusip_hist"] = crsp["ncusip"]
            if "cusip" in crsp.columns:
                crsp["cusip_hist"] = crsp["cusip_hist"].fillna(crsp["cusip"])
        crsp["cusip8"] = crsp["cusip_hist"].astype("string").str[:8]
        crsp_c = crsp[crsp["cusip8"].notna() & (crsp["cusip8"] != "")]
        opt_c = opt[opt["cusip8"].notna() & (opt["cusip8"] != "")]
        link_cusip = crsp_c.merge(
            opt_c,
            on="cusip8",
            how="inner",

            suffixes=("_crsp", "_opt"),
        ).copy()
        link_cusip["link_method"] = "cusip8"
        link_cusip = (
            link_cusip
            .sort_values(["permno", "secid", "namedt", "effect_date"])
            .drop_duplicates(subset=["permno", "secid"], keep="first")
            .reset_index(drop=True)
        )
        pieces = [link_cusip]
        if self.cfg.include_ticker_fallback:
            crsp_t = crsp[crsp["ticker"].notna() & (crsp["ticker"] != "")]
            opt_t = opt[opt["ticker"].notna() & (opt["ticker"] != "")]
            link_ticker = crsp_t.merge(
                opt_t,
                on="ticker",
                how="inner",
                suffixes=("_crsp", "_opt"),
            ).copy()
            link_ticker["link_method"] = "ticker"
            used = set(zip(link_cusip["permno"], link_cusip["secid"]))
            link_ticker = link_ticker[
                ~link_ticker.apply(lambda r: (r["permno"], r["secid"]) in used, axis=1)
            ].copy()
            link_ticker = (
                link_ticker
                .sort_values(["permno", "secid", "namedt", "effect_date"])
                .drop_duplicates(subset=["permno", "secid"], keep="first")
                .reset_index(drop=True)
            )
            pieces.append(link_ticker)
        link = pd.concat(pieces, ignore_index=True, sort=False)
        link["link_priority"] = link["link_method"].map({"cusip8": 1, "ticker": 2}).fillna(99)
        dominant = (
            link
            .sort_values(["permno", "link_priority", "namedt", "effect_date"])
            .drop_duplicates(subset=["permno"], keep="first")
            .reset_index(drop=True)
        )
        self.write_df(link, output_all, file_type="parquet")
        self.write_df(dominant, output_dom, file_type="parquet")
        return output_all, output_dom

    def build_linked_secid_file_from_top_liquid(
        self,
        top_liquid_file: Optional[Path] = None,
        link_file: Optional[Path] = None,
        output_file: Optional[Path] = None,
        replace: Optional[bool] = None,
    ) -> Path:
        replace = self.cfg.replace if replace is None else replace
        top_liquid_file = top_liquid_file or self.crsp_top_liquid_file
        link_file = link_file or self.crsp_optionm_link_dominant_file
        output_file = output_file or self.linked_secids_file
        if self.exists_and_skip(output_file, replace):
            return output_file
        con = duckdb.connect()
        try:
            con.execute(f"""
                COPY (
                    WITH liquid_permnos AS (
                        SELECT DISTINCT CAST(permno AS BIGINT) AS permno
                        FROM read_parquet('{top_liquid_file.as_posix()}')
                    ),
                    links AS (
                        SELECT
                            CAST(permno AS BIGINT) AS permno,
                            CAST(secid AS BIGINT) AS secid,
                            link_method
                        FROM read_parquet('{link_file.as_posix()}')
                    )
                    SELECT DISTINCT l.permno, l.secid, l.link_method
                    FROM links l
                    INNER JOIN liquid_permnos p
                        ON l.permno = p.permno
                    ORDER BY l.permno, l.secid
                )
                TO '{output_file.as_posix()}'
                (FORMAT PARQUET, COMPRESSION '{self.cfg.compression}')
            """)
        finally:
            con.close()
        return output_file

    def fetch_opprcd(
            self,
            secid_file: Optional[Path] = None,
            replace: Optional[bool] = None,
            final_output_file: Optional[Path] = None,
    ) -> Path:
        replace = self.cfg.replace if replace is None else replace
        secid_file = secid_file or self.linked_secids_file
        final_output_file = final_output_file or self.opprcd_final_file

        if self.exists_and_skip(final_output_file, replace):
            return final_output_file

        secid_df = pd.read_parquet(secid_file)
        secid_list = sorted(secid_df["secid"].dropna().astype(int).unique().tolist())
        if not secid_list:
            raise ValueError("No SECIDs found for opprcd fetch.")

        chunk_dir = self.path("opprcd_chunks")
        chunk_dir.mkdir(parents=True, exist_ok=True)

        select_cols = [
            "secid",
            "date",
            "exdate",
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
            "contract_size",
            "ss_flag",
            "forward_price",
            "expiry_indicator",
            "root",
            "suffix",
        ]

        month_specs = []
        secid_ranges = list(range(0, len(secid_list), self.cfg.optionm_secid_chunk_size))

        for year, month in self.iter_months():
            for chunk_idx, start_i in enumerate(secid_ranges):
                end_i = min(start_i + self.cfg.optionm_secid_chunk_size, len(secid_list))
                month_specs.append((year, month, chunk_idx, start_i, end_i))

        outer_pbar = tqdm(month_specs, desc="opprcd month-chunks", unit="chunk")

        for year, month, chunk_idx, start_i, end_i in outer_pbar:
            sub = secid_list[start_i:end_i]
            chunk_file = chunk_dir / f"opprcd_{year}_{month:02d}_chunk_{chunk_idx:05d}.parquet"

            if self.exists_and_skip(chunk_file, replace):
                outer_pbar.set_postfix_str(f"skip {year}-{month:02d} c{chunk_idx:05d}")
                continue

            start_date = pd.Timestamp(year=year, month=month, day=1)
            end_date = start_date + pd.offsets.MonthEnd(1)

            table_name = f"opprcd{year}"
            in_clause = self.sql_in_clause(sub)

            sql = f"""
                select {", ".join(select_cols)}
                from optionm_all.{table_name}
                where secid in ({in_clause})
                  and date between '{start_date.date()}' and '{end_date.date()}'
            """

            df_part = self.raw_sql(sql, date_cols=["date", "exdate"])
            self.write_df(df_part, chunk_file, file_type="parquet")
            outer_pbar.set_postfix_str(f"saved {year}-{month:02d} c{chunk_idx:05d}")

        if self.exists_and_skip(final_output_file, replace):
            return final_output_file

        chunk_files = sorted(chunk_dir.glob("opprcd_*.parquet"))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk parquet files found in {chunk_dir}")

        pq_glob = chunk_dir / "opprcd_*.parquet"
        con = duckdb.connect()
        try:
            con.execute(f"""
                COPY (
                    SELECT
                        CAST(secid AS BIGINT) AS secid,
                        CAST(date AS DATE) AS date,
                        CAST(exdate AS DATE) AS exdate,
                        cp_flag,
                        CAST(strike_price AS DOUBLE) AS strike_price,
                        CAST(best_bid AS DOUBLE) AS best_bid,
                        CAST(best_offer AS DOUBLE) AS best_offer,
                        CAST(volume AS DOUBLE) AS volume,
                        CAST(open_interest AS DOUBLE) AS open_interest,
                        CAST(impl_volatility AS DOUBLE) AS impl_volatility,
                        CAST(delta AS DOUBLE) AS delta,
                        CAST(gamma AS DOUBLE) AS gamma,
                        CAST(vega AS DOUBLE) AS vega,
                        CAST(theta AS DOUBLE) AS theta,
                        CAST(optionid AS BIGINT) AS optionid,
                        CAST(cfadj AS DOUBLE) AS cfadj,
                        CAST(contract_size AS DOUBLE) AS contract_size,
                        CAST(ss_flag AS INTEGER) AS ss_flag,
                        CAST(forward_price AS DOUBLE) AS forward_price,
                        expiry_indicator,
                        root,
                        suffix
                    FROM read_parquet('{pq_glob.as_posix()}')
                    ORDER BY date, secid, exdate, optionid
                )
                TO '{final_output_file.as_posix()}'
                (FORMAT PARQUET, COMPRESSION '{self.cfg.compression}')
            """)
        finally:
            con.close()

        if not self.cfg.keep_intermediate_csv:
            for f in chunk_files:
                f.unlink(missing_ok=True)

        return final_output_file

    def build_daily_net_gamma(
        self,
        opprcd_file: Optional[Path] = None,
        link_file: Optional[Path] = None,
        output_file: Optional[Path] = None,
        replace: Optional[bool] = None,
    ) -> Path:
        replace = self.cfg.replace if replace is None else replace
        opprcd_file = opprcd_file or self.opprcd_final_file
        link_file = link_file or self.crsp_optionm_link_dominant_file
        output_file = output_file or self.daily_net_gamma_file
        if self.exists_and_skip(output_file, replace):
            return output_file
        con = duckdb.connect()
        try:
            con.execute(f"""
                COPY (
                    WITH link AS (
                        SELECT
                            CAST(permno AS BIGINT) AS permno,
                            CAST(secid AS BIGINT) AS secid
                        FROM read_parquet('{link_file.as_posix()}')
                    ),
                    base AS (
                        SELECT
                            l.permno,
                            o.secid,
                            o.date,
                            o.exdate,
                            o.optionid,
                            o.cp_flag,
                            o.gamma,
                            o.open_interest,
                            o.contract_size,
                            CASE
                                WHEN o.cp_flag = 'C' THEN 1.0
                                WHEN o.cp_flag = 'P' THEN -1.0
                                ELSE NULL
                            END AS cp_sign
                        FROM read_parquet('{opprcd_file.as_posix()}') o
                        INNER JOIN link l
                            ON o.secid = l.secid
                        WHERE o.gamma IS NOT NULL
                          AND o.open_interest IS NOT NULL
                          AND o.contract_size IS NOT NULL
                    )
                    SELECT
                        permno,
                        secid,
                        date,
                        COUNT(*) AS n_opts,
                        SUM(gamma * open_interest * contract_size) AS gross_gamma_oi,
                        SUM(cp_sign * gamma * open_interest * contract_size) AS net_gamma_oi,
                        SUM(CASE WHEN cp_flag = 'C' THEN gamma * open_interest * contract_size ELSE 0 END) AS call_gamma_oi,
                        SUM(CASE WHEN cp_flag = 'P' THEN gamma * open_interest * contract_size ELSE 0 END) AS put_gamma_oi
                    FROM base
                    GROUP BY permno, secid, date
                    ORDER BY date, permno
                )
                TO '{output_file.as_posix()}'
                (FORMAT PARQUET, COMPRESSION '{self.cfg.compression}')
            """)
        finally:
            con.close()
        return output_file

    def summarize_link_coverage(self) -> None:
        link = pd.read_parquet(self.crsp_optionm_link_dominant_file)
        print("rows:", len(link))
        print("unique permno:", link["permno"].nunique())
        print("unique secid:", link["secid"].nunique())
        if "link_method" in link.columns:
            print(link["link_method"].value_counts(dropna=False))

    def validate_required_files(self, files: Optional[list[Path]] = None) -> None:
        files = files or [
            self.crsp_stocknames_file,
            self.crsp_id_master_file,
            self.optionm_secnmd_file,
            self.crsp_optionm_link_dominant_file,
        ]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    def run_identifier_pipeline(self) -> None:
        self.fetch_crsp_stocknames()
        self.build_crsp_id_master()
        self.extract_optionm_secnmd()
        self.build_crsp_optionm_link()

    def run_phase1_data_pipeline(self, top_pct: float = 0.30) -> None:
        self.run_identifier_pipeline()
        self.fetch_crsp_dsf(combine_final=True)
        self.build_crsp_liquidity_panel()
        self.build_top_liquid_stock_universe(top_pct=top_pct)
        self.build_linked_secid_file_from_top_liquid()
        self.fetch_opprcd()
        self.build_daily_net_gamma()

    def manifest_file(self, name: str = "manifest.parquet") -> Path:
        return self.path(name)

    def append_manifest_row(self, row: dict, manifest_name: str = "manifest.parquet") -> None:
        mf = self.manifest_file(manifest_name)
        row_df = pd.DataFrame([row])
        if mf.exists():
            old = pd.read_parquet(mf)
            new = pd.concat([old, row_df], ignore_index=True)
        else:
            new = row_df
        self.write_df(new, mf, file_type="parquet")

    def log_fetch_event(self, stage: str, status: str, extra: Optional[dict] = None) -> None:
        payload = {
            "stage": stage,
            "status": status,
            "start_date": self.cfg.start_date,
            "end_date": self.cfg.end_date,
            "file_type": self.cfg.file_type,
            "compression": self.cfg.compression,
        }
        if extra:
            payload.update(extra)
        self.append_manifest_row(payload)


if __name__ == "__main__":
    cfg = DataFetcherConfig(
        wrds_username="sniperw",
        data_dir="data",
        start_date="2024-01-01",
        end_date="2024-12-31",
        start_year=2024,
        end_year=2024,
        file_type="parquet",
        compression="zstd",
        replace=False,
        crsp_permno_chunk_size=500,
        optionm_secid_chunk_size=25,
        min_abs_prc=5.0,
        include_ticker_fallback=False,
        keep_intermediate_csv=False,
    )
    fetcher = DataFetcher(cfg)
    # fetcher.run_phase1_data_pipeline(top_pct=0.30)
    # fetcher.close()
