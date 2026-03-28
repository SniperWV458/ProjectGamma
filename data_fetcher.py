from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Literal
import threading
from queue import Queue, Empty
import time

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

    def query_sql(self, sql: str, date_cols: Optional[list[str]] = None) -> pd.DataFrame:
        if self.db is None:
            self.connect()
        return self.db.raw_sql(sql, date_cols=date_cols or [])

    def preview_table(self, library: str, table: str, n: int = 5) -> pd.DataFrame:
        q = f"select * from {library}.{table} limit {n}"
        df = self.query_sql(q)
        print(f"\n=== {library}.{table} ===")
        print(df.columns.tolist())
        print(df.head())
        return df

    def list_tables(self, library: str) -> list[str]:
        return self.connect().list_tables(library=library)

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

    def _run_tasks_with_connection_pool(
            self,
            tasks: list,
            worker_fn,
            desc: str,
            n_connections: int = 5,
    ) -> list:
        if n_connections < 1:
            raise ValueError("n_connections must be >= 1")

        self.close()

        task_queue: Queue = Queue()
        for task in tasks:
            task_queue.put(task)

        results = []
        errors = []
        lock = threading.Lock()
        pbar = tqdm(total=len(tasks), desc=desc, unit="task")

        def _worker():
            db = None
            try:
                db = wrds.Connection(wrds_username=self.cfg.wrds_username)
                while True:
                    try:
                        task = task_queue.get_nowait()
                    except Empty:
                        break

                    try:
                        worker_fn(db, task, results, errors, lock)
                    except Exception as e:
                        with lock:
                            errors.append((task, e))
                    finally:
                        with lock:
                            pbar.update(1)
                        task_queue.task_done()
            finally:
                if db is not None:
                    try:
                        db.close()
                    except Exception:
                        pass

        threads = [threading.Thread(target=_worker, daemon=True) for _ in range(n_connections)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        pbar.close()

        if errors:
            task, exc = errors[0]
            raise RuntimeError(f"Task failed: {task}") from exc

        return results

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
        df = self.query_sql(sql)
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
        df = self.query_sql(sql, date_cols=["namedt", "nameenddt", "st_date", "end_date"])
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
        import threading
        from queue import Queue, Empty

        replace = self.cfg.replace if replace is None else replace
        out_dir = self.monthly_dir(output_name)

        if permno_list is None:
            permno_list = self.get_common_stock_permnos()

        permno_list = [int(x) for x in permno_list if pd.notna(x)]
        if not permno_list:
            raise ValueError("No PERMNOs found for CRSP DSF fetch.")

        permno_chunks = self.chunk_list(permno_list, self.cfg.crsp_permno_chunk_size)

        # Ensure no persistent connection occupies an extra WRDS slot.
        self.close()

        tasks: list[tuple[int, int, int, list[int], Path]] = []
        month_row_counts: dict[tuple[int, int], int] = {}

        for year, month in self.iter_months():
            ym = f"{year}_{month:02d}"
            month_dir = out_dir / ym
            month_dir.mkdir(parents=True, exist_ok=True)
            month_row_counts[(year, month)] = 0

            for chunk_idx, sub_permnos in enumerate(permno_chunks):
                chunk_file = month_dir / f"crsp_dsf_{ym}_chunk_{chunk_idx:05d}.parquet"
                tasks.append((year, month, chunk_idx, list(sub_permnos), chunk_file))

        task_queue: Queue = Queue()
        for task in tasks:
            task_queue.put(task)

        errors: list[tuple[tuple[int, int, int, list[int], Path], Exception]] = []
        state_lock = threading.Lock()
        pbar_lock = threading.Lock()

        def worker(pbar) -> None:
            db = None
            try:
                db = wrds.Connection(wrds_username=self.cfg.wrds_username)

                while True:
                    try:
                        year, month, chunk_idx, sub_permnos, chunk_file = task_queue.get_nowait()
                    except Empty:
                        break

                    try:
                        if not self.exists_and_skip(chunk_file, replace):
                            start_date = pd.Timestamp(year=year, month=month, day=1)
                            end_date = start_date + pd.offsets.MonthEnd(1)

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

                            df_part = db.raw_sql(sql, date_cols=["date"])
                            self.write_df(df_part, chunk_file, file_type="parquet")

                            with state_lock:
                                month_row_counts[(year, month)] += len(df_part)

                        with pbar_lock:
                            pbar.update(1)
                            pbar.set_postfix_str(f"{year}-{month:02d} c{chunk_idx:05d}")

                    except Exception as e:
                        with state_lock:
                            errors.append(((year, month, chunk_idx, sub_permnos, chunk_file), e))
                        with pbar_lock:
                            pbar.update(1)
                            pbar.set_postfix_str(f"ERR {year}-{month:02d} c{chunk_idx:05d}")
                    finally:
                        task_queue.task_done()

            finally:
                if db is not None:
                    try:
                        db.close()
                    except Exception:
                        pass

        n_connections = 5
        with tqdm(total=len(tasks), desc="CRSP DSF month-chunks", unit="chunk") as pbar:
            threads = [threading.Thread(target=worker, args=(pbar,), daemon=True) for _ in range(n_connections)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

        if errors:
            task, exc = errors[0]
            raise RuntimeError(
                f"CRSP DSF fetch failed for year={task[0]}, month={task[1]:02d}, chunk={task[2]:05d}"
            ) from exc

        month_pbar = tqdm(self.iter_months(), desc="Combine CRSP DSF months", unit="month")
        for year, month in month_pbar:
            ym = f"{year}_{month:02d}"
            out_file = out_dir / f"crsp_dsf_{ym}.parquet"

            if self.exists_and_skip(out_file, replace):
                month_pbar.set_postfix_str(f"skip {ym}")
                continue

            month_dir = out_dir / ym
            pq_glob = month_dir / f"crsp_dsf_{ym}_chunk_*.parquet"
            chunk_files = sorted(month_dir.glob(f"crsp_dsf_{ym}_chunk_*.parquet"))

            if chunk_files:
                con = duckdb.connect()
                try:
                    con.execute(f"""
                        COPY (
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
                        )
                        TO '{out_file.as_posix()}'
                        (FORMAT PARQUET, COMPRESSION '{self.cfg.compression}')
                    """)
                finally:
                    con.close()

                if not self.cfg.keep_intermediate_csv:
                    for f in chunk_files:
                        f.unlink(missing_ok=True)
                    try:
                        month_dir.rmdir()
                    except OSError:
                        pass
            else:
                df_month = pd.DataFrame(
                    columns=["permno", "date", "prc", "ret", "retx", "shrout", "vol", "bidlo", "askhi"]
                )
                self.write_df(df_month, out_file, file_type="parquet")

            month_pbar.set_postfix_str(f"saved {ym} ({month_row_counts[(year, month)]:,} rows)")

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
        df = self.query_sql(sql, date_cols=["effect_date"])
        for col in ["cusip", "ticker", "class", "issuer", "issue"]:
            if col in df.columns:
                df[col] = df[col].astype("string").str.strip().str.upper()
        df["cusip8"] = df["cusip"].str[:8]
        df["cusip6"] = df["cusip"].str[:6]
        return self.write_df(df, output, file_type="parquet")

    def build_crsp_optionm_link(self, replace: Optional[bool] = None) -> tuple[Path, Path]:
        output_all = self.crsp_optionm_link_file
        output_dom = self.crsp_optionm_link_dominant_file

        do_replace = self.cfg.replace if replace is None else replace
        if output_all.exists() and output_dom.exists() and not do_replace:
            return output_all, output_dom

        crsp = pd.read_parquet(self.crsp_id_master_file)
        opt = pd.read_parquet(self.optionm_secnmd_file)

        # Standardize string columns
        for df in (crsp, opt):
            for col in df.columns:
                if df[col].dtype == object or str(df[col].dtype).startswith("string"):
                    df[col] = df[col].astype("string").str.strip().str.upper()

        # Ensure CRSP has a historical CUSIP source
        if "cusip_hist" not in crsp.columns:
            crsp["cusip_hist"] = crsp["ncusip"]
            if "cusip" in crsp.columns:
                crsp["cusip_hist"] = crsp["cusip_hist"].fillna(crsp["cusip"])

        crsp["cusip8"] = crsp["cusip_hist"].astype("string").str[:8]

        # CUSIP8 link
        crsp_c = crsp[crsp["cusip8"].notna() & (crsp["cusip8"] != "")]
        opt_c = opt[opt["cusip8"].notna() & (opt["cusip8"] != "")]

        link_cusip = crsp_c.merge(
            opt_c,
            on="cusip8",
            how="inner",
            suffixes=("_crsp", "_opt"),
        ).copy()

        link_cusip["link_method"] = "cusip8"

        # Add a unified ticker column
        if "ticker_crsp" in link_cusip.columns and "ticker_opt" in link_cusip.columns:
            link_cusip["ticker"] = link_cusip["ticker_crsp"].fillna(link_cusip["ticker_opt"])
        elif "ticker_crsp" in link_cusip.columns:
            link_cusip["ticker"] = link_cusip["ticker_crsp"]
        elif "ticker_opt" in link_cusip.columns:
            link_cusip["ticker"] = link_cusip["ticker_opt"]
        else:
            link_cusip["ticker"] = pd.Series(pd.NA, index=link_cusip.index, dtype="string")

        link_cusip = (
            link_cusip
            .sort_values(["permno", "secid", "namedt", "effect_date"])
            .drop_duplicates(subset=["permno", "secid"], keep="first")
            .reset_index(drop=True)
        )

        pieces = [link_cusip]

        # Optional ticker fallback
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

            # Ensure unified ticker column exists here too
            if "ticker" not in link_ticker.columns:
                if "ticker_crsp" in link_ticker.columns and "ticker_opt" in link_ticker.columns:
                    link_ticker["ticker"] = link_ticker["ticker_crsp"].fillna(link_ticker["ticker_opt"])
                elif "ticker_crsp" in link_ticker.columns:
                    link_ticker["ticker"] = link_ticker["ticker_crsp"]
                elif "ticker_opt" in link_ticker.columns:
                    link_ticker["ticker"] = link_ticker["ticker_opt"]
                else:
                    link_ticker["ticker"] = pd.Series(pd.NA, index=link_ticker.index, dtype="string")

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

        # Final safeguard: create a single unified ticker column on the concatenated frame
        ticker_candidates = [c for c in ["ticker", "ticker_crsp", "ticker_opt"] if c in link.columns]
        if ticker_candidates:
            link["ticker"] = link[ticker_candidates[0]]
            for c in ticker_candidates[1:]:
                link["ticker"] = link["ticker"].fillna(link[c])
        else:
            link["ticker"] = pd.Series(pd.NA, index=link.index, dtype="string")

        # Optional: place ticker near the front
        preferred_front = ["permno", "secid", "ticker", "link_method"]
        existing_front = [c for c in preferred_front if c in link.columns]
        other_cols = [c for c in link.columns if c not in existing_front]
        link = link[existing_front + other_cols]

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
            # Inspect schema of the link parquet first
            schema_df = con.execute(f"""
                DESCRIBE SELECT * FROM read_parquet('{link_file.as_posix()}')
            """).fetchdf()
            cols = set(schema_df["column_name"].tolist())

            # Build a robust ticker expression depending on available columns
            if "ticker" in cols:
                ticker_expr = 'CAST("ticker" AS VARCHAR) AS ticker'
            elif "ticker_crsp" in cols and "ticker_opt" in cols:
                ticker_expr = 'COALESCE(CAST("ticker_crsp" AS VARCHAR), CAST("ticker_opt" AS VARCHAR)) AS ticker'
            elif "ticker_crsp" in cols:
                ticker_expr = 'CAST("ticker_crsp" AS VARCHAR) AS ticker'
            elif "ticker_opt" in cols:
                ticker_expr = 'CAST("ticker_opt" AS VARCHAR) AS ticker'
            else:
                ticker_expr = 'CAST(NULL AS VARCHAR) AS ticker'

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
                            {ticker_expr},
                            link_method
                        FROM read_parquet('{link_file.as_posix()}')
                    )
                    SELECT DISTINCT
                        l.permno,
                        l.secid,
                        l.ticker,
                        l.link_method
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
        ]

        tasks = []
        secid_ranges = list(range(0, len(secid_list), self.cfg.optionm_secid_chunk_size))

        for year, month in self.iter_months():
            for chunk_idx, start_i in enumerate(secid_ranges):
                end_i = min(start_i + self.cfg.optionm_secid_chunk_size, len(secid_list))
                sub = secid_list[start_i:end_i]
                chunk_file = chunk_dir / f"opprcd_{year}_{month:02d}_chunk_{chunk_idx:05d}.parquet"
                tasks.append((year, month, chunk_idx, sub, chunk_file))

        def worker_fn(db, task, results, errors, lock):
            year, month, chunk_idx, sub, chunk_file = task

            if self.exists_and_skip(chunk_file, replace):
                with lock:
                    results.append((year, month, chunk_idx, "skipped"))
                return

            start_date = pd.Timestamp(year=year, month=month, day=1)
            end_date = start_date + pd.offsets.MonthEnd(1)
            table_name = f"opprcd{year}"
            in_clause = self.sql_in_clause(sub)

            sql = f"""
                select {", ".join(select_cols)}
                from optionm_all.{table_name}
                where secid in ({in_clause})
                  and date between '{start_date.date()}' and '{end_date.date()}'
                  and open_interest > 0
                  and gamma is not null
                  and gamma <> 0
                  and contract_size is not null
                  and contract_size > 0
                  and cfadj is not null
                  and cfadj > 0
            """

            df_part = db.raw_sql(sql, date_cols=["date", "exdate"])
            self.write_df(df_part, chunk_file, file_type="parquet")

            with lock:
                results.append((year, month, chunk_idx, "saved"))

        self._run_tasks_with_connection_pool(
            tasks=tasks,
            worker_fn=worker_fn,
            desc="opprcd month-chunks",
            n_connections=1,
        )

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
                        expiry_indicator
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

    def fetch_secprd(
            self,
            secid_file: Optional[Path] = None,
            replace: Optional[bool] = None,
            final_output_file: Optional[Path] = None,
    ) -> Path:
        replace = self.cfg.replace if replace is None else replace
        secid_file = secid_file or self.linked_secids_file
        final_output_file = final_output_file or self.path(
            f"secprd_linked_{self.cfg.start_year}_{self.cfg.end_year}.parquet"
        )

        if self.exists_and_skip(final_output_file, replace):
            return final_output_file

        secid_df = pd.read_parquet(secid_file)
        secid_list = sorted(secid_df["secid"].dropna().astype(int).unique().tolist())
        if not secid_list:
            raise ValueError("No SECIDs found for secprd fetch.")

        chunk_dir = self.path("secprd_chunks")
        chunk_dir.mkdir(parents=True, exist_ok=True)

        select_cols = [
            "secid",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "return",
            "cfadj",
            "cfret",
            "shrout",
        ]

        tasks = []
        secid_ranges = list(range(0, len(secid_list), self.cfg.optionm_secid_chunk_size))

        for year, month in self.iter_months():
            for chunk_idx, start_i in enumerate(secid_ranges):
                end_i = min(start_i + self.cfg.optionm_secid_chunk_size, len(secid_list))
                sub = secid_list[start_i:end_i]
                chunk_file = chunk_dir / f"secprd_{year}_{month:02d}_chunk_{chunk_idx:05d}.parquet"
                tasks.append((year, month, chunk_idx, sub, chunk_file))

        def worker_fn(db, task, results, errors, lock):
            year, month, chunk_idx, sub, chunk_file = task

            if self.exists_and_skip(chunk_file, replace):
                with lock:
                    results.append((year, month, chunk_idx, "skipped"))
                return

            start_date = pd.Timestamp(year=year, month=month, day=1)
            end_date = start_date + pd.offsets.MonthEnd(1)
            table_name = f"secprd{year}"
            in_clause = self.sql_in_clause(sub)

            sql = f"""
                select {", ".join(select_cols)}
                from optionm_all.{table_name}
                where secid in ({in_clause})
                  and date between '{start_date.date()}' and '{end_date.date()}'
            """

            df_part = db.raw_sql(sql, date_cols=["date"])
            self.write_df(df_part, chunk_file, file_type="parquet")

            with lock:
                results.append((year, month, chunk_idx, "saved"))

        self._run_tasks_with_connection_pool(
            tasks=tasks,
            worker_fn=worker_fn,
            desc="secprd month-chunks",
            n_connections=5,
        )

        if self.exists_and_skip(final_output_file, replace):
            return final_output_file

        chunk_files = sorted(chunk_dir.glob("secprd_*.parquet"))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk parquet files found in {chunk_dir}")

        pq_glob = chunk_dir / "secprd_*.parquet"
        con = duckdb.connect()
        try:
            con.execute(f"""
                COPY (
                    SELECT
                        CAST(secid AS BIGINT) AS secid,
                        CAST(date AS DATE) AS date,
                        CAST(open AS DOUBLE) AS open,
                        CAST(high AS DOUBLE) AS high,
                        CAST(low AS DOUBLE) AS low,
                        CAST(close AS DOUBLE) AS close,
                        CAST(volume AS DOUBLE) AS volume,
                        CAST(return AS DOUBLE) AS return,
                        CAST(cfadj AS DOUBLE) AS cfadj,
                        CAST(cfret AS DOUBLE) AS cfret,
                        CAST(shrout AS DOUBLE) AS shrout
                    FROM read_parquet('{pq_glob.as_posix()}')
                    ORDER BY date, secid
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

    def fetch_wrds_signals_raw_plus(
            self,
            link_file: Optional[Path] = None,
            replace: Optional[bool] = None,
            final_output_file: Optional[Path] = None,
    ) -> Path:
        replace = self.cfg.replace if replace is None else replace
        link_file = link_file or self.linked_secids_file
        final_output_file = final_output_file or self.path(
            f"wrds_signals_raw_plus_linked_{self.cfg.start_year}_{self.cfg.end_year}.parquet"
        )

        if self.exists_and_skip(final_output_file, replace):
            return final_output_file

        link_df = pd.read_parquet(link_file)

        required_cols = {"permno", "secid", "ticker"}
        missing = required_cols - set(link_df.columns)
        if missing:
            raise ValueError(f"link_file missing required columns: {sorted(missing)}")

        link_df = link_df[["permno", "secid", "ticker"]].copy()
        link_df = link_df.dropna(subset=["permno"])
        link_df["permno"] = link_df["permno"].astype("int64")

        # Keep one row per permno for the merge back
        link_df = (
            link_df.sort_values(["permno", "secid"])
            .drop_duplicates(subset=["permno"], keep="first")
            .reset_index(drop=True)
        )

        permno_list = sorted(link_df["permno"].unique().tolist())
        if not permno_list:
            raise ValueError("No PERMNOs found in linked file.")

        chunk_dir = self.path("wrds_signals_raw_plus_chunks")
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------
        # Discover schema from WRDS catalog
        # ------------------------------------------------------------
        self.connect()
        db = self.db

        meta_sql = """
            select
                column_name,
                data_type,
                ordinal_position
            from information_schema.columns
            where table_schema = 'wrdsapps'
              and table_name = 'signals_raw_plus'
            order by ordinal_position
        """
        meta = db.raw_sql(meta_sql)
        self.close()

        if meta.empty:
            raise ValueError("Could not retrieve schema for wrdsapps.signals_raw_plus")

        cols = meta["column_name"].astype(str).tolist()
        cols_lower = {c.lower(): c for c in cols}

        if "permno" not in cols_lower:
            raise ValueError("wrdsapps.signals_raw_plus does not contain a permno column")

        # Try common date column names
        date_candidates = ["date", "datadate", "signal_date", "month_end_date", "fdate"]
        date_col = next((cols_lower[c] for c in date_candidates if c in cols_lower), None)
        if date_col is None:
            raise ValueError(
                f"Could not identify date column in wrdsapps.signals_raw_plus. "
                f"Available columns: {cols[:20]}{'...' if len(cols) > 20 else ''}"
            )

        permno_col = cols_lower["permno"]

        # Exclude identifiers and administrative fields; fetch everything else as factors
        exclude_lower = {
            permno_col.lower(),
            date_col.lower(),
            "ticker",
            "secid",
            "gvkey",
            "cusip",
            "ncusip",
            "iid",
            "siccd",
            "namedt",
            "nameendt",
            "linkdt",
            "linkenddt",
        }

        factor_cols = [c for c in cols if c.lower() not in exclude_lower]
        if not factor_cols:
            raise ValueError("No factor columns detected in wrdsapps.signals_raw_plus")

        # ------------------------------------------------------------
        # Save local link table once for DuckDB merge later
        # ------------------------------------------------------------
        link_map_file = chunk_dir / "_permno_secid_ticker_map.parquet"
        self.write_df(link_df, link_map_file, file_type="parquet")

        tasks = []
        permno_ranges = list(range(0, len(permno_list), self.cfg.crsp_permno_chunk_size))

        for year, month in self.iter_months():
            for chunk_idx, start_i in enumerate(permno_ranges):
                end_i = min(start_i + self.cfg.crsp_permno_chunk_size, len(permno_list))
                sub = permno_list[start_i:end_i]
                chunk_file = chunk_dir / f"signals_raw_plus_{year}_{month:02d}_chunk_{chunk_idx:05d}.parquet"
                tasks.append((year, month, chunk_idx, sub, chunk_file))

        select_cols_sql = ",\n            ".join(
            [permno_col, date_col] + factor_cols
        )

        def worker_fn(db, task, results, errors, lock):
            year, month, chunk_idx, sub, chunk_file = task

            if self.exists_and_skip(chunk_file, replace):
                with lock:
                    results.append((year, month, chunk_idx, "skipped"))
                return

            start_date = pd.Timestamp(year=year, month=month, day=1)
            end_date = start_date + pd.offsets.MonthEnd(1)
            in_clause = self.sql_in_clause(sub)

            sql = f"""
                select
                    {select_cols_sql}
                from wrdsapps.signals_raw_plus
                where {permno_col} in ({in_clause})
                  and {date_col} between '{start_date.date()}' and '{end_date.date()}'
            """

            df_part = db.raw_sql(sql, date_cols=[date_col])

            # Standardize id/date names for downstream combine
            rename_map = {}
            if permno_col != "permno":
                rename_map[permno_col] = "permno"
            if date_col != "date":
                rename_map[date_col] = "date"
            if rename_map:
                df_part = df_part.rename(columns=rename_map)

            # Merge local secid/ticker here so each chunk is already aligned
            if not df_part.empty:
                df_part["permno"] = df_part["permno"].astype("int64")
                df_part = df_part.merge(link_df, on="permno", how="left")

            self.write_df(df_part, chunk_file, file_type="parquet")

            with lock:
                results.append((year, month, chunk_idx, "saved"))

        self._run_tasks_with_connection_pool(
            tasks=tasks,
            worker_fn=worker_fn,
            desc="signals_raw_plus month-chunks",
            n_connections=5,
        )

        if self.exists_and_skip(final_output_file, replace):
            return final_output_file

        chunk_files = sorted(chunk_dir.glob("signals_raw_plus_*.parquet"))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk parquet files found in {chunk_dir}")

        pq_glob = chunk_dir / "signals_raw_plus_*.parquet"

        # ------------------------------------------------------------
        # Build final parquet with stable typing
        # ------------------------------------------------------------
        con = duckdb.connect()
        try:
            # Inspect one chunk to get factor column names actually written
            sample_cols = pd.read_parquet(chunk_files[0]).columns.tolist()

            base_cols = ["secid", "permno", "ticker", "date"]
            factor_cols_final = [c for c in sample_cols if c not in base_cols]

            # Cast factors to DOUBLE by default; this is usually correct for signal tables
            factor_cast_sql = ",\n                        ".join(
                [f'CAST("{c}" AS DOUBLE) AS "{c}"' for c in factor_cols_final]
            )

            select_sql = f"""
                SELECT
                    CAST(secid AS BIGINT) AS secid,
                    CAST(permno AS BIGINT) AS permno,
                    CAST(ticker AS VARCHAR) AS ticker,
                    CAST(date AS DATE) AS date
                    {"," if factor_cast_sql else ""}
                    {factor_cast_sql}
                FROM read_parquet('{pq_glob.as_posix()}')
                ORDER BY date, permno, secid
            """

            con.execute(f"""
                COPY (
                    {select_sql}
                )
                TO '{final_output_file.as_posix()}'
                (FORMAT PARQUET, COMPRESSION '{self.cfg.compression}')
            """)
        finally:
            con.close()

        if not self.cfg.keep_intermediate_csv:
            for f in chunk_files:
                f.unlink(missing_ok=True)
            link_map_file.unlink(missing_ok=True)

        return final_output_file

    def summarize_link_coverage(self) -> None:
        link = pd.read_parquet(self.crsp_optionm_link_dominant_file)
        print("rows:", len(link))
        print("unique permno:", link["permno"].nunique())
        print("unique secid:", link["secid"].nunique())
        if "link_method" in link.columns:
            print(link["link_method"].value_counts(dropna=False))

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
