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

    keep_intermediate_csv: bool = True


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

    # =========================
    # Chunk helpers
    # =========================
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

    # =========================
    # I/O helpers
    # =========================
    def write_df(self, df: pd.DataFrame, filepath: Path, file_type: Optional[FileType] = None) -> Path:
        file_type = file_type or self.cfg.file_type
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if file_type == "csv":
            df.to_csv(filepath, index=False)
        elif file_type == "parquet":
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")
        return filepath

    def read_df(self, filepath: Path, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
        if filepath.suffix.lower() == ".csv":
            return pd.read_csv(filepath, parse_dates=parse_dates)
        if filepath.suffix.lower() == ".parquet":
            return pd.read_parquet(filepath)
        raise ValueError(f"Unsupported file: {filepath}")

    def combine_csvs_to_parquet_with_duckdb(
            self,
            input_glob: str,
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

    # =========================
    # SQL helpers
    # =========================
    def raw_sql(self, sql: str, date_cols: Optional[list[str]] = None) -> pd.DataFrame:
        db = self.connect()
        return db.raw_sql(sql, date_cols=date_cols or [])

    def preview_table(self, library: str, table: str, n: int = 5) -> pd.DataFrame:
        q = f"select * from {library}.{table} limit {n}"
        df = self.raw_sql(q)
        print(f"\n=== {library}.{table} ===")
        print(df.columns.tolist())
        print(df.head())
        return df

    def list_tables(self, library: str) -> list[str]:
        return self.connect().list_tables(library=library)

    # =========================
    # Core dataset paths
    # =========================
    @property
    def crsp_stocknames_file(self) -> Path:
        return self.path(f"crsp_common_stocknames_{self.cfg.start_year}_{self.cfg.end_year}.csv")

    @property
    def crsp_id_master_file(self) -> Path:
        return self.path(f"crsp_id_master_{self.cfg.start_year}_{self.cfg.end_year}.csv")

    @property
    def crsp_daily_file(self) -> Path:
        ext = self.output_ext()
        return self.path(f"crsp_daily_common_{self.cfg.start_year}_{self.cfg.end_year}.{ext}")

    @property
    def optionm_secnmd_file(self) -> Path:
        return self.path(f"optionm_secnmd_{self.cfg.start_year}_{self.cfg.end_year}.csv")

    @property
    def crsp_optionm_link_file(self) -> Path:
        return self.path("crsp_optionm_link.csv")

    @property
    def crsp_optionm_link_dominant_file(self) -> Path:
        return self.path("crsp_optionm_link_dominant.csv")

    @property
    def opprcd_final_file(self) -> Path:
        return self.path(f"opprcd_linked_{self.cfg.start_year}_{self.cfg.end_year}.parquet")

    @property
    def daily_net_gamma_file(self) -> Path:
        return self.path(f"daily_net_gamma_{self.cfg.start_year}_{self.cfg.end_year}.parquet")

    # =========================
    # CRSP methods
    # =========================
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
        return self.write_df(df, output, file_type="csv")

    def build_crsp_id_master(self, replace: Optional[bool] = None) -> Path:
        output = self.crsp_id_master_file
        if self.exists_and_skip(output, replace):
            return output

        df = pd.read_csv(
            self.crsp_stocknames_file,
            parse_dates=["namedt", "nameenddt", "st_date", "end_date"]
        )

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
            "hexcd", "shrcls", "namedt", "nameenddt"
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]

        out = (
            df[keep_cols]
            .drop_duplicates()
            .sort_values(["permno", "namedt", "nameenddt"])
            .reset_index(drop=True)
        )
        return self.write_df(out, output, file_type="csv")

    def fetch_crsp_dsf(
            self,
            permno_list: Optional[Sequence[int]] = None,
            replace: Optional[bool] = None,
            output_name: str = "crsp_dsf_monthly",
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
            out_file = out_dir / f"crsp_dsf_{ym}.csv"

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
                        vol
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
                    columns=["permno", "date", "prc", "ret", "retx", "shrout", "vol"]
                )

            self.write_df(df_month, out_file, file_type="csv")
            month_pbar.set_postfix_str(f"saved {ym} ({len(df_month):,} rows)")

        return out_dir

    # =========================
    # OptionMetrics methods
    # =========================
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

        return self.write_df(df, output, file_type="csv")

    def build_crsp_optionm_link(self, replace: Optional[bool] = None) -> tuple[Path, Path]:
        output_all = self.crsp_optionm_link_file
        output_dom = self.crsp_optionm_link_dominant_file

        if output_all.exists() and output_dom.exists() and not (self.cfg.replace if replace is None else replace):
            return output_all, output_dom

        crsp = pd.read_csv(self.crsp_id_master_file, parse_dates=["namedt", "nameenddt"])
        opt = pd.read_csv(self.optionm_secnmd_file, parse_dates=["effect_date"])

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
            suffixes=("_crsp", "_opt")
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
                suffixes=("_crsp", "_opt")
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

        self.write_df(link, output_all, file_type="csv")
        self.write_df(dominant, output_dom, file_type="csv")
        return output_all, output_dom

    def fetch_opprcd(self, replace: Optional[bool] = None) -> Path:
        output = self.opprcd_final_file
        if self.exists_and_skip(output, replace):
            return output
        raise NotImplementedError("#todo")

    # =========================
    # Downstream assembly
    # =========================
    def build_daily_net_gamma(self, replace: Optional[bool] = None) -> Path:
        output = self.daily_net_gamma_file
        if self.exists_and_skip(output, replace):
            return output

        raise NotImplementedError("#todo")

    # =========================
    # Diagnostics / validation
    # =========================
    def summarize_link_coverage(self) -> None:
        link = pd.read_csv(self.crsp_optionm_link_dominant_file)
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

    # =========================
    # End-to-end orchestration
    # =========================
    def run_identifier_pipeline(self) -> None:
        self.fetch_crsp_stocknames()
        self.build_crsp_id_master()
        self.extract_optionm_secnmd()
        self.build_crsp_optionm_link()

    def run_phase1_data_pipeline(self) -> None:
        self.run_identifier_pipeline()
        self.fetch_crsp_dsf()
        self.fetch_opprcd()
        self.build_daily_net_gamma()

    # =========================
    # Optional manifest / automation helpers
    # =========================
    def manifest_file(self, name: str = "manifest.csv") -> Path:
        return self.path(name)

    def append_manifest_row(self, row: dict, manifest_name: str = "manifest.csv") -> None:
        mf = self.manifest_file(manifest_name)
        row_df = pd.DataFrame([row])
        if mf.exists():
            old = pd.read_csv(mf)
            new = pd.concat([old, row_df], ignore_index=True)
        else:
            new = row_df
        new.to_csv(mf, index=False)

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
