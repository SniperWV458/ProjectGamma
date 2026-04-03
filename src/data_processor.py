from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union, Literal
import logging
import time

import duckdb
import pandas as pd


PathLike = Union[str, Path]


@dataclass
class OptionGammaConfig:
    option_parquet: PathLike
    spot_parquet: Optional[PathLike] = None
    output_dir: Optional[PathLike] = None

    # Core filters
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    secids: Optional[Sequence[int]] = None

    # Data quality controls
    require_standard_settlement: bool = True
    require_positive_oi: bool = True
    require_positive_contract_size: bool = True
    require_nonnull_gamma: bool = True
    require_valid_bidask: bool = False
    drop_zero_bidask_mid: bool = False
    require_spot: bool = True

    # Spot selection
    # choose from: open, high, low, close
    spot_price_field: Literal["open", "high", "low", "close"] = "close"

    # Storage / behavior
    temp_dir: Optional[PathLike] = None
    memory_limit: str = "8GB"
    threads: int = 4
    log_level: int = logging.INFO

    # Output behavior
    default_save_format: Literal["parquet", "csv"] = "parquet"

    # Research choices
    call_sign: int = 1
    put_sign: int = -1


class OptionGammaProcessor:
    def __init__(self, config: OptionGammaConfig):
        self.cfg = config
        self.option_parquet = Path(config.option_parquet)
        self.spot_parquet = Path(config.spot_parquet) if config.spot_parquet else None
        self.output_dir = Path(config.output_dir) if config.output_dir else None
        self.temp_dir = Path(config.temp_dir) if config.temp_dir else None

        self.logger = self._build_logger()
        self.con = self._build_connection()

        self._option_rel_name = "option_raw"
        self._spot_rel_name = "spot_raw"

        self._last_sql: Optional[str] = None
        self._register_sources() 

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        logger.setLevel(self.cfg.log_level)
        logger.propagate = False

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(self.cfg.log_level)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _build_connection(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(database=":memory:")
        con.execute(f"PRAGMA threads={self.cfg.threads}")
        con.execute(f"PRAGMA memory_limit='{self.cfg.memory_limit}'")
        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            con.execute(f"PRAGMA temp_directory='{self.temp_dir.as_posix()}'")
        return con

    def _register_sources(self) -> None:
        if not self.option_parquet.exists():
            raise FileNotFoundError(f"Option parquet not found: {self.option_parquet}")

        self.logger.info("Registering option parquet: %s", self.option_parquet)
        self.con.execute(
            f"""
            CREATE OR REPLACE VIEW {self._option_rel_name} AS
            SELECT *
            FROM read_parquet('{self.option_parquet.as_posix()}')
            """
        )

        if self.spot_parquet:
            if not self.spot_parquet.exists():
                raise FileNotFoundError(f"Spot parquet not found: {self.spot_parquet}")

            self.logger.info("Registering spot parquet: %s", self.spot_parquet)
            self.con.execute(
                f"""
                CREATE OR REPLACE VIEW {self._spot_rel_name} AS
                SELECT *
                FROM read_parquet('{self.spot_parquet.as_posix()}')
                """
            )

    @property
    def last_sql(self) -> Optional[str]:
        return self._last_sql

    @property
    def has_spot_source(self) -> bool:
        return self.spot_parquet is not None

    @property
    def spot_price_field(self) -> str:
        return self.cfg.spot_price_field

    def _timed_query(self, sql: str, fetch: Literal["df", "one", "none"] = "df"):
        self._last_sql = sql
        self.logger.debug("Executing SQL:\n%s", sql)

        t0 = time.perf_counter()
        if fetch == "df":
            out = self.con.execute(sql).fetchdf()
        elif fetch == "one":
            out = self.con.execute(sql).fetchone()
        elif fetch == "none":
            self.con.execute(sql)
            out = None
        else:
            raise ValueError(f"Unsupported fetch mode: {fetch}")
        dt = time.perf_counter() - t0

        self.logger.info("Query finished in %.2fs", dt)
        return out

    def _save_df(
        self,
        df: pd.DataFrame,
        filename: str,
        fmt: Optional[Literal["parquet", "csv"]] = None,
    ) -> Path:
        if self.output_dir is None:
            raise ValueError("output_dir is not configured")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fmt = fmt or self.cfg.default_save_format
        path = self.output_dir / f"{filename}.{fmt}"

        self.logger.info("Saving DataFrame to %s", path)
        if fmt == "parquet":
            df.to_parquet(path, index=False)
        elif fmt == "csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        return path

    def _base_where_clauses(self) -> list[str]:
        clauses = ["1=1"]

        if self.cfg.start_date:
            clauses.append(f"date >= DATE '{self.cfg.start_date}'")
        if self.cfg.end_date:
            clauses.append(f"date <= DATE '{self.cfg.end_date}'")

        if self.cfg.secids:
            secids_sql = ", ".join(str(int(x)) for x in self.cfg.secids)
            clauses.append(f"secid IN ({secids_sql})")

        if self.cfg.require_standard_settlement:
            clauses.append("(ss_flag = 0 OR ss_flag IS NULL)")

        if self.cfg.require_positive_oi:
            clauses.append("open_interest IS NOT NULL AND open_interest > 0")

        if self.cfg.require_positive_contract_size:
            clauses.append("contract_size IS NOT NULL AND contract_size > 0")

        if self.cfg.require_nonnull_gamma:
            clauses.append("gamma IS NOT NULL")

        if self.cfg.require_valid_bidask:
            clauses.append(
                "best_bid IS NOT NULL AND best_offer IS NOT NULL "
                "AND best_bid >= 0 AND best_offer >= best_bid"
            )

        if self.cfg.drop_zero_bidask_mid:
            clauses.append("((best_bid + best_offer) / 2.0) > 0")

        return clauses

    def _clean_option_sql(self) -> str:
        where_sql = "\n  AND ".join(self._base_where_clauses())

        return f"""
        SELECT
            secid,
            date,
            exdate,
            cp_flag,
            strike_price AS strike_raw,
            strike_price / 1000.0 AS strike,
            best_bid,
            best_offer,
            (best_bid + best_offer) / 2.0 AS mid,
            volume AS option_volume,
            open_interest,
            impl_volatility,
            delta,
            gamma,
            vega,
            theta,
            optionid,
            cfadj AS option_cfadj,
            contract_size,
            ss_flag,
            forward_price,
            expiry_indicator,
            datediff('day', date, exdate) AS dte
        FROM {self._option_rel_name}
        WHERE {where_sql}
        """

    def _spot_sql(self) -> str:
        if not self.has_spot_source:
            raise ValueError("spot_parquet is required for spot join")

        spot_px = self.spot_price_field

        return f"""
        SELECT
            secid,
            date,
            open  AS spot_open,
            high  AS spot_high,
            low   AS spot_low,
            close AS spot_close,
            {spot_px} AS spot,
            volume AS spot_volume,
            return AS spot_return,
            cfadj AS spot_cfadj,
            cfret AS spot_cfret,
            shrout AS spot_shrout
        FROM {self._spot_rel_name}
        """

    def _spot_join_sql(self) -> str:
        if not self.has_spot_source:
            raise ValueError("spot_parquet is required for spot join and GEX calculation")

        base = self._clean_option_sql()
        spot = self._spot_sql()

        sql = f"""
        WITH opt AS (
            {base}
        ),
        spot AS (
            {spot}
        )
        SELECT
            opt.*,
            spot.spot,
            spot.spot_open,
            spot.spot_high,
            spot.spot_low,
            spot.spot_close,
            spot.spot_volume,
            spot.spot_return,
            spot.spot_cfadj,
            spot.spot_cfret,
            spot.spot_shrout
        FROM opt
        LEFT JOIN spot
          ON opt.secid = spot.secid
         AND opt.date = spot.date
        """

        if self.cfg.require_spot:
            sql = f"""
            WITH joined AS (
                {sql}
            )
            SELECT *
            FROM joined
            WHERE spot IS NOT NULL AND spot > 0
            """

        return sql

    def _gex_sql(self) -> str:
        joined = self._spot_join_sql()

        return f"""
        WITH joined AS (
            {joined}
        )
        SELECT
            *,
            CASE
                WHEN upper(cp_flag) = 'C' THEN {self.cfg.call_sign}
                WHEN upper(cp_flag) = 'P' THEN {self.cfg.put_sign}
                ELSE NULL
            END AS cp_sign,

            gamma * contract_size AS unit_gamma_shares,
            gamma * open_interest * contract_size AS oi_gamma_shares,

            CASE
                WHEN upper(cp_flag) = 'C' THEN 1
                WHEN upper(cp_flag) = 'P' THEN -1
                ELSE NULL
            END
            * gamma * open_interest * contract_size * spot
            AS gex_dollar_1pt,

            CASE
                WHEN upper(cp_flag) = 'C' THEN 1
                WHEN upper(cp_flag) = 'P' THEN -1
                ELSE NULL
            END
            * gamma * open_interest * contract_size * spot * spot * 0.01
            AS gex_dollar_1pct,

            CASE
                WHEN strike IS NOT NULL AND strike > 0
                THEN spot / strike
                ELSE NULL
            END AS spot_moneyness,

            CASE
                WHEN strike IS NOT NULL AND strike > 0 AND forward_price IS NOT NULL
                THEN forward_price / strike
                ELSE NULL
            END AS forward_moneyness,

            CASE
                WHEN dte = 0 THEN '0DTE'
                WHEN dte = 1 THEN '1DTE'
                WHEN dte BETWEEN 2 AND 5 THEN '2_5DTE'
                WHEN dte BETWEEN 6 AND 21 THEN '6_21DTE'
                WHEN dte BETWEEN 22 AND 63 THEN '22_63DTE'
                WHEN dte >= 64 THEN '64PLUSDTE'
                ELSE 'INVALID_DTE'
            END AS dte_bucket
        FROM joined
        """

    def _duplicate_check_sql(self) -> str:
        return f"""
        WITH base AS (
            {self._clean_option_sql()}
        )
        SELECT
            date,
            optionid,
            COUNT(*) AS n_rows
        FROM base
        GROUP BY 1, 2
        HAVING COUNT(*) > 1
        ORDER BY n_rows DESC, date, optionid
        """

    def option_schema(self) -> pd.DataFrame:
        return self._timed_query(
            f"DESCRIBE SELECT * FROM {self._option_rel_name}",
            fetch="df",
        )

    def spot_schema(self) -> pd.DataFrame:
        if not self.has_spot_source:
            raise ValueError("No spot_parquet configured")
        return self._timed_query(
            f"DESCRIBE SELECT * FROM {self._spot_rel_name}",
            fetch="df",
        )

    def row_count(self) -> int:
        row = self._timed_query(
            f"SELECT COUNT(*) FROM {self._option_rel_name}",
            fetch="one",
        )
        return int(row[0])

    def clean_row_count(self) -> int:
        row = self._timed_query(
            f"""
            WITH base AS (
                {self._clean_option_sql()}
            )
            SELECT COUNT(*)
            FROM base
            """,
            fetch="one",
        )
        return int(row[0])

    def duplicate_optionid_date(self, limit: int = 1000) -> pd.DataFrame:
        sql = f"""
        {self._duplicate_check_sql()}
        LIMIT {int(limit)}
        """
        return self._timed_query(sql, fetch="df")

    def quality_report(self) -> pd.DataFrame:
        sql = f"""
        SELECT
            COUNT(*) AS n_total,
            SUM(CASE WHEN optionid IS NULL THEN 1 ELSE 0 END) AS n_optionid_null,
            SUM(CASE WHEN secid IS NULL THEN 1 ELSE 0 END) AS n_secid_null,
            SUM(CASE WHEN date IS NULL THEN 1 ELSE 0 END) AS n_date_null,
            SUM(CASE WHEN exdate IS NULL THEN 1 ELSE 0 END) AS n_exdate_null,
            SUM(CASE WHEN cp_flag IS NULL THEN 1 ELSE 0 END) AS n_cpflag_null,
            SUM(CASE WHEN gamma IS NULL THEN 1 ELSE 0 END) AS n_gamma_null,
            SUM(CASE WHEN open_interest IS NULL THEN 1 ELSE 0 END) AS n_oi_null,
            SUM(CASE WHEN open_interest <= 0 THEN 1 ELSE 0 END) AS n_oi_nonpositive,
            SUM(CASE WHEN contract_size IS NULL THEN 1 ELSE 0 END) AS n_contract_size_null,
            SUM(CASE WHEN contract_size <= 0 THEN 1 ELSE 0 END) AS n_contract_size_nonpositive,
            SUM(CASE WHEN ss_flag IS NOT NULL AND ss_flag <> 0 THEN 1 ELSE 0 END) AS n_nonstandard
        FROM {self._option_rel_name}
        """
        return self._timed_query(sql, fetch="df")

    def spot_quality_report(self) -> pd.DataFrame:
        if not self.has_spot_source:
            raise ValueError("No spot_parquet configured")

        sql = f"""
        SELECT
            COUNT(*) AS n_total,
            SUM(CASE WHEN secid IS NULL THEN 1 ELSE 0 END) AS n_secid_null,
            SUM(CASE WHEN date IS NULL THEN 1 ELSE 0 END) AS n_date_null,
            SUM(CASE WHEN open IS NULL THEN 1 ELSE 0 END) AS n_open_null,
            SUM(CASE WHEN high IS NULL THEN 1 ELSE 0 END) AS n_high_null,
            SUM(CASE WHEN low IS NULL THEN 1 ELSE 0 END) AS n_low_null,
            SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) AS n_close_null,
            SUM(CASE WHEN close <= 0 THEN 1 ELSE 0 END) AS n_close_nonpositive,
            SUM(CASE WHEN cfadj IS NULL THEN 1 ELSE 0 END) AS n_cfadj_null,
            SUM(CASE WHEN shrout IS NULL THEN 1 ELSE 0 END) AS n_shrout_null
        FROM {self._spot_rel_name}
        """
        return self._timed_query(sql, fetch="df")

    def spot_join_coverage_report(self) -> pd.DataFrame:
        if not self.has_spot_source:
            raise ValueError("No spot_parquet configured")

        sql = f"""
        WITH opt AS (
            {self._clean_option_sql()}
        ),
        spot AS (
            {self._spot_sql()}
        ),
        joined AS (
            SELECT
                opt.*,
                spot.spot
            FROM opt
            LEFT JOIN spot
              ON opt.secid = spot.secid
             AND opt.date = spot.date
        )
        SELECT
            COUNT(*) AS n_option_rows,
            SUM(CASE WHEN spot IS NOT NULL THEN 1 ELSE 0 END) AS n_matched_spot,
            SUM(CASE WHEN spot IS NULL THEN 1 ELSE 0 END) AS n_missing_spot,
            1.0 * SUM(CASE WHEN spot IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) AS matched_ratio
        FROM joined
        """
        return self._timed_query(sql, fetch="df")

    def fetch_clean_options(
        self,
        limit: Optional[int] = None,
        save: bool = False,
        filename: str = "option_clean",
    ) -> pd.DataFrame:
        sql = self._clean_option_sql()
        if limit is not None:
            sql += f"\nLIMIT {int(limit)}"

        df = self._timed_query(sql, fetch="df")
        self.logger.info("Fetched clean option rows: %s", len(df))

        if save:
            self._save_df(df, filename)
        return df

    def fetch_joined_with_spot(
        self,
        limit: Optional[int] = None,
        save: bool = False,
        filename: str = "option_with_spot",
    ) -> pd.DataFrame:
        sql = self._spot_join_sql()
        if limit is not None:
            sql += f"\nLIMIT {int(limit)}"

        df = self._timed_query(sql, fetch="df")
        self.logger.info("Fetched option+spot rows: %s", len(df))

        if save:
            self._save_df(df, filename)
        return df

    def fetch_contract_gex(
        self,
        limit: Optional[int] = None,
        save: bool = False,
        filename: str = "option_contract_gex_daily",
    ) -> pd.DataFrame:
        sql = self._gex_sql()
        if limit is not None:
            sql += f"\nLIMIT {int(limit)}"

        df = self._timed_query(sql, fetch="df")
        self.logger.info("Fetched contract GEX rows: %s", len(df))

        if save:
            self._save_df(df, filename)
        return df

    def aggregate_underlying_daily(
        self,
        save: bool = False,
        filename: str = "underlying_gex_daily",
    ) -> pd.DataFrame:
        sql = f"""
        WITH gex AS (
            {self._gex_sql()}
        )
        SELECT
            secid,
            date,
            MAX(spot) AS spot,
            MAX(spot_close) AS spot_close,
            MAX(spot_return) AS spot_return,
            MAX(spot_cfadj) AS spot_cfadj,
            MAX(spot_shrout) AS spot_shrout,

            COUNT(*) AS n_contracts,
            COUNT(DISTINCT optionid) AS n_optionids,
            COUNT(DISTINCT exdate) AS n_expiries,

            SUM(CASE WHEN upper(cp_flag) = 'C' THEN gex_dollar_1pct ELSE 0 END) AS call_gex_1pct,
            SUM(CASE WHEN upper(cp_flag) = 'P' THEN gex_dollar_1pct ELSE 0 END) AS put_gex_1pct,
            SUM(gex_dollar_1pct) AS net_gex_1pct,

            SUM(CASE WHEN upper(cp_flag) = 'C' THEN gex_dollar_1pt ELSE 0 END) AS call_gex_1pt,
            SUM(CASE WHEN upper(cp_flag) = 'P' THEN gex_dollar_1pt ELSE 0 END) AS put_gex_1pt,
            SUM(gex_dollar_1pt) AS net_gex_1pt,

            SUM(open_interest) AS total_open_interest,
            SUM(option_volume) AS total_option_volume
        FROM gex
        GROUP BY 1, 2
        ORDER BY secid, date
        """
        df = self._timed_query(sql, fetch="df")
        if save:
            self._save_df(df, filename)
        return df

    def aggregate_strike_daily(
        self,
        save: bool = False,
        filename: str = "option_strike_gex_daily",
    ) -> pd.DataFrame:
        sql = f"""
        WITH gex AS (
            {self._gex_sql()}
        )
        SELECT
            secid,
            date,
            strike,
            MAX(spot) AS spot,
            COUNT(*) AS n_contracts,
            SUM(open_interest) AS total_open_interest,

            SUM(CASE WHEN upper(cp_flag) = 'C' THEN gex_dollar_1pct ELSE 0 END) AS call_gex_1pct,
            SUM(CASE WHEN upper(cp_flag) = 'P' THEN gex_dollar_1pct ELSE 0 END) AS put_gex_1pct,
            SUM(gex_dollar_1pct) AS net_gex_1pct,

            SUM(CASE WHEN upper(cp_flag) = 'C' THEN gex_dollar_1pt ELSE 0 END) AS call_gex_1pt,
            SUM(CASE WHEN upper(cp_flag) = 'P' THEN gex_dollar_1pt ELSE 0 END) AS put_gex_1pt,
            SUM(gex_dollar_1pt) AS net_gex_1pt
        FROM gex
        GROUP BY 1, 2, 3
        ORDER BY secid, date, strike
        """
        df = self._timed_query(sql, fetch="df")
        if save:
            self._save_df(df, filename)
        return df

    def aggregate_expiry_daily(
        self,
        save: bool = False,
        filename: str = "option_expiry_gex_daily",
    ) -> pd.DataFrame:
        sql = f"""
        WITH gex AS (
            {self._gex_sql()}
        )
        SELECT
            secid,
            date,
            exdate,
            expiry_indicator,
            dte_bucket,
            MAX(spot) AS spot,
            COUNT(*) AS n_contracts,
            SUM(open_interest) AS total_open_interest,

            SUM(CASE WHEN upper(cp_flag) = 'C' THEN gex_dollar_1pct ELSE 0 END) AS call_gex_1pct,
            SUM(CASE WHEN upper(cp_flag) = 'P' THEN gex_dollar_1pct ELSE 0 END) AS put_gex_1pct,
            SUM(gex_dollar_1pct) AS net_gex_1pct,

            SUM(CASE WHEN upper(cp_flag) = 'C' THEN gex_dollar_1pt ELSE 0 END) AS call_gex_1pt,
            SUM(CASE WHEN upper(cp_flag) = 'P' THEN gex_dollar_1pt ELSE 0 END) AS put_gex_1pt,
            SUM(gex_dollar_1pt) AS net_gex_1pt
        FROM gex
        GROUP BY 1, 2, 3, 4, 5
        ORDER BY secid, date, exdate
        """
        df = self._timed_query(sql, fetch="df")
        if save:
            self._save_df(df, filename)
        return df

    def aggregate_dte_daily(
        self,
        save: bool = False,
        filename: str = "option_dte_gex_daily",
    ) -> pd.DataFrame:
        sql = f"""
        WITH gex AS (
            {self._gex_sql()}
        )
        SELECT
            secid,
            date,
            dte_bucket,
            MAX(spot) AS spot,
            COUNT(*) AS n_contracts,
            SUM(open_interest) AS total_open_interest,
            SUM(gex_dollar_1pct) AS net_gex_1pct,
            SUM(gex_dollar_1pt) AS net_gex_1pt
        FROM gex
        GROUP BY 1, 2, 3
        ORDER BY secid, date, dte_bucket
        """
        df = self._timed_query(sql, fetch="df")
        if save:
            self._save_df(df, filename)
        return df

    def export_query_to_parquet(self, sql: str, output_path: PathLike) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info("Exporting query to parquet: %s", output_path)
        copy_sql = f"""
        COPY (
            {sql}
        ) TO '{output_path.as_posix()}'
        (FORMAT PARQUET)
        """
        self._timed_query(copy_sql, fetch="none")
        return output_path

    def export_contract_gex_to_parquet(self, output_path: PathLike) -> Path:
        return self.export_query_to_parquet(self._gex_sql(), output_path)

    def export_underlying_daily_to_parquet(self, output_path: PathLike) -> Path:
        sql = f"""
        WITH gex AS (
            {self._gex_sql()}
        )
        SELECT
            secid,
            date,
            MAX(spot) AS spot,
            MAX(spot_close) AS spot_close,
            MAX(spot_return) AS spot_return,
            MAX(spot_cfadj) AS spot_cfadj,
            MAX(spot_shrout) AS spot_shrout,
            COUNT(*) AS n_contracts,
            COUNT(DISTINCT optionid) AS n_optionids,
            COUNT(DISTINCT exdate) AS n_expiries,
            SUM(CASE WHEN upper(cp_flag) = 'C' THEN gex_dollar_1pct ELSE 0 END) AS call_gex_1pct,
            SUM(CASE WHEN upper(cp_flag) = 'P' THEN gex_dollar_1pct ELSE 0 END) AS put_gex_1pct,
            SUM(gex_dollar_1pct) AS net_gex_1pct,
            SUM(CASE WHEN upper(cp_flag) = 'C' THEN gex_dollar_1pt ELSE 0 END) AS call_gex_1pt,
            SUM(CASE WHEN upper(cp_flag) = 'P' THEN gex_dollar_1pt ELSE 0 END) AS put_gex_1pt,
            SUM(gex_dollar_1pt) AS net_gex_1pt,
            SUM(open_interest) AS total_open_interest,
            SUM(option_volume) AS total_option_volume
        FROM gex
        GROUP BY 1, 2
        ORDER BY secid, date
        """
        return self.export_query_to_parquet(sql, output_path)

    def run_diagnostics(self) -> dict[str, pd.DataFrame | int]:
        out: dict[str, pd.DataFrame | int] = {
            "option_schema": self.option_schema(),
            "quality_report": self.quality_report(),
            "raw_row_count": self.row_count(),
            "clean_row_count": self.clean_row_count(),
            "duplicate_optionid_date": self.duplicate_optionid_date(limit=1000),
        }

        if self.has_spot_source:
            out["spot_schema"] = self.spot_schema()
            out["spot_quality_report"] = self.spot_quality_report()
            out["spot_join_coverage_report"] = self.spot_join_coverage_report()

        return out

    def run_baseline_pipeline(
        self,
        export_contract_gex: Optional[PathLike] = None,
        export_underlying_daily: Optional[PathLike] = None,
    ) -> dict[str, Optional[Path]]:
        outputs = {
            "contract_gex": None,
            "underlying_daily": None,
        }

        if export_contract_gex is not None:
            outputs["contract_gex"] = self.export_contract_gex_to_parquet(export_contract_gex)

        if export_underlying_daily is not None:
            outputs["underlying_daily"] = self.export_underlying_daily_to_parquet(export_underlying_daily)

        return outputs

    def close(self) -> None:
        self.logger.info("Closing DuckDB connection")
        self.con.close()