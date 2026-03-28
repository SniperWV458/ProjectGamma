from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from statsmodels.stats.sandwich_covariance import cov_hac


# ============================================================
# Config dataclasses
# ============================================================

@dataclass
class FactorSpec:
    """
    Specification for one factor source.

    Supports:
    - factors already in the base panel
    - external csv/parquet files
    - daily or monthly factor frequency
    """

    name: str
    path: Optional[str | Path] = None
    file_type: Optional[Literal["csv", "parquet"]] = None
    date_col: str = "date"
    id_col: str = "permno"
    factor_cols: Optional[list[str]] = None
    frequency: Literal["daily", "monthly"] = "daily"
    merge_how: Literal["left", "inner", "outer", "right"] = "left"
    already_in_panel: bool = False
    date_format: Optional[str] = None
    numeric_only: bool = True
    suffix: Optional[str] = None
    lag_days: int = 0
    lag_months: int = 0
    forward_fill: bool = False
    description: Optional[str] = None

    source_id_type: Literal["permno", "ticker", "secid"] = "permno"
    base_merge_id_col: str | None = None


@dataclass
class IdentifierConfig:
    """
    Identifier mapping configuration.

    Used when factor sources are not keyed by permno directly.
    Example: ticker-based factor files that need to be mapped to permno.
    """
    mapping_path: Optional[str | Path] = None
    mapping_file_type: Optional[Literal["csv", "parquet"]] = None

    permno_col: str = "permno"
    secid_col: str = "secid"
    ticker_col: str = "ticker"
    cusip_col: str = "cusip"
    ncusip_col: str = "ncusip"

    # optional metadata columns
    permco_col: Optional[str] = "permco"
    comnam_col: Optional[str] = "comnam"
    siccd_col: Optional[str] = "siccd"
    link_method_col: Optional[str] = "link_method"

    # string normalization
    uppercase_ticker: bool = True
    strip_ticker_whitespace: bool = True

    # duplicate handling after mapping
    duplicate_resolution: Literal["first", "last", "drop_ambiguous"] = "drop_ambiguous"


@dataclass
class DataConfig:
    """
    Core data paths.
    """

    # Anchor files
    underlying_gex_path: str | Path
    crsp_daily_path: str | Path

    # Optional date filter
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Base file types
    underlying_gex_file_type: Optional[Literal["csv", "parquet"]] = None
    crsp_daily_file_type: Optional[Literal["csv", "parquet"]] = None

    # Parsing / filters
    parse_dates: bool = True
    sort_values: bool = True

    # Optional row-level filters
    min_abs_price: Optional[float] = None
    drop_missing_permno: bool = True


@dataclass
class ColumnConfig:
    """
    Naming contract for the merged base panel.
    """

    id_col: str = "permno"
    date_col: str = "date"

    # Anchor return / signal cols
    gex_col: str = "net_gex_1pct"
    ret_col: str = "ret"

    # Price columns
    spot_col: str = "spot"
    crsp_price_col: str = "prc"

    # Optional controls
    industry_col: Optional[str] = None
    market_cap_col: Optional[str] = None
    volume_col: Optional[str] = "vol"

    # Explicit controls already in base panel if desired
    control_cols: list[str] = field(default_factory=list)


@dataclass
class PreprocessConfig:
    """
    Preprocessing behavior for factor and signal columns.
    """

    winsorize: bool = True
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99

    cross_sectional_zscore: bool = True
    zscore_suffix: str = "_z"

    min_cross_section_size: int = 10

    numeric_only: bool = True
    drop_duplicate_keys: bool = True
    allow_missing_factors: bool = True
    missing_threshold: Optional[float] = None
    fill_method: Optional[Literal["ffill", "bfill"]] = None

    # whether to take abs(prc) because CRSP prc can be negative signed quote convention
    use_abs_crsp_price: bool = True


@dataclass
class TargetConfig:
    """
    Forward target construction.
    """

    horizons: list[int] = field(default_factory=lambda: [1, 5, 20])

    create_signed_return: bool = True
    create_abs_return: bool = True
    create_squared_return: bool = True
    create_realized_vol: bool = True
    create_downside_semivar: bool = True
    create_tail_indicators: bool = True

    # For tail indicators
    tail_quantiles: list[float] = field(default_factory=lambda: [0.05, 0.95])
    tail_groupby: Literal["full_sample", "by_stock", "by_date"] = "full_sample"

    # Returns for forward compounding
    return_type: Literal["simple", "log"] = "simple"


@dataclass
class OutputConfig:
    """
    Output control.
    """

    output_dir: str | Path
    save_panel_snapshot: bool = True
    save_metadata: bool = True
    panel_snapshot_name: str = "phase1_panel.parquet"
    metadata_name: str = "phase1_metadata.json"
    verbose: bool = True


@dataclass
class RegressionConfig:
    """
    Regression and Phase 2 analysis settings.
    """

    use_fama_macbeth: bool = True
    nw_lags: int = 5
    add_intercept: bool = True

    # regression controls
    use_controls: bool = True
    control_cols: list[str] = field(default_factory=list)

    # minimum data requirements
    min_obs_per_date: int = 10
    min_dates_required: int = 20

    # regime validation defaults
    regime_validation_y_cols: list[str] = field(
        default_factory=lambda: [
            "ret_fwd_1d",
            "abs_ret_fwd_1d",
            "rv_fwd_5d",
            "tail_left_fwd_1d",
        ]
    )

    # interaction regression defaults
    interaction_y_cols: list[str] = field(
        default_factory=lambda: [
            "abs_ret_fwd_1d",
            "rv_fwd_5d",
            "tail_left_fwd_1d",
        ]
    )

    # sorting
    n_buckets: int = 5

    # output
    save_phase2_tables: bool = True
    save_phase2_plots: bool = True


@dataclass
class ClassificationConfig:
    """
    Phase 4 tail-classification settings.
    """

    enabled: bool = True

    # Targets to classify
    target_cols: list[str] = field(
        default_factory=lambda: [
            "tail_left_fwd_1d",
            "tail_left_fwd_5d",
            "tail_abs_fwd_1d",
        ]
    )

    # Train/test split by date
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None

    # If explicit split is not supplied, use fraction by sorted unique dates
    test_size: float = 0.3

    # Minimum rows after filtering
    min_train_rows: int = 200
    min_test_rows: int = 100

    # Models to run in first version
    model_names: list[str] = field(
        default_factory=lambda: [
            "logit_l2",
            "logit_elasticnet",
        ]
    )

    # Logistic settings
    max_iter: int = 2000
    C: float = 1.0
    l1_ratio: float = 0.5
    class_weight: Optional[str] = "balanced"

    # Feature construction
    use_controls: bool = True
    control_cols: list[str] = field(default_factory=list)
    include_factor_interaction: bool = True

    # Missing handling
    imputer_strategy: str = "median"

    # Output
    save_phase4_tables: bool = True
    save_phase4_plots: bool = True
    top_n_factor_plots: int = 15


# ============================================================
# Main experiment class - Phase 1 only
# ============================================================

class GEXCollaborativeEffectExperiment:
    """
    Phase 1 implementation for GEX collaborative effect experiments.

    Covered in this phase:
    - load anchor panel from underlying GEX + CRSP daily common
    - load external factor files (csv/parquet, daily/monthly)
    - merge factors into daily PERMNO-date panel
    - preprocess panel
    - build forward targets
    - build GEX regimes

    Notes on price usage:
    - `spot` from underlying GEX is preserved for option-underlying/GEX context
    - `prc` from CRSP is preserved for stock-operation context
    """

    def __init__(
            self,
            data_config: DataConfig,
            column_config: ColumnConfig,
            preprocess_config: PreprocessConfig,
            target_config: TargetConfig,
            regression_config: RegressionConfig,
            classification_config: ClassificationConfig,
            output_config: OutputConfig,
            identifier_config: Optional[IdentifierConfig] = None,
    ) -> None:
        self.data_config = data_config
        self.column_config = column_config
        self.preprocess_config = preprocess_config
        self.target_config = target_config
        self.regression_config = regression_config
        self.classification_config = classification_config
        self.output_config = output_config
        self.identifier_config = identifier_config or IdentifierConfig()

        self.output_dir = Path(self.output_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.base_panel: Optional[pd.DataFrame] = None
        self.panel: Optional[pd.DataFrame] = None

        self.factor_specs: list[FactorSpec] = []
        self.factor_cols_all: list[str] = []
        self.factor_cols_daily: list[str] = []
        self.factor_cols_monthly: list[str] = []
        self.factor_cols_selected = []
        self.factor_name_map = {}

        self.identifier_mapping: Optional[pd.DataFrame] = None

        self.metadata: dict[str, Any] = {}
        self.artifacts: dict[str, Any] = {}

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def run_phase1(
            self,
            factor_specs: Optional[list[FactorSpec]] = None,
            selected_factors: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Run Phase 1 only:
        1. load anchor panel
        2. merge factor sources
        3. preprocess
        4. build targets
        5. build GEX regimes
        6. save outputs
        """

        self.factor_specs = factor_specs or []

        self.base_panel = self.load_base_panel()
        self.panel = self.base_panel.copy()

        if self.factor_specs:
            self.panel = self.merge_factors(self.factor_specs)

        self.panel = self.preprocess_panel(selected_factors=selected_factors)
        self.panel = self.build_targets()
        self.panel = self.build_gex_regimes()

        self._build_metadata()
        self._save_phase1_outputs()

        self.artifacts = {
            "panel": self.panel,
            "panel_path": str(self.output_dir / self.output_config.panel_snapshot_name)
            if self.output_config.save_panel_snapshot else None,
            "metadata": self.metadata,
            "metadata_path": str(self.output_dir / self.output_config.metadata_name)
            if self.output_config.save_metadata else None,
            "factor_cols_all": self.factor_cols_all,
            "factor_cols_daily": self.factor_cols_daily,
            "factor_cols_monthly": self.factor_cols_monthly,
        }
        return self.artifacts

    # --------------------------------------------------------
    # Loading helpers
    # --------------------------------------------------------

    def load_base_panel(self) -> pd.DataFrame:
        """
        Load and merge:
        - underlying_gex_daily_with_permno
        - crsp_daily_common_2018_2024_linked_permnos

        Merge key:
        - permno
        - date
        """

        gex_path = Path(self.data_config.underlying_gex_path)
        crsp_path = Path(self.data_config.crsp_daily_path)

        gex_df = self._read_file(
            gex_path,
            self.data_config.underlying_gex_file_type,
        )
        crsp_df = self._read_file(
            crsp_path,
            self.data_config.crsp_daily_file_type,
        )

        gex_df = self._standardize_date_and_id(
            gex_df,
            id_col=self.column_config.id_col,
            date_col=self.column_config.date_col,
        )
        crsp_df = self._standardize_date_and_id(
            crsp_df,
            id_col=self.column_config.id_col,
            date_col=self.column_config.date_col,
        )

        if self.data_config.drop_missing_permno:
            gex_df = gex_df[gex_df[self.column_config.id_col].notna()].copy()
            crsp_df = crsp_df[crsp_df[self.column_config.id_col].notna()].copy()

        # CRSP prc may be signed; keep original and create abs version if configured
        if self.column_config.crsp_price_col in crsp_df.columns and self.preprocess_config.use_abs_crsp_price:
            crsp_df["prc_abs"] = crsp_df[self.column_config.crsp_price_col].abs()

        # Filter by date range
        gex_df = self._filter_date_range(gex_df, self.column_config.date_col)
        crsp_df = self._filter_date_range(crsp_df, self.column_config.date_col)

        # Optional price filter on CRSP absolute price
        if self.data_config.min_abs_price is not None:
            if "prc_abs" in crsp_df.columns:
                crsp_df = crsp_df[crsp_df["prc_abs"] >= self.data_config.min_abs_price].copy()
            elif self.column_config.crsp_price_col in crsp_df.columns:
                crsp_df = crsp_df[
                    crsp_df[self.column_config.crsp_price_col].abs() >= self.data_config.min_abs_price].copy()

        # Deduplicate before merge
        gex_df = self._deduplicate_key(gex_df)
        crsp_df = self._deduplicate_key(crsp_df)

        # Merge
        panel = pd.merge(
            gex_df,
            crsp_df,
            how="left",
            on=[self.column_config.id_col, self.column_config.date_col],
            suffixes=("", "_crsp"),
        )

        # Derived stock-operation columns
        if {"prc_abs", "shrout"}.issubset(panel.columns):
            panel["market_equity_crsp"] = panel["prc_abs"] * panel["shrout"]
        elif {self.column_config.crsp_price_col, "shrout"}.issubset(panel.columns):
            panel["market_equity_crsp"] = panel[self.column_config.crsp_price_col].abs() * panel["shrout"]

        # Explicit "operation" and "underlying" reference prices
        if self.column_config.spot_col in panel.columns:
            panel["price_underlying"] = panel[self.column_config.spot_col]
        if "prc_abs" in panel.columns:
            panel["price_stock"] = panel["prc_abs"]
        elif self.column_config.crsp_price_col in panel.columns:
            panel["price_stock"] = panel[self.column_config.crsp_price_col].abs()

        if self.data_config.sort_values:
            panel = panel.sort_values([self.column_config.id_col, self.column_config.date_col]).reset_index(drop=True)

        # Basic validation
        self.validate_required_columns(
            panel,
            [self.column_config.id_col, self.column_config.date_col, self.column_config.gex_col],
        )

        return panel

    def load_identifier_mapping(self) -> pd.DataFrame:
        """
        Load identifier mapping file for translating ticker/secid to permno.
        Expected columns include at least permno and ticker when ticker mapping is needed.
        """
        if self.identifier_mapping is not None:
            return self.identifier_mapping

        cfg = self.identifier_config
        if cfg.mapping_path is None:
            raise ValueError(
                "Identifier mapping is required but IdentifierConfig.mapping_path is None."
            )

        path = Path(cfg.mapping_path)
        file_type = cfg.mapping_file_type or self._infer_file_type(path)
        df = self._read_file(path, file_type)

        required = [cfg.permno_col]
        if cfg.ticker_col:
            required.append(cfg.ticker_col)

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Identifier mapping file is missing required columns: {missing}"
            )

        df = df.copy()

        # Standardize core identifier columns
        if cfg.permno_col in df.columns:
            df[cfg.permno_col] = pd.to_numeric(df[cfg.permno_col], errors="coerce").astype("Int64")

        if cfg.secid_col and cfg.secid_col in df.columns:
            df[cfg.secid_col] = pd.to_numeric(df[cfg.secid_col], errors="coerce").astype("Int64")

        if cfg.ticker_col and cfg.ticker_col in df.columns:
            df[cfg.ticker_col] = self._normalize_ticker_series(df[cfg.ticker_col])

        # Keep only relevant columns
        keep_cols = [
            c for c in [
                cfg.permno_col,
                cfg.secid_col,
                cfg.ticker_col,
                cfg.cusip_col,
                cfg.ncusip_col,
                cfg.permco_col,
                cfg.comnam_col,
                cfg.siccd_col,
                cfg.link_method_col,
            ]
            if c is not None and c in df.columns
        ]
        df = df[keep_cols].copy()

        self.identifier_mapping = df
        return df

    def _normalize_ticker_series(self, s: pd.Series) -> pd.Series:
        out = s.astype(str).str.strip()

        # common bad placeholders
        out = out.replace({
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "null": np.nan,
            "NA": np.nan,
            "N/A": np.nan,
        })

        if self.identifier_config.strip_ticker_whitespace:
            out = out.str.strip()

        if self.identifier_config.uppercase_ticker:
            out = out.str.upper()

        return out

    def _map_source_identifier_to_permno(
            self,
            df: pd.DataFrame,
            spec: FactorSpec,
    ) -> pd.DataFrame:
        """
        Translate source identifier to permno when source_id_type != permno.

        Supported:
        - ticker -> permno
        - secid -> permno
        """
        out = df.copy()
        source_col = spec.id_col
        target_col = self.column_config.id_col

        if spec.source_id_type == "permno":
            out[source_col] = pd.to_numeric(out[source_col], errors="coerce").astype("Int64")
            if source_col != target_col:
                out = out.rename(columns={source_col: target_col})
            return out

        mapping = self.load_identifier_mapping()
        cfg = self.identifier_config

        if spec.source_id_type == "ticker":
            if cfg.ticker_col not in mapping.columns:
                raise ValueError(
                    f"Mapping file does not contain ticker column '{cfg.ticker_col}'."
                )

            out[source_col] = self._normalize_ticker_series(out[source_col])

            map_df = mapping[[cfg.ticker_col, cfg.permno_col]].dropna().copy()

            # resolve duplicate ticker -> permno mappings
            dup_count = map_df.duplicated([cfg.ticker_col], keep=False).sum()
            if dup_count > 0:
                if cfg.duplicate_resolution == "first":
                    map_df = map_df.drop_duplicates(subset=[cfg.ticker_col], keep="first")
                elif cfg.duplicate_resolution == "last":
                    map_df = map_df.drop_duplicates(subset=[cfg.ticker_col], keep="last")
                elif cfg.duplicate_resolution == "drop_ambiguous":
                    counts = map_df.groupby(cfg.ticker_col)[cfg.permno_col].nunique()
                    good_tickers = counts[counts == 1].index
                    map_df = map_df[map_df[cfg.ticker_col].isin(good_tickers)].copy()
                    map_df = map_df.drop_duplicates(subset=[cfg.ticker_col], keep="first")
                else:
                    raise ValueError(
                        f"Unsupported duplicate_resolution '{cfg.duplicate_resolution}'."
                    )

            out = out.merge(
                map_df,
                how="left",
                left_on=source_col,
                right_on=cfg.ticker_col,
            )

            out = out.rename(columns={cfg.permno_col: target_col})
            drop_cols = [c for c in [cfg.ticker_col] if c in out.columns and c != source_col]
            if drop_cols:
                out = out.drop(columns=drop_cols)

            out[target_col] = pd.to_numeric(out[target_col], errors="coerce").astype("Int64")
            return out

        if spec.source_id_type == "secid":
            if cfg.secid_col not in mapping.columns:
                raise ValueError(
                    f"Mapping file does not contain secid column '{cfg.secid_col}'."
                )

            out[source_col] = pd.to_numeric(out[source_col], errors="coerce").astype("Int64")

            map_df = mapping[[cfg.secid_col, cfg.permno_col]].dropna().copy()

            dup_count = map_df.duplicated([cfg.secid_col], keep=False).sum()
            if dup_count > 0:
                if cfg.duplicate_resolution == "first":
                    map_df = map_df.drop_duplicates(subset=[cfg.secid_col], keep="first")
                elif cfg.duplicate_resolution == "last":
                    map_df = map_df.drop_duplicates(subset=[cfg.secid_col], keep="last")
                elif cfg.duplicate_resolution == "drop_ambiguous":
                    counts = map_df.groupby(cfg.secid_col)[cfg.permno_col].nunique()
                    good_secids = counts[counts == 1].index
                    map_df = map_df[map_df[cfg.secid_col].isin(good_secids)].copy()
                    map_df = map_df.drop_duplicates(subset=[cfg.secid_col], keep="first")
                else:
                    raise ValueError(
                        f"Unsupported duplicate_resolution '{cfg.duplicate_resolution}'."
                    )

            out = out.merge(
                map_df,
                how="left",
                left_on=source_col,
                right_on=cfg.secid_col,
            )

            out = out.rename(columns={cfg.permno_col: target_col})
            drop_cols = [c for c in [cfg.secid_col] if c in out.columns and c != source_col]
            if drop_cols:
                out = out.drop(columns=drop_cols)

            out[target_col] = pd.to_numeric(out[target_col], errors="coerce").astype("Int64")
            return out

        raise ValueError(
            f"Unsupported source_id_type '{spec.source_id_type}'. "
            f"Supported values: permno, ticker, secid."
        )

    def coerce_factor_columns_to_numeric(
            self,
            df: pd.DataFrame,
            factor_cols: list[str],
    ) -> pd.DataFrame:
        """
        Convert factor columns to numeric when possible.
        Handles strings like '1.42%' -> 0.0142.
        """
        out = df.copy()

        for c in factor_cols:
            if c not in out.columns:
                continue

            s = out[c]

            # Already numeric
            if pd.api.types.is_numeric_dtype(s):
                continue

            # Convert to string for cleaning
            s_str = s.astype(str).str.strip()

            # Replace common missing markers
            s_str = s_str.replace({
                "": np.nan,
                "nan": np.nan,
                "None": np.nan,
                "null": np.nan,
                "NA": np.nan,
                "N/A": np.nan,
            })

            # Handle percentage strings
            has_pct = s_str.str.contains("%", na=False)
            if has_pct.any():
                cleaned = s_str.str.replace("%", "", regex=False)
                out[c] = pd.to_numeric(cleaned, errors="coerce") / 100.0
            else:
                out[c] = pd.to_numeric(s_str, errors="coerce")

        return out

    def load_factor_source(self, spec: FactorSpec) -> pd.DataFrame:
        """
        Load one factor source from csv/parquet or from existing panel columns.

        Robustly supports source identifier types:
        - permno
        - ticker
        - secid

        Non-permno identifiers are mapped to permno using identifier mapping.
        """

        if spec.already_in_panel:
            if self.base_panel is None:
                raise ValueError("Base panel must be loaded before using already_in_panel factor specs.")

            df = self.base_panel.copy()
            factor_cols = spec.factor_cols or []

            needed = [self.column_config.id_col, self.column_config.date_col] + factor_cols
            missing = [c for c in needed if c not in df.columns]
            if missing:
                raise ValueError(
                    f"FactorSpec '{spec.name}' refers to in-panel columns that do not exist: {missing}"
                )

            return df[needed].copy()

        if spec.path is None:
            raise ValueError(
                f"FactorSpec '{spec.name}' requires a path when already_in_panel=False."
            )

        path = Path(spec.path)
        file_type = spec.file_type or self._infer_file_type(path)
        df = self._read_file(path, file_type)

        # Validate source columns exist before renaming/mapping
        if spec.id_col not in df.columns:
            raise ValueError(
                f"FactorSpec '{spec.name}' id_col '{spec.id_col}' not found in factor file."
            )
        if spec.date_col not in df.columns:
            raise ValueError(
                f"FactorSpec '{spec.name}' date_col '{spec.date_col}' not found in factor file."
            )

        # Standardize date column only first; keep original source id column for mapping
        df = self._standardize_date_and_id(
            df,
            id_col=spec.id_col,
            date_col=spec.date_col,
            out_id_col=spec.id_col,
            out_date_col=self.column_config.date_col,
            date_format=spec.date_format,
        )

        # Infer/select factor columns
        if spec.factor_cols is None:
            if spec.numeric_only:
                factor_cols = self.select_numeric_factor_columns(
                    df,
                    exclude_cols=[spec.id_col, self.column_config.date_col],
                )
            else:
                factor_cols = [
                    c for c in df.columns
                    if c not in [spec.id_col, self.column_config.date_col]
                ]
        else:
            factor_cols = [c for c in spec.factor_cols if c in df.columns]

        if not factor_cols:
            raise ValueError(f"No usable factor columns found for FactorSpec '{spec.name}'.")

        keep_cols = [spec.id_col, self.column_config.date_col] + factor_cols
        df = df[keep_cols].copy()

        # Convert factor columns to numeric where possible
        df = self.coerce_factor_columns_to_numeric(df, factor_cols)

        # Map source identifier to base id (permno)
        df = self._map_source_identifier_to_permno(df, spec)

        # Drop rows that failed mapping
        before = len(df)
        df = df[df[self.column_config.id_col].notna()].copy()
        after = len(df)

        if before > after:
            warnings.warn(
                f"FactorSpec '{spec.name}': dropped {before - after} rows because "
                f"{spec.source_id_type} could not be mapped to permno."
            )

        # Apply day lag for daily factors
        if spec.frequency == "daily" and spec.lag_days != 0:
            df[self.column_config.date_col] = (
                    df[self.column_config.date_col] + pd.to_timedelta(spec.lag_days, unit="D")
            )

        # Deduplicate on permno-date after mapping
        df = self._filter_date_range(df, self.column_config.date_col)
        df = self._deduplicate_key(df)

        # Add suffix if requested
        if spec.suffix:
            rename_map = {c: f"{c}{spec.suffix}" for c in factor_cols}
            df = df.rename(columns=rename_map)
            factor_cols = [rename_map[c] for c in factor_cols]

        # Final selection: now keyed by permno/date
        keep_cols = [self.column_config.id_col, self.column_config.date_col] + factor_cols
        df = df[keep_cols].copy()

        spec.factor_cols = factor_cols
        spec.base_merge_id_col = self.column_config.id_col

        return df

    def merge_factors(self, factor_specs: list[FactorSpec]) -> pd.DataFrame:
        """
        Merge all factor sources to self.panel.

        Daily factors:
        - direct merge on permno-date

        Monthly factors:
        - month-key merge after lag_months
        - optional forward-fill is handled structurally by month-key expansion
        """

        if self.panel is None:
            raise ValueError("Panel must be loaded before merging factors.")

        panel = self.panel.copy()

        for spec in factor_specs:
            factor_df = self.load_factor_source(spec)

            if spec.frequency == "daily":
                panel = self._merge_daily_factor_df(panel, factor_df, spec)
                self.factor_cols_daily.extend(spec.factor_cols or [])

            elif spec.frequency == "monthly":
                panel = self._merge_monthly_factor_df(panel, factor_df, spec)
                self.factor_cols_monthly.extend(spec.factor_cols or [])

            else:
                raise ValueError(f"Unsupported factor frequency '{spec.frequency}' for FactorSpec '{spec.name}'.")

            self.factor_cols_all.extend(spec.factor_cols or [])

        # Unique preserve order
        self.factor_cols_all = list(dict.fromkeys(self.factor_cols_all))
        self.factor_cols_daily = list(dict.fromkeys(self.factor_cols_daily))
        self.factor_cols_monthly = list(dict.fromkeys(self.factor_cols_monthly))

        self.factor_cols_selected = self.factor_cols_all.copy()

        return panel

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    def preprocess_panel(
            self,
            selected_factors: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Preprocess panel:
        - clean infinities
        - optional missingness filter
        - winsorize by date
        - z-score by date

        Important:
        - self.factor_cols_all remains the full merged factor universe
        - self.factor_cols_selected is the active subset used for preprocessing
        """

        if self.panel is None:
            raise ValueError("Panel must be present before preprocessing.")

        df = self.panel.copy()

        # Master merged factor universe: do not overwrite destructively
        master_factor_cols = [c for c in self.factor_cols_all if c in df.columns]

        # Resolve requested subset
        if selected_factors is None:
            factor_cols = master_factor_cols.copy()
        else:
            # allow selected_factors to match either exact merged names or base names
            factor_cols = self._resolve_requested_factor_cols(
                available_cols=master_factor_cols,
                requested_factors=selected_factors,
            )

        # Replace infs
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Extra coercion safety
        if factor_cols:
            df = self.coerce_factor_columns_to_numeric(df, factor_cols)

        # Missingness filter on active subset only
        if self.preprocess_config.missing_threshold is not None and factor_cols:
            keep_cols = []
            for c in factor_cols:
                missing_rate = df[c].isna().mean()
                if missing_rate <= self.preprocess_config.missing_threshold:
                    keep_cols.append(c)
                elif not self.preprocess_config.allow_missing_factors:
                    warnings.warn(
                        f"Dropping factor '{c}' due to missing rate {missing_rate:.2%} "
                        f"> threshold {self.preprocess_config.missing_threshold:.2%}"
                    )
            factor_cols = keep_cols

        # Optional fill
        if self.preprocess_config.fill_method is not None and factor_cols:
            df = df.sort_values([self.column_config.id_col, self.column_config.date_col]).copy()
            df[factor_cols] = (
                df.groupby(self.column_config.id_col, group_keys=False)[factor_cols]
                .apply(lambda x: x.fillna(method=self.preprocess_config.fill_method))
            )

        # Cross-sectional size filter
        cs_size = df.groupby(self.column_config.date_col)[self.column_config.id_col].transform("count")
        df = df[cs_size >= self.preprocess_config.min_cross_section_size].copy()

        # Winsorization
        winsor_cols = []
        if self.column_config.gex_col in df.columns:
            winsor_cols.append(self.column_config.gex_col)
        winsor_cols.extend(factor_cols)

        if self.preprocess_config.winsorize and winsor_cols:
            df = self.winsorize_by_date(df, winsor_cols)

        # Z-score
        z_cols = []
        if self.column_config.gex_col in df.columns:
            z_cols.append(self.column_config.gex_col)
        z_cols.extend(factor_cols)

        if self.preprocess_config.cross_sectional_zscore and z_cols:
            df = self.zscore_by_date(df, z_cols)

        # Save active subset separately
        self.factor_cols_selected = factor_cols
        self.panel = df
        return df

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    def build_targets(self) -> pd.DataFrame:
        """
        Create forward targets from CRSP return column by default.
        """

        if self.panel is None:
            raise ValueError("Panel must be present before building targets.")

        df = self.panel.copy()
        id_col = self.column_config.id_col
        ret_col = self.column_config.ret_col

        if ret_col not in df.columns:
            raise ValueError(f"Return column '{ret_col}' is not in panel.")

        df = df.sort_values([id_col, self.column_config.date_col]).copy()

        for h in self.target_config.horizons:
            # Signed forward return
            if self.target_config.create_signed_return:
                df[f"ret_fwd_{h}d"] = self.make_forward_return(df, ret_col=ret_col, horizon=h)

            # Absolute return
            if self.target_config.create_abs_return:
                if f"ret_fwd_{h}d" not in df.columns:
                    df[f"ret_fwd_{h}d"] = self.make_forward_return(df, ret_col=ret_col, horizon=h)
                df[f"abs_ret_fwd_{h}d"] = df[f"ret_fwd_{h}d"].abs()

            # Squared return
            if self.target_config.create_squared_return:
                if f"ret_fwd_{h}d" not in df.columns:
                    df[f"ret_fwd_{h}d"] = self.make_forward_return(df, ret_col=ret_col, horizon=h)
                df[f"sq_ret_fwd_{h}d"] = df[f"ret_fwd_{h}d"] ** 2

            # Realized volatility on future horizon
            if self.target_config.create_realized_vol:
                df[f"rv_fwd_{h}d"] = self.make_forward_realized_vol(df, ret_col=ret_col, horizon=h)

            # Downside semivariance
            if self.target_config.create_downside_semivar:
                df[f"downside_semivar_fwd_{h}d"] = self.make_forward_downside_semivar(df, ret_col=ret_col, horizon=h)

            # Tail indicators
            if self.target_config.create_tail_indicators:
                if f"ret_fwd_{h}d" not in df.columns:
                    df[f"ret_fwd_{h}d"] = self.make_forward_return(df, ret_col=ret_col, horizon=h)

                tail_map = self.make_forward_tail_indicators(
                    df,
                    y_col=f"ret_fwd_{h}d",
                    lower_q=min(self.target_config.tail_quantiles),
                    upper_q=max(self.target_config.tail_quantiles),
                )
                for k, v in tail_map.items():
                    df[f"{k}_fwd_{h}d"] = v

        self.panel = df
        return df

    # --------------------------------------------------------
    # Regimes
    # --------------------------------------------------------

    def build_gex_regimes(self, n_buckets: int = 5) -> pd.DataFrame:
        """
        Build GEX sign and quantile regimes.
        """

        if self.panel is None:
            raise ValueError("Panel must be present before building GEX regimes.")

        df = self.panel.copy()
        gex_col = self.column_config.gex_col
        gex_z_col = f"{gex_col}{self.preprocess_config.zscore_suffix}"

        if gex_col not in df.columns:
            raise ValueError(f"GEX column '{gex_col}' not found.")

        df["neg_gex_flag"] = (df[gex_col] < 0).astype("Int64")
        df["pos_gex_flag"] = (df[gex_col] >= 0).astype("Int64")
        df["gex_sign_regime"] = np.where(df[gex_col] < 0, "neg", "pos")

        bucket_source = gex_z_col if gex_z_col in df.columns else gex_col
        df["gex_q"] = self.assign_quantile_bucket_by_date(df, bucket_source, n_buckets=n_buckets)

        # Extreme bottom bucket as fragile negative regime candidate
        df["extreme_neg_gex_flag"] = (df["gex_q"] == 1).astype("Int64")

        self.panel = df
        return df

    # --------------------------------------------------------
    # File IO
    # --------------------------------------------------------

    def _read_file(
            self,
            path: str | Path,
            file_type: Optional[str] = None,
    ) -> pd.DataFrame:
        path = Path(path)
        file_type = file_type or self._infer_file_type(path)

        if file_type == "parquet":
            df = pd.read_parquet(path)
        elif file_type == "csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file type '{file_type}' for file {path}.")
        return df

    def _infer_file_type(self, path: str | Path) -> str:
        suffix = Path(path).suffix.lower()
        if suffix == ".parquet":
            return "parquet"
        if suffix == ".csv":
            return "csv"
        raise ValueError(f"Cannot infer file type from suffix '{suffix}' for path {path}.")

    # --------------------------------------------------------
    # Validation / standardization
    # --------------------------------------------------------

    def validate_required_columns(self, df: pd.DataFrame, required_cols: list[str]) -> None:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _standardize_date_and_id(
            self,
            df: pd.DataFrame,
            id_col: str,
            date_col: str,
            out_id_col: Optional[str] = None,
            out_date_col: Optional[str] = None,
            date_format: Optional[str] = None,
    ) -> pd.DataFrame:
        out_id_col = out_id_col or id_col
        out_date_col = out_date_col or date_col

        if id_col not in df.columns:
            raise ValueError(f"ID column '{id_col}' not found.")
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found.")

        df = df.copy()

        if out_id_col != id_col:
            df = df.rename(columns={id_col: out_id_col})
        if out_date_col != date_col:
            df = df.rename(columns={date_col: out_date_col})

        df[out_date_col] = pd.to_datetime(df[out_date_col], format=date_format, errors="coerce")
        if df[out_date_col].isna().any():
            bad = df[df[out_date_col].isna()].head()
            warnings.warn(f"Some dates could not be parsed. Example bad rows:\n{bad}")

        return df

    def _filter_date_range(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        out = df.copy()
        if self.data_config.start_date is not None:
            out = out[out[date_col] >= pd.Timestamp(self.data_config.start_date)].copy()
        if self.data_config.end_date is not None:
            out = out[out[date_col] <= pd.Timestamp(self.data_config.end_date)].copy()
        return out

    def _deduplicate_key(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.preprocess_config.drop_duplicate_keys:
            return df
        key = [self.column_config.id_col, self.column_config.date_col]
        return df.sort_values(key).drop_duplicates(subset=key, keep="last").reset_index(drop=True)

    def validate_unique_key(self, df: pd.DataFrame, id_col: str, date_col: str) -> None:
        dup = df.duplicated([id_col, date_col]).sum()
        if dup > 0:
            raise ValueError(f"Found {dup} duplicated ({id_col}, {date_col}) rows.")

    # --------------------------------------------------------
    # Factor utilities
    # --------------------------------------------------------

    def select_numeric_factor_columns(
            self,
            df: pd.DataFrame,
            exclude_cols: Optional[list[str]] = None,
    ) -> list[str]:
        exclude_cols = exclude_cols or []
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in num_cols if c not in exclude_cols]

    def _merge_daily_factor_df(
            self,
            panel: pd.DataFrame,
            factor_df: pd.DataFrame,
            spec: FactorSpec,
    ) -> pd.DataFrame:
        merge_id_col = self.column_config.id_col
        merge_date_col = self.column_config.date_col

        if merge_id_col not in panel.columns:
            raise ValueError(
                f"Base panel missing merge id column '{merge_id_col}' for factor source '{spec.name}'."
            )
        if merge_id_col not in factor_df.columns:
            raise ValueError(
                f"Factor df missing merge id column '{merge_id_col}' for factor source '{spec.name}'."
            )
        if merge_date_col not in panel.columns or merge_date_col not in factor_df.columns:
            raise ValueError(
                f"Merge date column '{merge_date_col}' missing for factor source '{spec.name}'."
            )

        overlap = [
            c for c in (spec.factor_cols or [])
            if c in panel.columns and c not in [merge_id_col, merge_date_col]
        ]
        if overlap:
            rename_map = {c: f"{c}__{spec.name}" for c in overlap}
            factor_df = factor_df.rename(columns=rename_map)
            spec.factor_cols = [rename_map.get(c, c) for c in (spec.factor_cols or [])]

        factor_df[merge_id_col] = pd.to_numeric(factor_df[merge_id_col], errors="coerce").astype("Int64")
        panel[merge_id_col] = pd.to_numeric(panel[merge_id_col], errors="coerce").astype("Int64")

        merged = pd.merge(
            panel,
            factor_df,
            how=spec.merge_how,
            on=[merge_id_col, merge_date_col],
        )
        return merged

    def _merge_monthly_factor_df(
            self,
            panel: pd.DataFrame,
            factor_df: pd.DataFrame,
            spec: FactorSpec,
    ) -> pd.DataFrame:
        """
        Monthly merge via effective year-month key after lag_months.
        Factor df is expected to already be mapped to permno.
        """
        p = panel.copy()
        f = factor_df.copy()

        date_col = self.column_config.date_col
        id_col = self.column_config.id_col

        if id_col not in p.columns or id_col not in f.columns:
            raise ValueError(
                f"Monthly merge for '{spec.name}' requires '{id_col}' in both panel and factor df."
            )

        p[id_col] = pd.to_numeric(p[id_col], errors="coerce").astype("Int64")
        f[id_col] = pd.to_numeric(f[id_col], errors="coerce").astype("Int64")

        p["ym"] = p[date_col].dt.to_period("M")
        f["ym"] = f[date_col].dt.to_period("M")

        if spec.lag_months != 0:
            f["ym"] = f["ym"] + spec.lag_months

        keep_cols = [id_col, "ym"] + (spec.factor_cols or [])
        f = f[keep_cols].copy()
        f = f.drop_duplicates(subset=[id_col, "ym"], keep="last")

        overlap = [
            c for c in (spec.factor_cols or [])
            if c in p.columns and c not in [id_col, date_col]
        ]
        if overlap:
            rename_map = {c: f"{c}__{spec.name}" for c in overlap}
            f = f.rename(columns=rename_map)
            spec.factor_cols = [rename_map.get(c, c) for c in (spec.factor_cols or [])]

        p = pd.merge(
            p,
            f,
            how=spec.merge_how,
            on=[id_col, "ym"],
        )

        p = p.drop(columns=["ym"])
        return p

    def _resolve_requested_factor_cols(
            self,
            available_cols: list[str],
            requested_factors: list[str],
    ) -> list[str]:
        """
        Resolve requested factor names against available merged factor columns.

        Supports:
        - exact merged column names
        - raw/base names before suffixes like '__bs'
        - raw/base names before z-score suffix

        Example:
        requested 'ivol' can match:
        - 'ivol'
        - 'ivol__bs'
        - 'ivol__foo'
        """

        if not requested_factors:
            return available_cols.copy()

        z_suffix = self.preprocess_config.zscore_suffix
        resolved = []

        for req in requested_factors:
            # exact match first
            if req in available_cols:
                resolved.append(req)
                continue

            matches = []
            for col in available_cols:
                base = col
                if base.endswith(z_suffix):
                    base = base[: -len(z_suffix)]
                raw_base = base.split("__")[0]
                if raw_base == req or base == req:
                    matches.append(col)

            if matches:
                resolved.extend(matches)

        # unique preserve order
        return list(dict.fromkeys([c for c in resolved if c in available_cols]))

    # --------------------------------------------------------
    # Preprocessing utilities
    # --------------------------------------------------------

    def winsorize_by_date(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        date_col = self.column_config.date_col
        lower = self.preprocess_config.winsor_lower
        upper = self.preprocess_config.winsor_upper

        for c in cols:
            if c not in out.columns:
                continue

            # Force numeric if possible
            out[c] = pd.to_numeric(out[c], errors="coerce")

            # Skip if still effectively unusable
            if not pd.api.types.is_numeric_dtype(out[c]):
                continue

            def _clip(s: pd.Series) -> pd.Series:
                s = pd.to_numeric(s, errors="coerce")
                if s.notna().sum() < 3:
                    return s
                lo = s.quantile(lower)
                hi = s.quantile(upper)
                return s.clip(lower=lo, upper=hi)

            out[c] = out.groupby(date_col, group_keys=False)[c].apply(_clip)

        return out

    def zscore_by_date(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        date_col = self.column_config.date_col
        suf = self.preprocess_config.zscore_suffix

        for c in cols:
            if c not in out.columns:
                continue

            def _z(s: pd.Series) -> pd.Series:
                std = s.std(ddof=0)
                if pd.isna(std) or std == 0:
                    return pd.Series(np.nan, index=s.index)
                return (s - s.mean()) / std

            out[f"{c}{suf}"] = out.groupby(date_col, group_keys=False)[c].apply(_z)

        return out

    # --------------------------------------------------------
    # Target builders
    # --------------------------------------------------------

    def make_forward_return(
            self,
            df: pd.DataFrame,
            ret_col: str,
            horizon: int,
    ) -> pd.Series:
        """
        Forward compounded return from t+1 to t+h using simple or log returns.
        """
        id_col = self.column_config.id_col

        def _forward_simple(s: pd.Series) -> pd.Series:
            vals = s.to_numpy(dtype=float)
            out = np.full(len(vals), np.nan)
            for i in range(len(vals)):
                start = i + 1
                end = i + 1 + horizon
                if end > len(vals):
                    continue
                window = vals[start:end]
                if np.isnan(window).any():
                    continue
                out[i] = np.prod(1.0 + window) - 1.0
            return pd.Series(out, index=s.index)

        def _forward_log(s: pd.Series) -> pd.Series:
            vals = s.to_numpy(dtype=float)
            out = np.full(len(vals), np.nan)
            for i in range(len(vals)):
                start = i + 1
                end = i + 1 + horizon
                if end > len(vals):
                    continue
                window = vals[start:end]
                if np.isnan(window).any():
                    continue
                out[i] = np.exp(np.log1p(window).sum()) - 1.0
            return pd.Series(out, index=s.index)

        func = _forward_simple if self.target_config.return_type == "simple" else _forward_log
        return df.groupby(id_col, group_keys=False)[ret_col].apply(func)

    def make_forward_realized_vol(
            self,
            df: pd.DataFrame,
            ret_col: str,
            horizon: int,
    ) -> pd.Series:
        id_col = self.column_config.id_col

        def _rv(s: pd.Series) -> pd.Series:
            vals = s.to_numpy(dtype=float)
            out = np.full(len(vals), np.nan)
            for i in range(len(vals)):
                start = i + 1
                end = i + 1 + horizon
                if end > len(vals):
                    continue
                window = vals[start:end]
                if np.isnan(window).any():
                    continue
                out[i] = np.sqrt(np.mean(window ** 2))
            return pd.Series(out, index=s.index)

        return df.groupby(id_col, group_keys=False)[ret_col].apply(_rv)

    def make_forward_downside_semivar(
            self,
            df: pd.DataFrame,
            ret_col: str,
            horizon: int,
    ) -> pd.Series:
        id_col = self.column_config.id_col

        def _dsv(s: pd.Series) -> pd.Series:
            vals = s.to_numpy(dtype=float)
            out = np.full(len(vals), np.nan)
            for i in range(len(vals)):
                start = i + 1
                end = i + 1 + horizon
                if end > len(vals):
                    continue
                window = vals[start:end]
                if np.isnan(window).any():
                    continue
                neg = np.minimum(window, 0.0)
                out[i] = np.mean(neg ** 2)
            return pd.Series(out, index=s.index)

        return df.groupby(id_col, group_keys=False)[ret_col].apply(_dsv)

    def make_forward_tail_indicators(
            self,
            df: pd.DataFrame,
            y_col: str,
            lower_q: float = 0.05,
            upper_q: float = 0.95,
    ) -> dict[str, pd.Series]:
        """
        Construct tail indicators from already-created forward return column.
        """

        y = df[y_col]

        if self.target_config.tail_groupby == "full_sample":
            lo = y.quantile(lower_q)
            hi = y.quantile(upper_q)
            tail_left = (y <= lo).astype("Int64")
            tail_right = (y >= hi).astype("Int64")
            tail_abs = ((y <= lo) | (y >= hi)).astype("Int64")

        elif self.target_config.tail_groupby == "by_date":
            date_col = self.column_config.date_col
            lo = df.groupby(date_col)[y_col].transform(lambda s: s.quantile(lower_q))
            hi = df.groupby(date_col)[y_col].transform(lambda s: s.quantile(upper_q))
            tail_left = (y <= lo).astype("Int64")
            tail_right = (y >= hi).astype("Int64")
            tail_abs = ((y <= lo) | (y >= hi)).astype("Int64")

        elif self.target_config.tail_groupby == "by_stock":
            id_col = self.column_config.id_col
            lo = df.groupby(id_col)[y_col].transform(lambda s: s.quantile(lower_q))
            hi = df.groupby(id_col)[y_col].transform(lambda s: s.quantile(upper_q))
            tail_left = (y <= lo).astype("Int64")
            tail_right = (y >= hi).astype("Int64")
            tail_abs = ((y <= lo) | (y >= hi)).astype("Int64")

        else:
            raise ValueError(f"Unsupported tail_groupby '{self.target_config.tail_groupby}'.")

        return {
            "tail_left": tail_left,
            "tail_right": tail_right,
            "tail_abs": tail_abs,
        }

    # --------------------------------------------------------
    # Regime utilities
    # --------------------------------------------------------

    def assign_quantile_bucket_by_date(
            self,
            df: pd.DataFrame,
            col: str,
            n_buckets: int = 5,
    ) -> pd.Series:
        date_col = self.column_config.date_col

        def _bucket(s: pd.Series) -> pd.Series:
            valid = s.notna()
            out = pd.Series(np.nan, index=s.index, dtype="float")
            if valid.sum() < n_buckets:
                return out
            try:
                out.loc[valid] = pd.qcut(
                    s.loc[valid],
                    q=n_buckets,
                    labels=False,
                    duplicates="drop",
                ) + 1
            except ValueError:
                return out
            return out

        return df.groupby(date_col, group_keys=False)[col].apply(_bucket).astype("Float64")

    # ========================================================
    # Phase 2 public runner
    # ========================================================

    def run_phase2(
            self,
            factor_cols: Optional[list[str]] = None,
            regime_validation_y_cols: Optional[list[str]] = None,
            interaction_y_cols: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Phase 2:
        1. regime validation
        2. interaction regression loop
        """

        if self.panel is None:
            raise ValueError("Run Phase 1 first so self.panel is available.")

        regime_validation_y_cols = regime_validation_y_cols or self.regression_config.regime_validation_y_cols
        interaction_y_cols = interaction_y_cols or self.regression_config.interaction_y_cols

        if factor_cols is None:
            factor_cols = self.get_active_factor_cols(prefer_selected=True)
        else:
            factor_cols = self._resolve_requested_factor_cols(
                available_cols=self.get_active_factor_cols(prefer_selected=False),
                requested_factors=factor_cols,
            )

        regime_validation = self.run_regime_validation(y_cols=regime_validation_y_cols)
        interaction_regression = self.run_interaction_regression(
            factor_cols=factor_cols,
            y_cols=interaction_y_cols,
        )

        phase2 = {
            "regime_validation": regime_validation,
            "interaction_regression": interaction_regression,
        }

        self.artifacts["phase2"] = phase2
        return phase2

    # ========================================================
    # Module 1: Regime validation
    # ========================================================

    def run_regime_validation(
            self,
            y_cols: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Validate that GEX behaves more like a risk/dispersion regime variable
        than a directional alpha.

        Outputs:
        - gex bucket sort tables
        - fama-macbeth style regression table on each Y
        """

        if self.panel is None:
            raise ValueError("Panel is not available.")

        df = self.panel.copy()
        y_cols = y_cols or self.regression_config.regime_validation_y_cols

        gex_col = self.column_config.gex_col
        gex_z_col = f"{gex_col}{self.preprocess_config.zscore_suffix}"
        gex_sort_col = gex_z_col if gex_z_col in df.columns else gex_col

        sort_tables = {}
        regression_rows = []
        plot_paths = {}

        for y_col in y_cols:
            if y_col not in df.columns:
                continue

            # sort table
            sort_df = self._make_univariate_bucket_table(
                df=df,
                signal_col=gex_sort_col,
                y_col=y_col,
                n_buckets=self.regression_config.n_buckets,
                label_prefix="gex_q",
            )
            sort_tables[y_col] = sort_df

            if self.regression_config.save_phase2_tables:
                self._save_table(
                    sort_df,
                    filename=f"phase2_regime_validation_sort_{y_col}.csv",
                )

            # regression
            x_cols = [gex_z_col] if gex_z_col in df.columns else [gex_col]
            if self.regression_config.use_controls:
                x_cols = x_cols + self._get_available_control_cols(df)

            reg_res = self.run_fama_macbeth(
                df=df,
                y_col=y_col,
                x_cols=x_cols,
                min_obs_per_date=self.regression_config.min_obs_per_date,
                nw_lags=self.regression_config.nw_lags,
            )

            reg_table = self.summarize_fama_macbeth_results(reg_res)
            reg_table["y_col"] = y_col
            regression_rows.append(reg_table)

            # plot
            if self.regression_config.save_phase2_plots:
                plot_path = self._plot_univariate_bucket_response(
                    sort_df=sort_df,
                    signal_name="GEX",
                    y_col=y_col,
                    filename=f"phase2_regime_validation_{y_col}.png",
                )
                plot_paths[y_col] = plot_path

        regression_table = (
            pd.concat(regression_rows, ignore_index=True)
            if regression_rows else pd.DataFrame()
        )

        if self.regression_config.save_phase2_tables and not regression_table.empty:
            self._save_table(
                regression_table,
                filename="phase2_regime_validation_regressions.csv",
            )

        out = {
            "sort_tables": sort_tables,
            "regression_table": regression_table,
            "plots": plot_paths,
        }

        self.artifacts["regime_validation"] = out
        return out

    # ========================================================
    # Module 2: interaction regression loop
    # ========================================================

    def run_interaction_regression(
            self,
            factor_cols: Optional[list[str]] = None,
            y_cols: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        For each factor F and each outcome Y, run cross-sectional regression:

            Y = a + b*gex_z + c*f_z + d*(gex_z*f_z) + controls + error

        Main output:
        - one summary row per (factor, y_col)
        """

        if self.panel is None:
            raise ValueError("Panel is not available.")

        df = self.panel.copy()
        y_cols = y_cols or self.regression_config.interaction_y_cols

        gex_col = self.column_config.gex_col
        gex_z_col = f"{gex_col}{self.preprocess_config.zscore_suffix}"
        if gex_z_col not in df.columns:
            raise ValueError(
                f"Expected z-scored GEX column '{gex_z_col}' not found. "
                "Make sure Phase 1 preprocessing created it."
            )
        if factor_cols is None:
            factor_cols = self.get_active_factor_cols(prefer_selected=True)
        else:
            factor_cols = self._resolve_requested_factor_cols(
                available_cols=self.get_active_factor_cols(prefer_selected=False),
                requested_factors=factor_cols,
            )

        factor_cols = [c for c in factor_cols if c in df.columns]

        if not factor_cols:
            warnings.warn(
                "No factor columns resolved for interaction regression. "
                "Check selected_factors naming, suffixes, and panel columns."
            )
        summary_rows = []
        full_tables = {}
        top_plot_paths = {}

        controls = self._get_available_control_cols(df) if self.regression_config.use_controls else []

        for y_col in y_cols:
            if y_col not in df.columns:
                continue

            per_y_rows = []

            for factor_col in factor_cols:
                factor_z_col = self._resolve_factor_z_col(df, factor_col)
                if factor_z_col is None:
                    continue

                work = df[[self.column_config.id_col, self.column_config.date_col, y_col, gex_z_col,
                           factor_z_col] + controls].copy()
                work["interaction"] = work[gex_z_col] * work[factor_z_col]

                x_cols = [gex_z_col, factor_z_col, "interaction"] + controls

                reg_res = self.run_fama_macbeth(
                    df=work,
                    y_col=y_col,
                    x_cols=x_cols,
                    min_obs_per_date=self.regression_config.min_obs_per_date,
                    nw_lags=self.regression_config.nw_lags,
                )

                coef_table = self.summarize_fama_macbeth_results(reg_res)
                coef_table["factor"] = factor_col
                coef_table["factor_z_col"] = factor_z_col
                coef_table["y_col"] = y_col

                # reshape to one-row summary for ranking
                summary_row = self._extract_interaction_summary_row(
                    coef_table=coef_table,
                    factor=factor_col,
                    factor_z_col=factor_z_col,
                    y_col=y_col,
                )
                per_y_rows.append(summary_row)

            if per_y_rows:
                per_y_df = pd.DataFrame(per_y_rows).sort_values(
                    by="abs_t_interaction", ascending=False
                ).reset_index(drop=True)
            else:
                per_y_df = pd.DataFrame()

            full_tables[y_col] = per_y_df
            summary_rows.append(per_y_df)

            if self.regression_config.save_phase2_tables and not per_y_df.empty:
                self._save_table(
                    per_y_df,
                    filename=f"phase2_interaction_summary_{y_col}.csv",
                )

            if self.regression_config.save_phase2_plots and not per_y_df.empty:
                plot_path = self._plot_top_interactions(
                    per_y_df=per_y_df,
                    y_col=y_col,
                    top_n=15,
                    filename=f"phase2_interaction_top_{y_col}.png",
                )
                top_plot_paths[y_col] = plot_path

        summary_table = (
            pd.concat(summary_rows, ignore_index=True)
            if summary_rows else pd.DataFrame()
        )

        if self.regression_config.save_phase2_tables and not summary_table.empty:
            self._save_table(
                summary_table,
                filename="phase2_interaction_summary_all.csv",
            )

        out = {
            "summary_table": summary_table,
            "tables_by_y": full_tables,
            "plots": top_plot_paths,
        }

        self.artifacts["interaction_regression"] = out
        return out

    # ========================================================
    # Fama-MacBeth style regression helpers
    # ========================================================

    def run_fama_macbeth(
            self,
            df: pd.DataFrame,
            y_col: str,
            x_cols: list[str],
            min_obs_per_date: Optional[int] = None,
            nw_lags: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Run daily cross-sectional OLS, then average slopes over time.
        Newey-West t-stats are computed on the time series of daily slopes.
        """

        min_obs_per_date = min_obs_per_date or self.regression_config.min_obs_per_date
        nw_lags = nw_lags if nw_lags is not None else self.regression_config.nw_lags

        date_col = self.column_config.date_col
        use_cols = [date_col, y_col] + x_cols
        work = df[use_cols].copy()

        beta_rows = []
        nobs_rows = []

        for dt, sub in work.groupby(date_col):
            sub = sub.dropna(subset=[y_col] + x_cols).copy()
            if len(sub) < min_obs_per_date:
                continue

            y = sub[y_col].astype(float)
            X = sub[x_cols].astype(float)

            if self.regression_config.add_intercept:
                X = sm.add_constant(X, has_constant="add")

            # skip singular or no variation cases gracefully
            try:
                model = sm.OLS(y, X, missing="drop")
                res = model.fit()
            except Exception:
                continue

            params = res.params.to_dict()
            params[date_col] = dt
            beta_rows.append(params)

            nobs_rows.append({
                date_col: dt,
                "n_obs": int(res.nobs),
                "r2": float(res.rsquared) if hasattr(res, "rsquared") else np.nan,
            })

        beta_df = pd.DataFrame(beta_rows)
        nobs_df = pd.DataFrame(nobs_rows)

        if beta_df.empty:
            return {
                "betas_by_date": pd.DataFrame(),
                "nobs_by_date": nobs_df,
                "summary": pd.DataFrame(),
            }

        # ensure date sorted
        beta_df = beta_df.sort_values(date_col).reset_index(drop=True)

        summary_rows = []
        coef_cols = [c for c in beta_df.columns if c != date_col]

        for coef_col in coef_cols:
            series = beta_df[[date_col, coef_col]].dropna().copy()
            T = len(series)

            if T < self.regression_config.min_dates_required:
                mean_beta = series[coef_col].mean() if T > 0 else np.nan
                summary_rows.append({
                    "term": coef_col,
                    "coef_mean": mean_beta,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "n_dates": T,
                })
                continue

            # HAC on mean-only regression of beta_t on constant
            y = series[coef_col].to_numpy(dtype=float)
            X = np.ones((len(y), 1), dtype=float)

            try:
                mean_model = sm.OLS(y, X).fit()
                hac_cov = cov_hac(mean_model, nlags=nw_lags)
                se = float(np.sqrt(hac_cov[0, 0]))
                coef_mean = float(mean_model.params[0])
                t_stat = coef_mean / se if se > 0 else np.nan
                p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=max(T - 1, 1))) if np.isfinite(t_stat) else np.nan
            except Exception:
                coef_mean = float(np.nanmean(y))
                t_stat = np.nan
                p_value = np.nan

            summary_rows.append({
                "term": coef_col,
                "coef_mean": coef_mean,
                "t_stat": t_stat,
                "p_value": p_value,
                "n_dates": T,
            })

        summary_df = pd.DataFrame(summary_rows)

        # attach average cross-sectional diagnostics
        if not nobs_df.empty:
            summary_df["avg_n_obs"] = nobs_df["n_obs"].mean()
            summary_df["avg_cs_r2"] = nobs_df["r2"].mean()

        return {
            "betas_by_date": beta_df,
            "nobs_by_date": nobs_df,
            "summary": summary_df,
        }

    def summarize_fama_macbeth_results(
            self,
            reg_res: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Return a standard tidy summary table.
        """

        summary = reg_res.get("summary", pd.DataFrame()).copy()
        if summary.empty:
            return pd.DataFrame(
                columns=[
                    "term", "coef_mean", "t_stat", "p_value",
                    "n_dates", "avg_n_obs", "avg_cs_r2"
                ]
            )

        cols = ["term", "coef_mean", "t_stat", "p_value", "n_dates"]
        if "avg_n_obs" in summary.columns:
            cols.append("avg_n_obs")
        if "avg_cs_r2" in summary.columns:
            cols.append("avg_cs_r2")

        return summary[cols].copy()

    def get_active_factor_cols(
            self,
            prefer_selected: bool = True,
    ) -> list[str]:
        """
        Resolve the currently usable raw factor columns from panel state.

        Priority:
        1. selected factors from current run
        2. full merged factor universe
        3. infer from z-score columns in panel
        """

        if self.panel is None:
            return []

        df = self.panel
        z_suffix = self.preprocess_config.zscore_suffix

        # 1. selected factors
        if prefer_selected and getattr(self, "factor_cols_selected", None):
            cols = [c for c in self.factor_cols_selected if c in df.columns]
            if cols:
                return cols

        # 2. full merged factors
        if getattr(self, "factor_cols_all", None):
            cols = [c for c in self.factor_cols_all if c in df.columns]
            if cols:
                return cols

        # 3. infer from z columns in panel
        exclude_prefixes = {
            self.column_config.gex_col,
            self.column_config.ret_col,
        }

        inferred = []
        for col in df.columns:
            if not col.endswith(z_suffix):
                continue
            raw = col[: -len(z_suffix)]
            if raw == self.column_config.gex_col:
                continue
            if raw.startswith("ret_fwd_") or raw.startswith("abs_ret_fwd_") or raw.startswith("rv_fwd_"):
                continue
            if raw.startswith("tail_") or raw.startswith("downside_semivar_") or raw.startswith("sq_ret_"):
                continue
            inferred.append(raw)

        return list(dict.fromkeys([c for c in inferred if c in df.columns]))

    # ========================================================
    # Sort helpers
    # ========================================================

    def _make_univariate_bucket_table(
            self,
            df: pd.DataFrame,
            signal_col: str,
            y_col: str,
            n_buckets: int = 5,
            label_prefix: str = "bucket",
    ) -> pd.DataFrame:
        """
        Create bucket-level average response table.
        """

        work = df[[self.column_config.date_col, signal_col, y_col]].dropna().copy()
        if work.empty:
            return pd.DataFrame()

        bucket_col = f"{label_prefix}_{signal_col}_{n_buckets}"
        work[bucket_col] = self.assign_quantile_bucket_by_date(
            work,
            col=signal_col,
            n_buckets=n_buckets,
        )

        work = work.dropna(subset=[bucket_col]).copy()

        bucket_ts = (
            work.groupby([self.column_config.date_col, bucket_col])[y_col]
            .mean()
            .reset_index()
        )

        summary = (
            bucket_ts.groupby(bucket_col)[y_col]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={bucket_col: "bucket"})
        )

        # spread top-bottom
        pivot = bucket_ts.pivot(
            index=self.column_config.date_col,
            columns=bucket_col,
            values=y_col,
        )

        if (1 in pivot.columns) and (n_buckets in pivot.columns):
            spread = pivot[n_buckets] - pivot[1]
            spread_row = pd.DataFrame([{
                "bucket": "top_minus_bottom",
                "mean": spread.mean(),
                "std": spread.std(ddof=1),
                "count": spread.notna().sum(),
            }])
            summary = pd.concat([summary, spread_row], ignore_index=True)

        summary["y_col"] = y_col
        summary["signal_col"] = signal_col
        return summary

    # ========================================================
    # Interaction summary helpers
    # ========================================================

    def _resolve_factor_z_col(
            self,
            df: pd.DataFrame,
            factor_col: str,
    ) -> Optional[str]:
        """
        Resolve factor column to its z-scored version if available.
        If factor_col already ends with z suffix and exists, use it.
        Otherwise try raw + suffix.
        """

        z_suffix = self.preprocess_config.zscore_suffix

        if factor_col in df.columns and factor_col.endswith(z_suffix):
            return factor_col

        candidate = f"{factor_col}{z_suffix}"
        if candidate in df.columns:
            return candidate

        # fallback: if raw exists and z-score missing, skip for phase 2
        return None

    def _extract_interaction_summary_row(
            self,
            coef_table: pd.DataFrame,
            factor: str,
            factor_z_col: str,
            y_col: str,
    ) -> dict[str, Any]:
        """
        Flatten coefficient table into one row.
        """

        def _grab(term: str, field: str) -> float:
            sub = coef_table.loc[coef_table["term"] == term, field]
            return float(sub.iloc[0]) if len(sub) else np.nan

        row = {
            "factor": factor,
            "factor_z_col": factor_z_col,
            "y_col": y_col,

            "coef_gex": _grab(f"{self.column_config.gex_col}{self.preprocess_config.zscore_suffix}", "coef_mean"),
            "t_gex": _grab(f"{self.column_config.gex_col}{self.preprocess_config.zscore_suffix}", "t_stat"),

            "coef_factor": _grab(factor_z_col, "coef_mean"),
            "t_factor": _grab(factor_z_col, "t_stat"),

            "coef_interaction": _grab("interaction", "coef_mean"),
            "t_interaction": _grab("interaction", "t_stat"),
            "p_interaction": _grab("interaction", "p_value"),

            "n_dates": _grab("interaction", "n_dates"),
            "avg_n_obs": _grab("interaction", "avg_n_obs") if "avg_n_obs" in coef_table.columns else np.nan,
            "avg_cs_r2": _grab("interaction", "avg_cs_r2") if "avg_cs_r2" in coef_table.columns else np.nan,
        }
        row["abs_t_interaction"] = abs(row["t_interaction"]) if pd.notna(row["t_interaction"]) else np.nan
        return row

    # ========================================================
    # Control helpers
    # ========================================================

    def _get_available_control_cols(self, df: pd.DataFrame) -> list[str]:
        """
        Controls come from:
        - regression_config.control_cols
        - column_config.control_cols
        de-duplicated and filtered to existing numeric columns
        """

        candidates = []
        candidates.extend(self.column_config.control_cols or [])
        candidates.extend(self.regression_config.control_cols or [])

        out = []
        seen = set()
        for c in candidates:
            if c in seen:
                continue
            seen.add(c)
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                out.append(c)
        return out

    # ========================================================
    # Saving helpers
    # ========================================================

    def _save_table(
            self,
            df: pd.DataFrame,
            filename: str,
    ) -> str:
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        return str(path)

    # ========================================================
    # Plot helpers
    # ========================================================

    def _plot_univariate_bucket_response(
            self,
            sort_df: pd.DataFrame,
            signal_name: str,
            y_col: str,
            filename: str,
    ) -> Optional[str]:
        if sort_df.empty:
            return None

        plot_df = sort_df[sort_df["bucket"] != "top_minus_bottom"].copy()
        if plot_df.empty:
            return None

        plt.figure(figsize=(8, 5))
        plt.plot(plot_df["bucket"].astype(str), plot_df["mean"], marker="o")
        plt.xlabel(f"{signal_name} bucket")
        plt.ylabel(f"Mean {y_col}")
        plt.title(f"{signal_name} buckets vs {y_col}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)

    def _plot_top_interactions(
            self,
            per_y_df: pd.DataFrame,
            y_col: str,
            top_n: int,
            filename: str,
    ) -> Optional[str]:
        if per_y_df.empty:
            return None

        plot_df = per_y_df.head(top_n).copy()
        plot_df = plot_df.sort_values("coef_interaction")

        plt.figure(figsize=(10, max(5, 0.35 * len(plot_df))))
        plt.barh(plot_df["factor"].astype(str), plot_df["coef_interaction"])
        plt.xlabel("Interaction coefficient")
        plt.ylabel("Factor")
        plt.title(f"Top {top_n} GEX × factor interactions for {y_col}")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)

    # ========================================================
    # Phase 3 public runner
    # ========================================================

    def run_phase3(
            self,
            factor_cols: Optional[list[str]] = None,
            regime_split_y_col: str = "ret_fwd_1d",
            double_sort_y_cols: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Phase 3:
        1. regime-split factor test
        2. double-sort analysis
        """

        if self.panel is None:
            raise ValueError("Run Phase 1 first so self.panel is available.")

        if factor_cols is None:
            factor_cols = self.get_active_factor_cols(prefer_selected=True)
        else:
            factor_cols = self._resolve_requested_factor_cols(
                available_cols=self.get_active_factor_cols(prefer_selected=False),
                requested_factors=factor_cols,
            )

        double_sort_y_cols = double_sort_y_cols or [
            "ret_fwd_1d",
            "abs_ret_fwd_1d",
            "rv_fwd_5d",
            "tail_left_fwd_1d",
        ]

        regime_split = self.run_regime_split_factor_test(
            factor_cols=factor_cols,
            y_col=regime_split_y_col,
        )
        double_sort = self.run_double_sort_analysis(
            factor_cols=factor_cols,
            y_cols=double_sort_y_cols,
        )

        phase3 = {
            "regime_split_factor_test": regime_split,
            "double_sort": double_sort,
        }
        self.artifacts["phase3"] = phase3
        return phase3

    # ========================================================
    # Module 3: Regime-split factor test
    # ========================================================

    def run_regime_split_factor_test(
            self,
            factor_cols: Optional[list[str]] = None,
            y_col: str = "ret_fwd_1d",
    ) -> dict[str, Any]:
        """
        Test factor efficacy separately inside negative-GEX and positive-GEX regimes.

        Metrics per factor and regime:
        - mean spread return
        - spread vol
        - spread sharpe
        - spread max drawdown
        - tail frequency
        - IC mean
        - rank IC mean
        """

        if self.panel is None:
            raise ValueError("Panel is not available.")

        df = self.panel.copy()

        if "gex_sign_regime" not in df.columns:
            raise ValueError("Run build_gex_regimes() in Phase 1 before Phase 3.")

        if y_col not in df.columns:
            raise ValueError(f"Target column '{y_col}' not found in panel.")

        if factor_cols is None:
            factor_cols = self.get_active_factor_cols(prefer_selected=True)
        else:
            factor_cols = self._resolve_requested_factor_cols(
                available_cols=self.get_active_factor_cols(prefer_selected=False),
                requested_factors=factor_cols,
            )

        factor_cols = [c for c in factor_cols if c in df.columns]
        if not factor_cols:
            warnings.warn("No factor columns resolved for regime-split factor test.")
            empty = {
                "factor_metrics_by_regime": pd.DataFrame(),
                "diff_table": pd.DataFrame(),
                "plots": {},
            }
            self.artifacts["regime_split_factor_test"] = empty
            return empty

        rows = []
        diff_rows = []
        plot_paths = {}

        for factor_col in factor_cols:
            factor_z_col = self._resolve_factor_z_col(df, factor_col)
            signal_col = factor_z_col if factor_z_col is not None else factor_col

            if signal_col not in df.columns:
                continue

            per_factor_regime = []

            for regime in ["neg", "pos"]:
                sub = df[df["gex_sign_regime"] == regime].copy()
                if sub.empty:
                    continue

                metrics = self._compute_factor_regime_metrics(
                    df=sub,
                    factor_col=signal_col,
                    y_col=y_col,
                    n_buckets=self.regression_config.n_buckets,
                )
                metrics["factor"] = factor_col
                metrics["signal_col"] = signal_col
                metrics["regime"] = regime

                rows.append(metrics)
                per_factor_regime.append(metrics)

            # diff table
            if len(per_factor_regime) == 2:
                neg_row = next((r for r in per_factor_regime if r["regime"] == "neg"), None)
                pos_row = next((r for r in per_factor_regime if r["regime"] == "pos"), None)

                if neg_row is not None and pos_row is not None:
                    for metric in [
                        "mean_spread_ret",
                        "spread_vol",
                        "spread_sharpe",
                        "spread_max_dd",
                        "tail_freq",
                        "ic_mean",
                        "rank_ic_mean",
                    ]:
                        diff_rows.append({
                            "factor": factor_col,
                            "signal_col": signal_col,
                            "metric": metric,
                            "neg_gex_value": neg_row.get(metric, np.nan),
                            "pos_gex_value": pos_row.get(metric, np.nan),
                            "difference": neg_row.get(metric, np.nan) - pos_row.get(metric, np.nan),
                        })

                    # plot factor spread cumulative by regime
                    if self.regression_config.save_phase2_plots:
                        ts_map = self._compute_factor_spread_timeseries_by_regime(
                            df=df,
                            factor_col=signal_col,
                            y_col=y_col,
                            n_buckets=self.regression_config.n_buckets,
                        )
                        if ts_map:
                            plot_path = self._plot_regime_split_spread_curves(
                                ts_map=ts_map,
                                factor_name=factor_col,
                                y_col=y_col,
                                filename=f"phase3_regime_split_{factor_col}_{y_col}.png",
                            )
                            plot_paths[factor_col] = plot_path

        factor_metrics_by_regime = pd.DataFrame(rows)
        diff_table = pd.DataFrame(diff_rows)

        if self.regression_config.save_phase2_tables:
            if not factor_metrics_by_regime.empty:
                self._save_table(
                    factor_metrics_by_regime,
                    filename=f"phase3_regime_split_metrics_{y_col}.csv",
                )
            if not diff_table.empty:
                self._save_table(
                    diff_table,
                    filename=f"phase3_regime_split_diffs_{y_col}.csv",
                )

        out = {
            "factor_metrics_by_regime": factor_metrics_by_regime,
            "diff_table": diff_table,
            "plots": plot_paths,
        }
        self.artifacts["regime_split_factor_test"] = out
        return out

    # ========================================================
    # Module 4: Double-sort analysis
    # ========================================================

    def run_double_sort_analysis(
            self,
            factor_cols: Optional[list[str]] = None,
            y_cols: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Two directions:
        A. factor bucket first, then GEX bucket
        B. GEX bucket first, then factor bucket

        Outputs:
        - cell mean matrices for each outcome
        - spread-in-spread tables
        - heatmap plots
        """

        if self.panel is None:
            raise ValueError("Panel is not available.")

        df = self.panel.copy()
        y_cols = y_cols or ["ret_fwd_1d", "abs_ret_fwd_1d", "rv_fwd_5d", "tail_left_fwd_1d"]

        if factor_cols is None:
            factor_cols = self.get_active_factor_cols(prefer_selected=True)
        else:
            factor_cols = self._resolve_requested_factor_cols(
                available_cols=self.get_active_factor_cols(prefer_selected=False),
                requested_factors=factor_cols,
            )

        factor_cols = [c for c in factor_cols if c in df.columns]
        if not factor_cols:
            warnings.warn("No factor columns resolved for double-sort analysis.")
            empty = {"results_by_factor": {}, "summary_table": pd.DataFrame(), "plots": {}}
            self.artifacts["double_sort"] = empty
            return empty

        gex_col = self.column_config.gex_col
        gex_z_col = f"{gex_col}{self.preprocess_config.zscore_suffix}"
        gex_signal_col = gex_z_col if gex_z_col in df.columns else gex_col

        results_by_factor = {}
        summary_rows = []
        plot_paths = {}

        for factor_col in factor_cols:
            factor_z_col = self._resolve_factor_z_col(df, factor_col)
            factor_signal_col = factor_z_col if factor_z_col is not None else factor_col

            if factor_signal_col not in df.columns:
                continue

            factor_res = {
                "sort_A": {},
                "sort_B": {},
                "spread_tables": {},
            }

            for y_col in y_cols:
                if y_col not in df.columns:
                    continue

                # A: factor first, GEX second
                sort_A = self._compute_double_sort_table(
                    df=df,
                    first_signal_col=factor_signal_col,
                    second_signal_col=gex_signal_col,
                    y_col=y_col,
                    n_buckets=self.regression_config.n_buckets,
                    first_name="factor",
                    second_name="gex",
                )

                # B: GEX first, factor second
                sort_B = self._compute_double_sort_table(
                    df=df,
                    first_signal_col=gex_signal_col,
                    second_signal_col=factor_signal_col,
                    y_col=y_col,
                    n_buckets=self.regression_config.n_buckets,
                    first_name="gex",
                    second_name="factor",
                )

                factor_res["sort_A"][y_col] = sort_A["cell_mean"]
                factor_res["sort_B"][y_col] = sort_B["cell_mean"]

                spread_row_A = sort_A["spread_summary"].copy()
                spread_row_A["factor"] = factor_col
                spread_row_A["signal_col"] = factor_signal_col
                spread_row_A["y_col"] = y_col
                spread_row_A["sort_direction"] = "factor_then_gex"

                spread_row_B = sort_B["spread_summary"].copy()
                spread_row_B["factor"] = factor_col
                spread_row_B["signal_col"] = factor_signal_col
                spread_row_B["y_col"] = y_col
                spread_row_B["sort_direction"] = "gex_then_factor"

                factor_res["spread_tables"][f"{y_col}_A"] = pd.DataFrame([spread_row_A])
                factor_res["spread_tables"][f"{y_col}_B"] = pd.DataFrame([spread_row_B])

                summary_rows.append(spread_row_A)
                summary_rows.append(spread_row_B)

                if self.regression_config.save_phase2_plots:
                    path_A = self._plot_double_sort_heatmap(
                        matrix=sort_A["cell_mean"],
                        title=f"{factor_col}: factor→GEX on {y_col}",
                        filename=f"phase3_double_sort_A_{factor_col}_{y_col}.png",
                    )
                    path_B = self._plot_double_sort_heatmap(
                        matrix=sort_B["cell_mean"],
                        title=f"{factor_col}: GEX→factor on {y_col}",
                        filename=f"phase3_double_sort_B_{factor_col}_{y_col}.png",
                    )
                    plot_paths[f"{factor_col}_{y_col}_A"] = path_A
                    plot_paths[f"{factor_col}_{y_col}_B"] = path_B

            results_by_factor[factor_col] = factor_res

        summary_table = pd.DataFrame(summary_rows)

        if self.regression_config.save_phase2_tables:
            if not summary_table.empty:
                self._save_table(
                    summary_table,
                    filename="phase3_double_sort_summary.csv",
                )

            # save matrices too
            for factor_name, res in results_by_factor.items():
                for direction_key in ["sort_A", "sort_B"]:
                    for y_col, mat in res[direction_key].items():
                        if isinstance(mat, pd.DataFrame) and not mat.empty:
                            self._save_table(
                                mat.reset_index(),
                                filename=f"phase3_{direction_key}_{factor_name}_{y_col}.csv",
                            )

        out = {
            "results_by_factor": results_by_factor,
            "summary_table": summary_table,
            "plots": plot_paths,
        }
        self.artifacts["double_sort"] = out
        return out

    # ========================================================
    # Regime-split helpers
    # ========================================================

    def _compute_factor_regime_metrics(
            self,
            df: pd.DataFrame,
            factor_col: str,
            y_col: str,
            n_buckets: int = 5,
    ) -> dict[str, Any]:
        """
        Compute factor efficacy metrics in one regime.
        """

        spread_ts = self._compute_factor_spread_timeseries(
            df=df,
            factor_col=factor_col,
            y_col=y_col,
            n_buckets=n_buckets,
        )

        ic_df = self._compute_daily_ic(
            df=df,
            factor_col=factor_col,
            y_col=y_col,
        )

        if spread_ts.empty:
            return {
                "mean_spread_ret": np.nan,
                "spread_vol": np.nan,
                "spread_sharpe": np.nan,
                "spread_max_dd": np.nan,
                "tail_freq": np.nan,
                "ic_mean": ic_df["ic"].mean() if not ic_df.empty else np.nan,
                "rank_ic_mean": ic_df["rank_ic"].mean() if not ic_df.empty else np.nan,
                "n_dates": ic_df["date"].nunique() if not ic_df.empty else 0,
            }

        perf = self._compute_performance_stats(spread_ts["spread_ret"])

        return {
            "mean_spread_ret": perf["mean_ret"],
            "spread_vol": perf["vol"],
            "spread_sharpe": perf["sharpe"],
            "spread_max_dd": perf["max_drawdown"],
            "tail_freq": float((spread_ts["spread_ret"] < 0).mean()),
            "ic_mean": ic_df["ic"].mean() if not ic_df.empty else np.nan,
            "rank_ic_mean": ic_df["rank_ic"].mean() if not ic_df.empty else np.nan,
            "n_dates": int(spread_ts["date"].nunique()),
        }

    def _compute_factor_spread_timeseries(
            self,
            df: pd.DataFrame,
            factor_col: str,
            y_col: str,
            n_buckets: int = 5,
    ) -> pd.DataFrame:
        """
        Daily long-short spread time series based on factor buckets.
        """

        work = df[[self.column_config.date_col, factor_col, y_col]].dropna().copy()
        if work.empty:
            return pd.DataFrame(columns=["date", "top", "bottom", "spread_ret"])

        bucket_col = "__factor_bucket"
        work[bucket_col] = self.assign_quantile_bucket_by_date(
            work,
            col=factor_col,
            n_buckets=n_buckets,
        )
        work = work.dropna(subset=[bucket_col]).copy()

        daily_bucket = (
            work.groupby([self.column_config.date_col, bucket_col])[y_col]
            .mean()
            .reset_index()
        )

        pivot = daily_bucket.pivot(
            index=self.column_config.date_col,
            columns=bucket_col,
            values=y_col,
        ).reset_index()

        if 1.0 not in pivot.columns or float(n_buckets) not in pivot.columns:
            if 1 not in pivot.columns or n_buckets not in pivot.columns:
                return pd.DataFrame(columns=["date", "top", "bottom", "spread_ret"])

        top_col = float(n_buckets) if float(n_buckets) in pivot.columns else n_buckets
        bot_col = 1.0 if 1.0 in pivot.columns else 1

        out = pd.DataFrame({
            "date": pivot[self.column_config.date_col],
            "top": pivot[top_col],
            "bottom": pivot[bot_col],
        })
        out["spread_ret"] = out["top"] - out["bottom"]
        return out.dropna(subset=["spread_ret"]).reset_index(drop=True)

    def _compute_factor_spread_timeseries_by_regime(
            self,
            df: pd.DataFrame,
            factor_col: str,
            y_col: str,
            n_buckets: int = 5,
    ) -> dict[str, pd.DataFrame]:
        out = {}
        for regime in ["neg", "pos"]:
            sub = df[df["gex_sign_regime"] == regime].copy()
            ts = self._compute_factor_spread_timeseries(
                df=sub,
                factor_col=factor_col,
                y_col=y_col,
                n_buckets=n_buckets,
            )
            if not ts.empty:
                out[regime] = ts
        return out

    def _compute_daily_ic(
            self,
            df: pd.DataFrame,
            factor_col: str,
            y_col: str,
    ) -> pd.DataFrame:
        """
        Daily Pearson IC and Spearman rank IC.
        """

        rows = []
        for dt, sub in df.groupby(self.column_config.date_col):
            sub = sub[[factor_col, y_col]].dropna().copy()
            if len(sub) < 5:
                continue

            if sub[factor_col].nunique() < 2 or sub[y_col].nunique() < 2:
                continue

            ic = sub[factor_col].corr(sub[y_col], method="pearson")
            rank_ic = sub[factor_col].corr(sub[y_col], method="spearman")

            rows.append({
                "date": dt,
                "ic": ic,
                "rank_ic": rank_ic,
            })

        return pd.DataFrame(rows)

    def _compute_performance_stats(
            self,
            ret_series: pd.Series,
    ) -> dict[str, float]:
        """
        Generic stats for daily series.
        """

        s = pd.Series(ret_series).dropna().astype(float)
        if s.empty:
            return {
                "mean_ret": np.nan,
                "vol": np.nan,
                "sharpe": np.nan,
                "downside_dev": np.nan,
                "max_drawdown": np.nan,
            }

        mean_ret = float(s.mean())
        vol = float(s.std(ddof=1))
        sharpe = mean_ret / vol if vol > 0 else np.nan

        downside = s[s < 0]
        downside_dev = float(np.sqrt((downside ** 2).mean())) if len(downside) else 0.0

        curve = (1.0 + s).cumprod()
        running_max = curve.cummax()
        drawdown = curve / running_max - 1.0
        max_dd = float(drawdown.min()) if not drawdown.empty else np.nan

        return {
            "mean_ret": mean_ret,
            "vol": vol,
            "sharpe": sharpe,
            "downside_dev": downside_dev,
            "max_drawdown": max_dd,
        }

    # ========================================================
    # Double-sort helpers
    # ========================================================

    def _compute_double_sort_table(
            self,
            df: pd.DataFrame,
            first_signal_col: str,
            second_signal_col: str,
            y_col: str,
            n_buckets: int = 5,
            first_name: str = "first",
            second_name: str = "second",
    ) -> dict[str, Any]:
        """
        Sequential double sort by date:
        1. bucket on first signal within date
        2. within each date-firstbucket group, bucket on second signal
        3. compute cell average of y
        """

        work = df[[self.column_config.date_col, first_signal_col, second_signal_col, y_col]].dropna().copy()

        work[first_signal_col] = pd.to_numeric(work[first_signal_col], errors="coerce")
        work[second_signal_col] = pd.to_numeric(work[second_signal_col], errors="coerce")
        work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
        work = work.dropna(subset=[first_signal_col, second_signal_col, y_col]).copy()

        median_date_count = work.groupby(self.column_config.date_col).size().median()
        if pd.notna(median_date_count) and median_date_count < n_buckets ** 2:
            warnings.warn(
                f"Double-sort: median cross-section size is {median_date_count:.0f} "
                f"but n_buckets²={n_buckets ** 2} — cells will be very thin or empty. "
                f"Consider reducing n_buckets (e.g. to {max(2, int(median_date_count ** 0.5))})."
            )

        if work.empty:
            return {
                "cell_mean": pd.DataFrame(),
                "spread_summary": {
                    "mean_top_minus_bottom_second_in_high_first": np.nan,
                    "mean_top_minus_bottom_second_in_low_first": np.nan,
                    "spread_in_spread": np.nan,
                },
            }

        first_bucket_col = f"__{first_name}_bucket"
        second_bucket_col = f"__{second_name}_bucket"

        work[first_bucket_col] = self.assign_quantile_bucket_by_date(
            work,
            col=first_signal_col,
            n_buckets=n_buckets,
        )
        work = work.dropna(subset=[first_bucket_col]).copy()

        def _assign_second(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            if len(group) < 2:
                group[second_bucket_col] = np.nan
                return group
            effective_q = min(n_buckets, len(group))
            try:
                group[second_bucket_col] = pd.qcut(
                    group[second_signal_col],
                    q=effective_q,
                    labels=False,
                    duplicates="drop",
                ) + 1
            except Exception:
                group[second_bucket_col] = np.nan
            return group

        work = (
            work.groupby([self.column_config.date_col, first_bucket_col], group_keys=False)
            .apply(_assign_second)
        )
        work = work.dropna(subset=[second_bucket_col]).copy()

        cell_ts = (
            work.groupby([self.column_config.date_col, first_bucket_col, second_bucket_col])[y_col]
            .mean()
            .reset_index()
        )

        cell_mean = (
            cell_ts.groupby([first_bucket_col, second_bucket_col])[y_col]
            .mean()
            .unstack(second_bucket_col)
            .sort_index()
        )

        cell_mean = cell_mean.apply(pd.to_numeric, errors="coerce")

        # spread-in-spread summary — use actual bucket range from data
        actual_first_buckets = sorted(cell_ts[first_bucket_col].dropna().unique())

        if len(actual_first_buckets) < 2:
            spread_summary = {
                "mean_top_minus_bottom_second_in_high_first": np.nan,
                "mean_top_minus_bottom_second_in_low_first": np.nan,
                "spread_in_spread": np.nan,
            }
            return {"cell_mean": cell_mean, "spread_summary": spread_summary}

        high_first = actual_first_buckets[-1]
        low_first = actual_first_buckets[0]

        high_rows = cell_ts[cell_ts[first_bucket_col] == high_first]
        low_rows = cell_ts[cell_ts[first_bucket_col] == low_first]

        def _compute_second_spread(sub: pd.DataFrame) -> pd.Series:
            piv = sub.pivot(
                index=self.column_config.date_col,
                columns=second_bucket_col,
                values=y_col,
            )
            sorted_cols = sorted(piv.columns)
            if len(sorted_cols) < 2:
                return pd.Series(dtype=float)
            return piv[sorted_cols[-1]] - piv[sorted_cols[0]]

        high_spread = _compute_second_spread(high_rows)
        low_spread = _compute_second_spread(low_rows)

        high_mean = float(high_spread.mean()) if len(high_spread) else np.nan
        low_mean = float(low_spread.mean()) if len(low_spread) else np.nan
        sis = high_mean - low_mean if pd.notna(high_mean) and pd.notna(low_mean) else np.nan

        spread_summary = {
            "mean_top_minus_bottom_second_in_high_first": high_mean,
            "mean_top_minus_bottom_second_in_low_first": low_mean,
            "spread_in_spread": sis,
        }

        return {
            "cell_mean": cell_mean,
            "spread_summary": spread_summary,
        }

    # ========================================================
    # Phase 3 plot helpers
    # ========================================================

    def _plot_regime_split_spread_curves(
            self,
            ts_map: dict[str, pd.DataFrame],
            factor_name: str,
            y_col: str,
            filename: str,
    ) -> Optional[str]:
        if not ts_map:
            return None

        plt.figure(figsize=(9, 5))

        for regime, ts in ts_map.items():
            if ts.empty:
                continue
            curve = (1.0 + ts["spread_ret"].fillna(0.0)).cumprod()
            plt.plot(ts["date"], curve, label=regime)

        plt.title(f"Factor spread curves by GEX regime: {factor_name} on {y_col}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative growth")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)

    def _plot_double_sort_heatmap(
            self,
            matrix: pd.DataFrame,
            title: str,
            filename: str,
    ) -> Optional[str]:
        if matrix is None or matrix.empty:
            return None

        plot_df = matrix.copy()

        # force numeric values
        plot_df = plot_df.apply(pd.to_numeric, errors="coerce")

        # drop fully empty rows/cols after coercion
        plot_df = plot_df.dropna(axis=0, how="all").dropna(axis=1, how="all")

        if plot_df.empty:
            return None

        values = plot_df.to_numpy(dtype=float)

        plt.figure(figsize=(7, 5))
        plt.imshow(values, aspect="auto")
        plt.colorbar()
        plt.title(title)
        plt.xlabel(plot_df.columns.name if plot_df.columns.name is not None else "Second bucket")
        plt.ylabel(plot_df.index.name if plot_df.index.name is not None else "First bucket")
        plt.xticks(range(len(plot_df.columns)), [str(c) for c in plot_df.columns])
        plt.yticks(range(len(plot_df.index)), [str(i) for i in plot_df.index])
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)

    # ========================================================
    # Phase 4 public runner
    # ========================================================

    def run_phase4(
            self,
            factor_cols: Optional[list[str]] = None,
            target_cols: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Phase 4:
        tail classification / regime classification.
        """

        if self.panel is None:
            raise ValueError("Run Phase 1 first so self.panel is available.")

        if factor_cols is None:
            factor_cols = self.get_active_factor_cols(prefer_selected=True)
        else:
            factor_cols = self._resolve_requested_factor_cols(
                available_cols=self.get_active_factor_cols(prefer_selected=False),
                requested_factors=factor_cols,
            )

        target_cols = target_cols or self.classification_config.target_cols

        out = self.run_tail_classification(
            factor_cols=factor_cols,
            target_cols=target_cols,
        )
        self.artifacts["phase4"] = out
        return out

    # ========================================================
    # Module 5: Tail classification
    # ========================================================

    def run_tail_classification(
            self,
            factor_cols: Optional[list[str]] = None,
            target_cols: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Compare simple classification setups for tail-event prediction.

        Feature sets per factor:
        1. factor_only
        2. gex_only
        3. factor_plus_gex
        4. factor_plus_gex_plus_interaction

        Models in initial version:
        - logistic regression (L2)
        - elastic-net logistic regression

        Leaves room for advanced models later.
        """

        if self.panel is None:
            raise ValueError("Panel is not available.")

        df = self.panel.copy()
        target_cols = target_cols or self.classification_config.target_cols

        gex_col = self.column_config.gex_col
        gex_z_col = f"{gex_col}{self.preprocess_config.zscore_suffix}"
        if gex_z_col not in df.columns:
            raise ValueError(
                f"Expected z-scored GEX column '{gex_z_col}' not found. "
                "Make sure Phase 1 preprocessing created it."
            )

        if factor_cols is None:
            factor_cols = self.get_active_factor_cols(prefer_selected=True)
        else:
            factor_cols = self._resolve_requested_factor_cols(
                available_cols=self.get_active_factor_cols(prefer_selected=False),
                requested_factors=factor_cols,
            )

        factor_cols = [c for c in factor_cols if c in df.columns]
        if not factor_cols:
            warnings.warn("No factor columns resolved for tail classification.")
            empty = {
                "model_scores": pd.DataFrame(),
                "predictions": {},
                "plots": {},
            }
            self.artifacts["tail_classification"] = empty
            return empty

        controls = self._get_available_classification_control_cols(df)
        scores_rows = []
        predictions = {}
        plot_paths = {}

        for target_col in target_cols:
            if target_col not in df.columns:
                continue

            if not self._is_binary_target(df[target_col]):
                warnings.warn(
                    f"Skipping target '{target_col}' because it is not binary after cleaning."
                )
                continue

            predictions[target_col] = {}

            for factor_col in factor_cols:
                factor_z_col = self._resolve_factor_z_col(df, factor_col)
                signal_col = factor_z_col if factor_z_col is not None else factor_col

                if signal_col not in df.columns:
                    continue

                feature_sets = self._build_tail_feature_sets(
                    df=df,
                    factor_col=signal_col,
                    gex_col=gex_z_col,
                    controls=controls,
                )

                for feature_set_name, x_cols in feature_sets.items():
                    if not x_cols:
                        continue

                    split = self._make_classification_split(
                        df=df,
                        feature_cols=x_cols,
                        target_col=target_col,
                    )
                    if split is None:
                        continue

                    X_train, y_train, X_test, y_test, split_meta = split

                    # keep raw predictions for best model selection later
                    local_pred_store = []

                    for model_name in self.classification_config.model_names:
                        model = self._build_classifier(model_name=model_name)
                        if model is None:
                            continue

                        try:
                            model.fit(X_train, y_train)
                            prob = model.predict_proba(X_test)[:, 1]
                            pred = (prob >= 0.5).astype(int)
                        except Exception as e:
                            warnings.warn(
                                f"Classification failed for target={target_col}, "
                                f"factor={factor_col}, feature_set={feature_set_name}, "
                                f"model={model_name}: {e}"
                            )
                            continue

                        metric_row = self._compute_classification_metrics(
                            y_true=y_test,
                            y_prob=prob,
                            y_pred=pred,
                        )
                        metric_row.update({
                            "target_col": target_col,
                            "factor": factor_col,
                            "signal_col": signal_col,
                            "feature_set": feature_set_name,
                            "model_name": model_name,
                            "n_train": len(y_train),
                            "n_test": len(y_test),
                            "train_pos_rate": float(np.mean(y_train)),
                            "test_pos_rate": float(np.mean(y_test)),
                            **split_meta,
                        })
                        scores_rows.append(metric_row)

                        pred_df = pd.DataFrame({
                            "y_true": y_test,
                            "y_prob": prob,
                            "y_pred": pred,
                        })
                        predictions[target_col][f"{factor_col}__{feature_set_name}__{model_name}"] = pred_df
                        local_pred_store.append((metric_row, pred_df))

            # plot best ROC and best score bars per target
            target_scores = pd.DataFrame([r for r in scores_rows if r["target_col"] == target_col])
            if not target_scores.empty:
                if self.classification_config.save_phase4_tables:
                    self._save_table(
                        target_scores,
                        filename=f"phase4_tail_scores_{target_col}.csv",
                    )

                if self.classification_config.save_phase4_plots:
                    # best overall ROC by AUC
                    best_row = target_scores.sort_values("auc", ascending=False).iloc[0].to_dict()
                    pred_key = f'{best_row["factor"]}__{best_row["feature_set"]}__{best_row["model_name"]}'
                    pred_df = predictions[target_col].get(pred_key)

                    if pred_df is not None and not pred_df.empty:
                        roc_path = self._plot_roc_curve(
                            y_true=pred_df["y_true"].to_numpy(),
                            y_prob=pred_df["y_prob"].to_numpy(),
                            title=(
                                f'Best ROC for {target_col}\n'
                                f'{best_row["factor"]} | {best_row["feature_set"]} | {best_row["model_name"]}'
                            ),
                            filename=f"phase4_roc_{target_col}.png",
                        )
                        plot_paths[f"{target_col}_roc"] = roc_path

                    bar_path = self._plot_classification_score_bars(
                        target_scores=target_scores,
                        target_col=target_col,
                        metric="auc",
                        top_n=self.classification_config.top_n_factor_plots,
                        filename=f"phase4_auc_bar_{target_col}.png",
                    )
                    plot_paths[f"{target_col}_auc_bar"] = bar_path

        model_scores = pd.DataFrame(scores_rows)

        if self.classification_config.save_phase4_tables and not model_scores.empty:
            self._save_table(
                model_scores,
                filename="phase4_tail_scores_all.csv",
            )

        out = {
            "model_scores": model_scores,
            "predictions": predictions,
            "plots": plot_paths,
        }
        self.artifacts["tail_classification"] = out
        return out

    # ========================================================
    # Classification helpers
    # ========================================================

    def _get_available_classification_control_cols(self, df: pd.DataFrame) -> list[str]:
        candidates = []
        if self.classification_config.use_controls:
            candidates.extend(self.column_config.control_cols or [])
            candidates.extend(self.regression_config.control_cols or [])
            candidates.extend(self.classification_config.control_cols or [])

        out = []
        seen = set()
        for c in candidates:
            if c in seen:
                continue
            seen.add(c)
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                out.append(c)
        return out

    def _is_binary_target(self, s: pd.Series) -> bool:
        clean = pd.to_numeric(s, errors="coerce").dropna()
        if clean.empty:
            return False
        uniq = set(clean.unique().tolist())
        return uniq.issubset({0, 1})

    def _build_tail_feature_sets(
            self,
            df: pd.DataFrame,
            factor_col: str,
            gex_col: str,
            controls: list[str],
    ) -> dict[str, list[str]]:
        feature_sets = {
            "factor_only": [factor_col] + controls,
            "gex_only": [gex_col] + controls,
            "factor_plus_gex": [factor_col, gex_col] + controls,
        }

        if self.classification_config.include_factor_interaction:
            interaction_col = f"__interaction__{factor_col}__x__{gex_col}"
            if interaction_col not in df.columns:
                df[interaction_col] = pd.to_numeric(df[factor_col], errors="coerce") * pd.to_numeric(df[gex_col],
                                                                                                     errors="coerce")
            feature_sets["factor_plus_gex_plus_interaction"] = [factor_col, gex_col, interaction_col] + controls

        # deduplicate while preserving order
        clean_feature_sets = {}
        for k, cols in feature_sets.items():
            clean_feature_sets[k] = list(dict.fromkeys([c for c in cols if c in df.columns]))
        return clean_feature_sets

    def _make_classification_split(
            self,
            df: pd.DataFrame,
            feature_cols: list[str],
            target_col: str,
    ) -> Optional[tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, dict[str, Any]]]:
        """
        Date-based train/test split to avoid random leakage across dates.
        """

        date_col = self.column_config.date_col
        work = df[[date_col, target_col] + feature_cols].copy()

        # numeric coercion for features / target
        for c in feature_cols:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        work[target_col] = pd.to_numeric(work[target_col], errors="coerce")

        work = work.dropna(subset=[target_col]).copy()
        if work.empty:
            return None

        dates = np.array(sorted(work[date_col].dropna().unique()))
        if len(dates) < 10:
            return None

        train_mask, test_mask, split_meta = self._build_date_split_masks(work[date_col])

        train_df = work.loc[train_mask].copy()
        test_df = work.loc[test_mask].copy()

        # drop rows missing all features
        train_df = train_df.dropna(subset=feature_cols, how="all")
        test_df = test_df.dropna(subset=feature_cols, how="all")

        if len(train_df) < self.classification_config.min_train_rows:
            return None
        if len(test_df) < self.classification_config.min_test_rows:
            return None

        # both classes must exist
        y_train = train_df[target_col].astype(int).to_numpy()
        y_test = test_df[target_col].astype(int).to_numpy()

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            return None

        X_train = train_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()

        return X_train, y_train, X_test, y_test, split_meta

    def _build_date_split_masks(
            self,
            date_series: pd.Series,
    ) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
        """
        Explicit date-range split if provided, otherwise chronological split by unique dates.
        """

        s = pd.to_datetime(date_series)
        cfg = self.classification_config

        if cfg.train_end is not None and cfg.test_start is not None:
            train_start = pd.Timestamp(cfg.train_start) if cfg.train_start else s.min()
            train_end = pd.Timestamp(cfg.train_end)
            test_start = pd.Timestamp(cfg.test_start)
            test_end = pd.Timestamp(cfg.test_end) if cfg.test_end else s.max()

            train_mask = (s >= train_start) & (s <= train_end)
            test_mask = (s >= test_start) & (s <= test_end)

            split_meta = {
                "split_type": "explicit_date_range",
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
            }
            return train_mask, test_mask, split_meta

        # fallback chronological split
        unique_dates = np.array(sorted(s.dropna().unique()))
        split_idx = max(1, int(len(unique_dates) * (1 - cfg.test_size)))
        split_idx = min(split_idx, len(unique_dates) - 1)

        train_dates = set(unique_dates[:split_idx])
        test_dates = set(unique_dates[split_idx:])

        train_mask = s.isin(train_dates)
        test_mask = s.isin(test_dates)

        split_meta = {
            "split_type": "chronological_fraction",
            "train_start": str(pd.Timestamp(min(train_dates)).date()),
            "train_end": str(pd.Timestamp(max(train_dates)).date()),
            "test_start": str(pd.Timestamp(min(test_dates)).date()),
            "test_end": str(pd.Timestamp(max(test_dates)).date()),
        }
        return train_mask, test_mask, split_meta

    def _build_classifier(
            self,
            model_name: str,
    ):
        """
        Simple first-version models only.
        Updated for sklearn penalty deprecation and better convergence.
        """

        from sklearn.preprocessing import StandardScaler

        cfg = self.classification_config

        if model_name == "logit_l2":
            clf = LogisticRegression(
                # penalty deprecated in sklearn >= 1.8
                l1_ratio=0.0,
                C=cfg.C,
                solver="lbfgs",
                max_iter=max(cfg.max_iter, 3000),
                class_weight=cfg.class_weight,
            )

        elif model_name == "logit_elasticnet":
            clf = LogisticRegression(
                # elastic net via l1_ratio
                l1_ratio=cfg.l1_ratio,
                C=cfg.C,
                solver="saga",
                max_iter=max(cfg.max_iter, 5000),
                class_weight=cfg.class_weight,
                tol=1e-3,  # slightly looser tolerance helps convergence
            )

        else:
            warnings.warn(f"Unsupported model_name '{model_name}' in current Phase 4.")
            return None

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=self.classification_config.imputer_strategy)),
            ("scaler", StandardScaler()),
            ("model", clf),
        ])
        return pipe

    def _compute_classification_metrics(
            self,
            y_true: np.ndarray,
            y_prob: np.ndarray,
            y_pred: np.ndarray,
    ) -> dict[str, float]:
        out = {
            "auc": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "brier": np.nan,
        }

        try:
            out["auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass

        try:
            out["accuracy"] = float(accuracy_score(y_true, y_pred))
        except Exception:
            pass

        try:
            out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        except Exception:
            pass

        try:
            out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        except Exception:
            pass

        try:
            out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        except Exception:
            pass

        try:
            out["brier"] = float(brier_score_loss(y_true, y_prob))
        except Exception:
            pass

        return out

    # ========================================================
    # Phase 4 plot helpers
    # ========================================================

    def _plot_roc_curve(
            self,
            y_true: np.ndarray,
            y_prob: np.ndarray,
            title: str,
            filename: str,
    ) -> Optional[str]:
        if len(y_true) == 0:
            return None

        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            return None

        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)

    def _plot_classification_score_bars(
            self,
            target_scores: pd.DataFrame,
            target_col: str,
            metric: str = "auc",
            top_n: int = 15,
            filename: str = "phase4_bar.png",
    ) -> Optional[str]:
        if target_scores.empty or metric not in target_scores.columns:
            return None

        plot_df = (
            target_scores
            .sort_values(metric, ascending=False)
            .head(top_n)
            .copy()
        )

        if plot_df.empty:
            return None

        plot_df["label"] = (
                plot_df["factor"].astype(str)
                + " | "
                + plot_df["feature_set"].astype(str)
                + " | "
                + plot_df["model_name"].astype(str)
        )

        plot_df = plot_df.sort_values(metric, ascending=True)

        plt.figure(figsize=(10, max(5, 0.35 * len(plot_df))))
        plt.barh(plot_df["label"], plot_df[metric])
        plt.xlabel(metric.upper())
        plt.ylabel("Specification")
        plt.title(f"Top {top_n} {metric.upper()} results for {target_col}")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)

    # --------------------------------------------------------
    # Metadata / outputs
    # --------------------------------------------------------

    def _build_metadata(self) -> None:
        panel = self.panel
        if panel is None:
            return

        meta = {
            "n_rows": int(len(panel)),
            "n_permnos": int(
                panel[self.column_config.id_col].nunique()) if self.column_config.id_col in panel.columns else None,
            "start_date": str(panel[
                                  self.column_config.date_col].min().date()) if self.column_config.date_col in panel.columns else None,
            "end_date": str(panel[
                                self.column_config.date_col].max().date()) if self.column_config.date_col in panel.columns else None,
            "factor_cols_all": self.factor_cols_all,
            "factor_cols_daily": self.factor_cols_daily,
            "factor_cols_monthly": self.factor_cols_monthly,
            "data_config": self._json_safe_dataclass(self.data_config),
            "column_config": self._json_safe_dataclass(self.column_config),
            "preprocess_config": self._json_safe_dataclass(self.preprocess_config),
            "target_config": self._json_safe_dataclass(self.target_config),
            "regression_config": self._json_safe_dataclass(self.regression_config),
            "classification_config": self._json_safe_dataclass(self.classification_config),
            "output_config": self._json_safe_dataclass(self.output_config),
            "identifier_config": self._json_safe_dataclass(self.identifier_config),
        }
        self.metadata = meta

    def _save_phase1_outputs(self) -> None:
        if self.panel is not None and self.output_config.save_panel_snapshot:
            panel_path = self.output_dir / self.output_config.panel_snapshot_name
            self.panel.to_parquet(panel_path, index=False)

        if self.metadata and self.output_config.save_metadata:
            meta_path = self.output_dir / self.output_config.metadata_name
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def _json_safe_dataclass(self, obj: Any) -> dict[str, Any]:
        d = asdict(obj)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, list):
                d[k] = [str(x) if isinstance(x, Path) else x for x in v]
        return d
