# ProjectGamma Data Pipeline (WRDS -> GEX -> Plots)

This repository contains a Python workflow for:

1. Fetching/linking equity and option data from WRDS (`data_fetcher.py`)
2. Computing contract-level and aggregate gamma exposure (GEX) with DuckDB (`data_processor.py`)
3. Exploring and visualizing outputs in an interactive desktop app (`result_plotter.py`)
4. Exporting samples and data inventory reports (`utils.py`)

---

## Files Covered

- `data_fetcher.py`: WRDS data extraction, identifier linking, and phase-1 dataset builds
- `data_processor.py`: Option-level cleaning, spot joins, GEX calculation, and aggregation
- `result_plotter.py`: Tkinter + Matplotlib GUI for time-series/scatter analysis
- `utils.py`: Small data utilities (sample export, parquet inventory, secid-name mapping)

---

## Requirements

Install dependencies in your environment:

```bash
pip install pandas duckdb wrds tqdm numpy matplotlib pyarrow
```

Notes:

- `wrds` requires WRDS credentials and network access.
- `result_plotter.py` uses `tkinter` (usually bundled with standard Python installs).
- Parquet read/write generally requires `pyarrow` (or compatible parquet engine).

---

## Typical Workflow

### 1) Fetch and link source data (`data_fetcher.py`)

Use `DataFetcherConfig` to define WRDS username, date range, output format, and chunk sizes.

Main high-level methods:

- `run_identifier_pipeline()`
  - `fetch_crsp_stocknames()`
  - `build_crsp_id_master()`
  - `extract_optionm_secnmd()`
  - `build_crsp_optionm_link()`
- `run_phase1_data_pipeline(top_pct=0.30)` adds:
  - `fetch_crsp_dsf()`
  - `build_crsp_liquidity_panel()`
  - `build_top_liquid_stock_universe()`
  - `build_linked_secid_file_from_top_liquid()`
  - `fetch_opprcd()`

Quick example:

```python
from data_fetcher import DataFetcher, DataFetcherConfig

cfg = DataFetcherConfig(
    wrds_username="your_wrds_username",
    data_dir="data",
    start_date="2024-01-01",
    end_date="2024-12-31",
    start_year=2024,
    end_year=2024,
    file_type="parquet",
    compression="zstd",
    replace=False,
)

fetcher = DataFetcher(cfg)
fetcher.run_phase1_data_pipeline(top_pct=0.30)
fetcher.close()
```

Common generated outputs include:

- `crsp_common_stocknames_YYYY_YYYY.parquet`
- `crsp_id_master_YYYY_YYYY.parquet`
- `crsp_daily_common_YYYY_YYYY.parquet`
- `crsp_liquidity_panel_YYYY_YYYY.parquet`
- `crsp_top_liquid_universe_YYYY_YYYY.parquet`
- `crsp_optionm_link_YYYY_YYYY.parquet`
- `crsp_optionm_link_dominant_*.parquet`
- `linked_secids_top_liquid_YYYY_YYYY.parquet`
- `opprcd_linked_YYYY_YYYY.parquet`
- optional manifest logging via `manifest.parquet`

---

### 2) Compute GEX and aggregates (`data_processor.py`)

`OptionGammaProcessor` reads option parquet (required) and spot parquet (optional but required for join/GEX methods), then computes:

- clean option view (`fetch_clean_options`)
- option+spot join (`fetch_joined_with_spot`)
- contract-level GEX (`fetch_contract_gex`)
- daily aggregates:
  - underlying-level (`aggregate_underlying_daily`)
  - strike-level (`aggregate_strike_daily`)
  - expiry-level (`aggregate_expiry_daily`)
  - DTE bucket-level (`aggregate_dte_daily`)

Quick example:

```python
from data_processor import OptionGammaConfig, OptionGammaProcessor

cfg = OptionGammaConfig(
    option_parquet="data/opprcd_linked_2024_2024.parquet",
    spot_parquet="data/secprd_linked_2024_2024.parquet",
    output_dir="results/option_gamma",
    start_date="2024-01-01",
    end_date="2024-12-31",
    require_spot=True,
    threads=4,
    memory_limit="8GB",
)

processor = OptionGammaProcessor(cfg)
underlying = processor.aggregate_underlying_daily(save=True, filename="underlying_gex_daily")
processor.close()
```

Convenience exports:

- `export_contract_gex_to_parquet(path)`
- `export_underlying_daily_to_parquet(path)`
- `run_baseline_pipeline(export_contract_gex=..., export_underlying_daily=...)`

---

### 3) Visualize results (`result_plotter.py`)

`ResultPlotter` is an interactive desktop app for filtering by SECID/date, smoothing series, lagging spot columns, and plotting:

- time series
- scatter
- combined time series + scatter

Expected columns are aligned with processor outputs (for example: `secid`, `date`, `spot`, `net_gex_1pct`, `net_gex_1pt`, and related aggregate fields).

Quick start from file:

```python
from result_plotter import ResultPlotter

app = ResultPlotter.from_file("results/option_gamma/underlying_gex_daily.parquet")
app.run()
```

Or load using a DuckDB query:

```python
from result_plotter import ResultPlotter

query = "SELECT * FROM read_parquet('results/option_gamma/underlying_gex_daily.parquet')"
app = ResultPlotter.from_duckdb_query(query=query)
app.run()
```

---

## Utility Helpers (`utils.py`)

- `export_parquet_sample_to_csv(...)`
  - Export either first `n` rows or a fraction of rows from a parquet file to CSV.
- `log_parquet_inventory(...)`
  - Produce a text report of parquet files, schema, row counts, and per-column stats.
- `export_secid_crsp_name_mapping(...)`
  - Join SECID link output with CRSP stock names and export a CSV mapping.

Example:

```python
from pathlib import Path
from utils import export_parquet_sample_to_csv, log_parquet_inventory

export_parquet_sample_to_csv(
    parquet_file="results/option_gamma/underlying_gex_daily.parquet",
    n=1000,
)

log_parquet_inventory(Path("data"))
```

---

## Practical Notes

- Prefer parquet outputs for large datasets and faster downstream processing.
- Keep `replace=False` in configs to avoid re-pulling large WRDS datasets unintentionally.
- Tune chunk sizes (`crsp_permno_chunk_size`, `optionm_secid_chunk_size`) based on WRDS response behavior.
- Use `run_diagnostics()` in `OptionGammaProcessor` to inspect schema/quality before heavy exports.
- Always call `.close()` on long-lived fetcher/processor objects to release connections cleanly.

---

## Suggested Directory Layout

```text
ProjectGamma/
  data/                    # fetched and linked parquet files
  results/
    option_gamma/          # processor outputs (contract/underlying aggregates)
  data_fetcher.py
  data_processor.py
  result_plotter.py
  utils.py
```
