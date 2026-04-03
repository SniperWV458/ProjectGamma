from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, List, Dict

import numpy as np
import pandas as pd
import duckdb

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.dates as mdates


@dataclass
class PlotConfig:
    figsize: tuple = (12, 7)
    dpi: int = 100
    default_gex_col: str = "net_gex_1pct"
    default_spot_col: str = "spot"
    default_smoothing_window: int = 1
    default_lag: int = 0
    default_spot_color: str = "#1f77b4"
    default_gex_color: str = "#ff7f0e"
    default_plot_type: str = "timeseries"


class ResultPlotter:
    EXPECTED_SCHEMA: Dict[str, str] = {
        "secid": "numeric",
        "date": "datetime",
        "spot": "numeric",
        "spot_close": "numeric",
        "spot_return": "numeric",
        "spot_cfadj": "numeric",
        "spot_shrout": "numeric",
        "n_contracts": "numeric",
        "n_optionids": "numeric",
        "n_expiries": "numeric",
        "call_gex_1pct": "numeric",
        "put_gex_1pct": "numeric",
        "net_gex_1pct": "numeric",
        "call_gex_1pt": "numeric",
        "put_gex_1pt": "numeric",
        "net_gex_1pt": "numeric",
        "total_open_interest": "numeric",
        "total_option_volume": "numeric",
    }

    GEX_COLUMNS = [
        "call_gex_1pct",
        "put_gex_1pct",
        "net_gex_1pct",
        "call_gex_1pt",
        "put_gex_1pt",
        "net_gex_1pt",
    ]

    SPOT_COLUMNS = [
        "spot",
        "spot_close",
        "spot_cfadj",
        "spot_return",
        "spot_return_1d",
        "spot_return_5d",
        "spot_log_return",
    ]

    PLOT_TYPES = [
        "timeseries",
        "scatter",
        "timeseries + scatter",
    ]

    def __init__(
        self,
        master: Optional[tk.Tk] = None,
        data: Optional[pd.DataFrame] = None,
        config: Optional[PlotConfig] = None,
    ) -> None:
        self.master = master if master is not None else tk.Tk()
        self.master.title("Result Plotter - GEX / Spot Viewer")
        self.master.geometry("1600x950")

        self.config = config or PlotConfig()

        self.df_raw: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None
        self.current_df: Optional[pd.DataFrame] = None

        self.figure: Optional[Figure] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.toolbar: Optional[NavigationToolbar2Tk] = None

        self._build_variables()
        self._build_layout()
        self._build_figure()

        if data is not None:
            self.set_data(data)

    def run(self) -> None:
        self.master.mainloop()

    def set_data(self, data: pd.DataFrame) -> None:
        df = data.copy()
        self.validate_schema(df)
        df = self.preprocess_data(df)
        self.df_raw = data.copy()
        self.df = df
        self._refresh_controls_from_data()
        self._set_status(f"Loaded in-memory dataframe with {len(df):,} rows.")

    def load_from_file(self, file_path: str | Path) -> None:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use .csv or .parquet")

        self.set_data(df)
        self._set_status(f"Loaded file: {file_path}")

    def load_from_duckdb(
        self,
        query: str,
        db_path: Optional[str | Path] = None,
    ) -> None:
        con = duckdb.connect() if db_path is None else duckdb.connect(str(db_path))
        try:
            df = con.execute(query).fetchdf()
        finally:
            con.close()

        self.set_data(df)
        self._set_status("Loaded data via DuckDB query.")

    def export_current_plot(self, save_path: str | Path) -> None:
        if self.figure is None:
            raise RuntimeError("No figure initialized.")
        self.figure.savefig(save_path, bbox_inches="tight")
        self._set_status(f"Plot exported to: {save_path}")

    def validate_schema(self, df: pd.DataFrame) -> None:
        required = [c for c in self.EXPECTED_SCHEMA if c in df.columns or c not in self.SPOT_COLUMNS]
        missing = [c for c in self.EXPECTED_SCHEMA if c not in df.columns]
        if missing:
            raise ValueError(
                "Input dataframe is missing required columns:\n"
                + "\n".join(f"  - {c}" for c in missing)
            )

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c], errors="ignore")

        numeric_cols = [c for c in df.columns if c != "date"]
        for c in numeric_cols:
            if c != "secid":
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["secid", "date"]).copy()
        df["secid"] = pd.to_numeric(df["secid"], errors="coerce")
        df = df.dropna(subset=["secid"])
        df["secid"] = df["secid"].astype("int64")

        df = df.sort_values(["secid", "date"]).reset_index(drop=True)

        if df.duplicated(subset=["secid", "date"]).sum() > 0:
            df = df.drop_duplicates(subset=["secid", "date"], keep="last").reset_index(drop=True)

        return df

    def get_available_secids(self) -> List[int]:
        if self.df is None:
            return []
        return sorted(self.df["secid"].dropna().astype(int).unique().tolist())

    def get_date_bounds(self) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        if self.df is None or self.df.empty:
            return None, None
        return self.df["date"].min(), self.df["date"].max()

    def get_available_spot_columns(self) -> List[str]:
        if self.df is None:
            return [c for c in self.SPOT_COLUMNS]
        return [c for c in self.SPOT_COLUMNS if c in self.df.columns]

    def get_available_gex_columns(self) -> List[str]:
        if self.df is None:
            return [c for c in self.GEX_COLUMNS]
        return [c for c in self.GEX_COLUMNS if c in self.df.columns]

    def filter_data(
        self,
        secid: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("No data loaded.")

        df = self.df[self.df["secid"] == secid].copy()

        if start_date:
            start_ts = pd.to_datetime(start_date, errors="coerce")
            if pd.isna(start_ts):
                raise ValueError(f"Invalid start date: {start_date}")
            df = df[df["date"] >= start_ts]

        if end_date:
            end_ts = pd.to_datetime(end_date, errors="coerce")
            if pd.isna(end_ts):
                raise ValueError(f"Invalid end date: {end_date}")
            df = df[df["date"] <= end_ts]

        return df.sort_values("date").reset_index(drop=True)

    def add_smoothed_columns(
        self,
        df: pd.DataFrame,
        columns: Iterable[str],
        window: int,
    ) -> pd.DataFrame:
        out = df.copy()
        if window <= 1:
            for c in columns:
                out[f"{c}__smoothed"] = out[c]
            return out

        for c in columns:
            out[f"{c}__smoothed"] = out[c].rolling(window=window, min_periods=1).mean()

        return out

    def add_lagged_spot_column(self, df: pd.DataFrame, spot_col: str, lag: int) -> pd.DataFrame:
        """
        Creates a lagged version of the selected spot column:
          lag = 0  => spot_col_t
          lag = 1  => spot_col_{t+1} aligned to gex_t
          lag = 5  => spot_col_{t+5} aligned to gex_t
          lag = -1 => spot_col_{t-1} aligned to gex_t
        """
        out = df.copy()
        out[f"{spot_col}__lagged"] = out[spot_col].shift(-lag)
        return out

    def compute_lagged_correlation(
        self,
        df: pd.DataFrame,
        gex_col: str,
        spot_col: str,
        lag: int,
    ) -> Optional[float]:
        tmp = self.add_lagged_spot_column(df, spot_col=spot_col, lag=lag)
        lagged_col = f"{spot_col}__lagged"
        aligned = tmp[[gex_col, lagged_col]].dropna()
        if len(aligned) < 2:
            return None
        corr = aligned[gex_col].corr(aligned[lagged_col])
        return None if pd.isna(corr) else float(corr)

    def get_scatter_dataframe(
        self,
        df: pd.DataFrame,
        gex_col: str,
        spot_col: str,
        lag: int,
    ) -> pd.DataFrame:
        tmp = self.add_lagged_spot_column(df, spot_col=spot_col, lag=lag)
        lagged_col = f"{spot_col}__lagged"
        return tmp[["date", gex_col, lagged_col]].dropna().reset_index(drop=True)

    def compute_summary_stats(
        self,
        df: pd.DataFrame,
        spot_col: str,
        gex_col: str,
        lag: int,
    ) -> Dict[str, str]:
        stats: Dict[str, str] = {}

        if df.empty:
            return {"status": "No data in current selection."}

        spot_s = df[spot_col].dropna()
        gex_s = df[gex_col].dropna()

        stats["Rows"] = f"{len(df):,}"
        stats["Date range"] = f"{df['date'].min().date()} to {df['date'].max().date()}"
        stats["Spot column"] = spot_col
        stats["GEX column"] = gex_col
        stats["Lag"] = str(lag)

        if not spot_s.empty:
            stats["Spot last"] = f"{spot_s.iloc[-1]:,.6f}"
            stats["Spot min"] = f"{spot_s.min():,.6f}"
            stats["Spot max"] = f"{spot_s.max():,.6f}"
            stats["Spot mean"] = f"{spot_s.mean():,.6f}"
            stats["Spot std"] = f"{spot_s.std():,.6f}"

        if not gex_s.empty:
            stats["GEX last"] = f"{gex_s.iloc[-1]:,.6f}"
            stats["GEX min"] = f"{gex_s.min():,.6f}"
            stats["GEX max"] = f"{gex_s.max():,.6f}"
            stats["GEX mean"] = f"{gex_s.mean():,.6f}"
            stats["GEX std"] = f"{gex_s.std():,.6f}"

        aligned_same_day = df[[spot_col, gex_col]].dropna()
        if len(aligned_same_day) >= 2:
            corr_same_day = aligned_same_day[spot_col].corr(aligned_same_day[gex_col])
            if pd.notna(corr_same_day):
                stats[f"Corr({spot_col}, {gex_col})"] = f"{corr_same_day:,.6f}"

        lag_corr = self.compute_lagged_correlation(
            df=df,
            gex_col=gex_col,
            spot_col=spot_col,
            lag=lag,
        )
        stats[f"Corr({gex_col}_t, {spot_col}_t+{lag})"] = (
            f"{lag_corr:,.6f}" if lag_corr is not None else "N/A"
        )

        scatter_df = self.get_scatter_dataframe(df, gex_col=gex_col, spot_col=spot_col, lag=lag)
        stats["Scatter usable rows"] = f"{len(scatter_df):,}"

        for c in [
            "total_open_interest",
            "total_option_volume",
            "n_contracts",
            "n_optionids",
            "n_expiries",
            "spot_shrout",
        ]:
            if c in df.columns and df[c].notna().any():
                s = df[c].dropna()
                if c.startswith("n_"):
                    stats[f"{c} last"] = f"{s.iloc[-1]:,.0f}"
                    stats[f"{c} mean"] = f"{s.mean():,.2f}"
                else:
                    stats[f"{c} last"] = f"{s.iloc[-1]:,.6f}"
                    stats[f"{c} mean"] = f"{s.mean():,.6f}"

        return stats

    def _build_variables(self) -> None:
        self.file_path_var = tk.StringVar(value="")
        self.secid_var = tk.StringVar(value="")
        self.spot_col_var = tk.StringVar(value=self.config.default_spot_col)
        self.gex_col_var = tk.StringVar(value=self.config.default_gex_col)
        self.plot_type_var = tk.StringVar(value=self.config.default_plot_type)

        self.start_date_var = tk.StringVar(value="")
        self.end_date_var = tk.StringVar(value="")
        self.smoothing_var = tk.IntVar(value=self.config.default_smoothing_window)
        self.lag_var = tk.IntVar(value=self.config.default_lag)

        self.show_zero_line_var = tk.BooleanVar(value=True)
        self.show_legend_var = tk.BooleanVar(value=True)

        self.spot_color_var = tk.StringVar(value=self.config.default_spot_color)
        self.gex_color_var = tk.StringVar(value=self.config.default_gex_color)

        self.status_var = tk.StringVar(value="Ready.")

    def _build_layout(self) -> None:
        self.master.columnconfigure(1, weight=1)
        self.master.rowconfigure(0, weight=1)

        self.control_frame = ttk.Frame(self.master, padding=10)
        self.control_frame.grid(row=0, column=0, sticky="nsw")

        self.plot_frame = ttk.Frame(self.master, padding=10)
        self.plot_frame.grid(row=0, column=1, sticky="nsew")

        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

        self._build_controls()

    def _build_controls(self) -> None:
        row = 0

        ttk.Label(self.control_frame, text="Data Source", font=("Segoe UI", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        row += 1

        ttk.Entry(self.control_frame, textvariable=self.file_path_var, width=38).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=2
        )
        ttk.Button(self.control_frame, text="Browse", command=self._on_browse_file).grid(
            row=row, column=2, sticky="ew", padx=(6, 0), pady=2
        )
        row += 1

        ttk.Button(self.control_frame, text="Load File", command=self._on_load_file).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(0, 12)
        )
        row += 1

        ttk.Separator(self.control_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8
        )
        row += 1

        ttk.Label(self.control_frame, text="Selection", font=("Segoe UI", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        row += 1

        ttk.Label(self.control_frame, text="SECID").grid(row=row, column=0, sticky="w", pady=2)
        self.secid_combo = ttk.Combobox(self.control_frame, textvariable=self.secid_var, state="readonly")
        self.secid_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        row += 1

        ttk.Label(self.control_frame, text="Spot column").grid(row=row, column=0, sticky="w", pady=2)
        self.spot_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.spot_col_var,
            values=self.SPOT_COLUMNS,
            state="readonly",
        )
        self.spot_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        row += 1

        ttk.Label(self.control_frame, text="GEX column").grid(row=row, column=0, sticky="w", pady=2)
        self.gex_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.gex_col_var,
            values=self.GEX_COLUMNS,
            state="readonly",
        )
        self.gex_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        row += 1

        ttk.Label(self.control_frame, text="Plot type").grid(row=row, column=0, sticky="w", pady=2)
        self.plot_type_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.plot_type_var,
            values=self.PLOT_TYPES,
            state="readonly",
        )
        self.plot_type_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        row += 1

        ttk.Label(self.control_frame, text="Start date").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(self.control_frame, textvariable=self.start_date_var).grid(
            row=row, column=1, columnspan=2, sticky="ew", pady=2
        )
        row += 1

        ttk.Label(self.control_frame, text="End date").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(self.control_frame, textvariable=self.end_date_var).grid(
            row=row, column=1, columnspan=2, sticky="ew", pady=2
        )
        row += 1

        ttk.Label(self.control_frame, text="Smoothing").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(
            self.control_frame,
            from_=1,
            to=252,
            textvariable=self.smoothing_var,
            width=10,
        ).grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        ttk.Label(self.control_frame, text="Lag").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(
            self.control_frame,
            from_=-60,
            to=60,
            textvariable=self.lag_var,
            width=10,
        ).grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        ttk.Label(self.control_frame, text="Spot color").grid(row=row, column=0, sticky="w", pady=2)
        self.spot_color_preview = tk.Label(
            self.control_frame,
            textvariable=self.spot_color_var,
            bg=self.spot_color_var.get(),
            fg="white",
            width=14,
        )
        self.spot_color_preview.grid(row=row, column=1, sticky="ew", pady=2)
        ttk.Button(self.control_frame, text="Pick", command=self._choose_spot_color).grid(
            row=row, column=2, sticky="ew", pady=2
        )
        row += 1

        ttk.Label(self.control_frame, text="GEX color").grid(row=row, column=0, sticky="w", pady=2)
        self.gex_color_preview = tk.Label(
            self.control_frame,
            textvariable=self.gex_color_var,
            bg=self.gex_color_var.get(),
            fg="white",
            width=14,
        )
        self.gex_color_preview.grid(row=row, column=1, sticky="ew", pady=2)
        ttk.Button(self.control_frame, text="Pick", command=self._choose_gex_color).grid(
            row=row, column=2, sticky="ew", pady=2
        )
        row += 1

        ttk.Checkbutton(
            self.control_frame,
            text="Show GEX zero line",
            variable=self.show_zero_line_var,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1

        ttk.Checkbutton(
            self.control_frame,
            text="Show legend",
            variable=self.show_legend_var,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1

        ttk.Button(self.control_frame, text="Plot", command=self.plot_current_selection).grid(
            row=row, column=0, sticky="ew", pady=(8, 2)
        )
        ttk.Button(self.control_frame, text="Reset", command=self.reset_filters).grid(
            row=row, column=1, sticky="ew", pady=(8, 2)
        )
        ttk.Button(self.control_frame, text="Refresh", command=self._refresh_controls_from_data).grid(
            row=row, column=2, sticky="ew", pady=(8, 2)
        )
        row += 1

        ttk.Button(self.control_frame, text="Export Plot", command=self._on_export_plot).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(2, 12)
        )
        row += 1

        ttk.Separator(self.control_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8
        )
        row += 1

        ttk.Label(self.control_frame, text="Summary", font=("Segoe UI", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        row += 1

        summary_container = ttk.Frame(self.control_frame)
        summary_container.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=2)

        self.stats_box = tk.Text(summary_container, width=45, height=20, wrap="word")
        self.stats_scrollbar = ttk.Scrollbar(summary_container, orient="vertical", command=self.stats_box.yview)
        self.stats_box.configure(yscrollcommand=self.stats_scrollbar.set)

        self.stats_box.grid(row=0, column=0, sticky="nsew")
        self.stats_scrollbar.grid(row=0, column=1, sticky="ns")

        summary_container.rowconfigure(0, weight=1)
        summary_container.columnconfigure(0, weight=1)

        self.stats_box.insert("1.0", "No data loaded.")
        self.stats_box.configure(state="disabled")
        row += 1

        ttk.Separator(self.control_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8
        )
        row += 1

        ttk.Label(self.control_frame, textvariable=self.status_var, foreground="blue").grid(
            row=row, column=0, columnspan=3, sticky="w"
        )

        for col in range(3):
            self.control_frame.columnconfigure(col, weight=1)

        self.control_frame.rowconfigure(row - 1, weight=1)

    def _build_figure(self) -> None:
        self.figure = Figure(figsize=self.config.figsize, dpi=self.config.dpi)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(self.plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self._draw_placeholder()

    def plot_current_selection(self) -> None:
        if self.df is None:
            messagebox.showwarning("No data", "Please load data first.")
            return

        secid_text = self.secid_var.get().strip()
        if not secid_text:
            messagebox.showwarning("Missing selection", "Please select a SECID.")
            return

        try:
            secid = int(secid_text)
            spot_col = self.spot_col_var.get().strip()
            gex_col = self.gex_col_var.get().strip()
            plot_type = self.plot_type_var.get().strip()
            start_date = self.start_date_var.get().strip() or None
            end_date = self.end_date_var.get().strip() or None
            smoothing = int(self.smoothing_var.get())
            lag = int(self.lag_var.get())
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        try:
            df_plot = self.filter_data(secid=secid, start_date=start_date, end_date=end_date)
            if df_plot.empty:
                self.current_df = df_plot
                self._draw_placeholder("No data for the selected filters.")
                self._update_stats({"status": "No data for the selected filters."})
                return

            smooth_cols = [gex_col]
            if plot_type in {"timeseries", "timeseries + scatter"}:
                smooth_cols.append(spot_col)

            df_plot = self.add_smoothed_columns(df_plot, smooth_cols, smoothing)
            self.current_df = df_plot

            self._render_plot(
                df=df_plot,
                secid=secid,
                spot_col=spot_col,
                gex_col=gex_col,
                lag=lag,
                smoothing=smoothing,
                plot_type=plot_type,
            )

            stats = self.compute_summary_stats(
                df=df_plot,
                spot_col=spot_col,
                gex_col=gex_col,
                lag=lag,
            )
            self._update_stats(stats)
            self._set_status(
                f"Plotted secid={secid}, plot_type={plot_type}, rows={len(df_plot):,}"
            )
        except Exception as e:
            messagebox.showerror("Plot error", str(e))

    def _render_plot(
        self,
        df: pd.DataFrame,
        secid: int,
        spot_col: str,
        gex_col: str,
        lag: int,
        smoothing: int,
        plot_type: str,
    ) -> None:
        self.figure.clear()

        spot_color = self.spot_color_var.get()
        gex_color = self.gex_color_var.get()

        if plot_type == "timeseries":
            ax_spot = self.figure.add_subplot(111)
            ax_gex = ax_spot.twinx()
            self._plot_timeseries(
                ax_spot=ax_spot,
                ax_gex=ax_gex,
                df=df,
                secid=secid,
                spot_col=spot_col,
                gex_col=gex_col,
                lag=lag,
                smoothing=smoothing,
                spot_color=spot_color,
                gex_color=gex_color,
            )

        elif plot_type == "scatter":
            ax = self.figure.add_subplot(111)
            self._plot_scatter(
                ax=ax,
                df=df,
                gex_col=gex_col,
                spot_col=spot_col,
                lag=lag,
                gex_color=gex_color,
            )

        elif plot_type == "timeseries + scatter":
            ax_spot = self.figure.add_subplot(211)
            ax_gex = ax_spot.twinx()
            self._plot_timeseries(
                ax_spot=ax_spot,
                ax_gex=ax_gex,
                df=df,
                secid=secid,
                spot_col=spot_col,
                gex_col=gex_col,
                lag=lag,
                smoothing=smoothing,
                spot_color=spot_color,
                gex_color=gex_color,
            )

            ax_scatter = self.figure.add_subplot(212)
            self._plot_scatter(
                ax=ax_scatter,
                df=df,
                gex_col=gex_col,
                spot_col=spot_col,
                lag=lag,
                gex_color=gex_color,
            )

        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_timeseries(
        self,
        ax_spot,
        ax_gex,
        df: pd.DataFrame,
        secid: int,
        spot_col: str,
        gex_col: str,
        lag: int,
        smoothing: int,
        spot_color: str,
        gex_color: str,
    ) -> None:
        x = df["date"]
        spot_series = df[f"{spot_col}__smoothed"]
        gex_series = df[f"{gex_col}__smoothed"]

        spot_label = spot_col if lag == 0 else f"{spot_col} (displayed at t, corr/scatter use t+{lag})"
        if smoothing > 1:
            spot_label = f"{spot_label} | MA{smoothing}"

        gex_label = gex_col if smoothing <= 1 else f"{gex_col} (MA{smoothing})"

        ax_spot.plot(
            x,
            spot_series,
            label=spot_label,
            linewidth=1.8,
            color=spot_color,
        )

        ax_gex.plot(
            x,
            gex_series,
            label=gex_label,
            linewidth=1.5,
            linestyle="--",
            color=gex_color,
        )

        if self.show_zero_line_var.get():
            ax_gex.axhline(0.0, linewidth=1.0, linestyle=":", color=gex_color, alpha=0.7)

        title = f"SECID {secid} | {spot_col} vs {gex_col} | lag={lag}"
        if smoothing > 1:
            title += f" | smoothing={smoothing}"

        ax_spot.set_title(title)
        ax_spot.set_xlabel("Date")
        ax_spot.set_ylabel(spot_col, color=spot_color)
        ax_gex.set_ylabel(gex_col, color=gex_color)

        ax_spot.tick_params(axis="y", labelcolor=spot_color)
        ax_gex.tick_params(axis="y", labelcolor=gex_color)
        ax_spot.grid(True, linestyle=":", alpha=0.6)

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax_spot.xaxis.set_major_locator(locator)
        ax_spot.xaxis.set_major_formatter(formatter)

        if self.show_legend_var.get():
            lines1, labels1 = ax_spot.get_legend_handles_labels()
            lines2, labels2 = ax_gex.get_legend_handles_labels()
            ax_spot.legend(lines1 + lines2, labels1 + labels2, loc="best")

    def _plot_scatter(
        self,
        ax,
        df: pd.DataFrame,
        gex_col: str,
        spot_col: str,
        lag: int,
        gex_color: str,
    ) -> None:
        scatter_df = self.get_scatter_dataframe(df, gex_col=gex_col, spot_col=spot_col, lag=lag)
        lagged_col = f"{spot_col}__lagged"

        if scatter_df.empty:
            ax.text(0.5, 0.5, "No data available for scatter plot.", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return

        x = scatter_df[gex_col].to_numpy()
        y = scatter_df[lagged_col].to_numpy()

        ax.scatter(x, y, alpha=0.6, color=gex_color, edgecolors="none")

        if len(scatter_df) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            xline = np.linspace(np.min(x), np.max(x), 200)
            yline = slope * xline + intercept
            ax.plot(xline, yline, color="black", linewidth=1.2, linestyle="-")

        corr = self.compute_lagged_correlation(
            df=df,
            gex_col=gex_col,
            spot_col=spot_col,
            lag=lag,
        )

        ax.set_title(
            f"Scatter: {gex_col}_t vs {spot_col}_t+{lag}"
            + (f" | corr={corr:.4f}" if corr is not None else "")
        )
        ax.set_xlabel(gex_col)
        ax.set_ylabel(f"{spot_col}_t+{lag}")
        ax.grid(True, linestyle=":", alpha=0.6)

    def _draw_placeholder(self, message: str = "Load data and select a SECID to plot.") -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw()

    def _choose_spot_color(self) -> None:
        color = colorchooser.askcolor(title="Choose spot line color", color=self.spot_color_var.get())
        if color and color[1]:
            self.spot_color_var.set(color[1])
            self.spot_color_preview.configure(bg=color[1])

    def _choose_gex_color(self) -> None:
        color = colorchooser.askcolor(title="Choose GEX line color", color=self.gex_color_var.get())
        if color and color[1]:
            self.gex_color_var.set(color[1])
            self.gex_color_preview.configure(bg=color[1])

    def _on_browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select result file",
            filetypes=[
                ("Data files", "*.csv *.parquet *.pq"),
                ("CSV files", "*.csv"),
                ("Parquet files", "*.parquet *.pq"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.file_path_var.set(path)

    def _on_load_file(self) -> None:
        path = self.file_path_var.get().strip()
        if not path:
            messagebox.showwarning("No path", "Please choose a file first.")
            return
        try:
            self.load_from_file(path)
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _on_export_plot(self) -> None:
        if self.figure is None:
            messagebox.showwarning("No plot", "No plot available to export.")
            return

        path = filedialog.asksaveasfilename(
            title="Export plot",
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("PDF file", "*.pdf"),
                ("SVG file", "*.svg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            self.export_current_plot(path)
            messagebox.showinfo("Exported", f"Plot saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _refresh_controls_from_data(self) -> None:
        if self.df is None or self.df.empty:
            self.secid_combo["values"] = []
            self._update_stats({"status": "No data loaded."})
            return

        secids = self.get_available_secids()
        self.secid_combo["values"] = [str(x) for x in secids]

        available_spot = self.get_available_spot_columns()
        available_gex = self.get_available_gex_columns()

        self.spot_combo["values"] = available_spot
        self.gex_combo["values"] = available_gex

        if secids:
            current = self.secid_var.get().strip()
            if not current or int(current) not in secids:
                self.secid_var.set(str(secids[0]))

        if available_spot and self.spot_col_var.get() not in available_spot:
            self.spot_col_var.set(available_spot[0])

        if available_gex and self.gex_col_var.get() not in available_gex:
            self.gex_col_var.set(available_gex[0])

        min_date, max_date = self.get_date_bounds()
        if min_date is not None and not self.start_date_var.get().strip():
            self.start_date_var.set(str(min_date.date()))
        if max_date is not None and not self.end_date_var.get().strip():
            self.end_date_var.set(str(max_date.date()))

        self._update_stats({
            "Rows": f"{len(self.df):,}",
            "SECIDs": f"{len(secids):,}",
            "Global date range": f"{min_date.date()} to {max_date.date()}",
            "Available spot columns": ", ".join(available_spot),
            "Available GEX columns": ", ".join(available_gex),
        })

    def reset_filters(self) -> None:
        if self.df is None:
            self.start_date_var.set("")
            self.end_date_var.set("")
            self.smoothing_var.set(1)
            self.lag_var.set(0)
            self.spot_col_var.set(self.config.default_spot_col)
            self.gex_col_var.set(self.config.default_gex_col)
            self.plot_type_var.set(self.config.default_plot_type)

            self.spot_color_var.set(self.config.default_spot_color)
            self.gex_color_var.set(self.config.default_gex_color)
            self.spot_color_preview.configure(bg=self.spot_color_var.get())
            self.gex_color_preview.configure(bg=self.gex_color_var.get())

            self._draw_placeholder()
            self._update_stats({"status": "No data loaded."})
            self._set_status("Reset complete.")
            return

        min_date, max_date = self.get_date_bounds()
        secids = self.get_available_secids()
        available_spot = self.get_available_spot_columns()
        available_gex = self.get_available_gex_columns()

        self.secid_var.set(str(secids[0]) if secids else "")
        self.spot_col_var.set(
            self.config.default_spot_col if self.config.default_spot_col in available_spot else (available_spot[0] if available_spot else "")
        )
        self.gex_col_var.set(
            self.config.default_gex_col if self.config.default_gex_col in available_gex else (available_gex[0] if available_gex else "")
        )
        self.plot_type_var.set(self.config.default_plot_type)

        self.start_date_var.set(str(min_date.date()) if min_date is not None else "")
        self.end_date_var.set(str(max_date.date()) if max_date is not None else "")
        self.smoothing_var.set(1)
        self.lag_var.set(0)

        self.spot_color_var.set(self.config.default_spot_color)
        self.gex_color_var.set(self.config.default_gex_color)
        self.spot_color_preview.configure(bg=self.spot_color_var.get())
        self.gex_color_preview.configure(bg=self.gex_color_var.get())

        self.show_zero_line_var.set(True)
        self.show_legend_var.set(True)

        self.current_df = None
        self._draw_placeholder()
        self._refresh_controls_from_data()
        self._set_status("Filters reset.")

    def _update_stats(self, stats: Dict[str, str]) -> None:
        self.stats_box.configure(state="normal")
        self.stats_box.delete("1.0", tk.END)
        for k, v in stats.items():
            self.stats_box.insert(tk.END, f"{k}: {v}\n")
        self.stats_box.see("1.0")
        self.stats_box.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    @classmethod
    def from_file(cls, file_path: str | Path) -> "ResultPlotter":
        app = cls()
        app.load_from_file(file_path)
        return app

    @classmethod
    def from_duckdb_query(
        cls,
        query: str,
        db_path: Optional[str | Path] = None,
    ) -> "ResultPlotter":
        app = cls()
        app.load_from_duckdb(query=query, db_path=db_path)
        return app


if __name__ == "__main__":
    app = ResultPlotter.from_file("results/option_gamma/underlying_gex_daily.parquet")
    app.run()