import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class ScrollableCheckFrame(ttk.LabelFrame):
    def __init__(self, master, title, items, default_value=False, height=260, width=260):
        super().__init__(master, text=title)
        self.vars = {}

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", padx=4, pady=4)

        ttk.Button(btn_row, text="All", command=self.select_all).pack(side="left", padx=2)
        ttk.Button(btn_row, text="None", command=self.clear_all).pack(side="left", padx=2)

        self.canvas = tk.Canvas(self, height=height, width=width)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        self.scrollbar.pack(side="right", fill="y", padx=(0, 4), pady=4)

        for item in items:
            var = tk.BooleanVar(value=default_value)
            self.vars[item] = var
            ttk.Checkbutton(self.inner, text=str(item), variable=var).pack(anchor="w", padx=6, pady=1)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

    def _on_mousewheel(self, event):
        try:
            widget = self.winfo_containing(event.x_root, event.y_root)
            if widget is None:
                return
            parent = widget
            inside = False
            while parent is not None:
                if parent == self:
                    inside = True
                    break
                parent = parent.master
            if inside:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def select_all(self):
        for v in self.vars.values():
            v.set(True)

    def clear_all(self):
        for v in self.vars.values():
            v.set(False)

    def get_selected(self):
        return [k for k, v in self.vars.items() if v.get()]


class BacktestGUI:
    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        factor_col: str = "factor",
        strategy_col: str = "strategy_name",
        ret_col: str = "portfolio_ret",
        ret_net_col: str = "portfolio_ret_net",
        annualization: int = 252,
        default_rolling_window: int = 63,
    ):
        self.df = df.copy()
        self.date_col = date_col
        self.factor_col = factor_col
        self.strategy_col = strategy_col
        self.ret_col = ret_col
        self.ret_net_col = ret_net_col
        self.annualization = annualization
        self.default_rolling_window = default_rolling_window

        self._prepare_data()

        self.root = tk.Tk()
        self.root.title("Backtest Time Series Viewer")
        self.root.geometry("1700x980")
        self.root.minsize(1400, 850)

        self._build_gui()
        self._draw_empty_plot()

    # --------------------------------------------------
    # Data prep
    # --------------------------------------------------
    def _prepare_data(self):
        required = [self.date_col, self.factor_col, self.strategy_col]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        if self.ret_col not in self.df.columns and self.ret_net_col not in self.df.columns:
            raise ValueError(f"Need at least one return column: {self.ret_col} or {self.ret_net_col}")

        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce")
        self.df = self.df.dropna(subset=[self.date_col]).copy()

        for c in [self.ret_col, self.ret_net_col]:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

        self.df[self.factor_col] = self.df[self.factor_col].astype(str)
        self.df[self.strategy_col] = self.df[self.strategy_col].astype(str)

        self.df = self.df.sort_values(
            [self.factor_col, self.strategy_col, self.date_col]
        ).reset_index(drop=True)

        self.all_factors = sorted(self.df[self.factor_col].dropna().unique().tolist())
        self.all_strategies = sorted(self.df[self.strategy_col].dropna().unique().tolist())

        self.min_date = self.df[self.date_col].min()
        self.max_date = self.df[self.date_col].max()

    # --------------------------------------------------
    # GUI
    # --------------------------------------------------
    def _build_gui(self):
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=1)
        self.root.columnconfigure(0, weight=1)

        # =========================
        # Top controls
        # =========================
        top = ttk.Frame(self.root)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        for i in range(10):
            top.columnconfigure(i, weight=0)
        top.columnconfigure(9, weight=1)

        ttk.Label(top, text="Start").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.start_date_var = tk.StringVar(value=self.min_date.strftime("%Y-%m-%d"))
        ttk.Entry(top, textvariable=self.start_date_var, width=14).grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(top, text="End").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        self.end_date_var = tk.StringVar(value=self.max_date.strftime("%Y-%m-%d"))
        ttk.Entry(top, textvariable=self.end_date_var, width=14).grid(row=0, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(top, text="Return").grid(row=0, column=4, sticky="w", padx=(18, 4), pady=4)
        ret_choices = []
        if self.ret_col in self.df.columns:
            ret_choices.append(self.ret_col)
        if self.ret_net_col in self.df.columns:
            ret_choices.append(self.ret_net_col)

        self.return_choice_var = tk.StringVar(value=ret_choices[0])
        self.return_combo = ttk.Combobox(
            top,
            textvariable=self.return_choice_var,
            values=ret_choices,
            state="readonly",
            width=20
        )
        self.return_combo.grid(row=0, column=5, sticky="w", padx=4, pady=4)

        ttk.Label(top, text="Metric").grid(row=0, column=6, sticky="w", padx=(18, 4), pady=4)
        self.metric_var = tk.StringVar(value="nav")
        self.metric_combo = ttk.Combobox(
            top,
            textvariable=self.metric_var,
            values=["return", "nav", "sharpe"],
            state="readonly",
            width=12
        )
        self.metric_combo.grid(row=0, column=7, sticky="w", padx=4, pady=4)

        ttk.Label(top, text="Sharpe Window").grid(row=0, column=8, sticky="w", padx=(18, 4), pady=4)
        self.rolling_window_var = tk.StringVar(value=str(self.default_rolling_window))
        ttk.Entry(top, textvariable=self.rolling_window_var, width=8).grid(row=0, column=9, sticky="w", padx=4, pady=4)

        self.normalize_nav_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top,
            text="Normalize NAV at first valid point",
            variable=self.normalize_nav_var
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=4, pady=4)

        ttk.Button(top, text="Plot", command=self.update_plot).grid(row=1, column=6, sticky="e", padx=4, pady=4)
        ttk.Button(top, text="Show Summary", command=self.show_summary).grid(row=1, column=7, sticky="e", padx=4, pady=4)
        ttk.Button(top, text="Quit", command=self.root.destroy).grid(row=1, column=8, sticky="e", padx=4, pady=4)

        # =========================
        # Selector row
        # =========================
        selector_row = ttk.Frame(self.root)
        selector_row.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        selector_row.columnconfigure(0, weight=1)
        selector_row.columnconfigure(1, weight=1)

        self.factor_frame = ScrollableCheckFrame(
            selector_row,
            title="Factors",
            items=self.all_factors,
            default_value=False,
            height=280,
            width=360
        )
        self.factor_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        self.strategy_frame = ScrollableCheckFrame(
            selector_row,
            title="Strategies",
            items=self.all_strategies,
            default_value=True,
            height=280,
            width=360
        )
        self.strategy_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        # =========================
        # Bottom area
        # =========================
        bottom = ttk.Frame(self.root)
        bottom.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        bottom.rowconfigure(0, weight=1)
        bottom.columnconfigure(0, weight=0)
        bottom.columnconfigure(1, weight=1)

        # Summary panel
        summary_box = ttk.LabelFrame(bottom, text="Summary")
        summary_box.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        summary_box.rowconfigure(0, weight=1)
        summary_box.columnconfigure(0, weight=1)

        self.summary_text = tk.Text(summary_box, wrap="none", width=52)
        self.summary_text.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)

        summary_scroll_y = ttk.Scrollbar(summary_box, orient="vertical", command=self.summary_text.yview)
        summary_scroll_y.grid(row=0, column=1, sticky="ns", pady=4)

        summary_scroll_x = ttk.Scrollbar(summary_box, orient="horizontal", command=self.summary_text.xview)
        summary_scroll_x.grid(row=1, column=0, sticky="ew", padx=(4, 0))

        self.summary_text.configure(
            yscrollcommand=summary_scroll_y.set,
            xscrollcommand=summary_scroll_x.set
        )

        # Plot panel
        plot_box = ttk.LabelFrame(bottom, text="Plot")
        plot_box.grid(row=0, column=1, sticky="nsew")
        plot_box.rowconfigure(0, weight=1)
        plot_box.columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(11, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_box)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.toolbar_frame = ttk.Frame(plot_box)
        self.toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _parse_date(self, s: str, field_name: str):
        try:
            return pd.to_datetime(s)
        except Exception:
            raise ValueError(f"Invalid {field_name}: {s}")

    def _get_filtered_df(self):
        selected_factors = self.factor_frame.get_selected()
        selected_strategies = self.strategy_frame.get_selected()

        if not selected_factors:
            raise ValueError("Please select at least one factor.")
        if not selected_strategies:
            raise ValueError("Please select at least one strategy.")

        start_dt = self._parse_date(self.start_date_var.get().strip(), "start date")
        end_dt = self._parse_date(self.end_date_var.get().strip(), "end date")

        if start_dt > end_dt:
            raise ValueError("Start date must be earlier than or equal to end date.")

        out = self.df[
            (self.df[self.factor_col].isin(selected_factors)) &
            (self.df[self.strategy_col].isin(selected_strategies)) &
            (self.df[self.date_col] >= start_dt) &
            (self.df[self.date_col] <= end_dt)
        ].copy()

        if out.empty:
            raise ValueError("No data available for the selected filters.")

        return out, start_dt, end_dt

    def _make_series_label(self, factor, strategy):
        return f"{factor} | {strategy}"

    def _build_wide_return_panel(self, df_filtered: pd.DataFrame, return_col: str):
        if return_col not in df_filtered.columns:
            raise ValueError(f"Return column not found: {return_col}")

        temp = df_filtered[[self.date_col, self.factor_col, self.strategy_col, return_col]].copy()
        temp["series_label"] = temp.apply(
            lambda x: self._make_series_label(x[self.factor_col], x[self.strategy_col]),
            axis=1
        )

        wide = temp.pivot_table(
            index=self.date_col,
            columns="series_label",
            values=return_col,
            aggfunc="last"
        ).sort_index()

        return wide

    def _calc_cum_return(self, ret_wide: pd.DataFrame):
        return (1.0 + ret_wide.fillna(0.0)).cumprod() - 1.0

    def _calc_nav(self, ret_wide: pd.DataFrame, normalize_at_first_valid: bool = True):
        nav = (1.0 + ret_wide.fillna(0.0)).cumprod()

        if not normalize_at_first_valid:
            return nav

        nav_norm = nav.copy()
        for col in nav_norm.columns:
            first_valid_idx = nav_norm[col].first_valid_index()
            if first_valid_idx is None:
                continue
            base = nav_norm.loc[first_valid_idx, col]
            if pd.notna(base) and base != 0:
                nav_norm[col] = nav_norm[col] / base
        return nav_norm

    def _calc_rolling_sharpe(self, ret_wide: pd.DataFrame, window: int):
        mean_ = ret_wide.rolling(window).mean()
        std_ = ret_wide.rolling(window).std()
        return np.sqrt(self.annualization) * mean_ / std_.replace(0, np.nan)

    def _performance_summary_table(self, df_filtered: pd.DataFrame, return_col: str):
        rows = []

        for (factor, strategy), g in df_filtered.groupby([self.factor_col, self.strategy_col]):
            g = g.sort_values(self.date_col)
            r = pd.to_numeric(g[return_col], errors="coerce").dropna()

            if len(r) == 0:
                continue

            nav = (1.0 + r).cumprod()
            total_return = nav.iloc[-1] - 1.0

            n = len(r)
            ann_return = nav.iloc[-1] ** (self.annualization / n) - 1.0 if n > 0 else np.nan
            ann_vol = r.std(ddof=1) * np.sqrt(self.annualization)
            sharpe = ann_return / ann_vol if pd.notna(ann_vol) and ann_vol != 0 else np.nan

            peak = nav.cummax()
            drawdown = nav / peak - 1.0
            max_dd = drawdown.min()

            rows.append({
                "factor": factor,
                "strategy": strategy,
                "n_obs": n,
                "total_return": total_return,
                "annualized_return": ann_return,
                "annualized_vol": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
            })

        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(["factor", "strategy"]).reset_index(drop=True)
        return out

    # --------------------------------------------------
    # Plotting
    # --------------------------------------------------
    def _draw_empty_plot(self):
        self.ax.clear()
        self.ax.set_title("Backtest Viewer")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Value")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def update_plot(self):
        try:
            df_filtered, start_dt, end_dt = self._get_filtered_df()
            metric = self.metric_var.get().strip().lower()
            return_col = self.return_choice_var.get().strip()

            wide_ret = self._build_wide_return_panel(df_filtered, return_col=return_col)

            self.ax.clear()

            if metric == "return":
                plot_df = self._calc_cum_return(wide_ret)
                ylabel = "Cumulative Return"
                title = f"Cumulative Return ({return_col}) | {start_dt.date()} to {end_dt.date()}"

            elif metric == "nav":
                plot_df = self._calc_nav(
                    wide_ret,
                    normalize_at_first_valid=self.normalize_nav_var.get()
                )
                ylabel = "NAV"
                title = f"NAV ({return_col}) | {start_dt.date()} to {end_dt.date()}"
                if self.normalize_nav_var.get():
                    title += " | normalized"

            elif metric == "sharpe":
                try:
                    window = int(self.rolling_window_var.get().strip())
                except Exception:
                    raise ValueError("Rolling window must be an integer.")
                if window <= 1:
                    raise ValueError("Rolling window must be greater than 1.")

                plot_df = self._calc_rolling_sharpe(wide_ret, window=window)
                ylabel = f"Rolling Sharpe ({window}d)"
                title = f"Rolling Sharpe ({return_col}) | {start_dt.date()} to {end_dt.date()}"

            else:
                raise ValueError(f"Unsupported metric: {metric}")

            if plot_df.empty or plot_df.shape[1] == 0:
                raise ValueError("No series available to plot.")

            for col in plot_df.columns:
                self.ax.plot(plot_df.index, plot_df[col], label=col, linewidth=1.5)

            self.ax.set_title(title)
            self.ax.set_xlabel("Date")
            self.ax.set_ylabel(ylabel)
            self.ax.grid(True, alpha=0.3)

            if metric in ["return", "sharpe"]:
                self.ax.axhline(0.0, linestyle="--", linewidth=1)

            if metric == "nav" and self.normalize_nav_var.get():
                self.ax.axhline(1.0, linestyle="--", linewidth=1)

            self.ax.legend(loc="best", fontsize=8)
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Plot Error", str(e))

    def show_summary(self):
        try:
            df_filtered, start_dt, end_dt = self._get_filtered_df()
            return_col = self.return_choice_var.get().strip()
            summary = self._performance_summary_table(df_filtered, return_col=return_col)

            self.summary_text.delete("1.0", tk.END)
            self.summary_text.insert(tk.END, f"Date range: {start_dt.date()} to {end_dt.date()}\n")
            self.summary_text.insert(tk.END, f"Return column: {return_col}\n")
            self.summary_text.insert(
                tk.END,
                f"NAV normalization: {'ON' if self.normalize_nav_var.get() else 'OFF'}\n\n"
            )

            if summary.empty:
                self.summary_text.insert(tk.END, "No summary available.\n")
                return

            display_df = summary.copy()
            for col in ["total_return", "annualized_return", "annualized_vol", "sharpe", "max_drawdown"]:
                display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")

            self.summary_text.insert(tk.END, display_df.to_string(index=False))

        except Exception as e:
            messagebox.showerror("Summary Error", str(e))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    bt_df = pd.read_csv("../results/gex_collab_mii/phase6_overlay_timeseries.csv")
    app = BacktestGUI(
        df=bt_df,
        date_col="date",
        factor_col="factor",
        strategy_col="strategy_name",
        ret_col="portfolio_ret",
        ret_net_col="portfolio_ret_net",
        annualization=252,
        default_rolling_window=63,
    )
    app.run()