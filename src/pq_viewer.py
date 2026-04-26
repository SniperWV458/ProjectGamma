import os
import math
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd

try:
    import duckdb
except ImportError:
    duckdb = None


class ParquetPreviewerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Parquet File Previewer")
        self.root.geometry("1300x800")

        self.file_path = None
        self.schema_df = None
        self.current_preview_df = None
        self.available_columns = []

        self._build_ui()

    def _build_ui(self):
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        ttk.Button(top_frame, text="Open Parquet File", command=self.open_file).pack(side="left")
        self.file_label = ttk.Label(top_frame, text="No file selected", width=100)
        self.file_label.pack(side="left", padx=10)

        control_frame = ttk.LabelFrame(self.root, text="Preview Controls", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(control_frame, text="Rows to preview:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.rows_var = tk.StringVar(value="200")
        ttk.Entry(control_frame, textvariable=self.rows_var, width=12).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(control_frame, text="Start row offset:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.offset_var = tk.StringVar(value="0")
        ttk.Entry(control_frame, textvariable=self.offset_var, width=12).grid(row=0, column=3, sticky="w", padx=5, pady=5)

        self.only_selected_cols_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            control_frame,
            text="Preview only selected columns",
            variable=self.only_selected_cols_var
        ).grid(row=0, column=4, sticky="w", padx=10, pady=5)

        ttk.Button(control_frame, text="Refresh Preview", command=self.refresh_preview).grid(
            row=0, column=5, sticky="w", padx=10, pady=5
        )

        ttk.Button(control_frame, text="Preview Random Sample", command=self.preview_random_sample).grid(
            row=0, column=6, sticky="w", padx=10, pady=5
        )

        info_frame = ttk.LabelFrame(self.root, text="File Information", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)

        self.info_text = tk.Text(info_frame, height=6, wrap="word")
        self.info_text.pack(fill="x")
        self.info_text.configure(state="disabled")

        middle_frame = ttk.Frame(self.root)
        middle_frame.pack(fill="both", expand=True, padx=10, pady=5)

        left_frame = ttk.LabelFrame(middle_frame, text="Columns", padding=10)
        left_frame.pack(side="left", fill="y", padx=(0, 5))

        self.column_listbox = tk.Listbox(left_frame, selectmode="extended", exportselection=False, width=35)
        self.column_listbox.pack(side="left", fill="y", expand=False)

        col_scroll = ttk.Scrollbar(left_frame, orient="vertical", command=self.column_listbox.yview)
        col_scroll.pack(side="right", fill="y")
        self.column_listbox.config(yscrollcommand=col_scroll.set)

        column_btn_frame = ttk.Frame(left_frame)
        column_btn_frame.pack(fill="x", pady=8)
        ttk.Button(column_btn_frame, text="Select All", command=self.select_all_columns).pack(fill="x", pady=2)
        ttk.Button(column_btn_frame, text="Clear Selection", command=self.clear_column_selection).pack(fill="x", pady=2)

        right_frame = ttk.LabelFrame(middle_frame, text="Preview Table", padding=10)
        right_frame.pack(side="left", fill="both", expand=True)

        self.tree = ttk.Treeview(right_frame, show="headings")
        self.tree.pack(side="left", fill="both", expand=True)

        y_scroll = ttk.Scrollbar(right_frame, orient="vertical", command=self.tree.yview)
        y_scroll.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=y_scroll.set)

        x_scroll = ttk.Scrollbar(self.root, orient="horizontal", command=self.tree.xview)
        x_scroll.pack(fill="x", padx=10, pady=(0, 5))
        self.tree.configure(xscrollcommand=x_scroll.set)

        bottom_frame = ttk.LabelFrame(self.root, text="Status", padding=10)
        bottom_frame.pack(fill="x", padx=10, pady=5)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(bottom_frame, textvariable=self.status_var).pack(anchor="w")

    def set_status(self, text: str):
        self.status_var.set(text)
        self.root.update_idletasks()

    def set_info_text(self, text: str):
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.configure(state="disabled")

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Select Parquet File",
            filetypes=[("Parquet files", "*.parquet"), ("All files", "*.*")]
        )
        if not path:
            return

        self.file_path = path
        self.file_label.config(text=path)
        self.set_status("Loading parquet metadata...")
        try:
            self.load_metadata()
            self.populate_column_list()
            self.refresh_preview()
        except Exception as e:
            self.set_status("Failed to open file.")
            messagebox.showerror("Error", f"Could not open parquet file.\n\n{e}")

    def load_metadata(self):
        if not self.file_path:
            return

        file_size = os.path.getsize(self.file_path)
        file_size_mb = file_size / (1024 * 1024)

        if duckdb is None:
            raise ImportError(
                "duckdb is required for efficient parquet preview.\n"
                "Install it with: pip install duckdb"
            )

        con = duckdb.connect()
        try:
            schema_df = con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{self._escape_path(self.file_path)}')"
            ).fetchdf()

            count_df = con.execute(
                f"SELECT COUNT(*) AS row_count FROM read_parquet('{self._escape_path(self.file_path)}')"
            ).fetchdf()

            row_count = int(count_df.loc[0, "row_count"])
        finally:
            con.close()

        self.schema_df = schema_df.copy()
        self.available_columns = schema_df["column_name"].tolist()

        info_lines = [
            f"File: {os.path.basename(self.file_path)}",
            f"Path: {self.file_path}",
            f"File size: {file_size_mb:.2f} MB",
            f"Estimated rows: {row_count:,}",
            f"Number of columns: {len(self.available_columns)}",
            "",
            "Schema:"
        ]

        for _, row in schema_df.iterrows():
            col_name = row.get("column_name", "")
            col_type = row.get("column_type", "")
            nullable = row.get("null", "")
            info_lines.append(f"  - {col_name}: {col_type} | null={nullable}")

        self.set_info_text("\n".join(info_lines))
        self.set_status("Metadata loaded.")

    def populate_column_list(self):
        self.column_listbox.delete(0, "end")
        for col in self.available_columns:
            self.column_listbox.insert("end", col)
        self.select_all_columns()

    def select_all_columns(self):
        self.column_listbox.select_set(0, "end")

    def clear_column_selection(self):
        self.column_listbox.select_clear(0, "end")

    def get_selected_columns(self):
        indices = self.column_listbox.curselection()
        return [self.column_listbox.get(i) for i in indices]

    def refresh_preview(self):
        if not self.file_path:
            messagebox.showwarning("No file", "Please open a parquet file first.")
            return

        try:
            rows = int(self.rows_var.get())
            offset = int(self.offset_var.get())
            if rows <= 0 or offset < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "Rows must be > 0 and offset must be >= 0.")
            return

        self.set_status("Loading preview...")
        try:
            selected_columns = self.get_selected_columns()
            use_selected = self.only_selected_cols_var.get()

            if use_selected:
                if not selected_columns:
                    messagebox.showwarning("No columns selected", "Please select at least one column.")
                    self.set_status("No columns selected.")
                    return
                columns = selected_columns
            else:
                columns = self.available_columns

            df = self.read_preview(columns=columns, limit=rows, offset=offset)
            self.current_preview_df = df
            self.display_dataframe(df)
            self.set_status(f"Showing rows {offset:,} to {offset + len(df) - 1:,}.")
        except Exception as e:
            self.set_status("Failed to load preview.")
            messagebox.showerror("Preview Error", f"{e}\n\n{traceback.format_exc()}")

    def preview_random_sample(self):
        if not self.file_path:
            messagebox.showwarning("No file", "Please open a parquet file first.")
            return

        try:
            rows = int(self.rows_var.get())
            if rows <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "Rows to preview must be > 0.")
            return

        selected_columns = self.get_selected_columns()
        use_selected = self.only_selected_cols_var.get()

        if use_selected:
            if not selected_columns:
                messagebox.showwarning("No columns selected", "Please select at least one column.")
                return
            columns = selected_columns
        else:
            columns = self.available_columns

        self.set_status("Loading random sample...")
        try:
            df = self.read_random_sample(columns=columns, limit=rows)
            self.current_preview_df = df
            self.display_dataframe(df)
            self.set_status(f"Showing random sample of {len(df):,} rows.")
        except Exception as e:
            self.set_status("Random sample failed.")
            messagebox.showerror("Sample Error", f"{e}\n\n{traceback.format_exc()}")

    def read_preview(self, columns, limit=200, offset=0) -> pd.DataFrame:
        if duckdb is None:
            raise ImportError("duckdb is required. Install it with: pip install duckdb")

        col_sql = ", ".join([self.quote_identifier(c) for c in columns])
        sql = f"""
            SELECT {col_sql}
            FROM read_parquet('{self._escape_path(self.file_path)}')
            LIMIT {int(limit)} OFFSET {int(offset)}
        """

        con = duckdb.connect()
        try:
            df = con.execute(sql).fetchdf()
        finally:
            con.close()

        return df

    def read_random_sample(self, columns, limit=200) -> pd.DataFrame:
        if duckdb is None:
            raise ImportError("duckdb is required. Install it with: pip install duckdb")

        col_sql = ", ".join([self.quote_identifier(c) for c in columns])

        sql = f"""
            SELECT {col_sql}
            FROM read_parquet('{self._escape_path(self.file_path)}')
            USING SAMPLE {int(limit)} ROWS
        """

        con = duckdb.connect()
        try:
            df = con.execute(sql).fetchdf()
        finally:
            con.close()

        return df

    def display_dataframe(self, df: pd.DataFrame):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df.columns)

        for col in df.columns:
            self.tree.heading(col, text=col)
            width = self.estimate_column_width(df[col], col)
            self.tree.column(col, width=width, anchor="w", stretch=False)

        if df.empty:
            return

        for _, row in df.iterrows():
            values = [self.safe_to_str(v) for v in row.tolist()]
            self.tree.insert("", "end", values=values)

    @staticmethod
    def safe_to_str(value):
        if pd.isna(value):
            return ""
        text = str(value)
        if len(text) > 200:
            text = text[:197] + "..."
        return text

    @staticmethod
    def estimate_column_width(series: pd.Series, col_name: str) -> int:
        max_len = len(str(col_name))
        sample = series.head(50)
        for v in sample:
            if pd.isna(v):
                continue
            max_len = max(max_len, len(str(v)))
        return min(max(80, max_len * 7), 350)

    @staticmethod
    def quote_identifier(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    @staticmethod
    def _escape_path(path: str) -> str:
        return path.replace("\\", "/").replace("'", "''")


def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = ParquetPreviewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()