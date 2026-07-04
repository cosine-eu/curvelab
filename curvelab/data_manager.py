"""Data loading and column access via pandas."""

import sqlite3
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


def _is_numeric_token(token: str) -> bool:
    """Return True if token parses as a number (int, float, or scientific notation)."""
    try:
        float(token)
        return True
    except ValueError:
        return False


# Tried in order for any plain-text file (CSV/TSV/TXT/DAT). latin-1 maps
# every byte 0x00-0xFF to a codepoint, so it can never raise
# UnicodeDecodeError and is guaranteed to terminate the fallback chain.
_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def _read_csv_with_encoding_fallback(path, **kwargs) -> pd.DataFrame:
    """Try pd.read_csv with common encodings until one decodes successfully."""
    for encoding in _TEXT_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise AssertionError("unreachable: latin-1 never raises UnicodeDecodeError")


def _open_text_with_encoding_fallback(path):
    """Open a text file, trying common encodings until one decodes successfully."""
    for encoding in _TEXT_ENCODINGS:
        try:
            with open(path, encoding=encoding) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    raise AssertionError("unreachable: latin-1 never raises UnicodeDecodeError")


class DataManager:
    """Wraps pandas DataFrames. Loads various tabular formats and exposes columns."""

    def __init__(self):
        self.datasets: dict[str, pd.DataFrame] = {}
        self.filepaths: dict[str, Path] = {}
        self.table_names: dict[str, str] = {}  # dataset_name -> SQLite table name

    def load(self, filepath: str | Path) -> tuple[str, list[str]]:
        """Load a data file and return (dataset_name, column_names).

        Supported formats: CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet,
        SQLite (.sqlite/.db).

        For SQLite files, all tables are loaded as separate datasets named
        ``filename::table_name``. The first table's info is returned.
        """
        filepath = Path(filepath)
        ext = filepath.suffix.lower()

        if ext in (".sqlite", ".db"):
            return self._load_sqlite(filepath)

        loaders = {
            ".csv": lambda p: _read_csv_with_encoding_fallback(p),
            ".tsv": lambda p: _read_csv_with_encoding_fallback(p, sep="\t"),
            ".xlsx": lambda p: pd.read_excel(p),
            ".xls": lambda p: pd.read_excel(p),
            ".ods": lambda p: pd.read_excel(p, engine="odf"),
            ".json": lambda p: pd.read_json(p),
            ".parquet": lambda p: pd.read_parquet(p),
            ".h5": lambda p: pd.read_hdf(p),
            ".hdf5": lambda p: pd.read_hdf(p),
            ".hdf": lambda p: pd.read_hdf(p),
        }
        loader = loaders.get(ext)
        if loader is None:
            loader = lambda p: self._load_text_columns(p)

        df = loader(filepath)

        # Deduplicate name
        base_name = filepath.name
        name = base_name
        counter = 2
        while name in self.datasets:
            name = f"{base_name} ({counter})"
            counter += 1

        self.datasets[name] = df
        self.filepaths[name] = filepath
        return name, list(df.columns)

    @staticmethod
    def _load_text_columns(filepath: Path) -> pd.DataFrame:
        """Load a whitespace-separated text file with comment line support.

        Handles .txt, .dat, and other plain-text formats. Lines starting with
        '#' are treated as comments. If the last comment line before data looks
        like column headers, those are used as column names.
        """
        lines = _open_text_with_encoding_fallback(filepath)

        # Separate comment lines and data lines
        comment_lines = []
        data_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                comment_lines.append(stripped)
            else:
                data_lines.append(line)

        if not data_lines:
            raise ValueError(f"No data found in '{filepath.name}'.")

        # Check if the last comment line looks like column headers
        header_names = None
        if comment_lines:
            last_comment = comment_lines[-1].lstrip("#").strip()
            tokens = last_comment.split()
            # Use as headers if token count matches the first data row's column count
            first_data_tokens = data_lines[0].split()
            if len(tokens) == len(first_data_tokens) and len(tokens) >= 2:
                # Verify tokens aren't all numeric (would be data, not headers)
                all_numeric = all(_is_numeric_token(t) for t in tokens)
                if not all_numeric:
                    header_names = tokens

        df = pd.read_csv(
            StringIO("".join(data_lines)),
            sep=r"\s+",
            header=None,
            engine="python",
        )

        if header_names is not None:
            df.columns = header_names
        else:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

        return df

    def _load_sqlite(self, filepath: Path) -> tuple[str, list[str]]:
        """Load all tables from a SQLite database as separate datasets."""
        conn = sqlite3.connect(filepath)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            if not tables:
                raise ValueError(f"No tables found in '{filepath.name}'.")

            first_name = None
            first_columns = None
            for table in tables:
                df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
                base_name = f"{filepath.name}::{table}"
                name = base_name
                counter = 2
                while name in self.datasets:
                    name = f"{base_name} ({counter})"
                    counter += 1
                self.datasets[name] = df
                self.filepaths[name] = filepath
                self.table_names[name] = table
                if first_name is None:
                    first_name = name
                    first_columns = list(df.columns)

            return first_name, first_columns
        finally:
            conn.close()

    def load_from_text(self, text: str) -> tuple[str, list[str]]:
        """Load tabular data from a raw text string (e.g. clipboard).

        Auto-detects delimiter and headers. Returns (dataset_name, column_names).
        """
        # Auto-detect delimiter: pick whichever produces the most columns consistently
        best_sep = None
        best_cols = 0
        for sep in ["\t", ",", ";", r"\s+"]:
            try:
                df_test = pd.read_csv(
                    StringIO(text), sep=sep, header=None, nrows=5,
                    engine="python" if sep == r"\s+" else "c",
                )
                ncols = df_test.shape[1]
                if ncols > best_cols:
                    best_cols = ncols
                    best_sep = sep
            except Exception:
                continue

        if best_sep is None:
            raise ValueError("Could not parse pasted text as tabular data.")

        engine = "python" if best_sep == r"\s+" else "c"

        # Auto-detect headers: check if first row has any non-numeric values
        first_line = text.strip().splitlines()[0]
        probe = pd.read_csv(
            StringIO(first_line), sep=best_sep, header=None, engine=engine,
        )
        has_header = any(
            isinstance(v, str) and not _is_numeric_token(v)
            for v in probe.iloc[0]
        )

        df = pd.read_csv(
            StringIO(text), sep=best_sep,
            header=0 if has_header else None,
            engine=engine,
        )

        if not has_header:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

        # Deduplicate name
        base_name = "clipboard"
        name = base_name
        counter = 2
        while name in self.datasets:
            name = f"{base_name} ({counter})"
            counter += 1

        self.datasets[name] = df
        return name, list(df.columns)

    def add_dataframe(self, name: str, df: pd.DataFrame) -> tuple[str, list[str]]:
        """Register a DataFrame directly (no file path). Returns (name, column_names).

        Uses the same name-deduplication logic as load().
        """
        base_name = name
        counter = 2
        while name in self.datasets:
            name = f"{base_name} ({counter})"
            counter += 1

        self.datasets[name] = df
        return name, list(df.columns)

    def column_names(self, dataset_name: str) -> list[str]:
        """Return column names for a dataset."""
        df = self.datasets.get(dataset_name)
        if df is None:
            return []
        return list(df.columns)

    def get_column(self, dataset_name: str, col_name: str) -> np.ndarray:
        """Return a column as a numpy array."""
        df = self.datasets.get(dataset_name)
        if df is None:
            raise ValueError(f"No dataset named '{dataset_name}'")
        return df[col_name].to_numpy(dtype=float)

    def remove_dataset(self, name: str):
        """Remove a dataset by name."""
        self.datasets.pop(name, None)
        self.filepaths.pop(name, None)
        self.table_names.pop(name, None)

    @property
    def dataset_names(self) -> list[str]:
        return list(self.datasets.keys())

    @property
    def is_loaded(self) -> bool:
        return len(self.datasets) > 0
