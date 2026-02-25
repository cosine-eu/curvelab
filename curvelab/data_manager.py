"""Data loading and column access via pandas."""

from pathlib import Path

import numpy as np
import pandas as pd


class DataManager:
    """Wraps pandas DataFrames. Loads various tabular formats and exposes columns."""

    def __init__(self):
        self.datasets: dict[str, pd.DataFrame] = {}
        self.filepaths: dict[str, Path] = {}

    def load(self, filepath: str | Path) -> tuple[str, list[str]]:
        """Load a data file and return (dataset_name, column_names).

        Supported formats: CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet.
        """
        filepath = Path(filepath)
        ext = filepath.suffix.lower()
        loaders = {
            ".csv": lambda p: pd.read_csv(p),
            ".tsv": lambda p: pd.read_csv(p, sep="\t"),
            ".xlsx": lambda p: pd.read_excel(p),
            ".xls": lambda p: pd.read_excel(p),
            ".json": lambda p: pd.read_json(p),
            ".parquet": lambda p: pd.read_parquet(p),
        }
        loader = loaders.get(ext)
        if loader is None:
            loader = loaders[".csv"]

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

    @property
    def dataset_names(self) -> list[str]:
        return list(self.datasets.keys())

    @property
    def is_loaded(self) -> bool:
        return len(self.datasets) > 0
