"""Tests for DataManager — the (previously untested) data-ingestion layer.

Covers file loading dispatch, the whitespace/comment text loader, SQLite
multi-table loading, clipboard-text parsing, name deduplication, column
access, and the small query/mutation helpers.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd


class DataManagerTestBase(unittest.TestCase):
    def setUp(self):
        from curvelab.data_manager import DataManager
        self.dm = DataManager()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _path(self, name):
        return os.path.join(self.tmp, name)

    def _write(self, name, text, encoding="utf-8"):
        p = self._path(name)
        with open(p, "w", encoding=encoding) as f:
            f.write(text)
        return p


class LoadFileFormatTests(DataManagerTestBase):
    def test_load_csv(self):
        p = self._write("d.csv", "x,y\n1,4\n2,5\n3,6\n")
        name, cols = self.dm.load(p)
        self.assertEqual(name, "d.csv")
        self.assertEqual(cols, ["x", "y"])
        np.testing.assert_allclose(self.dm.get_column(name, "y"), [4, 5, 6])

    def test_load_tsv(self):
        p = self._write("d.tsv", "x\ty\n1\t4\n2\t5\n")
        name, cols = self.dm.load(p)
        self.assertEqual(cols, ["x", "y"])
        np.testing.assert_allclose(self.dm.get_column(name, "x"), [1, 2])

    def test_load_json(self):
        p = self._write("d.json", '{"x": [1, 2, 3], "y": [4, 5, 6]}')
        name, cols = self.dm.load(p)
        self.assertEqual(sorted(cols), ["x", "y"])

    def test_load_parquet(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            try:
                import fastparquet  # noqa: F401
            except ImportError:
                self.skipTest("no parquet engine installed")
        p = self._path("d.parquet")
        pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_parquet(p)
        name, cols = self.dm.load(p)
        self.assertEqual(sorted(cols), ["x", "y"])

    def test_unknown_extension_falls_back_to_text_loader(self):
        p = self._write("d.dat", "1 2\n3 4\n5 6\n")
        name, cols = self.dm.load(p)
        # No header comment -> generated col_N names.
        self.assertEqual(cols, ["col_0", "col_1"])
        self.assertEqual(self.dm.datasets[name].shape, (3, 2))

    def test_duplicate_filename_is_deduped(self):
        p = self._write("d.csv", "x,y\n1,2\n")
        n1, _ = self.dm.load(p)
        n2, _ = self.dm.load(p)
        self.assertEqual(n1, "d.csv")
        self.assertEqual(n2, "d.csv (2)")


class TextColumnLoaderTests(DataManagerTestBase):
    def test_comment_header_used_as_column_names(self):
        p = self._write("d.txt", "# some file\n# x_col y_col\n1 2\n3 4\n")
        name, cols = self.dm.load(p)
        self.assertEqual(cols, ["x_col", "y_col"])

    def test_numeric_last_comment_not_used_as_header(self):
        # A numeric-looking last comment is data-like, not a header.
        p = self._write("d.txt", "# 1 2\n10 20\n30 40\n")
        name, cols = self.dm.load(p)
        self.assertEqual(cols, ["col_0", "col_1"])

    def test_blank_and_comment_lines_skipped(self):
        p = self._write("d.txt", "# header\n\n1 2\n\n3 4\n")
        name, cols = self.dm.load(p)
        self.assertEqual(self.dm.datasets[name].shape, (2, 2))

    def test_no_data_raises(self):
        p = self._write("d.txt", "# only comments\n# nothing else\n")
        with self.assertRaises(ValueError):
            self.dm.load(p)

    def test_header_token_count_mismatch_ignored(self):
        # 3 header tokens but 2 data columns -> not used as header.
        p = self._write("d.txt", "# a b c\n1 2\n3 4\n")
        name, cols = self.dm.load(p)
        self.assertEqual(cols, ["col_0", "col_1"])


class SqliteLoaderTests(DataManagerTestBase):
    def _make_db(self, tables: dict):
        p = self._path("d.sqlite")
        conn = sqlite3.connect(p)
        for tname, df in tables.items():
            df.to_sql(tname, conn, index=False)
        conn.close()
        return p

    def test_multiple_tables_loaded_as_separate_datasets(self):
        p = self._make_db({
            "alpha": pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
            "beta": pd.DataFrame({"a": [5, 6], "b": [7, 8]}),
        })
        first_name, first_cols = self.dm.load(p)
        # Two datasets registered, named filename::table.
        self.assertIn("d.sqlite::alpha", self.dm.dataset_names)
        self.assertIn("d.sqlite::beta", self.dm.dataset_names)
        # Returns the first table's info.
        self.assertEqual(first_name, "d.sqlite::alpha")
        self.assertEqual(first_cols, ["x", "y"])
        self.assertEqual(self.dm.table_names["d.sqlite::alpha"], "alpha")

    def test_empty_database_raises(self):
        p = self._path("empty.sqlite")
        sqlite3.connect(p).close()
        with self.assertRaises(ValueError):
            self.dm.load(p)

    def test_views_are_loaded_like_tables(self):
        p = self._make_db({"raw": pd.DataFrame({"x": [1, 2], "y": [3, 4]})})
        conn = sqlite3.connect(p)
        conn.execute("CREATE VIEW doubled AS SELECT x, y * 2 AS y FROM raw")
        conn.commit()
        conn.close()

        first_name, first_cols = self.dm.load(p)
        self.assertIn("d.sqlite::doubled", self.dm.dataset_names)
        self.assertEqual(self.dm.table_names["d.sqlite::doubled"], "doubled")
        # The view is queried, not just listed.
        self.assertEqual(
            list(self.dm.get_column("d.sqlite::doubled", "y")), [6.0, 8.0]
        )
        # "doubled" sorts before "raw", so it is what load() returns.
        self.assertEqual(first_name, "d.sqlite::doubled")
        self.assertEqual(first_cols, ["x", "y"])

    def test_tables_and_columns_are_alphabetical(self):
        p = self._make_db({
            "zulu": pd.DataFrame({"beta": [1], "Alpha": [2], "gamma": [3]}),
            "Mike": pd.DataFrame({"q": [1]}),
            "alpha": pd.DataFrame({"z": [1], "a": [2]}),
        })
        first_name, first_cols = self.dm.load(p)
        # Case-insensitive: alpha, Mike, zulu -- not alpha, zulu, Mike.
        self.assertEqual(
            self.dm.dataset_names,
            ["d.sqlite::alpha", "d.sqlite::Mike", "d.sqlite::zulu"],
        )
        self.assertEqual(first_name, "d.sqlite::alpha")
        self.assertEqual(first_cols, ["a", "z"])
        self.assertEqual(
            self.dm.column_names("d.sqlite::zulu"), ["Alpha", "beta", "gamma"]
        )
        # Sorting reorders the columns, it does not shuffle their values.
        self.assertEqual(list(self.dm.get_column("d.sqlite::zulu", "Alpha")), [2.0])


class LoadFromTextTests(DataManagerTestBase):
    def test_comma_delimited_with_header(self):
        name, cols = self.dm.load_from_text("a,b\n1,2\n3,4\n")
        self.assertEqual(cols, ["a", "b"])
        self.assertEqual(name, "clipboard")

    def test_tab_delimited_detected(self):
        name, cols = self.dm.load_from_text("a\tb\tc\n1\t2\t3\n")
        self.assertEqual(cols, ["a", "b", "c"])

    def test_semicolon_delimited_detected(self):
        name, cols = self.dm.load_from_text("a;b\n1;2\n3;4\n")
        self.assertEqual(cols, ["a", "b"])

    def test_headerless_gets_generated_names(self):
        name, cols = self.dm.load_from_text("1,2\n3,4\n5,6\n")
        self.assertEqual(cols, ["col_0", "col_1"])
        self.assertEqual(self.dm.datasets[name].shape, (3, 2))

    def test_clipboard_name_deduped(self):
        n1, _ = self.dm.load_from_text("a,b\n1,2\n")
        n2, _ = self.dm.load_from_text("a,b\n3,4\n")
        self.assertEqual(n1, "clipboard")
        self.assertEqual(n2, "clipboard (2)")

    def test_unparseable_text_raises(self):
        with self.assertRaises(ValueError):
            self.dm.load_from_text("")


class AddDataframeAndDedupeTests(DataManagerTestBase):
    def test_add_dataframe_registers_and_returns_columns(self):
        name, cols = self.dm.add_dataframe("mine", pd.DataFrame({"p": [1], "q": [2]}))
        self.assertEqual(name, "mine")
        self.assertEqual(cols, ["p", "q"])
        self.assertTrue(self.dm.is_loaded)

    def test_dedupe_name_suffixes_collisions(self):
        self.dm.add_dataframe("t", pd.DataFrame({"a": [1]}))
        self.assertEqual(self.dm._dedupe_name("t"), "t (2)")
        self.dm.add_dataframe("t", pd.DataFrame({"a": [1]}))
        self.assertEqual(self.dm._dedupe_name("t"), "t (3)")

    def test_dedupe_name_unused_name_unchanged(self):
        self.assertEqual(self.dm._dedupe_name("fresh"), "fresh")


class ColumnAccessTests(DataManagerTestBase):
    def setUp(self):
        super().setUp()
        self.name, _ = self.dm.add_dataframe(
            "d", pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]}))

    def test_column_names(self):
        self.assertEqual(self.dm.column_names("d"), ["x", "y"])

    def test_column_names_missing_dataset_returns_empty(self):
        self.assertEqual(self.dm.column_names("nope"), [])

    def test_get_column_returns_float_array(self):
        col = self.dm.get_column("d", "x")
        self.assertIsInstance(col, np.ndarray)
        self.assertEqual(col.dtype, np.float64)
        np.testing.assert_allclose(col, [1.0, 2.0, 3.0])

    def test_get_column_missing_dataset_raises(self):
        with self.assertRaises(ValueError):
            self.dm.get_column("nope", "x")


class QueryAndMutationTests(DataManagerTestBase):
    def test_is_loaded_and_dataset_names(self):
        self.assertFalse(self.dm.is_loaded)
        self.assertEqual(self.dm.dataset_names, [])
        self.dm.add_dataframe("a", pd.DataFrame({"c": [1]}))
        self.dm.add_dataframe("b", pd.DataFrame({"c": [1]}))
        self.assertTrue(self.dm.is_loaded)
        self.assertEqual(set(self.dm.dataset_names), {"a", "b"})

    def test_remove_dataset(self):
        p = self._write("d.csv", "x,y\n1,2\n")
        name, _ = self.dm.load(p)
        self.assertIn(name, self.dm.filepaths)
        self.dm.remove_dataset(name)
        self.assertNotIn(name, self.dm.dataset_names)
        self.assertNotIn(name, self.dm.filepaths)

    def test_remove_unknown_dataset_is_noop(self):
        self.dm.remove_dataset("nonexistent")  # must not raise


if __name__ == "__main__":
    unittest.main()
