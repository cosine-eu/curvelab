"""Dataset, series, and fit-session lifecycle handlers for CurveLabApp.

Mixed into CurveLabApp; these methods manage loading/removing datasets,
selecting series and sessions, session create/rename/delete, and
visibility toggles. They rely on the coordinator's state, panels, and
refresh helpers, so they are not usable standalone.
"""

import tkinter as tk
from tkinter import messagebox

import numpy as np
import pandas as pd

from .session import (
    SeriesRecord,
    make_series_id as _make_series_id,
    make_session_key as _make_session_key,
)


class SeriesSessionMixin:
    """Dataset loading and series/session lifecycle management."""

    # --- Dataset callbacks ---

    def _on_load_file(self, filepath: str):
        try:
            dataset_name, columns = self.data_mgr.load(filepath)
            self.data_panel.set_datasets(
                self.data_mgr.dataset_names, select=dataset_name
            )
            self.data_panel.set_columns(columns)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _on_paste_data(self, event=None):
        try:
            text = self.parent.clipboard_get()
        except tk.TclError:
            messagebox.showwarning("Paste Data", "Clipboard is empty.")
            return
        try:
            dataset_name, columns = self.data_mgr.load_from_text(text)
            self.data_panel.set_datasets(
                self.data_mgr.dataset_names, select=dataset_name
            )
            self.data_panel.set_columns(columns)
        except Exception as e:
            messagebox.showerror("Paste Error", str(e))

    def _on_dataset_selected(self, name: str):
        columns = self.data_mgr.column_names(name)
        self.data_panel.set_columns(columns)

    def _on_remove_dataset(self, name: str):
        to_remove = [
            sid for sid, rec in self._series_records.items()
            if rec.dataset_name == name
        ]
        n_sessions = sum(
            len(rec.fit_sessions) for sid in to_remove
            for rec in [self._series_records[sid]]
        )
        if n_sessions > 0:
            ok = messagebox.askyesno(
                "Confirm Remove",
                f"Dataset '{name}' has {n_sessions} fit session(s) "
                f"across {len(to_remove)} series.\n\n"
                f"Remove dataset and discard all fit sessions?",
            )
            if not ok:
                return
        for sid in to_remove:
            rec = self._series_records.pop(sid)
            for sess_name in list(rec.fit_sessions):
                skey = _make_session_key(sid, sess_name)
                self.plot_mgr.clear_fit_session(skey)
            if self._active_series_id == sid:
                self._active_series_id = None

        self.data_mgr.remove_dataset(name)
        self.data_panel.remove_series_for_dataset(name)

        names = self.data_mgr.dataset_names
        self.data_panel.set_datasets(names)
        if names:
            self._on_dataset_selected(names[0])
        else:
            self.data_panel.set_columns([])

        self._refresh_all_ui()

    # --- Series / Session callbacks ---

    def _on_series_selected(self, series_id: str):
        if series_id not in self._series_records:
            return
        self._active_series_id = series_id
        self._refresh_session_ui()

    def _on_session_selected(self, session_name: str):
        rec = self._active_record
        if rec is None or session_name not in rec.fit_sessions:
            return
        rec.active_session_name = session_name
        self._load_session_into_ui()

    def _ensure_placeholder_series(self):
        """Create a placeholder simulated series if no series exists."""
        if self._active_series_id is not None:
            return
        self._simulated_counter += 1
        n = self._simulated_counter
        df = pd.DataFrame({"x": pd.Series(dtype=float), "y": pd.Series(dtype=float)})
        ds_name, columns = self.data_mgr.add_dataframe(f"Simulated {n}", df)

        self.data_panel.set_datasets(
            self.data_mgr.dataset_names, select=ds_name
        )
        self.data_panel.set_columns(columns)

        series_info = {
            "dataset": ds_name,
            "x": "x", "y": "y", "yerr": "", "xerr": "",
            "marker": "o", "linestyle": "None", "color": "",
            "label": ds_name,
        }
        self.data_panel.add_series_entry(series_info)

        # Create the series record so it becomes the active series
        sid = _make_series_id(ds_name, "x", "y")
        rec = SeriesRecord(
            x=np.array([]), y=np.array([]),
            style=series_info, dataset_name=ds_name,
        )
        self._series_records[sid] = rec
        self._active_series_id = sid
        self._sync_series_combo()

    def _on_new_session(self, name: str):
        self._ensure_placeholder_series()
        rec = self._active_record
        if rec is None:
            return
        if name in rec.fit_sessions:
            messagebox.showwarning("Duplicate", f"Session '{name}' already exists.")
            return

        self._session_counter += 1
        rec.ensure_session(name)
        rec.active_session_name = name
        self._refresh_session_ui()

    def _on_rename_session(self, old_name: str, new_name: str):
        rec = self._active_record
        if rec is None or old_name not in rec.fit_sessions:
            return
        if new_name in rec.fit_sessions:
            messagebox.showwarning("Duplicate", f"Session '{new_name}' already exists.")
            return

        sess = rec.fit_sessions.pop(old_name)
        sess.name = new_name
        rec.fit_sessions[new_name] = sess

        # Re-key in PlotManager
        old_key = _make_session_key(self._active_series_id, old_name)
        new_key = _make_session_key(self._active_series_id, new_name)
        self.plot_mgr.rename_session_key(old_key, new_key)

        if rec.active_session_name == old_name:
            rec.active_session_name = new_name

        self._sync_session_list()

    def _on_delete_session(self, name: str):
        rec = self._active_record
        if rec is None or name not in rec.fit_sessions:
            return

        skey = _make_session_key(self._active_series_id, name)
        self.plot_mgr.clear_fit_session(skey)
        del rec.fit_sessions[name]

        if rec.active_session_name == name:
            rec.active_session_name = next(iter(rec.fit_sessions), None)

        self._refresh_session_ui()

    def _on_toggle_series_visible(self, idx: int):
        series_items = self.data_panel.series_list
        if idx < 0 or idx >= len(series_items):
            return
        s = series_items[idx]
        sid = _make_series_id(s["dataset"], s["x"], s["y"])
        rec = self._series_records.get(sid)
        if rec is None:
            return
        rec.visible = not rec.visible
        self._refresh_data_panel_labels()
        self._replot_all_series()

    def _refresh_data_panel_labels(self):
        """Update DataPanel listbox labels to reflect series visibility."""
        series_items = self.data_panel.series_list
        visibility = []
        for s in series_items:
            sid = _make_series_id(s["dataset"], s["x"], s["y"])
            rec = self._series_records.get(sid)
            visibility.append(rec.visible if rec else True)
        self.data_panel.set_series_visibility(visibility)

    def _on_toggle_session_visible(self, name: str):
        rec = self._active_record
        if rec is None or name not in rec.fit_sessions:
            return
        sess = rec.fit_sessions[name]
        sess.visible = not sess.visible
        self._sync_session_list()
        self._replot_all_series()

    def _on_toggle_series_and_sessions(self):
        """Toggle visibility of the active series and all its fit sessions."""
        rec = self._active_record
        if rec is None:
            return
        new_visible = not rec.visible
        rec.visible = new_visible
        for sess in rec.fit_sessions.values():
            sess.visible = new_visible
        self._refresh_data_panel_labels()
        self._sync_session_list()
        self._replot_all_series()
