"""Main application window - central coordinator."""

import csv
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

import numpy as np

from .data_manager import DataManager
from .fit_manager import FitManager, FitResult
from .plot_manager import PlotManager, SeriesStyle
from .session import FIT_COLORS, FitSession, ParamEdit, SeriesRecord
from .ui_panels import (
    DataPanel, PlotControlPanel, FitPanel, FitResultsPanel,
    FontDialog, ModelComparisonDialog,
)
from .workspace import WorkspaceEncoder, encode_value, decode_workspace


def _make_series_id(dataset: str, x_col: str, y_col: str) -> str:
    return f"{dataset}::{x_col}::{y_col}"


def _make_session_key(series_id: str, session_name: str) -> str:
    return f"{series_id}::{session_name}"


class CurveLabApp(ttk.Frame):
    """Central coordinator. Subclasses ttk.Frame for embeddability."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent

        # Managers (no GUI deps)
        self.data_mgr = DataManager()

        # Multi-series / multi-session state
        self._series_records: dict[str, SeriesRecord] = {}
        self._active_series_id: str | None = None
        self._session_counter: int = 0

        # Font state
        self._ui_family = "TkDefaultFont"
        self._ui_size = 10
        self._plot_family = "sans-serif"
        self._plot_size = 10

        self._build_layout()
        self._build_menu()

        # Keybindings
        self.parent.bind("<Control-z>", self._undo_param_edit)
        self.parent.bind("<Control-Z>", self._redo_param_edit)
        self.parent.bind("<Control-Shift-Z>", self._redo_param_edit)
        self.parent.bind("<Control-s>", self._save_workspace)
        self.parent.bind("<Control-o>", self._load_workspace)
        self.parent.bind("<Control-v>", self._on_paste_data)

    # --- Helper properties ---

    @property
    def _active_record(self) -> SeriesRecord | None:
        if self._active_series_id is None:
            return None
        return self._series_records.get(self._active_series_id)

    @property
    def _active_session(self) -> FitSession | None:
        rec = self._active_record
        return rec.active_session if rec else None

    @property
    def _active_fit_mgr(self) -> FitManager | None:
        sess = self._active_session
        return sess.fit_manager if sess else None

    def _active_session_key(self) -> str | None:
        rec = self._active_record
        if rec is None or rec.active_session_name is None:
            return None
        return _make_session_key(self._active_series_id, rec.active_session_name)

    # --- Layout ---

    def _build_layout(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Main horizontal paned window
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.grid(row=0, column=0, sticky="nsew")

        # --- LEFT PANE: Data + Fit panels stacked vertically ---
        left_pane = ttk.PanedWindow(main_pane, orient=tk.VERTICAL)
        main_pane.add(left_pane, weight=0)

        self.data_panel = DataPanel(
            left_pane,
            on_load=self._on_load_file,
            on_add_series=None,
            on_plot=self._on_plot,
            on_dataset_selected=self._on_dataset_selected,
            on_remove_dataset=self._on_remove_dataset,
        )
        left_pane.add(self.data_panel, weight=1)

        self.fit_panel = FitPanel(
            left_pane,
            on_add_component=self._on_add_component,
            on_remove_component=self._on_remove_component,
            on_edit_expression=self._on_edit_expression,
            on_auto_guess=self._on_auto_guess,
            on_fit=self._on_fit,
            on_clear_fit=self._on_clear_fit,
            on_series_selected=self._on_series_selected,
            on_session_selected=self._on_session_selected,
            on_new_session=self._on_new_session,
            on_rename_session=self._on_rename_session,
            on_delete_session=self._on_delete_session,
            on_batch_fit=self._on_batch_fit,
        )
        left_pane.add(self.fit_panel, weight=1)

        # --- RIGHT PANE: Plot + Controls + Results ---
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)

        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)  # plot expands

        # Plot area
        plot_frame = ttk.Frame(right_frame)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(1, weight=1)

        self.plot_mgr = PlotManager(plot_frame)
        self.plot_mgr.toolbar.pack(side=tk.TOP, fill=tk.X)
        self.plot_mgr.get_canvas_widget().pack(fill=tk.BOTH, expand=True)

        # Plot controls strip
        self.plot_controls = PlotControlPanel(
            right_frame,
            on_xscale=lambda s: self.plot_mgr.set_xscale(s),
            on_yscale=lambda s: self.plot_mgr.set_yscale(s),
            on_grid=lambda v: self.plot_mgr.set_grid(v),
            on_equal=lambda v: self.plot_mgr.set_equal_aspect(v),
            on_legend=lambda v: self.plot_mgr.set_legend(v),
            on_show_params_toggled=self._on_show_params_toggled,
            on_residuals_toggled=self._on_residuals_toggled,
            on_confidence_band_toggled=self._on_confidence_band_toggled,
            on_axis_labels=self._on_axis_labels,
        )
        self.plot_controls.grid(row=1, column=0, sticky="ew", pady=2)

        # Fit results panel
        self.fit_results = FitResultsPanel(
            right_frame, on_param_edited=self._on_param_edited
        )
        self.fit_results.grid(row=2, column=0, sticky="nsew")
        right_frame.rowconfigure(2, weight=1)

    def _build_menu(self):
        menubar = tk.Menu(self.parent)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label="Save Workspace...", command=self._save_workspace, accelerator="Ctrl+S"
        )
        file_menu.add_command(
            label="Load Workspace...", command=self._load_workspace, accelerator="Ctrl+O"
        )
        file_menu.add_command(
            label="Paste Data", command=self._on_paste_data, accelerator="Ctrl+V"
        )
        file_menu.add_separator()
        file_menu.add_command(label="Export Parameters...", command=self._export_params)
        file_menu.add_command(label="Export Fit Report...", command=self._export_report)
        file_menu.add_command(label="Save Plot...", command=self._save_plot)
        file_menu.add_separator()
        file_menu.add_command(label="Model Comparison...", command=self._show_model_comparison)
        menubar.add_cascade(label="File", menu=file_menu)

        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Fonts...", command=self._open_font_dialog)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        self.parent.config(menu=menubar)

    def _open_font_dialog(self):
        FontDialog(
            self,
            ui_family=self._ui_family,
            ui_size=self._ui_size,
            plot_family=self._plot_family,
            plot_size=self._plot_size,
            on_apply=self._apply_fonts,
        )

    def _apply_fonts(self, ui_family, ui_size, plot_family, plot_size):
        self._ui_family = ui_family
        self._ui_size = ui_size
        self._plot_family = plot_family
        self._plot_size = plot_size

        style = ttk.Style()
        style.configure(".", font=(ui_family, ui_size))
        style.configure("Treeview", font=(ui_family, ui_size))
        style.configure("Treeview.Heading", font=(ui_family, ui_size, "bold"))

        self.plot_mgr.set_font(plot_family, plot_size)

    # --- Export ---

    def _export_params(self):
        sess = self._active_session
        if sess is None or sess.result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            title="Export Parameters",
        )
        if not filepath:
            return
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "value", "stderr", "min", "max", "vary"])
            for name, info in sess.result.params.items():
                writer.writerow([
                    name, info["value"], info["stderr"],
                    info["min"], info["max"], info["vary"],
                ])

    def _export_report(self):
        sess = self._active_session
        if sess is None or sess.result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")],
            title="Export Fit Report",
        )
        if not filepath:
            return
        with open(filepath, "w") as f:
            f.write(sess.result.report)

    def _save_plot(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"), ("PDF", "*.pdf"),
                ("SVG", "*.svg"), ("All files", "*.*"),
            ],
            title="Save Plot",
        )
        if not filepath:
            return
        try:
            self.plot_mgr.fig.savefig(filepath, dpi=150, bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _show_model_comparison(self):
        rec = self._active_record
        if rec is None:
            messagebox.showwarning("No Series", "Select a series first.")
            return
        rows = []
        for sess_name, sess in rec.fit_sessions.items():
            if sess.result is None:
                continue
            model_desc = " + ".join(
                (f"{c.operator} " if i > 0 else "") + c.name
                for i, c in enumerate(sess.fit_manager.components)
            )
            gof = sess.result.gof
            rows.append({
                "session": sess_name,
                "model": model_desc,
                "n_params": len(sess.result.params),
                "chisqr": gof.get("chi-squared"),
                "redchi": gof.get("reduced chi-squared"),
                "aic": gof.get("AIC"),
                "bic": gof.get("BIC"),
            })
        if not rows:
            messagebox.showinfo("No Fits", "No completed fits to compare.")
            return
        ModelComparisonDialog(self, rows)

    # --- Sync helpers ---

    def _sync_series_combo(self):
        ids = list(self._series_records.keys())
        self.fit_panel.set_series_list(ids, select=self._active_series_id or "")

    def _sync_session_list(self):
        rec = self._active_record
        if rec is None:
            self.fit_panel.set_sessions([])
            return
        names = list(rec.fit_sessions.keys())
        self.fit_panel.set_sessions(names, select=rec.active_session_name or "")

    def _load_session_into_ui(self):
        sess = self._active_session
        if sess is None:
            self.fit_panel.set_components([])
            self.fit_results.clear()
            return

        self._update_component_list_from(sess.fit_manager)

        if sess.result is not None:
            self.fit_results.set_params(sess.result.params)
            self.fit_results.set_report(sess.result.report)
        else:
            self.fit_results.clear()

    def _update_component_list_from(self, fit_mgr: FitManager):
        labels = []
        for i, c in enumerate(fit_mgr.components):
            if c.name == "Expression" and c.expression:
                display = f"Expression: {c.expression}"
                if c.prefix:
                    display = f"{display} ({c.prefix})"
            else:
                display = f"{c.name} ({c.prefix})" if c.prefix else c.name
            if i > 0:
                display = f"{c.operator} {display}"
            labels.append(display)
        self.fit_panel.set_components(labels)

    def _update_component_list(self):
        fm = self._active_fit_mgr
        if fm is None:
            self.fit_panel.set_components([])
            return
        self._update_component_list_from(fm)

    # --- Residuals / confidence band helpers ---

    def _plot_residuals_for_session(self, skey: str, sess: FitSession, rec: SeriesRecord):
        """Compute and plot residuals for one session."""
        result = sess.result
        if result is None:
            return
        residuals = result.y_data - result.y_fit_data
        if result.yerr_data is not None:
            residuals = residuals / result.yerr_data
        self.plot_mgr.plot_residuals(skey, result.x_data, residuals, color=sess.color)

    def _plot_fit_for_session(self, skey: str, sess: FitSession, label: str):
        """Plot fit curve with optional confidence band."""
        result = sess.result
        if result is None:
            return
        show_band = self.plot_controls.confidence_band_var.get()
        self.plot_mgr.plot_fit(
            result.x_dense,
            result.y_fit_dense,
            label=label,
            color=sess.color,
            component_curves=result.component_curves,
            session_key=skey,
            y_uncertainty=result.y_uncertainty,
            show_band=show_band,
        )

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
        for sid in to_remove:
            rec = self._series_records.pop(sid)
            for sess_name in list(rec.fit_sessions):
                skey = _make_session_key(sid, sess_name)
                self.plot_mgr.clear_fit_session(skey)
            if self._active_series_id == sid:
                self._active_series_id = None

        self.data_mgr.remove_dataset(name)

        names = self.data_mgr.dataset_names
        self.data_panel.set_datasets(names)
        if names:
            self._on_dataset_selected(names[0])
        else:
            self.data_panel.set_columns([])

        self._replot_all_series()
        self._sync_series_combo()
        self._sync_session_list()
        self._load_session_into_ui()

    # --- Plot callback ---

    def _on_plot(self, series_list: list[dict]):
        if not self.data_mgr.is_loaded:
            messagebox.showwarning("No Data", "Load a data file first.")
            return
        if not series_list:
            messagebox.showwarning("No Series", "Add at least one series.")
            return

        self.plot_mgr.clear_all()

        new_records: dict[str, SeriesRecord] = {}

        for s in series_list:
            dataset = s["dataset"]
            sid = _make_series_id(dataset, s["x"], s["y"])
            try:
                x = self.data_mgr.get_column(dataset, s["x"])
                y = self.data_mgr.get_column(dataset, s["y"])
                yerr = self.data_mgr.get_column(dataset, s["yerr"]) if s.get("yerr") else None
                xerr = self.data_mgr.get_column(dataset, s["xerr"]) if s.get("xerr") else None
                style = SeriesStyle(
                    marker=s.get("marker", "o"),
                    linestyle=s.get("linestyle", "None"),
                    color=s.get("color", ""),
                    label=s.get("label", s["y"]),
                )
                self.plot_mgr.plot_series(x, y, yerr=yerr, xerr=xerr, style=style)

                if sid in self._series_records:
                    rec = self._series_records[sid]
                    rec.x = x
                    rec.y = y
                    rec.yerr = yerr
                    rec.xerr = xerr
                    rec.style = s
                else:
                    rec = SeriesRecord(
                        x=x, y=y, yerr=yerr, xerr=xerr,
                        style=s, dataset_name=dataset,
                    )
                new_records[sid] = rec

            except Exception as e:
                messagebox.showerror("Plot Error", f"Error plotting series: {e}")

        self._series_records = new_records
        show_resid = self.plot_controls.residuals_var.get()

        for sid, rec in self._series_records.items():
            for sess_name, sess in rec.fit_sessions.items():
                if sess.result is not None and sess.visible:
                    skey = _make_session_key(sid, sess_name)
                    label = f"{rec.style.get('label', sid)} \u2014 {sess_name}"
                    self._plot_fit_for_session(skey, sess, label)
                    if show_resid:
                        self._plot_residuals_for_session(skey, sess, rec)

        if self._active_series_id not in self._series_records:
            self._active_series_id = next(iter(self._series_records), None)

        self._sync_series_combo()
        self._sync_session_list()
        self._load_session_into_ui()

    def _replot_all_series(self):
        """Re-plot all current series and their fit curves."""
        self.plot_mgr.clear_all()
        show_resid = self.plot_controls.residuals_var.get()

        for sid, rec in self._series_records.items():
            s = rec.style
            style = SeriesStyle(
                marker=s.get("marker", "o"),
                linestyle=s.get("linestyle", "None"),
                color=s.get("color", ""),
                label=s.get("label", ""),
            )
            self.plot_mgr.plot_series(
                rec.x, rec.y, yerr=rec.yerr, xerr=rec.xerr, style=style
            )
            for sess_name, sess in rec.fit_sessions.items():
                if sess.result is not None and sess.visible:
                    skey = _make_session_key(sid, sess_name)
                    label = f"{s.get('label', sid)} \u2014 {sess_name}"
                    self._plot_fit_for_session(skey, sess, label)
                    if show_resid:
                        self._plot_residuals_for_session(skey, sess, rec)

        # Restore residuals visibility state
        self.plot_mgr.set_residuals_visible(show_resid)

    # --- Series / Session callbacks ---

    def _on_series_selected(self, series_id: str):
        if series_id not in self._series_records:
            return
        self._active_series_id = series_id
        self._sync_session_list()
        self._load_session_into_ui()

    def _on_session_selected(self, session_name: str):
        rec = self._active_record
        if rec is None or session_name not in rec.fit_sessions:
            return
        rec.active_session_name = session_name
        self._load_session_into_ui()

    def _on_new_session(self, name: str):
        rec = self._active_record
        if rec is None:
            messagebox.showwarning("No Series", "Plot a series first, then select it.")
            return
        if name in rec.fit_sessions:
            messagebox.showwarning("Duplicate", f"Session '{name}' already exists.")
            return

        self._session_counter += 1
        color = FIT_COLORS[len(rec.fit_sessions) % len(FIT_COLORS)]
        sess = FitSession(name=name, color=color)
        rec.fit_sessions[name] = sess
        rec.active_session_name = name
        self._sync_session_list()
        self._load_session_into_ui()

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
        for d in (self.plot_mgr._fit_lines, self.plot_mgr._annotations,
                  self.plot_mgr._residual_lines):
            if old_key in d:
                d[new_key] = d.pop(old_key)

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

        self._sync_session_list()
        self._load_session_into_ui()

    # --- Fit callbacks ---

    def _on_add_component(self, model_name: str, operator: str = "+", expression: str = ""):
        fm = self._active_fit_mgr
        if fm is None:
            messagebox.showwarning("No Session", "Create a fit session first.")
            return
        fm.add_component(model_name, operator=operator, expression=expression)
        self._update_component_list()

    def _on_remove_component(self, index: int):
        fm = self._active_fit_mgr
        if fm is None:
            return
        fm.remove_component(index)
        self._update_component_list()

    def _on_edit_expression(self, index: int, new_expr: str):
        fm = self._active_fit_mgr
        if fm is None:
            return
        fm.edit_expression(index, new_expr)
        self._update_component_list()

    def _get_fit_data(self, rec: SeriesRecord):
        """Return (x, y, yerr) cleaned and optionally masked to visible range.

        Guard-rails applied:
        1. Filter out NaN/inf in x or y (and corresponding yerr entries)
        2. Sort by x
        3. Warn (once per fit) if duplicate x values exist
        """
        x, y, yerr = rec.x.copy(), rec.y.copy(), rec.yerr.copy() if rec.yerr is not None else None

        # 1. Visible-range mask (applied first so guard-rails only touch relevant data)
        if self.plot_controls.fit_visible_var.get():
            xmin, xmax = self.plot_mgr.ax.get_xlim()
            mask = (x >= xmin) & (x <= xmax)
            x, y = x[mask], y[mask]
            yerr = yerr[mask] if yerr is not None else None

        # 2. Filter NaN / inf
        finite_mask = np.isfinite(x) & np.isfinite(y)
        if yerr is not None:
            finite_mask &= np.isfinite(yerr)
        n_dropped = int((~finite_mask).sum())
        if n_dropped > 0:
            x, y = x[finite_mask], y[finite_mask]
            yerr = yerr[finite_mask] if yerr is not None else None
            messagebox.showinfo(
                "Data Cleaned",
                f"Removed {n_dropped} point(s) with NaN/inf values.",
            )

        # 3. Sort by x
        order = np.argsort(x)
        x, y = x[order], y[order]
        yerr = yerr[order] if yerr is not None else None

        # 4. Warn on duplicate x values
        if len(x) > 0:
            n_dup = len(x) - len(np.unique(x))
            if n_dup > 0:
                messagebox.showwarning(
                    "Duplicate X Values",
                    f"{n_dup} duplicate x-value(s) detected. "
                    "This may cause issues with some models.",
                )

        return x, y, yerr

    def _on_auto_guess(self):
        rec = self._active_record
        fm = self._active_fit_mgr
        if rec is None or fm is None:
            messagebox.showwarning("No Session", "Create a fit session first.")
            return
        if not fm.components:
            messagebox.showwarning("No Model", "Add at least one model component.")
            return

        try:
            x, y, _ = self._get_fit_data(rec)
            params = fm.auto_guess(x, y)
            params_info = {}
            for name, par in params.items():
                params_info[name] = {
                    "value": par.value,
                    "stderr": None,
                    "min": par.min,
                    "max": par.max,
                    "vary": par.vary,
                }
            self.fit_results.set_params(params_info)
        except Exception as e:
            messagebox.showerror("Guess Error", str(e))

    def _on_fit(self):
        rec = self._active_record
        sess = self._active_session
        if rec is None or sess is None:
            messagebox.showwarning("No Session", "Create a fit session first.")
            return
        fm = sess.fit_manager
        if not fm.components:
            messagebox.showwarning("No Model", "Add at least one model component.")
            return

        try:
            x, y, yerr = self._get_fit_data(rec)
            result = fm.run_fit(x, y, yerr=yerr)
            sess.result = result

            skey = _make_session_key(self._active_series_id, sess.name)
            series_label = rec.style.get("label", self._active_series_id)
            label = f"{series_label} \u2014 {sess.name}"

            # Clear previous fit for this session, then plot new
            self.plot_mgr.clear_fit_session(skey)
            self._plot_fit_for_session(skey, sess, label)

            # Residuals
            if self.plot_controls.residuals_var.get():
                self._plot_residuals_for_session(skey, sess, rec)

            # Show results
            self.fit_results.set_params(result.params)
            self.fit_results.set_report(result.report)

            # Show params on plot if toggled
            if self.plot_controls.show_params_var.get():
                self.plot_mgr.annotate_params(
                    result.params, gof=result.gof, session_key=skey
                )

        except Exception as e:
            messagebox.showerror("Fit Error", str(e))

    def _on_batch_fit(self):
        """Apply the active session's model to all plotted series."""
        rec = self._active_record
        sess = self._active_session
        if rec is None or sess is None:
            messagebox.showwarning("No Session", "Create a fit session first.")
            return
        source_fm = sess.fit_manager
        if not source_fm.components:
            messagebox.showwarning("No Model", "Add at least one model component.")
            return
        if len(self._series_records) < 1:
            messagebox.showwarning("No Series", "Plot at least one series.")
            return

        session_name = sess.name
        summary_rows = []
        show_resid = self.plot_controls.residuals_var.get()

        for sid, target_rec in self._series_records.items():
            # Ensure target series has a session with the same name
            if session_name not in target_rec.fit_sessions:
                color = FIT_COLORS[len(target_rec.fit_sessions) % len(FIT_COLORS)]
                target_rec.fit_sessions[session_name] = FitSession(
                    name=session_name, color=color,
                )
                if target_rec.active_session_name is None:
                    target_rec.active_session_name = session_name

            target_sess = target_rec.fit_sessions[session_name]
            target_fm = target_sess.fit_manager

            # Clone model components from source
            source_fm.clone_components_to(target_fm)

            try:
                x, y, yerr = self._get_fit_data(target_rec)
                target_fm.auto_guess(x, y)
                result = target_fm.run_fit(x, y, yerr=yerr)
                target_sess.result = result

                skey = _make_session_key(sid, session_name)
                series_label = target_rec.style.get("label", sid)
                label = f"{series_label} \u2014 {session_name}"

                self.plot_mgr.clear_fit_session(skey)
                self._plot_fit_for_session(skey, target_sess, label)
                if show_resid:
                    self._plot_residuals_for_session(skey, target_sess, target_rec)

                model_desc = " + ".join(
                    (f"{c.operator} " if i > 0 else "") + c.name
                    for i, c in enumerate(target_fm.components)
                )
                gof = result.gof
                summary_rows.append({
                    "session": f"{series_label} / {session_name}",
                    "model": model_desc,
                    "n_params": len(result.params),
                    "chisqr": gof.get("chi-squared"),
                    "redchi": gof.get("reduced chi-squared"),
                    "aic": gof.get("AIC"),
                    "bic": gof.get("BIC"),
                })
            except Exception as e:
                series_label = target_rec.style.get("label", sid)
                summary_rows.append({
                    "session": f"{series_label} / {session_name}",
                    "model": "ERROR",
                    "n_params": 0,
                    "chisqr": None,
                    "redchi": None,
                    "aic": None,
                    "bic": None,
                })
                messagebox.showwarning(
                    "Batch Fit Warning",
                    f"Fit failed for {series_label}: {e}",
                )

        # Sync UI to the currently active session
        self._sync_session_list()
        self._load_session_into_ui()

        if summary_rows:
            ModelComparisonDialog(self, summary_rows)

    def _on_clear_fit(self):
        sess = self._active_session
        if sess is None:
            return
        skey = self._active_session_key()
        if skey:
            self.plot_mgr.clear_fit_session(skey)
        sess.fit_manager.clear_components()
        sess.result = None
        sess.undo_stack.clear()
        sess.redo_stack.clear()
        self.fit_panel.set_components([])
        self.fit_results.clear()

    def _on_show_params_toggled(self, show: bool):
        sess = self._active_session
        skey = self._active_session_key()
        if skey is None:
            return
        if show and sess and sess.result:
            self.plot_mgr.annotate_params(
                sess.result.params, gof=sess.result.gof, session_key=skey
            )
        else:
            self.plot_mgr.remove_annotation(session_key=skey)

    def _on_residuals_toggled(self, show: bool):
        self.plot_mgr.set_residuals_visible(show)
        if show:
            # Plot residuals for all sessions that have results
            for sid, rec in self._series_records.items():
                for sess_name, sess in rec.fit_sessions.items():
                    if sess.result is not None and sess.visible:
                        skey = _make_session_key(sid, sess_name)
                        if skey not in self.plot_mgr._residual_lines:
                            self._plot_residuals_for_session(skey, sess, rec)
        else:
            self.plot_mgr.clear_all_residuals()

    def _on_confidence_band_toggled(self, show: bool):
        self._replot_all_series()

    def _on_axis_labels(self, xlabel: str, ylabel: str):
        self.plot_mgr.set_axis_labels(xlabel, ylabel)

    def _on_param_edited(self, param_name: str, field: str, value: str):
        fm = self._active_fit_mgr
        sess = self._active_session
        if fm is None or fm.params is None or sess is None:
            return
        if param_name not in fm.params:
            return
        try:
            par = fm.params[param_name]
            if field == "vary":
                old_value = par.vary
                new_value = value.lower() in ("yes", "true", "1")
                fm.set_param(param_name, vary=new_value)
            elif field == "value":
                old_value = par.value
                new_value = float(value)
                fm.set_param(param_name, value=new_value)
            elif field == "min":
                old_value = par.min
                new_value = float("-inf") if value in ("-inf", "") else float(value)
                fm.set_param(param_name, min=new_value)
            elif field == "max":
                old_value = par.max
                new_value = float("inf") if value in ("inf", "") else float(value)
                fm.set_param(param_name, max=new_value)
            else:
                return
            edit = ParamEdit(param_name=param_name, field=field,
                             old_value=old_value, new_value=new_value)
            sess.undo_stack.append(edit)
            sess.redo_stack.clear()
        except ValueError:
            pass

    def _refresh_param_display(self):
        """Refresh the parameter table from the live FitManager params."""
        fm = self._active_fit_mgr
        if fm is None or fm.params is None:
            return
        params_info = {}
        for name, par in fm.params.items():
            params_info[name] = {
                "value": par.value,
                "stderr": par.stderr,
                "min": par.min,
                "max": par.max,
                "vary": par.vary,
            }
        self.fit_results.set_params(params_info)

    def _undo_param_edit(self, event=None):
        sess = self._active_session
        fm = self._active_fit_mgr
        if sess is None or fm is None or not sess.undo_stack:
            return
        edit = sess.undo_stack.pop()
        fm.set_param(edit.param_name, **{edit.field: edit.old_value})
        sess.redo_stack.append(edit)
        self._refresh_param_display()

    def _redo_param_edit(self, event=None):
        sess = self._active_session
        fm = self._active_fit_mgr
        if sess is None or fm is None or not sess.redo_stack:
            return
        edit = sess.redo_stack.pop()
        fm.set_param(edit.param_name, **{edit.field: edit.new_value})
        sess.undo_stack.append(edit)
        self._refresh_param_display()

    # --- Workspace persistence ---

    def _serialize_workspace(self) -> dict:
        """Build a workspace dict from current app state."""
        data_filepaths = {
            name: str(path) for name, path in self.data_mgr.filepaths.items()
        }

        series = {}
        for sid, rec in self._series_records.items():
            fit_sessions = {}
            for sess_name, sess in rec.fit_sessions.items():
                sess_data = {
                    "name": sess.name,
                    "color": sess.color,
                    "visible": sess.visible,
                    **sess.fit_manager.serialize(),
                }
                if sess.result is not None:
                    r = sess.result
                    sess_data["result"] = encode_value({
                        "x_dense": r.x_dense,
                        "y_fit_dense": r.y_fit_dense,
                        "x_data": r.x_data,
                        "y_data": r.y_data,
                        "y_fit_data": r.y_fit_data,
                        "yerr_data": r.yerr_data,
                        "y_uncertainty": r.y_uncertainty,
                        "component_curves": {
                            k: v for k, v in r.component_curves.items()
                        },
                        "params": r.params,
                        "gof": r.gof,
                        "report": r.report,
                    })
                fit_sessions[sess_name] = sess_data

            series[sid] = {
                "dataset_name": rec.dataset_name,
                "style": rec.style,
                "fit_sessions": fit_sessions,
                "active_session_name": rec.active_session_name,
            }

        return {
            "version": 1,
            "data_filepaths": data_filepaths,
            "series": series,
            "active_series_id": self._active_series_id,
            "plot_controls": {
                "xscale": self.plot_controls.xscale_var.get(),
                "yscale": self.plot_controls.yscale_var.get(),
                "grid": self.plot_controls.grid_var.get(),
                "equal": self.plot_controls.equal_var.get(),
                "legend": self.plot_controls.legend_var.get(),
                "show_params": self.plot_controls.show_params_var.get(),
                "fit_visible": self.plot_controls.fit_visible_var.get(),
                "residuals": self.plot_controls.residuals_var.get(),
                "confidence_band": self.plot_controls.confidence_band_var.get(),
                "xlabel": self.plot_controls.xlabel_var.get(),
                "ylabel": self.plot_controls.ylabel_var.get(),
            },
            "fonts": {
                "ui_family": self._ui_family,
                "ui_size": self._ui_size,
                "plot_family": self._plot_family,
                "plot_size": self._plot_size,
            },
        }

    def _save_workspace(self, event=None):
        """Save current workspace to a .clw JSON file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".clw",
            filetypes=[("CurveLab Workspace", "*.clw"), ("All files", "*.*")],
            title="Save Workspace",
        )
        if not filepath:
            return
        try:
            workspace = self._serialize_workspace()
            with open(filepath, "w") as f:
                json.dump(workspace, f, cls=WorkspaceEncoder, indent=2)
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _load_workspace(self, event=None):
        """Load a workspace from a .clw JSON file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("CurveLab Workspace", "*.clw"), ("All files", "*.*")],
            title="Load Workspace",
        )
        if not filepath:
            return
        try:
            with open(filepath, "r") as f:
                ws = json.load(f, object_hook=decode_workspace)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return

        # 1. Reload datasets from saved filepaths
        self.data_mgr = DataManager()
        dataset_name_map = {}  # old name -> new name (may differ on reload)
        for name, fpath in ws.get("data_filepaths", {}).items():
            try:
                new_name, columns = self.data_mgr.load(fpath)
                dataset_name_map[name] = new_name
            except Exception as e:
                messagebox.showwarning(
                    "Missing File",
                    f"Could not reload '{fpath}':\n{e}\n\nSeries from this dataset will be skipped.",
                )

        self.data_panel.set_datasets(self.data_mgr.dataset_names)
        if self.data_mgr.dataset_names:
            first = self.data_mgr.dataset_names[0]
            self.data_panel.set_columns(self.data_mgr.column_names(first))

        # 2. Reconstruct SeriesRecord objects
        self._series_records.clear()
        for sid, sdata in ws.get("series", {}).items():
            ds_name = sdata.get("dataset_name", "")
            actual_ds = dataset_name_map.get(ds_name)
            if actual_ds is None:
                continue  # dataset couldn't be reloaded

            style = sdata.get("style", {})
            try:
                x_col = style.get("x", "")
                y_col = style.get("y", "")
                x = self.data_mgr.get_column(actual_ds, x_col)
                y = self.data_mgr.get_column(actual_ds, y_col)
                yerr_col = style.get("yerr")
                xerr_col = style.get("xerr")
                yerr = self.data_mgr.get_column(actual_ds, yerr_col) if yerr_col else None
                xerr = self.data_mgr.get_column(actual_ds, xerr_col) if xerr_col else None
            except Exception:
                # Fall back: cannot reconstruct data arrays
                continue

            rec = SeriesRecord(
                x=x, y=y, yerr=yerr, xerr=xerr,
                style=style, dataset_name=actual_ds,
                active_session_name=sdata.get("active_session_name"),
            )

            # Reconstruct fit sessions
            for sess_name, sess_data in sdata.get("fit_sessions", {}).items():
                fm = FitManager.deserialize(sess_data)

                sess = FitSession(
                    name=sess_data.get("name", sess_name),
                    fit_manager=fm,
                    color=sess_data.get("color", ""),
                    visible=sess_data.get("visible", True),
                )

                # Reconstruct FitResult if present
                rdata = sess_data.get("result")
                if rdata is not None:
                    # Ensure arrays are numpy
                    def to_array(v):
                        if isinstance(v, np.ndarray):
                            return v
                        if isinstance(v, list):
                            return np.array(v)
                        return v

                    component_curves = {}
                    for k, v in rdata.get("component_curves", {}).items():
                        component_curves[k] = to_array(v)

                    sess.result = FitResult(
                        x_dense=to_array(rdata.get("x_dense", [])),
                        y_fit_dense=to_array(rdata.get("y_fit_dense", [])),
                        x_data=to_array(rdata.get("x_data", [])),
                        y_data=to_array(rdata.get("y_data", [])),
                        y_fit_data=to_array(rdata.get("y_fit_data", [])),
                        yerr_data=to_array(rdata["yerr_data"]) if rdata.get("yerr_data") is not None else None,
                        y_uncertainty=to_array(rdata["y_uncertainty"]) if rdata.get("y_uncertainty") is not None else None,
                        component_curves=component_curves,
                        params=rdata.get("params", {}),
                        gof=rdata.get("gof", {}),
                        report=rdata.get("report", ""),
                    )

                rec.fit_sessions[sess_name] = sess

            self._series_records[sid] = rec

        # 3. Restore active series
        self._active_series_id = ws.get("active_series_id")
        if self._active_series_id not in self._series_records:
            self._active_series_id = next(iter(self._series_records), None)

        # 4. Restore plot controls
        pc = ws.get("plot_controls", {})
        self.plot_controls.xscale_var.set(pc.get("xscale", "linear"))
        self.plot_controls.yscale_var.set(pc.get("yscale", "linear"))
        self.plot_controls.grid_var.set(pc.get("grid", True))
        self.plot_controls.equal_var.set(pc.get("equal", False))
        self.plot_controls.legend_var.set(pc.get("legend", True))
        self.plot_controls.show_params_var.set(pc.get("show_params", False))
        self.plot_controls.fit_visible_var.set(pc.get("fit_visible", False))
        self.plot_controls.residuals_var.set(pc.get("residuals", False))
        self.plot_controls.confidence_band_var.set(pc.get("confidence_band", False))
        self.plot_controls.xlabel_var.set(pc.get("xlabel", ""))
        self.plot_controls.ylabel_var.set(pc.get("ylabel", ""))

        # Apply plot control states
        self.plot_mgr.set_xscale(pc.get("xscale", "linear"))
        self.plot_mgr.set_yscale(pc.get("yscale", "linear"))
        self.plot_mgr.set_grid(pc.get("grid", True))
        self.plot_mgr.set_equal_aspect(pc.get("equal", False))
        self.plot_mgr.set_legend(pc.get("legend", True))
        self.plot_mgr.set_axis_labels(pc.get("xlabel", ""), pc.get("ylabel", ""))

        # 5. Restore fonts
        fonts = ws.get("fonts", {})
        if fonts:
            self._apply_fonts(
                ui_family=fonts.get("ui_family", self._ui_family),
                ui_size=fonts.get("ui_size", self._ui_size),
                plot_family=fonts.get("plot_family", self._plot_family),
                plot_size=fonts.get("plot_size", self._plot_size),
            )

        # 6. Replot everything and sync UI
        self._replot_all_series()
        self._sync_series_combo()
        self._sync_session_list()
        self._load_session_into_ui()

    @classmethod
    def launch(cls):
        """Standalone launch: creates Tk root and runs mainloop."""
        root = tk.Tk()
        root.title("CurveLab - Data Plotter & Curve Fitter")
        root.geometry("1200x850")
        root.minsize(800, 600)

        app = cls(root)
        app.pack(fill=tk.BOTH, expand=True)

        root.mainloop()
