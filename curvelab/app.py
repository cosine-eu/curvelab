"""Main application window - central coordinator."""

import threading
import tkinter as tk
import warnings
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

import numpy as np
import pandas as pd

from .data_manager import DataManager
from .fit_manager import FitManager
from .plot_manager import PlotManager, SeriesStyle
from .session import (
    FIT_COLORS, FitSession, ParamEdit, SeriesRecord,
    make_series_id as _make_series_id,
    make_session_key as _make_session_key,
)
from .ui_panels import (
    DataPanel, PlotControlPanel, FitPanel, FitResultsPanel,
)
from .ui_dialogs_analysis import FontDialog
from .app_analysis_handlers import AnalysisHandlersMixin
from .app_data_tools import DataToolsMixin
from .app_workspace import WorkspaceMixin
from .app_fit_handlers import FitHandlersMixin


class CurveLabApp(
    AnalysisHandlersMixin, DataToolsMixin, WorkspaceMixin,
    FitHandlersMixin, ttk.Frame,
):
    """Central coordinator. Subclasses ttk.Frame for embeddability."""

    # Max click-to-point distance (display pixels) for point exclusion.
    _CLICK_HIT_RADIUS_PX = 10
    # Poll interval (ms) for checking on a background fit thread.
    _FIT_POLL_INTERVAL_MS = 100

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent

        # Managers (no GUI deps)
        self.data_mgr = DataManager()

        # Multi-series / multi-session state
        self._series_records: dict[str, SeriesRecord] = {}
        self._active_series_id: str | None = None
        self._session_counter: int = 0
        self._simulated_counter: int = 0

        # Font state
        self._ui_family = "TkDefaultFont"
        self._ui_size = 10
        self._plot_family = "sans-serif"
        self._plot_size = 10

        # Suppress noisy lmfit/scipy runtime warnings by default
        self._show_warnings = False
        self._warnings_filter_installed = False
        self._install_warnings_filter()

        # Threading state for async fits
        self._fit_thread: threading.Thread | None = None
        self._fit_abort = threading.Event()

        self._build_layout()
        self._build_menu()

        # Keybindings
        self.parent.bind("<Control-z>", self._undo_param_edit)
        self.parent.bind("<Control-Z>", self._redo_param_edit)
        self.parent.bind("<Control-Shift-Z>", self._redo_param_edit)
        self.parent.bind("<Control-s>", self._save_workspace)
        self.parent.bind("<Control-o>", self._load_workspace)
        self.parent.bind("<Control-v>", self._on_paste_data)
        self.parent.bind("<Control-q>", lambda e: self.parent.destroy())

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

    # --- Guard helpers: return the resolved object, or None after warning ---
    # A non-None _active_session implies a non-None _active_record, so callers
    # of the session/fit-result guards may use self._active_record freely.

    def _require_session(self) -> FitSession | None:
        sess = self._active_session
        if sess is None:
            messagebox.showwarning("No Session", "Create a fit session first.")
            return None
        return sess

    def _require_fit_result(self) -> FitSession | None:
        sess = self._active_session
        if sess is None or sess.result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return None
        return sess

    def _require_last_result(self) -> FitManager | None:
        """Like _require_fit_result, but also needs the live lmfit result
        (absent after a workspace load until refit)."""
        sess = self._require_fit_result()
        if sess is None:
            return None
        fm = sess.fit_manager
        if fm._last_result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return None
        return fm

    def _require_model(self, fm: FitManager) -> bool:
        if not fm.components:
            messagebox.showwarning("No Model", "Add at least one model component.")
            return False
        return True

    # --- Shared fit-data preparation (used by fit, analysis, and data tools) ---

    def _get_fit_data(self, rec: SeriesRecord):
        """Return (x, y, yerr, xerr) cleaned and optionally masked to fit range."""
        from .preprocessing import prepare_fit_data

        # Determine x range from UI
        x_range = None
        xmin_str = self.plot_controls.fit_xmin_var.get().strip()
        xmax_str = self.plot_controls.fit_xmax_var.get().strip()
        if xmin_str or xmax_str:
            try:
                xmin = float(xmin_str) if xmin_str else -np.inf
                xmax = float(xmax_str) if xmax_str else np.inf
                x_range = (xmin, xmax)
            except ValueError:
                messagebox.showwarning(
                    "Invalid Fit Range",
                    f"Could not parse fit range ('{xmin_str}', '{xmax_str}') "
                    "as numbers. Fitting the full data range instead.",
                )
        elif self.plot_controls.fit_visible_var.get():
            x_range = self.plot_mgr.ax.get_xlim()

        x, y, yerr, xerr, warnings = prepare_fit_data(rec, x_range=x_range)

        # Show warnings via UI
        for w in warnings:
            if "NaN" in w:
                messagebox.showinfo("Data Cleaned", w)
            else:
                messagebox.showwarning("Duplicate X Values", w)

        return x, y, yerr, xerr

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
            on_toggle_series_visible=self._on_toggle_series_visible,
            on_column_calc=self._on_column_calc,
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
            on_toggle_session_visible=self._on_toggle_session_visible,
            on_toggle_series_visible=self._on_toggle_series_and_sessions,
            on_abort=self._abort_fit,
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
        self.plot_mgr.canvas.mpl_connect("button_press_event", self._on_plot_click)
        self.plot_mgr.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)

        self._coord_var = tk.StringVar(value="")
        ttk.Label(plot_frame, textvariable=self._coord_var,
                  font=("TkFixedFont", 9)).pack(side=tk.BOTTOM, anchor=tk.W)

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
            on_data_toggled=self._on_data_toggled,
            on_weighted_resid_toggled=self._on_weighted_resid_toggled,
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
        file_menu.add_command(label="Export Curve Data...", command=self._export_curve_data)
        file_menu.add_separator()
        file_menu.add_command(label="Export Model Result...", command=self._export_model_result)
        file_menu.add_command(label="Import Model Result...", command=self._import_model_result)
        file_menu.add_separator()
        file_menu.add_command(label="Save Plot...", command=self._save_plot)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.parent.destroy, accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=file_menu)

        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_command(
            label="Confidence Intervals...", command=self._show_confidence_intervals
        )
        analysis_menu.add_command(
            label="Correlation Matrix...", command=self._show_correlations
        )
        analysis_menu.add_command(
            label="Covariance Matrix...", command=self._show_covariance
        )
        analysis_menu.add_command(
            label="Diagnostic Plots...", command=self._show_diagnostic_plots
        )
        analysis_menu.add_command(
            label="2D Confidence Contours...", command=self._show_confidence_contours
        )
        analysis_menu.add_command(
            label="Profile Likelihood...", command=self._show_profile_likelihood
        )
        analysis_menu.add_command(
            label="Bootstrap CI...", command=self._show_bootstrap
        )
        analysis_menu.add_separator()
        analysis_menu.add_command(
            label="Global Fit...", command=self._on_global_fit
        )
        analysis_menu.add_command(
            label="Uncertainty Propagation...", command=self._show_uncertainty_propagation
        )
        analysis_menu.add_separator()
        analysis_menu.add_command(
            label="Model Comparison...", command=self._show_model_comparison
        )
        analysis_menu.add_command(
            label="F-Test (Nested Models)...", command=self._show_f_test
        )
        analysis_menu.add_separator()
        analysis_menu.add_command(
            label="Simulate Data...", command=self._on_simulate_data
        )
        analysis_menu.add_command(
            label="Evaluate Model...", command=self._evaluate_model
        )
        analysis_menu.add_command(
            label="Find Peaks...", command=self._find_peaks
        )
        analysis_menu.add_command(
            label="Derivative / Integral...", command=self._show_derivative_integral
        )
        analysis_menu.add_command(
            label="Smooth / Outlier Detection...", command=self._smooth_outlier_dialog
        )
        analysis_menu.add_separator()
        analysis_menu.add_command(
            label="Clear Exclusions (Active Series)", command=self._clear_exclusions
        )
        analysis_menu.add_command(
            label="Clear Exclusions (All Series)", command=self._clear_all_exclusions
        )
        menubar.add_cascade(label="Analysis", menu=analysis_menu)

        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Fonts...", command=self._open_font_dialog)
        self._show_warnings_var = tk.BooleanVar(value=self._show_warnings)
        settings_menu.add_checkbutton(
            label="Show numeric warnings",
            variable=self._show_warnings_var,
            command=self._toggle_warnings,
        )
        menubar.add_cascade(label="Settings", menu=settings_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About CurveLab", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.parent.config(menu=menubar)

    def _show_about(self):
        from . import __version__
        dlg = tk.Toplevel(self)
        dlg.title("About CurveLab")
        dlg.resizable(False, False)
        dlg.transient(self.parent)

        # Load logo
        logo_path = Path(__file__).parent / "logo_small.png"
        if logo_path.exists():
            img = tk.PhotoImage(file=str(logo_path))
            dlg._logo_img = img  # prevent garbage collection
            tk.Label(dlg, image=img, bg="#1a1a2e").pack(padx=20, pady=(20, 10))

        tk.Label(
            dlg, text=f"CurveLab {__version__}",
            font=("Helvetica", 16, "bold"),
        ).pack(pady=(5, 5))
        tk.Label(
            dlg,
            text="A GUI interface for lmfit and an experiment\nin AI-assisted coding.",
            justify=tk.CENTER,
        ).pack(pady=(0, 10))
        tk.Label(dlg, text="By G. Vacanti").pack()
        tk.Label(
            dlg, text="https://cosine.eu",
            fg="blue", cursor="hand2",
        ).pack(pady=(0, 10))
        ttk.Button(dlg, text="OK", command=dlg.destroy).pack(pady=(5, 15))

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

    def _install_warnings_filter(self):
        """Suppress RuntimeWarning from lmfit/scipy/uncertainties."""
        if not self._warnings_filter_installed:
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning,
                module=r"(lmfit|scipy|uncertainties)\.",
            )
            self._warnings_filter_installed = True

    def _remove_warnings_filter(self):
        """Remove the RuntimeWarning suppression."""
        warnings.filterwarnings(
            "default", category=RuntimeWarning,
            module=r"(lmfit|scipy|uncertainties)\.",
        )
        self._warnings_filter_installed = False

    def _toggle_warnings(self):
        self._show_warnings = self._show_warnings_var.get()
        if self._show_warnings:
            self._remove_warnings_filter()
        else:
            self._install_warnings_filter()

    # --- Sync helpers ---

    def _sync_series_combo(self):
        ids = list(self._series_records.keys())
        self.fit_panel.set_series_list(ids, select=self._active_series_id or "")

    def _sync_session_list(self):
        rec = self._active_record
        if rec is None:
            self.fit_panel.set_sessions([], series_label="")
            return
        names = list(rec.fit_sessions.keys())
        visibility = {n: s.visible for n, s in rec.fit_sessions.items()}
        series_label = rec.style.get("label", self._active_series_id or "")
        self.fit_panel.set_sessions(names, select=rec.active_session_name or "",
                                    visibility=visibility,
                                    series_label=series_label)

    def _load_session_into_ui(self):
        sess = self._active_session
        if sess is None:
            self.fit_panel.set_components([])
            self.fit_results.clear()
            return

        self._update_component_list_from(sess.fit_manager)

        if sess.result is not None:
            # Enrich params with init_value for display
            params_display = {}
            for name, info in sess.result.params.items():
                entry = dict(info)
                if sess.result.init_params and name in sess.result.init_params:
                    entry["init_value"] = sess.result.init_params[name]
                params_display[name] = entry
            self.fit_results.set_params(params_display)
            self.fit_results.set_gof(sess.result.gof)
            self.fit_results.set_report(sess.result.report)
        else:
            self.fit_results.clear()

    def _update_component_list_from(self, fit_mgr: FitManager):
        self.fit_panel.set_components(fit_mgr.component_labels())

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
        if result.yerr_data is not None and self.plot_controls.weighted_resid_var.get():
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

        self._replot_all_series()
        self._sync_series_combo()
        self._sync_session_list()
        self._load_session_into_ui()

    # --- Plot callback ---

    def _on_plot(self, series_list: list[dict]):
        if not self.data_mgr.is_loaded:
            messagebox.showwarning("No Data", "Load a data file first.")
            return

        self.plot_mgr.clear_all()

        if not series_list:
            messagebox.showwarning("No Series", "Add at least one series.")

        new_records: dict[str, SeriesRecord] = {}
        latest_new_sid: str | None = None

        for s in series_list:
            dataset = s["dataset"]
            sid = _make_series_id(dataset, s["x"], s["y"])
            try:
                x = self.data_mgr.get_column(dataset, s["x"])
                y = self.data_mgr.get_column(dataset, s["y"])
                yerr = self.data_mgr.get_column(dataset, s["yerr"]) if s.get("yerr") else None
                xerr = self.data_mgr.get_column(dataset, s["xerr"]) if s.get("xerr") else None
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
                    latest_new_sid = sid

                if rec.visible:
                    style = SeriesStyle(
                        marker=s.get("marker", "o"),
                        linestyle=s.get("linestyle", "None"),
                        color=s.get("color", ""),
                        label=s.get("label", s["y"]),
                    )
                    self.plot_mgr.plot_series(x, y, yerr=yerr, xerr=xerr, style=style)

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

        # Switch to the most recently added series so new sessions target it
        if latest_new_sid is not None:
            self._active_series_id = latest_new_sid
        elif self._active_series_id not in self._series_records:
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
            if rec.visible:
                mask = rec.mask
                if mask is not None and not mask.all():
                    # Plot included points normally
                    style = SeriesStyle(
                        marker=s.get("marker", "o"),
                        linestyle=s.get("linestyle", "None"),
                        color=s.get("color", ""),
                        label=s.get("label", ""),
                    )
                    inc_yerr = rec.yerr[mask] if rec.yerr is not None else None
                    inc_xerr = rec.xerr[mask] if rec.xerr is not None else None
                    self.plot_mgr.plot_series(
                        rec.x[mask], rec.y[mask],
                        yerr=inc_yerr, xerr=inc_xerr, style=style,
                    )
                    # Plot excluded points as dimmed
                    exc = ~mask
                    exc_style = SeriesStyle(
                        marker=s.get("marker", "o"),
                        linestyle="None",
                        color="gray",
                        markersize=3.0,
                    )
                    exc_yerr = rec.yerr[exc] if rec.yerr is not None else None
                    exc_xerr = rec.xerr[exc] if rec.xerr is not None else None
                    self.plot_mgr.plot_series(
                        rec.x[exc], rec.y[exc],
                        yerr=exc_yerr, xerr=exc_xerr, style=exc_style,
                    )
                else:
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

        # Respect data visibility toggle
        if not self.plot_controls.data_var.get():
            self.plot_mgr.set_data_visible(False)

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

        self._sync_session_list()
        self._load_session_into_ui()

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

    # --- Fit callbacks ---

    def _on_add_component(self, model_name: str, operator: str = "+", expression: str = ""):
        sess = self._require_session()
        if sess is None:
            return
        fm = sess.fit_manager
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

    def _on_data_toggled(self, show: bool):
        self.plot_mgr.set_data_visible(show)

    def _on_mouse_motion(self, event):
        """Update coordinate readout on mouse motion."""
        if event.inaxes is not None and event.xdata is not None:
            self._coord_var.set(f"x={event.xdata:.6g}  y={event.ydata:.6g}")
        else:
            self._coord_var.set("")

    def _on_plot_click(self, event):
        """Handle click on plot — toggle point exclusion when in exclude mode."""
        if not self.plot_controls.exclude_var.get():
            return
        if event.inaxes != self.plot_mgr.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Find the closest point across all visible series
        best_dist = float("inf")
        best_sid = None
        best_idx = None

        # Get axis display transform for distance calculation
        ax = self.plot_mgr.ax
        click_display = ax.transData.transform((event.xdata, event.ydata))
        for sid, rec in self._series_records.items():
            if not rec.visible or len(rec.x) == 0:
                continue
            # Transform all of this series' points to display coords in one
            # call instead of once per point -- click_display is constant
            # per click, so it's computed outside both loops.
            pts_display = ax.transData.transform(np.column_stack((rec.x, rec.y)))
            dists = np.hypot(
                pts_display[:, 0] - click_display[0],
                pts_display[:, 1] - click_display[1],
            )
            i = int(np.argmin(dists))
            dist = dists[i]
            if dist < best_dist:
                best_dist = dist
                best_sid = sid
                best_idx = i

        # Only toggle if click is within 10 pixels of a point
        if best_sid is None or best_dist > self._CLICK_HIT_RADIUS_PX:
            return

        rec = self._series_records[best_sid]
        if rec.mask is None:
            rec.mask = np.ones(len(rec.x), dtype=bool)
        rec.mask[best_idx] = not rec.mask[best_idx]

        n_excluded = int((~rec.mask).sum())
        self._replot_all_series()
        self.plot_mgr.canvas.draw_idle()
        # Brief status in title
        self.parent.title(f"CurveLab — {n_excluded} point(s) excluded")

    def _clear_exclusions(self):
        """Clear point exclusions from the active series."""
        rec = self._active_record
        if rec is None:
            return
        rec.mask = None
        self._replot_all_series()
        self.plot_mgr.canvas.draw()
        self.parent.title("CurveLab")

    def _clear_all_exclusions(self):
        """Clear point exclusions from all series."""
        for rec in self._series_records.values():
            rec.mask = None
        self._replot_all_series()
        self.plot_mgr.canvas.draw()
        self.parent.title("CurveLab")

    def _on_residuals_toggled(self, show: bool):
        self.plot_mgr.set_residuals_visible(show)
        if show:
            # Plot residuals for all sessions that have results
            for sid, rec in self._series_records.items():
                for sess_name, sess in rec.fit_sessions.items():
                    if sess.result is not None and sess.visible:
                        skey = _make_session_key(sid, sess_name)
                        self._plot_residuals_for_session(skey, sess, rec)
        else:
            self.plot_mgr.clear_all_residuals()

    def _on_weighted_resid_toggled(self, _weighted: bool):
        """Re-plot residuals when weighted/raw toggle changes."""
        if not self.plot_controls.residuals_var.get():
            return
        # Clear and re-plot all residuals
        self.plot_mgr.clear_all_residuals()
        for sid, rec in self._series_records.items():
            for sess_name, sess in rec.fit_sessions.items():
                if sess.result is not None and sess.visible:
                    skey = _make_session_key(sid, sess_name)
                    self._plot_residuals_for_session(skey, sess, rec)
        self.plot_mgr.canvas.draw_idle()

    def _on_confidence_band_toggled(self, show: bool):
        xlim = self.plot_mgr.ax.get_xlim()
        ylim = self.plot_mgr.ax.get_ylim()
        self._replot_all_series()
        self.plot_mgr.ax.set_xlim(xlim)
        self.plot_mgr.ax.set_ylim(ylim)
        self.plot_mgr.canvas.draw_idle()

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
                fm.set_param_hint(param_name, vary=new_value)
            elif field == "value":
                old_value = par.value
                new_value = float(value)
                fm.set_param(param_name, value=new_value)
                fm.set_param_hint(param_name, value=new_value)
            elif field == "min":
                old_value = par.min
                new_value = float("-inf") if value in ("-inf", "") else float(value)
                fm.set_param(param_name, min=new_value)
                fm.set_param_hint(param_name, min=new_value)
            elif field == "max":
                old_value = par.max
                new_value = float("inf") if value in ("inf", "") else float(value)
                fm.set_param(param_name, max=new_value)
                fm.set_param_hint(param_name, max=new_value)
            elif field == "expr":
                old_value = par.expr or ""
                new_value = value.strip()
                if new_value:
                    fm.set_param(param_name, expr=new_value)
                    fm.set_param_hint(param_name, expr=new_value)
                else:
                    fm.set_param(param_name, expr="", vary=True)
                    fm.set_param_hint(param_name, expr="", vary=True)
            else:
                return
            edit = ParamEdit(param_name=param_name, field=field,
                             old_value=old_value, new_value=new_value)
            sess.undo_stack.append(edit)
            sess.redo_stack.clear()
        except ValueError:
            self._refresh_param_display()

    def _refresh_param_display(self):
        """Refresh the parameter table from the live FitManager params."""
        fm = self._active_fit_mgr
        if fm is None or fm.params is None:
            return
        self.fit_results.set_params(FitManager.params_to_info(fm.params))

    def _undo_param_edit(self, event=None):
        sess = self._active_session
        fm = self._active_fit_mgr
        if sess is None or fm is None or not sess.undo_stack:
            return
        edit = sess.undo_stack.pop()
        fm.set_param_hint(edit.param_name, **{edit.field: edit.old_value})
        sess.redo_stack.append(edit)
        self._refresh_param_display()

    def _redo_param_edit(self, event=None):
        sess = self._active_session
        fm = self._active_fit_mgr
        if sess is None or fm is None or not sess.redo_stack:
            return
        edit = sess.redo_stack.pop()
        fm.set_param_hint(edit.param_name, **{edit.field: edit.new_value})
        sess.undo_stack.append(edit)
        self._refresh_param_display()

    @classmethod
    def launch(cls):
        """Standalone launch: creates Tk root and runs mainloop."""
        root = tk.Tk()
        root.withdraw()  # hide main window during splash

        # Show splash screen
        logo_path = Path(__file__).parent / "logo.png"
        splash = None
        if logo_path.exists():
            splash = tk.Toplevel(root)
            splash.overrideredirect(True)
            img = tk.PhotoImage(file=str(logo_path))
            splash._img = img
            tk.Label(splash, image=img, bg="#1a1a2e").pack()
            # Center on screen
            splash.update_idletasks()
            sw = splash.winfo_screenwidth()
            sh = splash.winfo_screenheight()
            w, h = img.width(), img.height()
            splash.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        root.title("CurveLab - Data Plotter & Curve Fitter")
        root.geometry("1400x950")
        root.minsize(900, 700)

        app = cls(root)
        app.pack(fill=tk.BOTH, expand=True)

        # Close splash and show main window
        if splash is not None:
            root.after(1500, lambda: (splash.destroy(), root.deiconify()))
        else:
            root.deiconify()

        root.mainloop()
