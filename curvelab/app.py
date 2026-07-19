"""Main application window - central coordinator."""

import threading
import tkinter as tk
import warnings
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

import numpy as np

from .data_manager import DataManager
from .fit_manager import FitManager
from .plot_manager import PlotManager, SeriesStyle
from .session import (
    FitSession, ParamEdit, SeriesRecord,
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
from .app_menu import MenuMixin
from .app_series import SeriesSessionMixin


class CurveLabApp(
    MenuMixin, SeriesSessionMixin, AnalysisHandlersMixin, DataToolsMixin,
    WorkspaceMixin, FitHandlersMixin, ttk.Frame,
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

    def _require_series(self) -> SeriesRecord | None:
        rec = self._active_record
        if rec is None:
            messagebox.showwarning("No Series", "Plot a series first.")
            return None
        return rec

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
            self._display_result(sess.result)
        else:
            self.fit_results.clear()

    def _display_result(self, result):
        """Push a fit result to the results panel, enriching the parameter
        table with each parameter's initial value when available."""
        params_display = {}
        for name, info in result.params.items():
            entry = dict(info)
            if result.init_params and name in result.init_params:
                entry["init_value"] = result.init_params[name]
            params_display[name] = entry
        self.fit_results.set_params(params_display)
        self.fit_results.set_gof(result.gof)
        self.fit_results.set_report(result.report)

    def _update_component_list_from(self, fit_mgr: FitManager):
        self.fit_panel.set_components(fit_mgr.component_labels())

    def _update_component_list(self):
        fm = self._active_fit_mgr
        if fm is None:
            self.fit_panel.set_components([])
            return
        self._update_component_list_from(fm)

    # --- Composite refreshers: resync UI after state changes ---

    def _refresh_session_ui(self):
        """Resync the session list and load the active session into the panels."""
        self._sync_session_list()
        self._load_session_into_ui()

    def _refresh_series_ui(self):
        """Resync the series combo, session list, and active-session panels."""
        self._sync_series_combo()
        self._refresh_session_ui()

    def _refresh_all_ui(self):
        """Replot everything and resync all series/session panels."""
        self._replot_all_series()
        self._refresh_series_ui()

    # --- Residuals / confidence band helpers ---

    def _plot_residuals_for_session(self, skey: str, sess: FitSession, rec: SeriesRecord):
        """Compute and plot residuals for one session."""
        result = sess.result
        if result is None:
            return
        if self.plot_controls.weighted_resid_var.get():
            residuals = result.weighted_residuals()
        else:
            residuals = result.residuals()
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

    def _show_fit_on_plot(self, sid: str, sess: FitSession, rec: SeriesRecord) -> str:
        """Clear and redraw one session's fit curve, plus residuals when the
        residuals panel is shown. Returns the session key of the artists."""
        skey = _make_session_key(sid, sess.name)
        label = f"{rec.style.get('label', sid)} — {sess.name}"
        self.plot_mgr.clear_fit_session(skey)
        self._plot_fit_for_session(skey, sess, label)
        if self.plot_controls.residuals_var.get():
            self._plot_residuals_for_session(skey, sess, rec)
        return skey

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

        for sid, rec in self._series_records.items():
            for sess in rec.fit_sessions.values():
                if sess.result is not None and sess.visible:
                    self._show_fit_on_plot(sid, sess, rec)

        # Switch to the most recently added series so new sessions target it
        if latest_new_sid is not None:
            self._active_series_id = latest_new_sid
        elif self._active_series_id not in self._series_records:
            self._active_series_id = next(iter(self._series_records), None)

        self._refresh_series_ui()

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
            for sess in rec.fit_sessions.values():
                if sess.result is not None and sess.visible:
                    self._show_fit_on_plot(sid, sess, rec)

        # Restore residuals visibility state
        self.plot_mgr.set_residuals_visible(show_resid)

        # Respect data visibility toggle
        if not self.plot_controls.data_var.get():
            self.plot_mgr.set_data_visible(False)

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
