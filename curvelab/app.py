"""Main application window - central coordinator."""

import threading
import tkinter as tk
import warnings
from tkinter import ttk, messagebox
from pathlib import Path

import numpy as np

from .data_manager import DataManager
from .fit_manager import FitManager
from .plot_manager import PlotManager
from .session import (
    FitSession, ParamEdit, SeriesRecord,
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
from .app_plotting import PlottingMixin


# Editable parameter-table fields: how to read the current value off an
# lmfit Parameter, and how to parse the text typed into the cell.
_PARAM_FIELD_EDITORS = {
    "vary": (lambda par: par.vary,
             lambda text: text.lower() in ("yes", "true", "1")),
    "value": (lambda par: par.value, float),
    "min": (lambda par: par.min,
            lambda text: float("-inf") if text in ("-inf", "") else float(text)),
    "max": (lambda par: par.max,
            lambda text: float("inf") if text in ("inf", "") else float(text)),
    "expr": (lambda par: par.expr or "", lambda text: text.strip()),
}


class CurveLabApp(
    MenuMixin, SeriesSessionMixin, PlottingMixin, AnalysisHandlersMixin,
    DataToolsMixin, WorkspaceMixin, FitHandlersMixin, ttk.Frame,
):
    """Central coordinator. Subclasses ttk.Frame for embeddability."""

    # Max click-to-point distance (display pixels) for point exclusion.
    _CLICK_HIT_RADIUS_PX = 10
    # Poll interval (ms) for checking on a background fit thread.
    _FIT_POLL_INTERVAL_MS = 100
    # How long the splash screen stays up before the main window appears.
    _SPLASH_DURATION_MS = 1500

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent

        # Managers (no GUI deps)
        self.data_mgr = DataManager()

        # Multi-series / multi-session state
        self._series_records: dict[str, SeriesRecord] = {}
        self._active_series_id: str | None = None
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

    def _get_fit_data(self, rec: SeriesRecord, warnings_out: list[str] | None = None):
        """Return (x, y, yerr, xerr) cleaned and optionally masked to fit range.

        Preparation warnings pop up as dialogs. Callers that prepare many
        series in a loop (batch and global fit) pass warnings_out instead:
        messages are labelled by series and appended there, so the caller
        can report them once rather than one dialog per series.
        """
        from .preprocessing import prepare_fit_data

        # Determine x range from UI
        x_range = None
        range_warning = None
        xmin_str = self.plot_controls.fit_xmin_var.get().strip()
        xmax_str = self.plot_controls.fit_xmax_var.get().strip()
        if xmin_str or xmax_str:
            try:
                xmin = float(xmin_str) if xmin_str else -np.inf
                xmax = float(xmax_str) if xmax_str else np.inf
                x_range = (xmin, xmax)
            except ValueError:
                range_warning = (
                    f"Could not parse fit range ('{xmin_str}', '{xmax_str}') "
                    "as numbers. Fitting the full data range instead."
                )
        elif self.plot_controls.fit_visible_var.get():
            x_range = self.plot_mgr.ax.get_xlim()

        x, y, yerr, xerr, warnings = prepare_fit_data(rec, x_range=x_range)

        if warnings_out is not None:
            label = rec.style.get("label") or ""
            messages = ([range_warning] if range_warning else []) + warnings
            warnings_out.extend(f"{label}: {m}" if label else m for m in messages)
        else:
            if range_warning:
                messagebox.showwarning("Invalid Fit Range", range_warning)
            for w in warnings:
                if "NaN" in w:
                    messagebox.showinfo("Data Cleaned", w)
                else:
                    messagebox.showwarning("Duplicate X Values", w)

        return x, y, yerr, xerr

    def _report_collected_warnings(self, title: str, messages: list[str]):
        """Show messages collected over a multi-series run as one dialog."""
        if not messages:
            return
        unique = list(dict.fromkeys(messages))
        messagebox.showwarning(title, "\n".join(unique))

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
            on_plot=self._on_plot,
            on_dataset_selected=self._on_dataset_selected,
            on_remove_dataset=self._on_remove_dataset,
            on_toggle_series_visible=self._on_toggle_series_visible,
            on_column_calc=self._on_column_calc,
            on_confirm_remove_series=self._confirm_remove_series,
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
            on_title=self._on_title,
            on_axis_limits=self._on_axis_limits,
        )
        self.plot_controls.grid(row=1, column=0, sticky="ew", pady=2)

        # Fit results panel
        self.fit_results = FitResultsPanel(
            right_frame, on_param_edited=self._on_param_edited,
            on_use_as_start=self._on_use_result_as_start,
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

    def _on_param_edited(self, param_name: str, field: str, value: str):
        fm = self._active_fit_mgr
        sess = self._active_session
        if fm is None or fm.params is None or sess is None:
            return
        if param_name not in fm.params:
            return
        editor = _PARAM_FIELD_EDITORS.get(field)
        if editor is None:
            return
        read_current, parse = editor
        old_value = read_current(fm.params[param_name])
        try:
            new_value = parse(value)
            kwargs = self._param_field_kwargs(field, new_value)
            fm.set_param(param_name, **kwargs)
            fm.set_param_hint(param_name, **kwargs)
        except ValueError:
            self._refresh_param_display()
            return
        # A manually edited value becomes the fit's new starting point, so
        # the reset-to-guess in run_fit doesn't discard it.
        if field == "value":
            fm.set_start_value(param_name, new_value)
        sess.undo_stack.append(ParamEdit(
            param_name=param_name, field=field,
            old_value=old_value, new_value=new_value,
        ))
        sess.redo_stack.clear()

    @staticmethod
    def _param_field_kwargs(field: str, value) -> dict:
        """Parameter attributes to write for one edited field.

        Clearing an expression also restores vary: lmfit forces vary=False
        when an expression is set and never restores it when the expression
        is removed, so an undone expression edit would leave the parameter
        frozen."""
        if field == "expr" and not value:
            return {"expr": "", "vary": True}
        return {field: value}

    def _on_use_result_as_start(self):
        """Adopt the last fit's parameters as the starting point for the next
        fit. Fits otherwise restart from the guess each time; this is the
        opt-in way to chain a refinement."""
        sess = self._require_fit_result()
        if sess is None:
            return
        fm = sess.fit_manager
        if fm.params is None:
            return
        fm.capture_start_values()
        # Reflect the adopted start in the Initial column as feedback.
        params_display = {}
        for name, info in sess.result.params.items():
            entry = dict(info)
            if name in fm.params:
                entry["init_value"] = fm.params[name].value
            params_display[name] = entry
        self.fit_results.set_params(params_display)

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
        fm.set_param_hint(
            edit.param_name, **self._param_field_kwargs(edit.field, edit.old_value)
        )
        if edit.field == "value":
            fm.set_start_value(edit.param_name, edit.old_value)
        sess.redo_stack.append(edit)
        self._refresh_param_display()

    def _redo_param_edit(self, event=None):
        sess = self._active_session
        fm = self._active_fit_mgr
        if sess is None or fm is None or not sess.redo_stack:
            return
        edit = sess.redo_stack.pop()
        fm.set_param_hint(
            edit.param_name, **self._param_field_kwargs(edit.field, edit.new_value)
        )
        if edit.field == "value":
            fm.set_start_value(edit.param_name, edit.new_value)
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
            root.after(cls._SPLASH_DURATION_MS,
                       lambda: (splash.destroy(), root.deiconify()))
        else:
            root.deiconify()

        root.mainloop()
