"""Main application window - central coordinator."""

import csv
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

import numpy as np
import pandas as pd

from .data_manager import DataManager
from .fit_manager import FitManager, FitResult, REDUCE_FUNCTIONS
from .plot_manager import PlotManager, SeriesStyle
from .session import FIT_COLORS, FitSession, ParamEdit, SeriesRecord
from .ui_panels import (
    DataPanel, PlotControlPanel, FitPanel, FitResultsPanel,
    FontDialog, ModelComparisonDialog,
    ConfidenceIntervalDialog, CorrelationMatrixDialog,
    BruteCandidatesDialog, EmceeSummaryDialog,
    DiagnosticPlotsDialog, ConfidenceContourDialog,
    GlobalFitDialog, UncertaintyPropagationDialog,
    SimulateDataDialog, ColumnCalculatorDialog, FTestDialog,
    ProfileLikelihoodDialog, BootstrapDialog, CovarianceMatrixDialog,
    EvaluateModelDialog, FindPeaksDialog, DerivativeIntegralDialog,
    SmoothOutlierDialog,
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
        self._simulated_counter: int = 0

        # Font state
        self._ui_family = "TkDefaultFont"
        self._ui_size = 10
        self._plot_family = "sans-serif"
        self._plot_size = 10

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
            label="Clear Point Exclusions", command=self._clear_exclusions
        )
        menubar.add_cascade(label="Analysis", menu=analysis_menu)

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

    def _export_curve_data(self):
        """Export fit curve, residuals, and component curves as CSV."""
        sess = self._active_session
        if sess is None or sess.result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("TSV", "*.tsv"), ("All files", "*.*")],
            title="Export Curve Data",
        )
        if not filepath:
            return

        result = sess.result

        # Build dense curve data (fit curve + components)
        dense_data = {"x": result.x_dense, "y_fit": result.y_fit_dense}
        if result.y_uncertainty is not None:
            dense_data["y_uncertainty"] = result.y_uncertainty
        for comp_name, comp_curve in result.component_curves.items():
            dense_data[f"component_{comp_name.rstrip('_')}"] = comp_curve

        # Build data-point residuals
        residuals = result.y_data - result.y_fit_data
        weighted_residuals = None
        if result.yerr_data is not None:
            safe_yerr = np.maximum(np.abs(result.yerr_data), 1e-12)
            weighted_residuals = residuals / safe_yerr

        try:
            import pandas as pd

            # Sheet 1: dense fit curve
            df_curve = pd.DataFrame(dense_data)

            # Sheet 2: data-point residuals
            resid_data = {
                "x": result.x_data,
                "y_data": result.y_data,
                "y_fit": result.y_fit_data,
                "residuals": residuals,
            }
            if weighted_residuals is not None:
                resid_data["weighted_residuals"] = weighted_residuals
            df_resid = pd.DataFrame(resid_data)

            if filepath.endswith(".tsv"):
                sep = "\t"
            else:
                sep = ","

            # Write both tables separated by a blank line
            with open(filepath, "w", newline="") as f:
                f.write("# Fit curve (dense grid)\n")
                df_curve.to_csv(f, sep=sep, index=False)
                f.write("\n# Data points and residuals\n")
                df_resid.to_csv(f, sep=sep, index=False)

        except Exception as e:
            messagebox.showerror("Export Error", str(e))

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

    def _show_f_test(self):
        rec = self._active_record
        if rec is None:
            messagebox.showwarning("No Series", "Select a series first.")
            return
        sessions = {}
        for sess_name, sess in rec.fit_sessions.items():
            if sess.result is None:
                continue
            gof = sess.result.gof
            n_vary = sum(1 for p in sess.result.params.values() if p.get("vary", True))
            sessions[sess_name] = {
                "n_params": n_vary,
                "chisqr": gof.get("chi-squared", 0),
                "n_data": len(sess.result.x_data),
            }
        if len(sessions) < 2:
            messagebox.showinfo("Need 2+ Fits", "Need at least two completed fits to compare.")
            return
        FTestDialog(self, sessions)

    # --- Simulate Data ---

    def _evaluate_model(self):
        """Evaluate the fitted model at user-specified x values."""
        sess = self._active_session
        fm = self._active_fit_mgr
        if sess is None or sess.result is None or fm is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        EvaluateModelDialog(self, fm)

    def _find_peaks(self):
        """Auto-detect peaks in active series and add Gaussian components."""
        rec = self._active_record
        fm = self._active_fit_mgr
        if rec is None or fm is None:
            messagebox.showwarning("No Data", "Load data and create a session first.")
            return
        x, y, yerr, xerr = self._get_fit_data(rec)
        if len(x) < 3:
            messagebox.showwarning("Insufficient Data", "Need at least 3 data points.")
            return
        FindPeaksDialog(self, x, y, fm, on_done=self._update_component_list)

    def _smooth_outlier_dialog(self):
        rec = self._active_record
        if rec is None:
            messagebox.showwarning("No Data", "Plot a series first.")
            return
        SmoothOutlierDialog(
            self,
            x=rec.x.copy(),
            y=rec.y.copy(),
            yerr=rec.yerr.copy() if rec.yerr is not None else None,
            mask=rec.mask.copy() if rec.mask is not None else None,
            ax=self.plot_mgr.ax,
            canvas=self.plot_mgr.canvas,
            on_apply_mask=self._apply_smooth_mask,
            on_export_series=self._export_smooth_series,
        )

    def _apply_smooth_mask(self, new_mask):
        rec = self._active_record
        if rec is None:
            return
        rec.mask = new_mask
        self._replot_all_series()
        self.plot_mgr.canvas.draw_idle()

    def _export_smooth_series(self, x, y, label_suffix):
        import pandas as pd
        rec = self._active_record
        ds_base = rec.dataset_name if rec else "data"
        ds_name = f"{ds_base} ({label_suffix})"
        df = pd.DataFrame({"x": x, "y": y})
        ds_name, columns = self.data_mgr.add_dataframe(ds_name, df)
        self.data_panel.set_datasets(self.data_mgr.dataset_names, select=ds_name)
        self.data_panel.set_columns(columns)
        series_info = {
            "dataset": ds_name,
            "x": "x", "y": "y", "yerr": "", "xerr": "",
            "marker": "o", "linestyle": "-", "color": "",
            "label": ds_name,
        }
        self.data_panel.add_series_entry(series_info)
        self._on_plot(self.data_panel.series_list)

    def _show_derivative_integral(self):
        """Show derivative and integral of the fitted curve."""
        sess = self._active_session
        fm = self._active_fit_mgr
        if sess is None or sess.result is None or fm is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        DerivativeIntegralDialog(self, fm, sess.result)

    def _on_simulate_data(self):
        fm = self._active_fit_mgr
        if fm is None or not fm.components or fm.params is None:
            messagebox.showwarning(
                "No Model", "Set up a model with parameters first."
            )
            return

        # Default x-range from active series or plot limits
        rec = self._active_record
        if rec is not None and len(rec.x) > 0:
            x_min, x_max = float(rec.x.min()), float(rec.x.max())
        else:
            x_min, x_max = 0.0, 10.0

        SimulateDataDialog(
            self,
            x_min=x_min,
            x_max=x_max,
            n_points=200,
            on_generate=self._generate_simulated_data,
        )

    def _generate_simulated_data(self, x_min, x_max, n_points, noise_cfg):
        fm = self._active_fit_mgr
        x = np.linspace(x_min, x_max, n_points)
        xerr = None

        # X-jitter
        if noise_cfg.get("jitter"):
            sigma_x = noise_cfg["jitter_sigma"]
            x = x + np.random.normal(0, sigma_x, size=n_points)
            xerr = np.full(n_points, sigma_x)

        y = fm.model.eval(fm.params, x=x)

        variance = np.zeros(n_points)

        # Poisson noise
        if noise_cfg.get("poisson"):
            scale = noise_cfg["poisson_scale"]
            poisson_sigma = scale * np.sqrt(np.abs(y))
            y = y + poisson_sigma * np.random.normal(0, 1, size=n_points)
            variance += poisson_sigma ** 2

        # Gaussian noise
        if noise_cfg.get("gaussian"):
            sigma = noise_cfg["gaussian_sigma"]
            y = y + np.random.normal(0, sigma, size=n_points)
            variance += sigma ** 2

        yerr = np.sqrt(variance) if variance.any() else None

        # Sort by x (jitter may reorder)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        if yerr is not None:
            yerr = yerr[order]
        if xerr is not None:
            xerr = xerr[order]

        # Check if active series is an empty placeholder — reuse its dataset
        cur_rec = self._active_record
        reuse_placeholder = (
            cur_rec is not None
            and len(cur_rec.x) == 0
            and cur_rec.dataset_name in self.data_mgr.datasets
        )

        if reuse_placeholder:
            ds_name = cur_rec.dataset_name
            data = {"x": x, "y": y}
            if yerr is not None:
                data["yerr"] = yerr
            if xerr is not None:
                data["xerr"] = xerr
            self.data_mgr.datasets[ds_name] = pd.DataFrame(data)
            # Update column combos for this dataset
            self.data_panel.set_columns(self.data_mgr.column_names(ds_name))
            # Update the series entry in DataPanel to reflect new error columns
            for item in self.data_panel._series_items:
                if item["dataset"] == ds_name:
                    item["yerr"] = "yerr" if yerr is not None else ""
                    item["xerr"] = "xerr" if xerr is not None else ""
                    break
        else:
            self._simulated_counter += 1
            n = self._simulated_counter
            data = {"x": x, "y": y}
            if yerr is not None:
                data["yerr"] = yerr
            if xerr is not None:
                data["xerr"] = xerr
            df = pd.DataFrame(data)
            ds_name, columns = self.data_mgr.add_dataframe(f"Simulated {n}", df)

            self.data_panel.set_datasets(
                self.data_mgr.dataset_names, select=ds_name
            )
            self.data_panel.set_columns(columns)

            series_info = {
                "dataset": ds_name,
                "x": "x", "y": "y",
                "yerr": "yerr" if yerr is not None else "",
                "xerr": "xerr" if xerr is not None else "",
                "marker": "o", "linestyle": "None", "color": "",
                "label": ds_name,
            }
            self.data_panel.add_series_entry(series_info)

        self._on_plot(self.data_panel.series_list)

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
        visibility = {n: s.visible for n, s in rec.fit_sessions.items()}
        self.fit_panel.set_sessions(names, select=rec.active_session_name or "",
                                    visibility=visibility)

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
        labels = []
        for i, c in enumerate(fit_mgr.components):
            if c.name == "Expression" and c.expression:
                display = f"Expression: {c.expression}"
                if c.prefix:
                    display = f"{display} ({c.prefix})"
            elif c.name == "Spline" and c.expression:
                display = f"Spline [{c.expression}]"
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

    def _on_column_calc(self):
        dataset = self.data_panel.dataset_var.get()
        if not dataset:
            messagebox.showwarning("No Dataset", "Load a dataset first.")
            return
        columns = self.data_mgr.column_names(dataset)

        _SAFE_NAMES = {
            "abs": np.abs, "sqrt": np.sqrt, "log": np.log, "log2": np.log2,
            "log10": np.log10, "exp": np.exp, "sin": np.sin, "cos": np.cos,
            "tan": np.tan, "arcsin": np.arcsin, "arccos": np.arccos,
            "arctan": np.arctan, "arctan2": np.arctan2,
            "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
            "floor": np.floor, "ceil": np.ceil, "round": np.round_,
            "sign": np.sign, "clip": np.clip,
            "pi": np.pi, "e": np.e, "inf": np.inf, "nan": np.nan,
            "diff": np.diff, "cumsum": np.cumsum,
            "mean": np.mean, "std": np.std, "min": np.min, "max": np.max,
            "where": np.where, "isnan": np.isnan, "isinf": np.isinf,
        }

        def on_apply(name, expr, preview_only):
            if not expr:
                raise ValueError("Enter an expression.")
            df = self.data_mgr.datasets[dataset]
            namespace = dict(_SAFE_NAMES)
            namespace["__builtins__"] = {}
            for col in df.columns:
                namespace[col] = df[col].to_numpy(dtype=float)
            result = eval(expr, namespace)  # noqa: S307
            result = np.asarray(result, dtype=float)
            if result.ndim == 0:
                result = np.full(len(df), result)
            if len(result) != len(df):
                raise ValueError(
                    f"Result has {len(result)} values, expected {len(df)}. "
                    f"(Functions like diff reduce length by 1.)"
                )
            if preview_only:
                return result
            df[name] = result
            # Refresh column dropdowns
            self.data_panel.set_columns(list(df.columns))
            return result

        ColumnCalculatorDialog(self, columns=columns, on_apply=on_apply)

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
        """Return (x, y, yerr, xerr) cleaned and optionally masked to visible range.

        Guard-rails applied:
        1. Filter out NaN/inf in x or y (and corresponding yerr/xerr entries)
        2. Sort by x
        3. Warn (once per fit) if duplicate x values exist
        """
        x, y = rec.x.copy(), rec.y.copy()
        yerr = rec.yerr.copy() if rec.yerr is not None else None
        xerr = rec.xerr.copy() if rec.xerr is not None else None

        # 0. Apply point exclusion mask
        if rec.mask is not None:
            x, y = x[rec.mask], y[rec.mask]
            yerr = yerr[rec.mask] if yerr is not None else None
            xerr = xerr[rec.mask] if xerr is not None else None

        # 1. Fit range mask: explicit typed range takes priority, then visible range
        xmin_str = self.plot_controls.fit_xmin_var.get().strip()
        xmax_str = self.plot_controls.fit_xmax_var.get().strip()
        if xmin_str or xmax_str:
            try:
                xmin = float(xmin_str) if xmin_str else -np.inf
                xmax = float(xmax_str) if xmax_str else np.inf
            except ValueError:
                xmin, xmax = -np.inf, np.inf
            mask = (x >= xmin) & (x <= xmax)
            x, y = x[mask], y[mask]
            yerr = yerr[mask] if yerr is not None else None
            xerr = xerr[mask] if xerr is not None else None
        elif self.plot_controls.fit_visible_var.get():
            xmin, xmax = self.plot_mgr.ax.get_xlim()
            mask = (x >= xmin) & (x <= xmax)
            x, y = x[mask], y[mask]
            yerr = yerr[mask] if yerr is not None else None
            xerr = xerr[mask] if xerr is not None else None

        # 2. Filter NaN / inf
        finite_mask = np.isfinite(x) & np.isfinite(y)
        if yerr is not None:
            finite_mask &= np.isfinite(yerr)
        if xerr is not None:
            finite_mask &= np.isfinite(xerr)
        n_dropped = int((~finite_mask).sum())
        if n_dropped > 0:
            x, y = x[finite_mask], y[finite_mask]
            yerr = yerr[finite_mask] if yerr is not None else None
            xerr = xerr[finite_mask] if xerr is not None else None
            messagebox.showinfo(
                "Data Cleaned",
                f"Removed {n_dropped} point(s) with NaN/inf values.",
            )

        # 3. Sort by x
        order = np.argsort(x)
        x, y = x[order], y[order]
        yerr = yerr[order] if yerr is not None else None
        xerr = xerr[order] if xerr is not None else None

        # 4. Warn on duplicate x values
        if len(x) > 0:
            n_dup = len(x) - len(np.unique(x))
            if n_dup > 0:
                messagebox.showwarning(
                    "Duplicate X Values",
                    f"{n_dup} duplicate x-value(s) detected. "
                    "This may cause issues with some models.",
                )

        return x, y, yerr, xerr

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
            x, y, _, _ = self._get_fit_data(rec)
            params = fm.auto_guess(x, y)
            params_info = {}
            for name, par in params.items():
                params_info[name] = {
                    "value": par.value,
                    "stderr": None,
                    "min": par.min,
                    "max": par.max,
                    "vary": par.vary,
                    "expr": par.expr or "",
                }
            self.fit_results.set_params(params_info)
        except Exception as e:
            messagebox.showerror("Guess Error", str(e))

    _SLOW_METHODS = {"emcee", "brute", "differential_evolution", "basinhopping",
                      "dual_annealing", "shgo", "ampgo"}

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

        method = self.fit_panel.method_var.get()

        # Brute validation: all varied params need finite bounds
        if method == "brute":
            if fm.params is not None:
                for name, par in fm.params.items():
                    if par.vary and (par.min == float("-inf") or par.max == float("inf")):
                        messagebox.showwarning(
                            "Brute Requires Bounds",
                            f"Parameter '{name}' needs finite min and max bounds "
                            f"for brute-force search.",
                        )
                        return

        if method == "odr":
            self._run_fit_odr(rec, sess)
        elif method in self._SLOW_METHODS:
            self._run_fit_async(rec, sess, method)
        else:
            self._run_fit_sync(rec, sess, method)

    def _get_fit_options(self):
        """Read reduce function, weight mode, max_nfev, band_sigma, scale_covar from UI."""
        reduce_fcn = REDUCE_FUNCTIONS.get(self.fit_panel.reduce_var.get())
        weight_mode = self.fit_panel.weight_var.get()
        max_nfev_str = self.fit_panel.max_nfev_var.get().strip()
        max_nfev = int(max_nfev_str) if max_nfev_str else None
        band_sigma = int(self.plot_controls.band_sigma_var.get())
        scale_covar = self.fit_panel.scale_covar_var.get()
        return reduce_fcn, weight_mode, max_nfev, band_sigma, scale_covar

    def _run_fit_sync(self, rec, sess, method):
        """Run fit synchronously (fast methods)."""
        fm = sess.fit_manager
        try:
            x, y, yerr, xerr = self._get_fit_data(rec)
            reduce_fcn, weight_mode, max_nfev, band_sigma, scale_covar = self._get_fit_options()
            result = fm.run_fit(
                x, y, yerr=yerr, xerr=xerr, method=method,
                reduce_fcn=reduce_fcn, weight_mode=weight_mode,
                max_nfev=max_nfev, band_sigma=band_sigma,
                scale_covar=scale_covar,
            )
            sess.result = result
            self._post_fit_update(sess, rec)
        except Exception as e:
            messagebox.showerror("Fit Error", str(e))

    def _run_fit_odr(self, rec, sess):
        """Run ODR fit using odrpack."""
        fm = sess.fit_manager
        try:
            x, y, yerr, xerr = self._get_fit_data(rec)
            _, _, _, band_sigma, _ = self._get_fit_options()
            result = fm.run_odr(
                x, y, yerr=yerr, xerr=xerr, band_sigma=band_sigma,
            )
            sess.result = result
            self._post_fit_update(sess, rec)
        except ImportError as e:
            messagebox.showerror("Missing Package", str(e))
        except Exception as e:
            messagebox.showerror("ODR Error", str(e))

    def _run_fit_async(self, rec, sess, method):
        """Run fit in a background thread (slow methods)."""
        if self._fit_thread is not None and self._fit_thread.is_alive():
            messagebox.showwarning("Busy", "A fit is already running.")
            return

        self._fit_abort.clear()
        self.fit_panel.set_fitting_state(True)

        fm = sess.fit_manager
        try:
            x, y, yerr, xerr = self._get_fit_data(rec)
        except Exception as e:
            messagebox.showerror("Fit Error", str(e))
            self.fit_panel.set_fitting_state(False)
            return

        # Build iter_cb that checks abort flag
        def iter_cb(params, iter, resid, *args, **kw):
            if self._fit_abort.is_set():
                return True

        # Build fit_kws for emcee
        fit_kws = {}
        if method == "emcee":
            fit_kws["is_weighted"] = yerr is not None

        reduce_fcn, weight_mode, max_nfev, band_sigma, scale_covar = self._get_fit_options()

        # Container for result/error from the thread
        container = {"result": None, "error": None}

        def _run():
            try:
                result = fm.run_fit(
                    x, y, yerr=yerr, xerr=xerr, method=method,
                    iter_cb=iter_cb, fit_kws=fit_kws,
                    reduce_fcn=reduce_fcn, weight_mode=weight_mode,
                    max_nfev=max_nfev, band_sigma=band_sigma,
                    scale_covar=scale_covar,
                )
                container["result"] = result
            except Exception as e:
                container["error"] = e

        self._fit_thread = threading.Thread(target=_run, daemon=True)
        self._fit_thread.start()

        def _poll():
            if self._fit_thread.is_alive():
                self.after(100, _poll)
                return
            self._fit_thread = None
            self.fit_panel.set_fitting_state(False)

            if container["error"] is not None:
                if not self._fit_abort.is_set():
                    messagebox.showerror("Fit Error", str(container["error"]))
                return
            if self._fit_abort.is_set():
                return

            sess.result = container["result"]
            self._post_fit_update(sess, rec)

            # Auto-show special result dialogs
            if sess.result.candidates:
                self._show_candidates_dialog(sess)
            if sess.result.flatchain is not None:
                self._show_emcee_summary_dialog(sess)

        self.after(100, _poll)

    def _abort_fit(self):
        """Signal the background fit to stop."""
        self._fit_abort.set()

    def _post_fit_update(self, sess, rec):
        """Update plot, params, and report after a fit completes."""
        result = sess.result
        skey = _make_session_key(self._active_series_id, sess.name)
        series_label = rec.style.get("label", self._active_series_id)
        label = f"{series_label} \u2014 {sess.name}"

        self.plot_mgr.clear_fit_session(skey)
        self._plot_fit_for_session(skey, sess, label)

        if self.plot_controls.residuals_var.get():
            self._plot_residuals_for_session(skey, sess, rec)

        # Enrich params with init_value for the results panel
        params_display = {}
        for name, info in result.params.items():
            entry = dict(info)
            if result.init_params and name in result.init_params:
                entry["init_value"] = result.init_params[name]
            params_display[name] = entry

        self.fit_results.set_params(params_display)
        self.fit_results.set_gof(result.gof)
        self.fit_results.set_report(result.report)

        if self.plot_controls.show_params_var.get():
            self.plot_mgr.annotate_params(
                result.params, gof=result.gof, session_key=skey
            )

    # --- Analysis handlers ---

    def _show_confidence_intervals(self):
        sess = self._active_session
        if sess is None or sess.result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        fm = sess.fit_manager
        try:
            ci_text = fm.compute_confidence_intervals()
            ConfidenceIntervalDialog(self, ci_text)
        except Exception as e:
            messagebox.showerror("CI Error", str(e))

    def _show_correlations(self):
        sess = self._active_session
        if sess is None or sess.result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        fm = sess.fit_manager
        try:
            correlations = fm.get_correlations()
            if not correlations:
                messagebox.showinfo("No Correlations", "No parameter correlations available.")
                return
            CorrelationMatrixDialog(self, correlations)
        except Exception as e:
            messagebox.showerror("Correlation Error", str(e))

    def _show_covariance(self):
        sess = self._active_session
        if sess is None or sess.result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        fm = sess.fit_manager
        result = fm.get_covariance_matrix()
        if result is None:
            messagebox.showinfo("No Covariance", "Covariance matrix not available.")
            return
        param_names, cov_matrix = result
        CovarianceMatrixDialog(self, param_names, cov_matrix)

    def _show_diagnostic_plots(self):
        sess = self._active_session
        if sess is None or sess.result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        DiagnosticPlotsDialog(self, sess.result)

    def _show_confidence_contours(self):
        sess = self._active_session
        if sess is None or sess.result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        fm = sess.fit_manager
        if fm._last_result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        # Collect varied parameters
        vary_params = [
            name for name, par in fm._last_result.params.items()
            if par.vary
        ]
        if len(vary_params) < 2:
            messagebox.showwarning(
                "Not Enough Parameters",
                "Need at least 2 varied parameters for contour plots.",
            )
            return
        ConfidenceContourDialog(self, fm._last_result, vary_params)

    def _show_profile_likelihood(self):
        sess = self._active_session
        if sess is None or sess.result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        fm = sess.fit_manager
        if fm._last_result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        try:
            profiles = fm.compute_ci_profiles()
            if not profiles:
                messagebox.showinfo("No Profiles",
                                    "Could not compute profile traces.")
                return
            best_chi2 = fm._last_result.chisqr
            ProfileLikelihoodDialog(self, profiles, best_chi2)
        except Exception as e:
            messagebox.showerror("Profile Error", str(e))

    def _show_bootstrap(self):
        sess = self._active_session
        rec = self._active_record
        if sess is None or sess.result is None or rec is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        fm = sess.fit_manager

        def on_run(n_boot, boot_type):
            x, y, yerr, xerr = self._get_fit_data(rec)
            weight_mode = self.fit_panel.weight_var.get()
            method = self.fit_panel.method_var.get()
            return fm.run_bootstrap(
                x, y, yerr=yerr, n_boot=n_boot,
                method=method, boot_type=boot_type,
                weight_mode=weight_mode,
            )

        BootstrapDialog(self, on_run=on_run)

    def _show_candidates_dialog(self, sess):
        """Show brute-force candidates dialog with option to load values."""
        if sess.result is None or not sess.result.candidates:
            return

        def on_select(params_dict):
            fm = sess.fit_manager
            for name, val in params_dict.items():
                fm.set_param(name, value=val)
            self._refresh_param_display()

        BruteCandidatesDialog(self, sess.result.candidates, on_select=on_select)

    def _show_emcee_summary_dialog(self, sess):
        """Show emcee MCMC summary dialog."""
        if sess.result is None or sess.result.flatchain is None:
            return
        EmceeSummaryDialog(self, sess.result.flatchain, sess.result.params)

    def _on_global_fit(self):
        """Open Global Fit dialog for simultaneous fitting across series."""
        rec = self._active_record
        sess = self._active_session
        if rec is None or sess is None:
            messagebox.showwarning("No Session", "Create a fit session first.")
            return
        fm = sess.fit_manager
        if not fm.components or fm.params is None:
            messagebox.showwarning("No Model", "Add at least one model component.")
            return
        if len(self._series_records) < 2:
            messagebox.showwarning("Need Series", "Plot at least 2 series for global fitting.")
            return

        series_info = []
        for sid, r in self._series_records.items():
            series_info.append({"id": sid, "label": r.style.get("label", sid)})

        base_param_names = list(fm.params.keys())

        def on_fit(selected_ids, shared):
            datasets = []
            selected_recs = []
            for sid in selected_ids:
                r = self._series_records[sid]
                x, y, yerr, xerr = self._get_fit_data(r)
                datasets.append((x, y, yerr, xerr))
                selected_recs.append((sid, r))

            _, weight_mode, max_nfev, _, _ = self._get_fit_options()
            method = self.fit_panel.method_var.get()
            results = fm.run_global_fit(
                datasets, shared, method=method,
                max_nfev=max_nfev, weight_mode=weight_mode,
            )

            session_name = sess.name
            show_resid = self.plot_controls.residuals_var.get()

            for (sid, r), result in zip(selected_recs, results):
                if session_name not in r.fit_sessions:
                    color = FIT_COLORS[len(r.fit_sessions) % len(FIT_COLORS)]
                    r.fit_sessions[session_name] = FitSession(
                        name=session_name, color=color,
                    )
                    if r.active_session_name is None:
                        r.active_session_name = session_name

                target_sess = r.fit_sessions[session_name]
                target_sess.result = result

                skey = _make_session_key(sid, session_name)
                series_label = r.style.get("label", sid)
                label = f"{series_label} \u2014 {session_name}"
                self.plot_mgr.clear_fit_session(skey)
                self._plot_fit_for_session(skey, target_sess, label)
                if show_resid:
                    self._plot_residuals_for_session(skey, target_sess, r)

            self._sync_session_list()
            self._load_session_into_ui()

        GlobalFitDialog(self, series_info, base_param_names, on_fit=on_fit)

    def _show_uncertainty_propagation(self):
        """Open Uncertainty Propagation dialog."""
        sess = self._active_session
        if sess is None or sess.result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        fm = sess.fit_manager
        if fm._last_result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        try:
            uvars = fm._last_result.uvars
        except Exception:
            messagebox.showwarning(
                "No Uncertainties",
                "Uncertainty variables not available. Ensure covariance was estimated.",
            )
            return
        if not uvars:
            messagebox.showwarning(
                "No Uncertainties",
                "No parameters with uncertainties available.",
            )
            return
        UncertaintyPropagationDialog(self, uvars)

    def _export_model_result(self):
        """Export lmfit ModelResult to a .sav file."""
        sess = self._active_session
        if sess is None or sess.result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        fm = sess.fit_manager
        if fm._last_result is None:
            messagebox.showwarning("No Fit", "Run a fit first.")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".sav",
            filetypes=[("lmfit Model Result", "*.sav"), ("All files", "*.*")],
            title="Export Model Result",
        )
        if not filepath:
            return
        try:
            from lmfit.model import save_modelresult
            save_modelresult(fm._last_result, filepath)
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _import_model_result(self):
        """Import lmfit ModelResult from a .sav file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("lmfit Model Result", "*.sav"), ("All files", "*.*")],
            title="Import Model Result",
        )
        if not filepath:
            return
        try:
            from lmfit.model import load_modelresult
            loaded = load_modelresult(filepath)

            # Display params and report in the results panel
            params_info = {}
            for name, par in loaded.params.items():
                params_info[name] = {
                    "value": par.value,
                    "stderr": par.stderr,
                    "min": par.min,
                    "max": par.max,
                    "vary": par.vary,
                    "expr": par.expr or "",
                }
            self.fit_results.set_params(params_info)
            gof = {
                "chi-squared": getattr(loaded, "chisqr", None),
                "reduced chi-squared": getattr(loaded, "redchi", None),
                "R-squared": getattr(loaded, "rsquared", None),
                "AIC": getattr(loaded, "aic", None),
                "BIC": getattr(loaded, "bic", None),
            }
            self.fit_results.set_gof(gof)
            self.fit_results.set_report(loaded.fit_report())
        except Exception as e:
            messagebox.showerror("Import Error", str(e))

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
        reduce_fcn, weight_mode, max_nfev, band_sigma, scale_covar = self._get_fit_options()

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
                x, y, yerr, xerr = self._get_fit_data(target_rec)
                target_fm.auto_guess(x, y)
                method = self.fit_panel.method_var.get()
                result = target_fm.run_fit(
                    x, y, yerr=yerr, xerr=xerr, method=method,
                    reduce_fcn=reduce_fcn, weight_mode=weight_mode,
                    max_nfev=max_nfev, band_sigma=band_sigma,
                    scale_covar=scale_covar,
                )
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
        for sid, rec in self._series_records.items():
            if not rec.visible:
                continue
            for i in range(len(rec.x)):
                # Transform data coords to display coords for fair distance
                dx_display = ax.transData.transform((rec.x[i], rec.y[i]))
                click_display = ax.transData.transform((event.xdata, event.ydata))
                dist = ((dx_display[0] - click_display[0]) ** 2 +
                        (dx_display[1] - click_display[1]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_sid = sid
                    best_idx = i

        # Only toggle if click is within 10 pixels of a point
        if best_sid is None or best_dist > 10:
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
        """Clear all point exclusions from the active series."""
        rec = self._active_record
        if rec is None:
            return
        rec.mask = None
        self._replot_all_series()
        self.plot_mgr.canvas.draw_idle()
        self.parent.title("CurveLab")

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
        self.plot_mgr.draw()

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
                else:
                    fm.set_param(param_name, expr="", vary=True)
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
        params_info = {}
        for name, par in fm.params.items():
            params_info[name] = {
                "value": par.value,
                "stderr": par.stderr,
                "min": par.min,
                "max": par.max,
                "vary": par.vary,
                "expr": par.expr or "",
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
                        "candidates": r.candidates,
                        "init_params": r.init_params,
                        # flatchain (emcee DataFrame) is not serialized
                    })
                fit_sessions[sess_name] = sess_data

            sdata = {
                "dataset_name": rec.dataset_name,
                "style": rec.style,
                "visible": rec.visible,
                "fit_sessions": fit_sessions,
                "active_session_name": rec.active_session_name,
            }
            if rec.mask is not None:
                sdata["mask"] = encode_value(rec.mask)
            series[sid] = sdata

        return {
            "version": 1,
            "data_filepaths": data_filepaths,
            "table_names": self.data_mgr.table_names,
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
                "fit_xmin": self.plot_controls.fit_xmin_var.get(),
                "fit_xmax": self.plot_controls.fit_xmax_var.get(),
                "residuals": self.plot_controls.residuals_var.get(),
                "confidence_band": self.plot_controls.confidence_band_var.get(),
                "fit_method": self.fit_panel.method_var.get(),
                "reduce_fcn": self.fit_panel.reduce_var.get(),
                "weight_mode": self.fit_panel.weight_var.get(),
                "max_nfev": self.fit_panel.max_nfev_var.get(),
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
        loaded_sqlite_files: set[str] = set()  # avoid loading same .sqlite/.db twice
        saved_table_names = ws.get("table_names", {})
        for name, fpath in ws.get("data_filepaths", {}).items():
            try:
                fpath_str = str(fpath)
                ext = Path(fpath).suffix.lower()
                if ext in (".sqlite", ".db"):
                    if fpath_str in loaded_sqlite_files:
                        # Already loaded — find the matching dataset by table name
                        table = saved_table_names.get(name, "")
                        for ds_name, tbl in self.data_mgr.table_names.items():
                            if tbl == table and str(self.data_mgr.filepaths.get(ds_name)) == fpath_str:
                                dataset_name_map[name] = ds_name
                                break
                        continue
                    loaded_sqlite_files.add(fpath_str)
                    self.data_mgr.load(fpath)
                    # Map all old names for this file to their new dataset names
                    for old_name, old_fpath in ws.get("data_filepaths", {}).items():
                        if str(old_fpath) == fpath_str:
                            table = saved_table_names.get(old_name, "")
                            for ds_name, tbl in self.data_mgr.table_names.items():
                                if tbl == table and str(self.data_mgr.filepaths.get(ds_name)) == fpath_str:
                                    dataset_name_map[old_name] = ds_name
                                    break
                else:
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
        self._simulated_counter = 0
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
                continue
            ds_name = actual_ds

            saved_mask = sdata.get("mask")
            if saved_mask is not None:
                saved_mask = np.asarray(saved_mask, dtype=bool)

            rec = SeriesRecord(
                x=x, y=y, yerr=yerr, xerr=xerr,
                style=style, dataset_name=ds_name,
                visible=sdata.get("visible", True),
                mask=saved_mask,
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
                        candidates=rdata.get("candidates"),
                        init_params=rdata.get("init_params"),
                        # flatchain is not serialized
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
        self.plot_controls.fit_xmin_var.set(pc.get("fit_xmin", ""))
        self.plot_controls.fit_xmax_var.set(pc.get("fit_xmax", ""))
        self.plot_controls.residuals_var.set(pc.get("residuals", False))
        self.plot_controls.confidence_band_var.set(pc.get("confidence_band", False))
        self.fit_panel.method_var.set(pc.get("fit_method", "leastsq"))
        self.fit_panel.reduce_var.set(pc.get("reduce_fcn", "Chi-square (default)"))
        self.fit_panel.weight_var.set(pc.get("weight_mode", "1/yerr (default)"))
        self.fit_panel.max_nfev_var.set(pc.get("max_nfev", ""))
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
        root.geometry("1400x950")
        root.minsize(900, 700)

        app = cls(root)
        app.pack(fill=tk.BOTH, expand=True)

        root.mainloop()
