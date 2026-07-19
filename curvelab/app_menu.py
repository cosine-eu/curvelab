"""Menu bar and About dialog for CurveLabApp.

Mixed into CurveLabApp; wires the File/Analysis/Settings/Help menus to
handlers defined on the coordinator and its sibling mixins.
"""

import tkinter as tk
from pathlib import Path
from tkinter import ttk


class MenuMixin:
    """Builds the menu bar and the About dialog."""

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
