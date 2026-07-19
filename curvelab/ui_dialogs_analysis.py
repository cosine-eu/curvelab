"""Analysis and statistics dialog classes for CurveLab."""

import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont

from .analysis_tools import compute_diagnostic_stats
from .ui_common import BaseDialog, set_readonly_text


def build_scrollable_text_viewer(dialog, content: str, font=("Courier", 10)) -> tk.Text:
    """Fill a Toplevel with a read-only monospace Text, scrollbars, and a Close button."""
    text = tk.Text(dialog, wrap=tk.NONE, font=font)
    text.insert("1.0", content)
    text.config(state=tk.DISABLED)

    xscroll = ttk.Scrollbar(dialog, orient=tk.HORIZONTAL, command=text.xview)
    yscroll = ttk.Scrollbar(dialog, orient=tk.VERTICAL, command=text.yview)
    text.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)

    text.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=(10, 0))
    yscroll.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=(10, 0))
    xscroll.grid(row=1, column=0, sticky="ew", padx=(10, 0))

    dialog.columnconfigure(0, weight=1)
    dialog.rowconfigure(0, weight=1)

    ttk.Button(dialog, text="Close", command=dialog.destroy).grid(
        row=2, column=0, columnspan=2, pady=10
    )
    return text


class FontDialog(BaseDialog):
    """Dialog for setting UI and plot fonts."""

    SIZES = [7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 24]

    def __init__(self, parent, ui_family="", ui_size=10, plot_family="", plot_size=10, on_apply=None):
        super().__init__(parent, "Font Settings", resizable=(False, False), modal=True)

        self._on_apply = on_apply
        families = sorted(tkfont.families())

        # --- UI Font ---
        ui_frame = ttk.LabelFrame(self, text="UI Font", padding=10)
        ui_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        ttk.Label(ui_frame, text="Family:").grid(row=0, column=0, sticky=tk.W)
        self.ui_family_var = tk.StringVar(value=ui_family)
        ui_fam = ttk.Combobox(ui_frame, textvariable=self.ui_family_var, values=families, width=25)
        ui_fam.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(ui_frame, text="Size:").grid(row=1, column=0, sticky=tk.W)
        self.ui_size_var = tk.IntVar(value=ui_size)
        ttk.Combobox(
            ui_frame, textvariable=self.ui_size_var, values=self.SIZES, width=5
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # --- Plot Font ---
        plot_frame = ttk.LabelFrame(self, text="Plot Font", padding=10)
        plot_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(plot_frame, text="Family:").grid(row=0, column=0, sticky=tk.W)
        self.plot_family_var = tk.StringVar(value=plot_family)
        plot_fam = ttk.Combobox(plot_frame, textvariable=self.plot_family_var, values=families, width=25)
        plot_fam.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(plot_frame, text="Size:").grid(row=1, column=0, sticky=tk.W)
        self.plot_size_var = tk.IntVar(value=plot_size)
        ttk.Combobox(
            plot_frame, textvariable=self.plot_size_var, values=self.SIZES, width=5
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # --- Buttons ---
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Apply", command=self._apply).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)

    def _apply(self):
        if self._on_apply:
            self._on_apply(
                ui_family=self.ui_family_var.get(),
                ui_size=self.ui_size_var.get(),
                plot_family=self.plot_family_var.get(),
                plot_size=self.plot_size_var.get(),
            )
        self.destroy()


class ModelComparisonDialog(BaseDialog):
    """Side-by-side comparison of fit sessions: AIC, BIC, reduced chi-squared."""

    def __init__(self, parent, rows: list[dict]):
        """rows: list of dicts with keys session, model, n_params, chisqr, redchi, aic, bic."""
        super().__init__(parent, "Model Comparison", size="700x300")

        columns = ("model", "n_params", "chisqr", "redchi", "aic", "bic")
        tree = ttk.Treeview(self, columns=columns, show=("tree", "headings"), height=10)
        tree.column("#0", width=120, stretch=False)
        tree.heading("#0", text="Session")
        tree.heading("model", text="Model")
        tree.column("model", width=150)
        tree.heading("n_params", text="N params")
        tree.column("n_params", width=70)
        tree.heading("chisqr", text="\u03c7\u00b2")
        tree.column("chisqr", width=90)
        tree.heading("redchi", text="\u03c7\u00b2/\u03bd")
        tree.column("redchi", width=90)
        tree.heading("aic", text="AIC")
        tree.column("aic", width=90)
        tree.heading("bic", text="BIC")
        tree.column("bic", width=90)

        tree.tag_configure("best_aic", background="#d4edda")
        tree.tag_configure("even", background="#f0f0f0")
        tree.tag_configure("odd", background="#ffffff")

        # Find best AIC for highlighting
        aic_vals = [r["aic"] for r in rows if r["aic"] is not None]
        best_aic = min(aic_vals) if aic_vals else None

        for i, r in enumerate(rows):
            tags = []
            if best_aic is not None and r["aic"] == best_aic:
                tags.append("best_aic")
            else:
                tags.append("even" if i % 2 == 0 else "odd")
            tree.insert(
                "", tk.END, text=r["session"],
                values=(
                    r["model"],
                    r["n_params"],
                    f"{r['chisqr']:.4g}" if r["chisqr"] is not None else "",
                    f"{r['redchi']:.4g}" if r["redchi"] is not None else "",
                    f"{r['aic']:.4g}" if r["aic"] is not None else "",
                    f"{r['bic']:.4g}" if r["bic"] is not None else "",
                ),
                tags=tuple(tags),
            )

        scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 10))


class FTestDialog(BaseDialog):
    """F-test for nested model comparison between two fit sessions."""

    def __init__(self, parent, sessions: dict[str, dict]):
        """sessions: {name: {n_params, chisqr, n_data}}."""
        super().__init__(parent, "F-Test for Nested Models", resizable=(False, False))
        self._sessions = sessions
        names = list(sessions.keys())

        # Session selectors
        sel_frame = ttk.LabelFrame(self, text="Select two sessions (simpler vs more complex)", padding=5)
        sel_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(sel_frame, text="Reduced model (fewer params):").grid(row=0, column=0, sticky=tk.W)
        self._reduced_var = tk.StringVar(value=names[0] if names else "")
        ttk.Combobox(sel_frame, textvariable=self._reduced_var,
                     values=names, state="readonly", width=25).grid(row=0, column=1, padx=5)

        ttk.Label(sel_frame, text="Full model (more params):").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self._full_var = tk.StringVar(value=names[1] if len(names) > 1 else "")
        ttk.Combobox(sel_frame, textvariable=self._full_var,
                     values=names, state="readonly", width=25).grid(row=1, column=1, padx=5, pady=(5, 0))

        ttk.Button(self, text="Compute", command=self._compute).pack(pady=5)

        self._result_text = tk.Text(self, height=10, width=60, wrap=tk.WORD,
                                     font=("Courier", 10), state=tk.DISABLED)
        self._result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 10))

    def _compute(self):
        from .analysis_tools import f_test

        r_name = self._reduced_var.get()
        f_name = self._full_var.get()
        if r_name == f_name:
            self._show_result("Select two different sessions.")
            return
        if r_name not in self._sessions or f_name not in self._sessions:
            self._show_result("Select valid sessions.")
            return

        r = self._sessions[r_name]
        f = self._sessions[f_name]

        p1, p2 = r["n_params"], f["n_params"]
        chi1, chi2 = r["chisqr"], f["chisqr"]
        n = r["n_data"]

        # Ensure reduced model has fewer params
        if p1 >= p2:
            self._show_result(
                f"Reduced model ({r_name}) has {p1} params, "
                f"full model ({f_name}) has {p2} params.\n\n"
                f"The reduced model must have fewer parameters than the full model."
            )
            return

        if chi2 >= chi1:
            self._show_result(
                f"Full model has equal or worse \u03c7\u00b2 ({chi2:.4g}) "
                f"than reduced model ({chi1:.4g}).\n\n"
                f"The extra parameters do not improve the fit."
            )
            return

        try:
            f_stat, p_value, df1, df2 = f_test(chi1, p1, chi2, p2, n)
        except ValueError as e:
            self._show_result(str(e))
            return

        lines = [
            f"Reduced model: {r_name}",
            f"  Parameters: {p1},  \u03c7\u00b2 = {chi1:.6g}",
            f"",
            f"Full model: {f_name}",
            f"  Parameters: {p2},  \u03c7\u00b2 = {chi2:.6g}",
            f"",
            f"Extra parameters: {df1}",
            f"Residual DOF:     {df2}",
            f"",
            f"F-statistic: {f_stat:.4f}",
            f"p-value:     {p_value:.6g}",
            f"",
        ]
        if p_value < 0.01:
            lines.append("The extra parameters significantly improve the fit (p < 0.01).")
        elif p_value < 0.05:
            lines.append("The extra parameters marginally improve the fit (p < 0.05).")
        else:
            lines.append("The extra parameters do NOT significantly improve the fit.")
            lines.append("The simpler model is preferred.")

        self._show_result("\n".join(lines))

    def _show_result(self, text: str):
        set_readonly_text(self._result_text, text)


class CovarianceMatrixDialog(BaseDialog):
    """Display the full parameter covariance matrix."""

    def __init__(self, parent, param_names: list[str], cov_matrix):
        super().__init__(parent, "Covariance Matrix", size="700x400")

        import numpy as np

        n = len(param_names)
        # Format matrix as aligned text
        max_name = max(len(n) for n in param_names)
        col_width = 14

        lines = []
        # Header row
        header = " " * (max_name + 2) + "".join(f"{n:>{col_width}}" for n in param_names)
        lines.append(header)
        lines.append("-" * len(header))
        # Data rows
        for i, name in enumerate(param_names):
            row = f"{name:>{max_name}}  "
            row += "".join(f"{cov_matrix[i, j]:>{col_width}.6g}" for j in range(n))
            lines.append(row)

        build_scrollable_text_viewer(self, "\n".join(lines))


class ConfidenceIntervalDialog(BaseDialog):
    """Display confidence interval report in monospace text."""

    def __init__(self, parent, ci_text: str):
        super().__init__(parent, "Confidence Intervals", size="600x400")

        build_scrollable_text_viewer(self, ci_text)


class CorrelationMatrixDialog(BaseDialog):
    """Display parameter correlation matrix in a Treeview."""

    def __init__(self, parent, correlations: dict[str, dict[str, float]]):
        super().__init__(parent, "Correlation Matrix", size="700x400")

        # Collect all parameter names
        all_params = list(correlations.keys())
        if not all_params:
            ttk.Label(self, text="No correlations available.").pack(padx=20, pady=20)
            ttk.Button(self, text="Close", command=self.destroy).pack(pady=10)
            return

        tree = ttk.Treeview(self, columns=all_params, show=("tree", "headings"))
        tree.column("#0", width=120, stretch=False)
        tree.heading("#0", text="Parameter")

        for p in all_params:
            tree.heading(p, text=p)
            tree.column(p, width=80)

        tree.tag_configure("even", background="#f0f0f0")
        tree.tag_configure("odd", background="#ffffff")

        for i, name in enumerate(all_params):
            vals = []
            for other in all_params:
                if name == other:
                    vals.append("1.000")
                elif other in correlations.get(name, {}):
                    vals.append(f"{correlations[name][other]:.3f}")
                else:
                    vals.append("")
            tag = "even" if i % 2 == 0 else "odd"
            tree.insert("", tk.END, text=name, values=tuple(vals), tags=(tag,))

        scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 10))


class BruteCandidatesDialog(BaseDialog):
    """Display brute-force candidates with option to load one."""

    def __init__(self, parent, candidates: list[dict], on_select=None):
        super().__init__(parent, "Brute-Force Candidates", size="700x400")
        self._on_select = on_select
        self._candidates = candidates

        if not candidates:
            ttk.Label(self, text="No candidates available.").pack(padx=20, pady=20)
            ttk.Button(self, text="Close", command=self.destroy).pack(pady=10)
            return

        # Get parameter names from first candidate
        param_names = list(candidates[0]["params"].keys())
        columns = ("score",) + tuple(param_names)

        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        self._tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=15
        )
        self._tree.heading("score", text="Score")
        self._tree.column("score", width=100)
        for p in param_names:
            self._tree.heading(p, text=p)
            self._tree.column(p, width=90)

        self._tree.tag_configure("even", background="#f0f0f0")
        self._tree.tag_configure("odd", background="#ffffff")
        self._tree.tag_configure("best", background="#d4edda")

        for i, cand in enumerate(candidates):
            vals = [f"{cand['score']:.6g}"]
            for p in param_names:
                vals.append(f"{cand['params'][p]:.6g}")
            tag = "best" if i == 0 else ("even" if i % 2 == 0 else "odd")
            self._tree.insert("", tk.END, values=tuple(vals), tags=(tag,))

        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Load Selected", command=self._load_selected).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(btn_frame, text="Close", command=self.destroy).pack(side=tk.LEFT)

    def _load_selected(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = self._tree.index(sel[0])
        if self._on_select and idx < len(self._candidates):
            self._on_select(self._candidates[idx]["params"])
            self.destroy()


class EmceeSummaryDialog(BaseDialog):
    """Display emcee MCMC summary statistics."""

    def __init__(self, parent, flatchain, params_info: dict):
        super().__init__(parent, "MCMC (emcee) Summary", size="600x400")

        try:
            import pandas as pd
            if isinstance(flatchain, pd.DataFrame):
                lines = [f"{'Parameter':<20s} {'Median':>12s} {'Mean':>12s} "
                         f"{'Std':>12s} {'2.5%':>12s} {'97.5%':>12s}"]
                lines.append("-" * 80)
                for col in flatchain.columns:
                    data = flatchain[col]
                    lines.append(
                        f"{col:<20s} {data.median():>12.6g} {data.mean():>12.6g} "
                        f"{data.std():>12.6g} {data.quantile(0.025):>12.6g} "
                        f"{data.quantile(0.975):>12.6g}"
                    )
                content = "\n".join(lines)
            else:
                content = "Flatchain data not available as DataFrame."
        except Exception as e:
            content = f"Error processing emcee results: {e}"

        build_scrollable_text_viewer(self, content)


class DiagnosticPlotsDialog(BaseDialog):
    """2x2 diagnostic plot grid: residuals vs fitted, Q-Q, scale-location, ACF."""

    def __init__(self, parent, fit_result):
        super().__init__(parent, "Fit Diagnostic Plots", size="800x600")

        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        import scipy.stats as stats

        r = fit_result
        residuals = r.y_data - r.y_fit_data
        fitted = r.y_fit_data

        # Standardize residuals
        std_resid = residuals - residuals.mean()
        s = residuals.std()
        if s > 0:
            std_resid = std_resid / s

        fig = Figure(figsize=(8, 6))

        # 1. Residuals vs Fitted
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.scatter(fitted, residuals, s=12, alpha=0.7)
        ax1.axhline(0, color="red", linestyle="--", linewidth=0.8)
        ax1.set_xlabel("Fitted values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs Fitted")

        # 2. Normal Q-Q
        ax2 = fig.add_subplot(2, 2, 2)
        stats.probplot(std_resid, plot=ax2)
        ax2.set_title("Normal Q-Q")

        # 3. Scale-Location
        ax3 = fig.add_subplot(2, 2, 3)
        sqrt_abs_resid = np.sqrt(np.abs(std_resid))
        ax3.scatter(fitted, sqrt_abs_resid, s=12, alpha=0.7)
        ax3.set_xlabel("Fitted values")
        ax3.set_ylabel("\u221a|Standardized residuals|")
        ax3.set_title("Scale-Location")

        # 4. Autocorrelation
        ax4 = fig.add_subplot(2, 2, 4)
        n = len(residuals)
        max_lag = min(20, n - 1)
        mean_r = residuals.mean()
        var_r = np.sum((residuals - mean_r) ** 2)
        acf = []
        for lag in range(max_lag + 1):
            c = np.sum((residuals[:n - lag] - mean_r) * (residuals[lag:] - mean_r))
            acf.append(c / var_r if var_r > 0 else 0.0)
        lags = np.arange(max_lag + 1)
        ax4.bar(lags, acf, width=0.4, color="steelblue")
        ci = 1.96 / np.sqrt(n)
        ax4.axhline(ci, color="red", linestyle="--", linewidth=0.8)
        ax4.axhline(-ci, color="red", linestyle="--", linewidth=0.8)
        ax4.axhline(0, color="black", linewidth=0.5)
        ax4.set_xlabel("Lag")
        ax4.set_ylabel("ACF")
        ax4.set_title("Autocorrelation")

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Residual statistics summary ---
        stats_lines = compute_diagnostic_stats(residuals)
        if stats_lines:
            stats_frame = ttk.LabelFrame(self, text="Residual Statistics", padding=5)
            stats_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
            for line in stats_lines:
                ttk.Label(stats_frame, text=line, wraplength=750,
                          justify=tk.LEFT).pack(anchor=tk.W)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=5)


class ConfidenceContourDialog(BaseDialog):
    """Interactive 2D confidence contour plot using lmfit.conf_interval2d."""

    def __init__(self, parent, last_result, vary_params: list[str]):
        super().__init__(parent, "2D Confidence Contours", size="700x600")

        self._last_result = last_result
        self._vary_params = vary_params

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        # Controls frame
        ctrl = ttk.Frame(self, padding=5)
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="Param X:").pack(side=tk.LEFT)
        self._x_var = tk.StringVar(value=vary_params[0])
        ttk.Combobox(
            ctrl, textvariable=self._x_var, values=vary_params,
            state="readonly", width=15,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="Param Y:").pack(side=tk.LEFT, padx=(10, 0))
        self._y_var = tk.StringVar(value=vary_params[1] if len(vary_params) > 1 else vary_params[0])
        ttk.Combobox(
            ctrl, textvariable=self._y_var, values=vary_params,
            state="readonly", width=15,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="Grid:").pack(side=tk.LEFT, padx=(10, 0))
        self._grid_var = tk.IntVar(value=10)
        ttk.Spinbox(
            ctrl, textvariable=self._grid_var, from_=5, to=50, width=4,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(ctrl, text="Compute", command=self._compute).pack(side=tk.LEFT, padx=(10, 0))

        # Status
        self._status_var = tk.StringVar(value="Select parameters and click Compute.")
        ttk.Label(self, textvariable=self._status_var).pack(fill=tk.X, padx=10)

        # Plot area
        self._fig = Figure(figsize=(6, 5))
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=5)

    def _compute(self):
        x_name = self._x_var.get()
        y_name = self._y_var.get()
        if x_name == y_name:
            self._status_var.set("Select two different parameters.")
            return
        nx = ny = self._grid_var.get()
        self._status_var.set("Computing...")
        self.update_idletasks()

        try:
            from lmfit import conf_interval2d
            x_arr, y_arr, grid = conf_interval2d(
                self._last_result, self._last_result,
                x_name, y_name, nx=nx, ny=ny,
            )
            self._ax.clear()
            self._ax.contourf(x_arr, y_arr, grid, cmap="coolwarm")
            self._ax.set_xlabel(x_name)
            self._ax.set_ylabel(y_name)
            self._ax.set_title(f"Confidence: {x_name} vs {y_name}")
            self._fig.tight_layout()
            self._canvas.draw()
            self._status_var.set("Done.")
        except Exception as e:
            self._status_var.set(f"Error: {e}")


class BootstrapDialog(BaseDialog):
    """Bootstrap confidence intervals with parameter histograms."""

    def __init__(self, parent, on_run=None):
        super().__init__(parent, "Bootstrap Confidence Intervals", size="900x650")
        self._on_run = on_run

        # Controls
        ctrl = ttk.Frame(self, padding=5)
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="N bootstrap:").pack(side=tk.LEFT)
        self._n_var = tk.StringVar(value="200")
        ttk.Entry(ctrl, textvariable=self._n_var, width=8).pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, text="Method:").pack(side=tk.LEFT, padx=(10, 0))
        self._type_var = tk.StringVar(value="residual")
        ttk.Combobox(ctrl, textvariable=self._type_var,
                     values=["residual", "case"], state="readonly",
                     width=10).pack(side=tk.LEFT, padx=5)

        self._run_btn = ttk.Button(ctrl, text="Run", command=self._run)
        self._run_btn.pack(side=tk.LEFT, padx=10)
        self._status_var = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self._status_var).pack(side=tk.LEFT, fill=tk.X)

        # Results area
        self._results_frame = ttk.Frame(self)
        self._results_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=5)

    def _run(self):
        if self._on_run is None:
            return
        try:
            n_boot = int(self._n_var.get())
        except ValueError:
            self._status_var.set("Invalid N.")
            return
        boot_type = self._type_var.get()
        self._run_btn.config(state=tk.DISABLED)
        self._status_var.set("Running...")
        self.update_idletasks()
        try:
            distributions, n_failed = self._on_run(n_boot, boot_type)
            self._show_results(distributions)
            if n_failed:
                self._status_var.set(
                    f"Done ({n_boot - n_failed}/{n_boot} resamples converged, "
                    f"{n_failed} failed)."
                )
            else:
                self._status_var.set(f"Done ({n_boot} resamples).")
        except Exception as e:
            self._status_var.set(f"Error: {e}")
        finally:
            self._run_btn.config(state=tk.NORMAL)

    def _show_results(self, distributions: dict):
        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        # Clear previous results
        for w in self._results_frame.winfo_children():
            w.destroy()

        n = len(distributions)
        if n == 0:
            ttk.Label(self._results_frame, text="No results.").pack()
            return

        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig = Figure(figsize=(4 * ncols, 3 * nrows))

        summary_lines = []
        for i, (pname, values) in enumerate(distributions.items()):
            if len(values) < 2:
                continue
            ax = fig.add_subplot(nrows, ncols, i + 1)
            ax.hist(values, bins=min(30, len(values) // 5 + 1),
                    color="steelblue", alpha=0.7, edgecolor="white")
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            ci_lo, ci_hi = np.percentile(values, [2.5, 97.5])
            ax.axvline(mean, color="red", linewidth=1.5, label=f"mean={mean:.4g}")
            ax.axvline(ci_lo, color="orange", linestyle="--", linewidth=1)
            ax.axvline(ci_hi, color="orange", linestyle="--", linewidth=1)
            ax.set_xlabel(pname)
            ax.set_title(f"{pname}\n{mean:.4g} \u00b1 {std:.4g}", fontsize=9)
            summary_lines.append(
                f"{pname}: {mean:.6g} \u00b1 {std:.4g}  "
                f"[95% CI: {ci_lo:.4g}, {ci_hi:.4g}]"
            )

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self._results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Summary text
        if summary_lines:
            summary_frame = ttk.LabelFrame(self._results_frame,
                                            text="Summary", padding=5)
            summary_frame.pack(fill=tk.X, padx=5, pady=5)
            text = tk.Text(summary_frame, height=min(len(summary_lines) + 1, 8),
                          font=("Courier", 9), wrap=tk.NONE)
            text.insert("1.0", "\n".join(summary_lines))
            text.config(state=tk.DISABLED)
            text.pack(fill=tk.X)


class ProfileLikelihoodDialog(BaseDialog):
    """Plot chi-squared profiles for each parameter."""

    def __init__(self, parent, profiles: dict[str, list[tuple[float, float]]],
                 best_chi2: float):
        super().__init__(parent, "Profile Likelihood", size="900x600")

        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        n = len(profiles)
        if n == 0:
            ttk.Label(self, text="No profile data available.").pack(pady=20)
            ttk.Button(self, text="Close", command=self.destroy).pack(pady=5)
            return

        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig = Figure(figsize=(4 * ncols, 3 * nrows))

        for i, (pname, points) in enumerate(profiles.items()):
            if not points:
                continue
            ax = fig.add_subplot(nrows, ncols, i + 1)
            pvals = [p[0] for p in points]
            chi2s = [p[1] for p in points]
            ax.plot(pvals, chi2s, "o-", markersize=3, color="steelblue")
            ax.axhline(best_chi2, color="red", linestyle="--", linewidth=0.8,
                       label=f"\u03c7\u00b2_min = {best_chi2:.4g}")
            # 1-sigma threshold
            ax.axhline(best_chi2 + 1, color="orange", linestyle=":",
                       linewidth=0.8, label="\u03c7\u00b2_min + 1")
            ax.set_xlabel(pname)
            ax.set_ylabel("\u03c7\u00b2")
            ax.set_title(pname)
            ax.legend(fontsize=7)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        ttk.Button(self, text="Close", command=self.destroy).pack(pady=5)


class GlobalFitDialog(BaseDialog):
    """Dialog for global fitting across multiple series with shared parameters."""

    def __init__(self, parent, series_info: list[dict], base_param_names: list[str],
                 on_fit=None):
        """
        series_info: list of dicts with keys 'id' and 'label'.
        base_param_names: list of parameter names from the model.
        on_fit: callback(selected_series_ids, shared_param_names).
        """
        super().__init__(parent, "Global Fit", size="500x500")
        self._on_fit = on_fit

        # --- Series selection ---
        series_frame = ttk.LabelFrame(self, text="Series", padding=5)
        series_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        self._series_vars = {}
        for info in series_info:
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(series_frame, text=info["label"], variable=var).pack(
                anchor=tk.W
            )
            self._series_vars[info["id"]] = var

        # --- Parameter sharing ---
        param_frame = ttk.LabelFrame(self, text="Shared Parameters", padding=5)
        param_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        canvas = tk.Canvas(param_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(param_frame, orient=tk.VERTICAL, command=canvas.yview)
        inner_frame = ttk.Frame(canvas)

        inner_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._param_vars = {}
        for name in base_param_names:
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(inner_frame, text=name, variable=var).pack(anchor=tk.W)
            self._param_vars[name] = var

        # --- Buttons ---
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self._status_var = tk.StringVar(value="")
        ttk.Label(btn_frame, textvariable=self._status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(btn_frame, text="Fit", command=self._do_fit).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)

    def _do_fit(self):
        selected_ids = [sid for sid, var in self._series_vars.items() if var.get()]
        shared = {name for name, var in self._param_vars.items() if var.get()}
        if len(selected_ids) < 2:
            self._status_var.set("Select at least 2 series.")
            return
        self._status_var.set("Fitting...")
        self.update_idletasks()
        if self._on_fit:
            try:
                self._on_fit(selected_ids, shared)
                self.destroy()
            except Exception as e:
                self._status_var.set(f"Error: {e}")


class UncertaintyPropagationDialog(BaseDialog):
    """Evaluate expressions with propagated uncertainties using ufloats."""

    def __init__(self, parent, uvars: dict):
        """uvars: dict mapping param name to ufloat (from lmfit result.uvars)."""
        super().__init__(parent, "Uncertainty Propagation", size="550x400")
        self._uvars = uvars

        # --- Available variables ---
        var_frame = ttk.LabelFrame(self, text="Available Variables", padding=5)
        var_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        var_list = tk.Listbox(var_frame, height=8)
        var_scroll = ttk.Scrollbar(var_frame, orient=tk.VERTICAL, command=var_list.yview)
        var_list.configure(yscrollcommand=var_scroll.set)
        var_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        var_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        for name, uval in uvars.items():
            try:
                var_list.insert(tk.END, f"{name} = {uval.nominal_value:.6g} \u00b1 {uval.std_dev:.6g}")
            except AttributeError:
                var_list.insert(tk.END, f"{name} = {uval}")

        # --- Expression entry ---
        expr_frame = ttk.Frame(self, padding=5)
        expr_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(expr_frame, text="Expression:").pack(side=tk.LEFT)
        self._expr_var = tk.StringVar()
        expr_entry = ttk.Entry(expr_frame, textvariable=self._expr_var, width=40)
        expr_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        expr_entry.bind("<Return>", lambda e: self._evaluate())

        ttk.Button(expr_frame, text="Evaluate", command=self._evaluate).pack(side=tk.LEFT)

        # --- Result ---
        self._result_var = tk.StringVar(value="Enter an expression using parameter names above.")
        ttk.Label(self, textvariable=self._result_var, wraplength=500, justify=tk.LEFT).pack(
            fill=tk.X, padx=10, pady=5
        )

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 10))

    def _evaluate(self):
        expr = self._expr_var.get().strip()
        if not expr:
            self._result_var.set("Enter an expression.")
            return
        try:
            from uncertainties import umath
            from asteval import Interpreter
            aeval = Interpreter()
            # Add uvars and umath functions to interpreter
            for k, v in self._uvars.items():
                aeval.symtable[k] = v
            for fname in dir(umath):
                if not fname.startswith("_"):
                    aeval.symtable[fname] = getattr(umath, fname)
            result = aeval(expr)
            if aeval.error:
                raise ValueError(aeval.error[0].get_error()[1])
            try:
                self._result_var.set(
                    f"{expr} = {result.nominal_value:.6g} \u00b1 {result.std_dev:.6g}"
                )
            except AttributeError:
                self._result_var.set(f"{expr} = {result}")
        except Exception as e:
            self._result_var.set(f"Error: {e}")


