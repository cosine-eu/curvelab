"""Analysis dialog launchers for CurveLabApp.

Mixed into CurveLabApp; these methods rely on the coordinator's session
accessors (_active_record, _require_fit_result, ...) and panels (fit_results,
fit_panel), so they are not usable standalone.
"""

from tkinter import messagebox, filedialog

from .fit_manager import FitManager
from .ui_dialogs_analysis import (
    ModelComparisonDialog, FTestDialog,
    ConfidenceIntervalDialog, CorrelationMatrixDialog, CovarianceMatrixDialog,
    DiagnosticPlotsDialog, ConfidenceContourDialog, ProfileLikelihoodDialog,
    BootstrapDialog, BruteCandidatesDialog, EmceeSummaryDialog,
    UncertaintyPropagationDialog,
)


class AnalysisHandlersMixin:
    """Menu/button handlers that open statistical-analysis dialogs."""

    def _analysis_title(self, base_title: str) -> str:
        """Build a dialog title that includes the active session name."""
        sess = self._active_session
        if sess is not None:
            return f"{base_title} — {sess.name}"
        return base_title

    def _show_model_comparison(self):
        rec = self._active_record
        if rec is None:
            messagebox.showwarning("No Series", "Select a series first.")
            return
        rows = []
        for sess_name, sess in rec.fit_sessions.items():
            if sess.result is None:
                continue
            model_desc = sess.fit_manager.model_description()
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
        dlg = ModelComparisonDialog(self, rows)
        series_label = rec.style.get("label", self._active_series_id or "")
        dlg.title(f"Model Comparison — {series_label}")

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
        dlg = FTestDialog(self, sessions)
        series_label = rec.style.get("label", self._active_series_id or "")
        dlg.title(f"F-Test for Nested Models — {series_label}")

    def _show_confidence_intervals(self):
        sess = self._require_fit_result()
        if sess is None:
            return
        fm = sess.fit_manager
        try:
            ci_text = fm.compute_confidence_intervals()
            dlg = ConfidenceIntervalDialog(self, ci_text)
            dlg.title(self._analysis_title("Confidence Intervals"))
        except Exception as e:
            messagebox.showerror("CI Error", str(e))

    def _show_correlations(self):
        sess = self._require_fit_result()
        if sess is None:
            return
        fm = sess.fit_manager
        try:
            correlations = fm.get_correlations()
            if not correlations:
                messagebox.showinfo("No Correlations", "No parameter correlations available.")
                return
            dlg = CorrelationMatrixDialog(self, correlations)
            dlg.title(self._analysis_title("Correlation Matrix"))
        except Exception as e:
            messagebox.showerror("Correlation Error", str(e))

    def _show_covariance(self):
        sess = self._require_fit_result()
        if sess is None:
            return
        fm = sess.fit_manager
        result = fm.get_covariance_matrix()
        if result is None:
            messagebox.showinfo("No Covariance", "Covariance matrix not available.")
            return
        param_names, cov_matrix = result
        dlg = CovarianceMatrixDialog(self, param_names, cov_matrix)
        dlg.title(self._analysis_title("Covariance Matrix"))

    def _show_diagnostic_plots(self):
        sess = self._require_fit_result()
        if sess is None:
            return
        dlg = DiagnosticPlotsDialog(self, sess.result)
        dlg.title(self._analysis_title("Fit Diagnostic Plots"))

    def _show_confidence_contours(self):
        fm = self._require_last_result()
        if fm is None:
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
        dlg = ConfidenceContourDialog(self, fm._last_result, vary_params)
        dlg.title(self._analysis_title("2D Confidence Contours"))

    def _show_profile_likelihood(self):
        fm = self._require_last_result()
        if fm is None:
            return
        try:
            profiles = fm.compute_ci_profiles()
            if not profiles:
                messagebox.showinfo("No Profiles",
                                    "Could not compute profile traces.")
                return
            best_chi2 = fm._last_result.chisqr
            dlg = ProfileLikelihoodDialog(self, profiles, best_chi2)
            dlg.title(self._analysis_title("Profile Likelihood"))
        except Exception as e:
            messagebox.showerror("Profile Error", str(e))

    def _show_bootstrap(self):
        sess = self._require_fit_result()
        if sess is None:
            return
        rec = self._active_record
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

        dlg = BootstrapDialog(self, on_run=on_run)
        dlg.title(self._analysis_title("Bootstrap Confidence Intervals"))

    def _show_candidates_dialog(self, sess):
        """Show brute-force candidates dialog with option to load values."""
        if sess.result is None or not sess.result.candidates:
            return

        def on_select(params_dict):
            fm = sess.fit_manager
            for name, val in params_dict.items():
                fm.set_param(name, value=val)
            self._refresh_param_display()

        dlg = BruteCandidatesDialog(self, sess.result.candidates, on_select=on_select)
        dlg.title(self._analysis_title("Brute-Force Candidates"))

    def _show_emcee_summary_dialog(self, sess):
        """Show emcee MCMC summary dialog."""
        if sess.result is None or sess.result.flatchain is None:
            return
        dlg = EmceeSummaryDialog(self, sess.result.flatchain, sess.result.params)
        dlg.title(self._analysis_title("MCMC (emcee) Summary"))

    def _show_uncertainty_propagation(self):
        """Open Uncertainty Propagation dialog."""
        fm = self._require_last_result()
        if fm is None:
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
        dlg = UncertaintyPropagationDialog(self, uvars)
        dlg.title(self._analysis_title("Uncertainty Propagation"))

    def _export_model_result(self):
        """Export lmfit ModelResult to a .sav file."""
        fm = self._require_last_result()
        if fm is None:
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
            self.fit_results.set_params(FitManager.params_to_info(loaded.params))
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
