"""Fit-execution handlers for CurveLabApp.

Mixed into CurveLabApp; these methods drive fitting (sync, ODR, and background
threaded runs), auto-guess, global fit, and batch fit. They rely on the
coordinator's session accessors, panels, plot manager, and the fit-thread
state (_fit_thread, _fit_abort, _FIT_POLL_INTERVAL_MS) held on CurveLabApp.
"""

import threading
from tkinter import messagebox

from .fit_manager import (
    DEFAULT_F_SCALE, FitManager, SLOW_METHODS,
    objective_kwargs, validate_fit_setup,
)
from .ui_dialogs_analysis import (
    GlobalFitDialog, ModelComparisonDialog, comparison_row,
)


class FitHandlersMixin:
    """Fit execution: single/ODR/async runs, auto-guess, global and batch fit."""

    def _on_auto_guess(self):
        sess = self._require_session()
        if sess is None:
            return
        rec = self._active_record
        fm = sess.fit_manager
        if not self._require_model(fm):
            return

        try:
            x, y, _, _ = self._get_fit_data(rec)
            params = fm.auto_guess(x, y)
            self.fit_results.set_params(FitManager.params_to_info(params))
        except Exception as e:
            messagebox.showerror("Guess Error", str(e))

    def _on_fit(self):
        sess = self._require_session()
        if sess is None:
            return
        rec = self._active_record
        if not rec.visible:
            messagebox.showwarning(
                "Hidden Series",
                "The active series is hidden. The fit curve will be shown "
                "but the underlying data points are not visible on the plot.",
            )
        fm = sess.fit_manager
        if not self._require_model(fm):
            return

        method = self.fit_panel.method_var.get()

        # Reject configurations the method can't run (missing backend package,
        # or a bounds-requiring method with an unbounded parameter); warn about
        # ones that will run but not as configured.
        errors, warnings = validate_fit_setup(
            method, fm.params,
            has_xerr=rec.xerr is not None,
            weight_mode=self.fit_panel.weight_var.get(),
        )
        if errors:
            messagebox.showwarning("Cannot Fit", "\n".join(errors))
            return
        if warnings:
            messagebox.showinfo("Fit Warning", "\n".join(warnings))

        # Capture the series id now, since an async fit can outlive the
        # user's current selection (e.g. they switch to another series
        # while a slow method like emcee is still running).
        sid = self._active_series_id

        if method == "odr":
            self._run_fit_odr(rec, sess, sid)
        elif method in SLOW_METHODS:
            self._run_fit_async(rec, sess, method, sid)
        else:
            self._run_fit_sync(rec, sess, method, sid)

    def _get_fit_options(self):
        """Read objective, weight mode, max_nfev, band_sigma, scale_covar from UI."""
        method = self.fit_panel.method_var.get()
        try:
            f_scale = float(self.fit_panel.f_scale_var.get())
        except ValueError:
            f_scale = DEFAULT_F_SCALE
        objective_kws = objective_kwargs(
            method, self.fit_panel.objective_var.get(), f_scale)
        weight_mode = self.fit_panel.weight_var.get()
        max_nfev_str = self.fit_panel.max_nfev_var.get().strip()
        max_nfev = int(max_nfev_str) if max_nfev_str else None
        band_sigma = int(self.plot_controls.band_sigma_var.get())
        scale_covar = self.fit_panel.scale_covar_var.get()
        return objective_kws, weight_mode, max_nfev, band_sigma, scale_covar

    def _run_fit_sync(self, rec, sess, method, sid):
        """Run fit synchronously (fast methods)."""
        fm = sess.fit_manager
        try:
            x, y, yerr, xerr = self._get_fit_data(rec)
            objective_kws, weight_mode, max_nfev, band_sigma, scale_covar = self._get_fit_options()
            result = fm.run_fit(
                x, y, yerr=yerr, xerr=xerr, method=method,
                objective_kws=objective_kws, weight_mode=weight_mode,
                max_nfev=max_nfev, band_sigma=band_sigma,
                scale_covar=scale_covar,
            )
            sess.result = result
            self._post_fit_update(sess, rec, sid)
        except Exception as e:
            messagebox.showerror("Fit Error", str(e))

    def _run_fit_odr(self, rec, sess, sid):
        """Run ODR fit using odrpack."""
        fm = sess.fit_manager
        try:
            x, y, yerr, xerr = self._get_fit_data(rec)
            _, _, _, band_sigma, _ = self._get_fit_options()
            result = fm.run_odr(
                x, y, yerr=yerr, xerr=xerr, band_sigma=band_sigma,
            )
            sess.result = result
            self._post_fit_update(sess, rec, sid)
        except ImportError as e:
            messagebox.showerror("Missing Package", str(e))
        except Exception as e:
            messagebox.showerror("ODR Error", str(e))

    def _set_fit_running(self, running: bool):
        """Toggle the Fit/Abort button and lock out every control that
        mutates the model or parameters while a background fit thread is
        reading/writing them (add/remove component, auto guess, batch fit,
        clear, session changes, and direct parameter-table edits)."""
        self.fit_panel.set_fitting_state(running)
        self.fit_results.set_locked(running)

    def _run_fit_async(self, rec, sess, method, sid):
        """Run fit in a background thread (slow methods)."""
        if self._fit_thread is not None and self._fit_thread.is_alive():
            messagebox.showwarning("Busy", "A fit is already running.")
            return

        self._fit_abort.clear()
        self._set_fit_running(True)

        fm = sess.fit_manager
        try:
            x, y, yerr, xerr = self._get_fit_data(rec)
        except Exception as e:
            messagebox.showerror("Fit Error", str(e))
            self._set_fit_running(False)
            return

        # Build iter_cb that checks abort flag
        def iter_cb(params, iter, resid, *args, **kw):
            if self._fit_abort.is_set():
                return True

        # Build fit_kws for emcee
        fit_kws = {}
        if method == "emcee":
            fit_kws["is_weighted"] = yerr is not None

        objective_kws, weight_mode, max_nfev, band_sigma, scale_covar = self._get_fit_options()

        # Container for result/error from the thread
        container = {"result": None, "error": None}

        def _run():
            try:
                result = fm.run_fit(
                    x, y, yerr=yerr, xerr=xerr, method=method,
                    iter_cb=iter_cb, fit_kws=fit_kws,
                    objective_kws=objective_kws, weight_mode=weight_mode,
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
                self.after(self._FIT_POLL_INTERVAL_MS, _poll)
                return
            self._fit_thread = None
            self._set_fit_running(False)

            if container["error"] is not None:
                if not self._fit_abort.is_set():
                    messagebox.showerror("Fit Error", str(container["error"]))
                return
            if self._fit_abort.is_set():
                return

            sess.result = container["result"]
            self._post_fit_update(sess, rec, sid)

            # Auto-show special result dialogs
            if sess.result.candidates:
                self._show_candidates_dialog(sess)
            if sess.result.flatchain is not None:
                self._show_emcee_summary_dialog(sess)

        self.after(self._FIT_POLL_INTERVAL_MS, _poll)

    def _abort_fit(self):
        """Signal the background fit to stop."""
        self._fit_abort.set()

    def _post_fit_update(self, sess, rec, sid):
        """Update plot, params, and report after a fit completes.

        sid is the series id the fit was launched against, captured at
        launch time -- not necessarily self._active_series_id, since an
        async fit (emcee, brute, ...) can outlive the user's selection.
        """
        result = sess.result
        skey = self._show_fit_on_plot(sid, sess, rec)

        self._display_result(result)

        if self.plot_controls.show_params_var.get():
            self.plot_mgr.annotate_params(
                result.params, gof=result.gof, session_key=skey
            )

    def _on_global_fit(self):
        """Open Global Fit dialog for simultaneous fitting across series."""
        sess = self._require_session()
        if sess is None:
            return
        fm = sess.fit_manager
        if not self._require_model(fm):
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
            data_warnings: list[str] = []
            for sid in selected_ids:
                r = self._series_records[sid]
                x, y, yerr, xerr = self._get_fit_data(r, warnings_out=data_warnings)
                datasets.append((x, y, yerr, xerr))
                selected_recs.append((sid, r))
            self._report_collected_warnings("Data Warnings", data_warnings)

            _, weight_mode, max_nfev, _, _ = self._get_fit_options()
            method = self.fit_panel.method_var.get()
            results = fm.run_global_fit(
                datasets, shared, method=method,
                max_nfev=max_nfev, weight_mode=weight_mode,
            )

            session_name = sess.name

            for (sid, r), result in zip(selected_recs, results):
                target_sess = r.ensure_session(session_name)
                target_sess.result = result
                self._show_fit_on_plot(sid, target_sess, r)

            self._refresh_session_ui()

        GlobalFitDialog(self, series_info, base_param_names, on_fit=on_fit)

    def _on_batch_fit(self):
        """Apply the active session's model to all plotted series."""
        sess = self._require_session()
        if sess is None:
            return
        source_fm = sess.fit_manager
        if not self._require_model(source_fm):
            return
        if len(self._series_records) < 1:
            messagebox.showwarning("No Series", "Plot at least one series.")
            return

        session_name = sess.name
        summary_rows = []
        data_warnings: list[str] = []
        fit_errors: list[str] = []
        objective_kws, weight_mode, max_nfev, band_sigma, scale_covar = self._get_fit_options()

        for sid, target_rec in self._series_records.items():
            target_sess = target_rec.ensure_session(session_name)
            target_fm = target_sess.fit_manager

            # Clone model components from source
            source_fm.clone_components_to(target_fm)

            try:
                x, y, yerr, xerr = self._get_fit_data(
                    target_rec, warnings_out=data_warnings
                )
                target_fm.auto_guess(x, y)
                method = self.fit_panel.method_var.get()
                result = target_fm.run_fit(
                    x, y, yerr=yerr, xerr=xerr, method=method,
                    objective_kws=objective_kws, weight_mode=weight_mode,
                    max_nfev=max_nfev, band_sigma=band_sigma,
                    scale_covar=scale_covar,
                )
                target_sess.result = result
                self._show_fit_on_plot(sid, target_sess, target_rec)

                series_label = target_rec.style.get("label", sid)
                summary_rows.append(comparison_row(
                    f"{series_label} / {session_name}",
                    target_fm.model_description(), result,
                ))
            except Exception as e:
                series_label = target_rec.style.get("label", sid)
                summary_rows.append(comparison_row(
                    f"{series_label} / {session_name}", "ERROR",
                ))
                fit_errors.append(f"{series_label}: {e}")

        # Sync UI to the currently active session
        self._refresh_session_ui()

        self._report_collected_warnings("Data Warnings", data_warnings)
        self._report_collected_warnings("Batch Fit Warnings", fit_errors)

        if summary_rows:
            ModelComparisonDialog(self, summary_rows)
