"""Fit-execution handlers for CurveLabApp.

Mixed into CurveLabApp; these methods drive fitting (sync, ODR, and background
threaded runs), auto-guess, global fit, and batch fit. They rely on the
coordinator's session accessors, panels, plot manager, and the fit-thread
state (_fit_thread, _fit_abort, _FIT_POLL_INTERVAL_MS) held on CurveLabApp.
"""

import threading
from tkinter import messagebox

from .fit_manager import (
    DEFAULT_F_SCALE, FitManager, METHOD_CAPS, OBJ_SCALAR, SLOW_METHODS,
    objective_kwargs, validate_fit_setup,
)
from .ui_dialogs_analysis import (
    GlobalFitDialog, ModelComparisonDialog, comparison_row,
)


def fit_outcome_notes(result, method: str) -> list[str]:
    """Plain-language notes about how a fit ended, for the post-fit dialog.

    Distinguishes the causes that used to share one 'not identifiable'
    message: the minimizer stopped before converging; the method gives no
    covariance without numdifftools; a parameter ended on a bound; and the
    genuine case, a covariance the data cannot support.
    """
    import importlib.util

    notes = []
    if not result.success:
        evals = f" after {result.nfev} function evaluations" if result.nfev is not None else ""
        notes.append(
            f"The fit did not converge{evals}: {result.message or 'no reason given'}.\n"
            "The parameter values are where the minimizer stopped, not a "
            "best fit. Raise Max nfev, improve the starting values (Auto "
            "Guess), or try another method."
        )
    if getattr(result, "n_omitted", 0):
        notes.append(
            f"{result.n_omitted} data point(s) were left out because the model "
            "is not finite there (a log or square root of a negative value, a "
            "division by zero). Chi-square, AIC and BIC refer to the remaining "
            "points; check the model or the fit range."
        )
    at_bound = [f"{name} (at {info['at_bound']})" for name, info in result.params.items()
                if info.get("at_bound")]
    if at_bound:
        notes.append(
            "Parameter(s) at a bound: " + ", ".join(at_bound) + ".\n"
            "A parameter on a bound is not at an interior minimum, so its "
            "StdErr is not its uncertainty. Loosen the bound if it was a "
            "guess, or fix the parameter if the bound is physical."
        )
    if result.success and not result.errorbars:
        caps = METHOD_CAPS.get(method)
        scalar = caps is not None and caps.objective == OBJ_SCALAR
        if scalar and importlib.util.find_spec("numdifftools") is None:
            notes.append(
                f"No uncertainties: the '{method}' method does not produce a "
                "covariance matrix, and estimating one afterwards needs the "
                "'numdifftools' package, which is not installed. Install it "
                "(pip install numdifftools) or use least_squares/leastsq."
            )
        else:
            notes.append(
                "No uncertainties: the covariance matrix could not be "
                "estimated (the StdErr column is empty). This usually means the "
                "parameters are not all identifiable from the data, for example "
                "an over-parameterized model or a parameter the data cannot "
                "constrain. The curve may look good while individual values are "
                "unreliable. Consider fixing or removing a parameter, or using a "
                "simpler model."
            )
    return notes


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
        if self._refuse_while_fitting():
            return
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

        # Validate the option entries before any fit path touches UI state:
        # _run_fit_async locks the panel first, and an exception after that
        # would leave the Fit button stuck on "Abort".
        if self._fit_options_or_warn() is None:
            return

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
        """Read objective, weight mode, max_nfev, band_sigma, scale_covar from UI.

        Raises ValueError with a user-readable message for a Max nfev that
        is not a positive whole number. Fit entry points must call
        _fit_options_or_warn before changing any UI state, so a bad entry
        is reported instead of escaping from a Tk callback halfway through.
        """
        method = self.fit_panel.method_var.get()
        try:
            f_scale = float(self.fit_panel.f_scale_var.get())
        except ValueError:
            f_scale = DEFAULT_F_SCALE
        objective_kws = objective_kwargs(
            method, self.fit_panel.objective_var.get(), f_scale)
        weight_mode = self.fit_panel.weight_var.get()
        max_nfev_str = self.fit_panel.max_nfev_var.get().strip()
        max_nfev = None
        if max_nfev_str:
            try:
                max_nfev = int(max_nfev_str)
            except ValueError:
                max_nfev = 0
            if max_nfev <= 0:
                raise ValueError(
                    f"Max nfev must be a positive whole number or empty, "
                    f"got '{max_nfev_str}'."
                )
        band_sigma = int(self.plot_controls.band_sigma_var.get())
        scale_covar = self.fit_panel.scale_covar_var.get()
        return objective_kws, weight_mode, max_nfev, band_sigma, scale_covar

    def _fit_options_or_warn(self):
        """_get_fit_options, reporting an invalid entry to the user and
        returning None instead of raising."""
        try:
            return self._get_fit_options()
        except ValueError as e:
            messagebox.showwarning("Invalid Fit Option", str(e))
            return None

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
        """Toggle the Fit/Abort button and lock out the panel controls that
        mutate the model or parameters while a background fit thread is
        reading/writing them (add/remove component, auto guess, batch fit,
        clear, session changes, and direct parameter-table edits). Entry
        points that are not panel buttons (menu items, key bindings) cannot
        be disabled this way and call _refuse_while_fitting instead."""
        self.fit_panel.set_fitting_state(running)
        self.fit_results.set_locked(running)

    def _fit_in_progress(self) -> bool:
        return self._fit_thread is not None and self._fit_thread.is_alive()

    def _refuse_while_fitting(self) -> bool:
        """True (after telling the user) if a background fit is running and
        the caller must not touch the model or parameters."""
        if not self._fit_in_progress():
            return False
        messagebox.showwarning(
            "Busy", "A fit is running. Wait for it to finish, or abort it.")
        return True

    def _abort_iter_cb(self):
        """An lmfit iter_cb that stops the minimizer once Abort is pressed."""
        def iter_cb(params, iter, resid, *args, **kw):
            if self._fit_abort.is_set():
                return True
        return iter_cb

    def _run_in_background(self, job, on_done, error_title="Fit Error") -> bool:
        """Run job() on the fit thread with the panel locked, then call
        on_done(result) on the Tk thread. An exception is shown as a dialog
        (unless the user aborted, which is not an error), and an aborted job
        is dropped. Returns False if a fit is already running."""
        if self._fit_in_progress():
            messagebox.showwarning("Busy", "A fit is already running.")
            return False

        self._fit_abort.clear()
        self._set_fit_running(True)
        container = {"result": None, "error": None}

        def _run():
            try:
                container["result"] = job()
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
            if self._fit_abort.is_set():
                return
            if container["error"] is not None:
                messagebox.showerror(error_title, str(container["error"]))
                return
            on_done(container["result"])

        self.after(self._FIT_POLL_INTERVAL_MS, _poll)
        return True

    def _run_fit_async(self, rec, sess, method, sid):
        """Run fit in a background thread (slow methods)."""
        if self._fit_in_progress():
            messagebox.showwarning("Busy", "A fit is already running.")
            return

        fm = sess.fit_manager
        try:
            x, y, yerr, xerr = self._get_fit_data(rec)
        except Exception as e:
            messagebox.showerror("Fit Error", str(e))
            return

        # emcee's sampler settings come from the MCMC Settings dialog;
        # is_weighted and the walker start are filled in by run_fit from
        # the weights actually used.
        fit_kws = dict(self._emcee_settings) if method == "emcee" else {}

        objective_kws, weight_mode, max_nfev, band_sigma, scale_covar = self._get_fit_options()
        iter_cb = self._abort_iter_cb()

        def job():
            return fm.run_fit(
                x, y, yerr=yerr, xerr=xerr, method=method,
                iter_cb=iter_cb, fit_kws=fit_kws,
                objective_kws=objective_kws, weight_mode=weight_mode,
                max_nfev=max_nfev, band_sigma=band_sigma,
                scale_covar=scale_covar,
            )

        def on_done(result):
            sess.result = result
            self._post_fit_update(sess, rec, sid)
            # Auto-show special result dialogs
            if sess.result.candidates:
                self._show_candidates_dialog(sess)
            if sess.result.flatchain is not None:
                self._show_emcee_summary_dialog(sess)

        self._run_in_background(job, on_done)

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

        notes = fit_outcome_notes(result, self.fit_panel.method_var.get())
        if notes:
            messagebox.showwarning("Fit Warning", "\n\n".join(notes))

    def _on_global_fit(self):
        """Open Global Fit dialog for simultaneous fitting across series."""
        if self._refuse_while_fitting():
            return
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

            options = self._fit_options_or_warn()
            if options is None:
                return
            objective_kws, weight_mode, max_nfev, _, scale_covar = options
            method = self.fit_panel.method_var.get()
            session_name = sess.name
            iter_cb = self._abort_iter_cb()

            # A global fit is several datasets' worth of work, so it runs on
            # the fit thread like a slow single fit: the panel locks and the
            # Fit button turns into Abort.
            def job():
                return fm.run_global_fit(
                    datasets, shared, method=method,
                    max_nfev=max_nfev, weight_mode=weight_mode,
                    objective_kws=objective_kws, scale_covar=scale_covar,
                    iter_cb=iter_cb,
                )

            def on_done(results):
                for (sid, r), result in zip(selected_recs, results):
                    target_sess = r.ensure_session(session_name)
                    target_sess.result = result
                    self._show_fit_on_plot(sid, target_sess, r)
                self._refresh_session_ui()

            self._run_in_background(job, on_done, error_title="Global Fit Error")

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
        options = self._fit_options_or_warn()
        if options is None:
            return
        objective_kws, weight_mode, max_nfev, band_sigma, scale_covar = options

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
            # Rows are different series: their AICs are not commensurable.
            ModelComparisonDialog(self, summary_rows, highlight_best=False)
