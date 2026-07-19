"""Fit-execution handlers for CurveLabApp.

Mixed into CurveLabApp; these methods drive fitting (sync, ODR, and background
threaded runs), auto-guess, global fit, and batch fit. They rely on the
coordinator's session accessors, panels, plot manager, and the fit-thread
state (_fit_thread, _fit_abort, _FIT_POLL_INTERVAL_MS) held on CurveLabApp.
"""

import threading
from tkinter import messagebox

from .fit_manager import FitManager, REDUCE_FUNCTIONS
from .session import (
    FIT_COLORS, FitSession,
    make_session_key as _make_session_key,
)
from .ui_dialogs_analysis import ModelComparisonDialog, GlobalFitDialog


class FitHandlersMixin:
    """Fit execution: single/ODR/async runs, auto-guess, global and batch fit."""

    _SLOW_METHODS = {"emcee", "brute", "differential_evolution", "basinhopping",
                      "dual_annealing", "shgo", "ampgo"}

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

        # Capture the series id now, since an async fit can outlive the
        # user's current selection (e.g. they switch to another series
        # while a slow method like emcee is still running).
        sid = self._active_series_id

        if method == "odr":
            self._run_fit_odr(rec, sess, sid)
        elif method in self._SLOW_METHODS:
            self._run_fit_async(rec, sess, method, sid)
        else:
            self._run_fit_sync(rec, sess, method, sid)

    def _get_fit_options(self):
        """Read reduce function, weight mode, max_nfev, band_sigma, scale_covar from UI."""
        reduce_fcn = REDUCE_FUNCTIONS.get(self.fit_panel.reduce_var.get())
        weight_mode = self.fit_panel.weight_var.get()
        max_nfev_str = self.fit_panel.max_nfev_var.get().strip()
        max_nfev = int(max_nfev_str) if max_nfev_str else None
        band_sigma = int(self.plot_controls.band_sigma_var.get())
        scale_covar = self.fit_panel.scale_covar_var.get()
        return reduce_fcn, weight_mode, max_nfev, band_sigma, scale_covar

    def _run_fit_sync(self, rec, sess, method, sid):
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
        skey = _make_session_key(sid, sess.name)
        series_label = rec.style.get("label", sid)
        label = f"{series_label} — {sess.name}"

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
                label = f"{series_label} — {session_name}"
                self.plot_mgr.clear_fit_session(skey)
                self._plot_fit_for_session(skey, target_sess, label)
                if show_resid:
                    self._plot_residuals_for_session(skey, target_sess, r)

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
                label = f"{series_label} — {session_name}"

                self.plot_mgr.clear_fit_session(skey)
                self._plot_fit_for_session(skey, target_sess, label)
                if show_resid:
                    self._plot_residuals_for_session(skey, target_sess, target_rec)

                model_desc = target_fm.model_description()
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
        self._refresh_session_ui()

        if summary_rows:
            ModelComparisonDialog(self, summary_rows)
