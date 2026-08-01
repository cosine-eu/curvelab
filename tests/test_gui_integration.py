"""Headless GUI integration tests for CurveLabApp.

These construct a real Tk app (no pixels asserted — only resulting state) and
drive the coordinator's handlers to cover cross-component workflows that pure
unit tests can't reach. They are the regression net for the GUI-layer fixes
made on this branch (async-fit series attribution, control lockout during a
fit, removed-series replot, the parameter-cell double-commit guard, and
workspace-load making analysis tools available).

The whole module skips cleanly when no display / Tk is available, so a headless
run without Xvfb doesn't fail.
"""

import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd


def _display_available() -> bool:
    try:
        import tkinter as tk
        root = tk.Tk()
        root.destroy()
        return True
    except Exception:
        return False


DISPLAY_OK = _display_available()


@unittest.skipUnless(DISPLAY_OK, "no display available for Tk")
class GuiTestBase(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from curvelab.app import CurveLabApp
        self.root = tk.Tk()
        self.app = CurveLabApp(self.root)
        self.app.pack()

    def tearDown(self):
        self.root.destroy()

    def _add_fitted_series(self, sid="ds::x::y", label="s1", dataset="ds", run=True):
        """Register a dataset + series + session with a Linear model (optionally fit)."""
        from curvelab.session import SeriesRecord, FitSession
        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 40)
        y = 2.0 * x + 1.0 + rng.normal(0, 0.1, 40)
        self.app.data_mgr.add_dataframe(dataset, pd.DataFrame({"x": x, "y": y}))
        rec = SeriesRecord(
            x=x, y=y, yerr=None, xerr=None, dataset_name=dataset,
            style={"dataset": dataset, "x": "x", "y": "y", "yerr": "", "xerr": "",
                   "label": label, "marker": "o", "linestyle": "None", "color": ""},
        )
        self.app._series_records[sid] = rec
        self.app._active_series_id = sid
        sess = FitSession(name="Fit 1", color="C0")
        rec.fit_sessions["Fit 1"] = sess
        rec.active_session_name = "Fit 1"
        sess.fit_manager.add_component("Linear")
        sess.fit_manager.auto_guess(x, y)
        if run:
            sess.result = sess.fit_manager.run_fit(x, y, weight_mode="No weights")
        self.app._sync_series_combo()
        self.app._sync_session_list()
        self.app._load_session_into_ui()
        return rec, sess


class AsyncFitAttributionTests(GuiTestBase):
    """8cdaf32: a fit must be plotted under the series it was launched against,
    not whichever series happens to be active when it completes."""

    def test_post_fit_update_uses_launch_series(self):
        rec_a, sess_a = self._add_fitted_series(sid="ds::x::y", label="A", dataset="ds")
        # A second series that becomes "active" while A's fit is in flight.
        self._add_fitted_series(sid="ds2::x::y", label="B", dataset="ds2", run=False)
        self.app._active_series_id = "ds2::x::y"

        # Complete A's fit while B is active, threading A's captured sid through.
        self.app._post_fit_update(sess_a, rec_a, "ds::x::y")
        self.root.update()

        self.assertIn("ds::x::y::Fit 1", self.app.plot_mgr._fit_lines)
        self.assertNotIn("ds2::x::y::Fit 1", self.app.plot_mgr._fit_lines)


class ControlLockoutTests(GuiTestBase):
    """923d021: model/parameter-mutating controls lock while a fit runs."""

    def test_controls_disabled_during_fit_and_restored(self):
        self._add_fitted_series(run=False)
        fp = self.app.fit_panel
        buttons = [fp._add_comp_btn, fp._remove_comp_btn, fp._auto_guess_btn,
                   fp._batch_fit_btn, fp._clear_fit_btn, fp._new_session_btn,
                   fp._rename_session_btn, fp._delete_session_btn]

        self.app._set_fit_running(True)
        for b in buttons:
            self.assertEqual(str(b["state"]), "disabled")
        self.assertTrue(self.app.fit_results._locked)
        self.assertEqual(str(fp._fit_btn["text"]), "Abort")

        self.app._set_fit_running(False)
        for b in buttons:
            self.assertEqual(str(b["state"]), "normal")
        self.assertFalse(self.app.fit_results._locked)
        self.assertEqual(str(fp._fit_btn["text"]), "Fit")


class RemoveSeriesReplotTests(GuiTestBase):
    """ff1ddb4: removing a series from the panel replots immediately."""

    def test_removing_series_updates_app_state(self):
        dp = self.app.data_panel
        df = pd.DataFrame({"x": np.linspace(0, 10, 20),
                           "y1": np.linspace(0, 10, 20) * 2,
                           "y2": np.linspace(0, 10, 20) * 3})
        self.app.data_mgr.add_dataframe("ds", df)
        dp.dataset_var.set("ds")
        for ycol, label in (("y1", "s1"), ("y2", "s2")):
            dp.x_var.set("x"); dp.y_var.set(ycol); dp.label_var.set(label)
            dp._add_series()
        dp._plot()
        self.root.update()
        self.assertEqual(len(self.app._series_records), 2)

        dp.series_listbox.selection_clear(0, "end")
        dp.series_listbox.selection_set(0)
        dp._remove_series()
        self.root.update()
        self.assertEqual(len(self.app._series_records), 1)

        # Removing the last one leaves a clean, empty state (no crash).
        dp.series_listbox.selection_clear(0, "end")
        dp.series_listbox.selection_set(0)
        dp._remove_series()
        self.root.update()
        self.assertEqual(len(self.app._series_records), 0)
        self.assertIsNone(self.app._active_series_id)


class ParamEditorGuardTests(GuiTestBase):
    """766c337: the parameter cell editor must not double-commit onto a
    destroyed widget (Return + deferred FocusOut) or when locked mid-edit."""

    def _open_editor(self):
        panel = self.app.fit_results
        panel.set_params({
            "slope": {"value": 2.0, "init_value": 2.0, "stderr": 0.1,
                      "min": float("-inf"), "max": float("inf"),
                      "vary": True, "expr": ""},
        })
        self.root.update()
        item = panel.param_tree.get_children()[0]
        bbox = panel.param_tree.bbox(item, "#1")  # the "value" cell
        event = mock.Mock(x=bbox[0] + bbox[2] // 2, y=bbox[1] + bbox[3] // 2)
        panel._on_double_click(event)
        self.root.update()
        return panel

    def test_return_commit_records_once_and_cleans_up(self):
        edits = []
        panel = self.app.fit_results
        panel._on_param_edited = lambda name, field, val: edits.append((name, field, val))
        self._open_editor()
        self.assertIsNotNone(panel._editing_entry)

        panel._editing_entry.event_generate("<Return>")
        self.root.update()

        self.assertIsNone(panel._editing_entry)      # entry torn down
        self.assertTrue(panel._editing_done)
        self.assertEqual(len(edits), 1)              # exactly one edit recorded

    def test_set_locked_mid_edit_does_not_raise(self):
        panel = self._open_editor()
        self.assertIsNotNone(panel._editing_entry)
        panel.set_locked(True)   # would hit a destroyed widget without the guard
        self.root.update()
        self.assertIsNone(panel._editing_entry)
        self.assertTrue(panel._editing_done)


class ParamUndoRedoTests(GuiTestBase):
    """lmfit forces vary=False when an expression is set and never restores
    it, so undoing an expression edit must restore vary explicitly -- else
    the parameter stays frozen with no visible cause."""

    def test_undo_expression_edit_restores_vary(self):
        rec, sess = self._add_fitted_series()
        fm = sess.fit_manager

        self.app._on_param_edited("slope", "expr", "intercept*2")
        self.assertEqual(fm.params["slope"].expr, "intercept*2")
        self.assertFalse(fm.params["slope"].vary)

        self.app._undo_param_edit()

        self.assertIn(fm.params["slope"].expr, (None, ""))
        self.assertTrue(fm.params["slope"].vary)

    def test_redo_reapplies_expression(self):
        rec, sess = self._add_fitted_series()
        fm = sess.fit_manager

        self.app._on_param_edited("slope", "expr", "intercept*2")
        self.app._undo_param_edit()
        self.app._redo_param_edit()

        self.assertEqual(fm.params["slope"].expr, "intercept*2")
        self.assertFalse(fm.params["slope"].vary)

    def test_field_parsing_and_rejection(self):
        rec, sess = self._add_fitted_series()
        fm = sess.fit_manager

        self.app._on_param_edited("slope", "min", "")        # empty means -inf
        self.app._on_param_edited("slope", "max", "inf")
        self.app._on_param_edited("slope", "vary", "No")
        self.assertEqual(fm.params["slope"].min, float("-inf"))
        self.assertEqual(fm.params["slope"].max, float("inf"))
        self.assertFalse(fm.params["slope"].vary)

        n_undo = len(sess.undo_stack)
        self.app._on_param_edited("slope", "value", "banana")
        self.assertEqual(len(sess.undo_stack), n_undo)       # nothing recorded
        self.app._on_param_edited("slope", "nonsense", "1")
        self.assertEqual(len(sess.undo_stack), n_undo)

    def test_undo_value_edit_unchanged(self):
        rec, sess = self._add_fitted_series()
        fm = sess.fit_manager
        original = fm.params["slope"].value

        self.app._on_param_edited("slope", "value", "42.0")
        self.assertEqual(fm.params["slope"].value, 42.0)

        self.app._undo_param_edit()

        self.assertAlmostEqual(fm.params["slope"].value, original)


class ExclusionRenderTests(GuiTestBase):
    """Excluded points are dimmed on every drawing path. _on_plot used to
    ignore the mask, so pressing Plot showed all points as included while
    fits still used the reduced set."""

    def _series_with_exclusion(self):
        rec, sess = self._add_fitted_series(run=False)
        rec.mask = np.ones(len(rec.x), dtype=bool)
        rec.mask[0] = False
        self.app.data_panel.add_series_entry(rec.style)
        return rec

    def test_plot_dims_excluded_points(self):
        rec = self._series_with_exclusion()
        self.app._on_plot(self.app.data_panel.series_list)

        lines = self.app.plot_mgr._series_lines
        self.assertEqual(len(lines), 2)                    # included + excluded
        self.assertIn("gray", [ln.get_color() for ln in lines])
        self.assertEqual(len(lines[1].get_xdata()), 1)     # the one excluded point

    def test_replot_matches_plot(self):
        rec = self._series_with_exclusion()
        self.app._on_plot(self.app.data_panel.series_list)
        n_after_plot = len(self.app.plot_mgr._series_lines)

        self.app._replot_all_series()

        self.assertEqual(len(self.app.plot_mgr._series_lines), n_after_plot)

    def test_stale_mask_dropped_when_column_length_changes(self):
        import pandas as pd
        rec = self._series_with_exclusion()
        # Reload the dataset with fewer rows, as a re-import would.
        self.app.data_mgr.datasets["ds"] = pd.DataFrame(
            {"x": np.linspace(0, 10, 5), "y": np.linspace(0, 10, 5)}
        )

        self.app._on_plot(self.app.data_panel.series_list)

        self.assertIsNone(rec.mask)


class RemoveSeriesConfirmTests(GuiTestBase):
    """Removing a series from the DataPanel discards its fit sessions, so it
    asks first -- as removing a dataset already did."""

    def _series_with_session(self):
        rec, sess = self._add_fitted_series()
        self.app.data_panel.add_series_entry(rec.style)
        self.app.data_panel.series_listbox.selection_set(0)
        return rec

    def test_declining_keeps_series_and_sessions(self):
        rec = self._series_with_session()
        with mock.patch("curvelab.app_series.messagebox.askyesno",
                        return_value=False) as ask:
            self.app.data_panel._remove_series()

        ask.assert_called_once()
        self.assertIn("ds::x::y", self.app._series_records)
        self.assertEqual(len(self.app.data_panel.series_list), 1)

    def test_accepting_removes_series(self):
        self._series_with_session()
        with mock.patch("curvelab.app_series.messagebox.askyesno",
                        return_value=True):
            self.app.data_panel._remove_series()

        self.assertNotIn("ds::x::y", self.app._series_records)
        self.assertEqual(len(self.app.data_panel.series_list), 0)

    def test_series_without_sessions_removed_without_asking(self):
        rec, sess = self._add_fitted_series(run=False)
        rec.fit_sessions.clear()
        rec.active_session_name = None
        self.app.data_panel.add_series_entry(rec.style)
        self.app.data_panel.series_listbox.selection_set(0)

        with mock.patch("curvelab.app_series.messagebox.askyesno") as ask:
            self.app.data_panel._remove_series()

        ask.assert_not_called()
        self.assertEqual(len(self.app.data_panel.series_list), 0)


class BatchFitWarningTests(GuiTestBase):
    """Batch fit prepares every series in a loop; data warnings and failures
    are collected and reported once instead of one modal dialog per series."""

    def _add_series_with_nans(self, n_series=3):
        import pandas as pd
        from curvelab.session import SeriesRecord, FitSession
        for i in range(n_series):
            x = np.linspace(0, 10, 20)
            y = 2.0 * x + 1.0
            y[i] = np.nan                       # one bad point per series
            ds = f"ds{i}"
            self.app.data_mgr.add_dataframe(ds, pd.DataFrame({"x": x, "y": y}))
            sid = f"{ds}::x::y"
            rec = SeriesRecord(
                x=x, y=y, dataset_name=ds,
                style={"dataset": ds, "x": "x", "y": "y", "label": f"s{i}"},
            )
            self.app._series_records[sid] = rec
            sess = FitSession(name="Fit 1", color="C0")
            rec.fit_sessions["Fit 1"] = sess
            rec.active_session_name = "Fit 1"
            sess.fit_manager.add_component("Linear")
            if i == 0:
                self.app._active_series_id = sid

    def test_batch_fit_reports_data_warnings_once(self):
        self._add_series_with_nans()
        with mock.patch("curvelab.app.messagebox") as app_mb, \
             mock.patch("curvelab.app_fit_handlers.ModelComparisonDialog"):
            self.app._on_batch_fit()

        self.assertEqual(app_mb.showinfo.call_count, 0)      # no per-series info
        self.assertEqual(app_mb.showwarning.call_count, 1)   # one summary
        body = app_mb.showwarning.call_args[0][1]
        for label in ("s0", "s1", "s2"):
            self.assertIn(label, body)

    def test_single_fit_still_shows_dialog(self):
        self._add_series_with_nans(n_series=1)
        rec = self.app._series_records["ds0::x::y"]
        with mock.patch("curvelab.app.messagebox") as app_mb:
            self.app._get_fit_data(rec)

        self.assertEqual(app_mb.showinfo.call_count, 1)

    def test_collected_warnings_are_labelled_and_deduplicated(self):
        self._add_series_with_nans(n_series=1)
        rec = self.app._series_records["ds0::x::y"]
        collected = []
        with mock.patch("curvelab.app.messagebox") as app_mb:
            self.app._get_fit_data(rec, warnings_out=collected)
            self.app._get_fit_data(rec, warnings_out=collected)
            app_mb.showinfo.assert_not_called()
            self.app._report_collected_warnings("Data Warnings", collected)

        self.assertEqual(len(collected), 2)
        self.assertTrue(all(m.startswith("s0: ") for m in collected))
        # Identical messages collapse into one line in the dialog.
        self.assertEqual(app_mb.showwarning.call_args[0][1].count("\n"), 0)


class PlotControlPersistenceTests(GuiTestBase):
    """Controls that change what is plotted or computed must survive a save
    and load; several were written to the widget but never to the file."""

    NON_DEFAULTS = {
        "weighted_resid_var": False,
        "data_var": False,
        "band_sigma_var": "3",
        "residuals_var": True,
        "confidence_band_var": True,
    }

    def test_plot_controls_round_trip(self):
        for name, value in self.NON_DEFAULTS.items():
            getattr(self.app.plot_controls, name).set(value)
        self.app.fit_panel.scale_covar_var.set(False)

        ws = self.app._serialize_workspace()

        # A fresh app starts at the defaults, then restores from the dict.
        import tkinter as tk
        from curvelab.app import CurveLabApp
        root2 = tk.Tk()
        try:
            app2 = CurveLabApp(root2)
            app2._load_workspace_plot_controls(ws)
            for name, value in self.NON_DEFAULTS.items():
                self.assertEqual(getattr(app2.plot_controls, name).get(), value, name)
            self.assertFalse(app2.fit_panel.scale_covar_var.get())
        finally:
            root2.destroy()

    def test_missing_keys_fall_back_to_defaults(self):
        self.app._load_workspace_plot_controls({"plot_controls": {}})

        self.assertTrue(self.app.plot_controls.weighted_resid_var.get())
        self.assertTrue(self.app.plot_controls.data_var.get())
        self.assertEqual(self.app.plot_controls.band_sigma_var.get(), "1")
        self.assertTrue(self.app.fit_panel.scale_covar_var.get())


class SimulatedSeriesTests(GuiTestBase):
    """Generated series (simulate, export smoothed, placeholder) all register
    through one helper: dataset, column combos, and panel entry together."""

    def test_simulated_data_registers_dataset_and_series(self):
        rec, sess = self._add_fitted_series(run=False)
        self.app.data_panel.add_series_entry(rec.style)

        self.app._generate_simulated_data(
            0.0, 10.0, 50, {"gaussian": True, "gaussian_sigma": 0.1}
        )

        self.assertIn("Simulated 1", self.app.data_mgr.datasets)
        sid = "Simulated 1::x::y"
        self.assertIn(sid, self.app._series_records)
        self.assertEqual(len(self.app._series_records[sid].x), 50)
        # Noise produced a yerr column, and the series entry points at it.
        entry = [s for s in self.app.data_panel.series_list
                 if s["dataset"] == "Simulated 1"][0]
        self.assertEqual(entry["yerr"], "yerr")

    def test_exported_smooth_series_is_registered(self):
        rec, sess = self._add_fitted_series(run=False)
        self.app.data_panel.add_series_entry(rec.style)

        self.app._export_smooth_series(rec.x, rec.y, "Smoothed")

        name = f"{rec.dataset_name} (Smoothed)"
        self.assertIn(name, self.app.data_mgr.datasets)
        entry = [s for s in self.app.data_panel.series_list
                 if s["dataset"] == name][0]
        self.assertEqual(entry["linestyle"], "-")


class SqliteWorkspaceReloadTests(GuiTestBase):
    """A SQLite file loads all its tables as separate datasets, so a
    workspace reload has to map each saved name back by (file, table)."""

    def _make_db(self):
        import os, sqlite3, tempfile
        path = os.path.join(tempfile.mkdtemp(), "meas.sqlite")
        conn = sqlite3.connect(path)
        for table in ("run1", "run2"):
            conn.execute(f"CREATE TABLE {table} (x REAL, y REAL)")
            conn.executemany(f"INSERT INTO {table} VALUES (?, ?)",
                             [(float(i), 2.0 * i) for i in range(5)])
        conn.commit()
        conn.close()
        return path

    def test_both_tables_map_back_to_their_datasets(self):
        path = self._make_db()
        name = os.path.basename(path)
        ws = {
            "data_filepaths": {f"{name}::run1": path, f"{name}::run2": path},
            "table_names": {f"{name}::run1": "run1", f"{name}::run2": "run2"},
        }

        mapping = self.app._load_workspace_datasets(ws)

        self.assertEqual(mapping[f"{name}::run1"], f"{name}::run1")
        self.assertEqual(mapping[f"{name}::run2"], f"{name}::run2")
        # The file is opened once, not once per saved dataset.
        self.assertEqual(len(self.app.data_mgr.datasets), 2)


class ObjectiveControlTests(GuiTestBase):
    """The Objective control follows the method, f_scale enables only for
    robust losses, and the app turns the selection into fit_kws."""

    def _state(self, w):
        return str(w.cget("state"))

    def test_objective_choices_and_control_states_follow_method(self):
        fp = self.app.fit_panel
        fp.method_var.set("least_squares"); fp._on_method_changed()
        self.assertIn("Cauchy", fp._objective_combo["values"])
        self.assertEqual(self._state(fp._f_scale_entry), "disabled")  # linear

        fp.objective_var.set("Cauchy"); fp._on_objective_changed()
        self.assertEqual(self._state(fp._f_scale_entry), "normal")

        fp.method_var.set("odr"); fp._on_method_changed()
        self.assertEqual(fp._objective_combo["values"], ("Orthogonal distance",))
        self.assertEqual(self._state(fp._weight_combo), "disabled")
        self.assertEqual(self._state(fp._scale_covar_check), "disabled")

        # Remembered loss restored when returning to least_squares.
        fp.method_var.set("least_squares"); fp._on_method_changed()
        self.assertEqual(fp.objective_var.get(), "Cauchy")
        self.assertEqual(self._state(fp._weight_combo), "readonly")

    def test_get_fit_options_builds_objective_kws(self):
        self._add_fitted_series(run=False)
        fp = self.app.fit_panel
        fp.method_var.set("least_squares"); fp._on_method_changed()
        fp.objective_var.set("Cauchy"); fp._on_objective_changed()
        fp.f_scale_var.set("3.0")

        objective_kws, *_ = self.app._get_fit_options()
        self.assertEqual(objective_kws, {"loss": "cauchy", "f_scale": 3.0})

        fp.method_var.set("nelder"); fp._on_method_changed()
        fp.objective_var.set("Neg. entropy"); fp._on_objective_changed()
        objective_kws, *_ = self.app._get_fit_options()
        self.assertIn("reduce_fcn", objective_kws)

    def test_bad_f_scale_falls_back_to_default(self):
        self._add_fitted_series(run=False)
        fp = self.app.fit_panel
        fp.method_var.set("least_squares"); fp._on_method_changed()
        fp.objective_var.set("Cauchy"); fp._on_objective_changed()
        fp.f_scale_var.set("not a number")
        objective_kws, *_ = self.app._get_fit_options()
        self.assertEqual(objective_kws["f_scale"], 1.0)


class ObjectiveWorkspaceTests(GuiTestBase):
    """The objective survives a workspace round trip, and an old reduce_fcn
    key still loads."""

    def test_objective_round_trip(self):
        fp = self.app.fit_panel
        fp.method_var.set("least_squares"); fp._on_method_changed()
        fp.objective_var.set("Huber"); fp._on_objective_changed()
        fp.f_scale_var.set("2.5")
        ws = self.app._serialize_workspace()

        import tkinter as tk
        from curvelab.app import CurveLabApp
        root2 = tk.Tk()
        try:
            app2 = CurveLabApp(root2)
            app2._load_workspace_plot_controls(ws)
            self.assertEqual(app2.fit_panel.method_var.get(), "least_squares")
            self.assertEqual(app2.fit_panel.objective_var.get(), "Huber")
            self.assertEqual(app2.fit_panel.f_scale_var.get(), "2.5")
        finally:
            root2.destroy()

    def test_legacy_reduce_fcn_key_still_loads(self):
        # A pre-objective workspace: scalar method + reduce label.
        ws = {"plot_controls": {"fit_method": "nelder",
                                "reduce_fcn": "Neg. entropy"}}
        self.app._load_workspace_plot_controls(ws)
        self.assertEqual(self.app.fit_panel.method_var.get(), "nelder")
        self.assertEqual(self.app.fit_panel.objective_var.get(), "Neg. entropy")


class FitValidationTests(GuiTestBase):
    """_on_fit refuses a method whose requirements aren't met, and doesn't
    launch a fit in that case."""

    def test_bounds_required_method_blocks_fit(self):
        rec, sess = self._add_fitted_series(run=False)
        # Linear params default to unbounded, and DE needs finite bounds.
        self.app.fit_panel.method_var.set("differential_evolution")

        with mock.patch("curvelab.app_fit_handlers.messagebox.showwarning") as warn, \
             mock.patch.object(self.app, "_run_fit_sync") as sync, \
             mock.patch.object(self.app, "_run_fit_async") as async_:
            self.app._on_fit()

        warn.assert_called_once()
        self.assertIn("finite", warn.call_args[0][1])
        sync.assert_not_called()
        async_.assert_not_called()

    def test_bounded_params_allow_fit(self):
        rec, sess = self._add_fitted_series(run=False)
        fm = sess.fit_manager
        for name in fm.params:
            fm.set_param(name, min=-100.0, max=100.0)
        self.app.fit_panel.method_var.set("differential_evolution")

        with mock.patch("curvelab.app_fit_handlers.messagebox.showwarning") as warn, \
             mock.patch.object(self.app, "_run_fit_async") as async_:
            self.app._on_fit()

        warn.assert_not_called()
        async_.assert_called_once()   # DE is a slow method -> async path


class ScaleRoundTripTests(GuiTestBase):
    """matplotlib 3.6 keeps a line's log-transformed path cache across a
    scale change once a draw happened in log scale, rendering lines at log
    positions on the restored linear axis. PlotManager.set_xscale/yscale
    recache all lines to defeat this; the round trip must render exactly
    like the original."""

    def _render(self):
        import io
        buf = io.BytesIO()
        self.app.plot_mgr.fig.savefig(buf, format="png")
        return buf.getvalue()

    def test_log_linear_round_trip_renders_identically(self):
        self._add_fitted_series()
        self.app._replot_all_series()
        pm = self.app.plot_mgr
        pm.canvas.draw()
        self.root.update()
        baseline = self._render()

        for scale_setter in (pm.set_xscale, pm.set_yscale):
            scale_setter("log")
            pm.canvas.draw()          # the bug needs a real draw in log scale
            self.root.update()
            scale_setter("linear")
            pm.canvas.draw()
            self.root.update()
            self.assertEqual(self._render(), baseline)


class TitleAndAxisLimitTests(GuiTestBase):
    """The Title and X/Y Range entries pin the plot title and axis limits;
    empty fields mean automatic, and pinned limits survive a replot."""

    def test_title_is_applied_and_survives_replot(self):
        self._add_fitted_series()
        self.app._on_title("My Measurement")
        self.assertEqual(self.app.plot_mgr.ax.get_title(), "My Measurement")
        self.app._replot_all_series()  # clear_all + redraw must keep the title
        self.assertEqual(self.app.plot_mgr.ax.get_title(), "My Measurement")

    def test_limits_pin_and_clear(self):
        self._add_fitted_series()
        self.app._replot_all_series()
        ax = self.app.plot_mgr.ax

        self.app._on_axis_limits("1", "5", "-2", "8")
        self.assertEqual(ax.get_xlim(), (1.0, 5.0))
        self.assertEqual(ax.get_ylim(), (-2.0, 8.0))

        # Pinned limits survive a replot.
        self.app._replot_all_series()
        self.assertEqual(ax.get_xlim(), (1.0, 5.0))
        self.assertEqual(ax.get_ylim(), (-2.0, 8.0))

        # Partial spec: only one side pinned, the other stays automatic.
        self.app._on_axis_limits("", "5", "", "")
        self.assertEqual(ax.get_xlim()[1], 5.0)
        self.assertLess(ax.get_xlim()[0], 1.0)  # auto again (data starts at 0)

        # All empty restores full autoscale (data spans 0..10).
        self.app._on_axis_limits("", "", "", "")
        self.assertLess(ax.get_xlim()[0], 1.0)
        self.assertGreater(ax.get_xlim()[1], 9.0)

    def test_pinned_limits_survive_equal_aspect_toggle(self):
        self._add_fitted_series()
        self.app._replot_all_series()
        ax = self.app.plot_mgr.ax
        self.app._on_axis_limits("1", "5", "-2", "8")

        self.app.plot_mgr.set_equal_aspect(True)
        self.app.plot_mgr.set_equal_aspect(False)

        self.assertEqual(ax.get_xlim(), (1.0, 5.0))
        self.assertEqual(ax.get_ylim(), (-2.0, 8.0))

    def test_bad_limit_warns_and_keeps_previous(self):
        self._add_fitted_series()
        self.app._replot_all_series()
        self.app._on_axis_limits("1", "5", "", "")
        with mock.patch("curvelab.app_plotting.messagebox.showwarning") as warn:
            self.app._on_axis_limits("banana", "5", "", "")
            warn.assert_called_once()
        self.assertEqual(self.app.plot_mgr.ax.get_xlim(), (1.0, 5.0))


class WorkspaceLoadAnalysisTests(GuiTestBase):
    """The experiment doc's canonical bug: load a workspace, then open an
    analysis dialog — the fit result (and reconstructed lmfit result) must be
    available so tools don't wrongly report 'run a fit first'."""

    def test_analysis_available_after_workspace_load(self):
        import os, tempfile
        from curvelab.session import SeriesRecord, FitSession
        # The workspace records reloadable file paths, so the series must be
        # backed by a real file (not add_dataframe, which has no path).
        d = tempfile.mkdtemp()
        csvp = os.path.join(d, "data.csv")
        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 40)
        y = 2.0 * x + 1.0 + rng.normal(0, 0.1, 40)
        pd.DataFrame({"x": x, "y": y}).to_csv(csvp, index=False)
        name, _ = self.app.data_mgr.load(csvp)
        sid = f"{name}::x::y"
        rec = SeriesRecord(
            x=x, y=y, yerr=None, xerr=None, dataset_name=name,
            style={"dataset": name, "x": "x", "y": "y", "yerr": "", "xerr": "",
                   "label": "s1", "marker": "o", "linestyle": "None", "color": ""},
        )
        self.app._series_records[sid] = rec
        self.app._active_series_id = sid
        sess = FitSession(name="Fit 1", color="C0")
        rec.fit_sessions["Fit 1"] = sess
        rec.active_session_name = "Fit 1"
        sess.fit_manager.add_component("Linear")
        sess.fit_manager.auto_guess(x, y)
        sess.result = sess.fit_manager.run_fit(x, y, weight_mode="No weights")
        self.app.plot_controls.residuals_var.set(True)

        wsp = os.path.join(d, "ws.clw")
        try:
            with mock.patch("curvelab.app_workspace.filedialog.asksaveasfilename",
                            return_value=wsp):
                self.app._save_workspace()

            # Fresh app loads it.
            import tkinter as tk
            from curvelab.app import CurveLabApp
            root2 = tk.Tk()
            app2 = CurveLabApp(root2)
            app2.pack()
            try:
                with mock.patch("curvelab.app_workspace.filedialog.askopenfilename",
                                return_value=wsp):
                    app2._load_workspace()

                sess2 = app2._require_fit_result()  # None + warning if broken
                self.assertIsNotNone(sess2)
                # refit reconstructed the lmfit result analysis tools depend on.
                self.assertIsNotNone(sess2.fit_manager._last_result)

                # Opening an analysis handler must not warn "No Fit".
                with mock.patch(
                    "curvelab.app_analysis_handlers.messagebox.showwarning"
                ) as warn:
                    app2._show_confidence_intervals()
                    self.assertFalse(
                        any("No Fit" in str(c) for c in warn.call_args_list))
            finally:
                root2.destroy()
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
