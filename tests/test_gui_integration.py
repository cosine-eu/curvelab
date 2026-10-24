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
        # messagebox is patched because _post_fit_update warns when a fit
        # reports no uncertainties: a real modal dialog has nobody to dismiss
        # it under a headless X server and would hang the run forever.
        with mock.patch("curvelab.app_fit_handlers.messagebox"):
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


    def test_menu_and_key_entry_points_refuse_while_a_fit_runs(self):
        # Panel buttons are disabled by the lock; menu items and key bindings
        # cannot be, so they must check for a live fit thread themselves.
        # Ctrl+Z on the parameters while the worker is inside model.fit on
        # the same Parameters object is exactly the race the lock exists for.
        from curvelab.session import ParamEdit
        self._add_fitted_series(run=False)
        sess = self.app._active_session
        fm = sess.fit_manager
        sess.undo_stack.append(ParamEdit(param_name="slope", field="value",
                                         old_value=9.0, new_value=fm.params["slope"].value))
        live = mock.Mock()
        live.is_alive.return_value = True
        self.app._fit_thread = live
        try:
            with mock.patch("curvelab.app_fit_handlers.messagebox") as mb, \
                    mock.patch("curvelab.app_workspace.filedialog") as fd, \
                    mock.patch("curvelab.app_analysis_handlers.filedialog") as fd2:
                self.app._undo_param_edit()
                self.app._on_fit()
                self.app._on_global_fit()
                self.app._load_workspace()
                self.app._import_model_result()
            self.assertEqual(mb.showwarning.call_count, 5)
            self.assertTrue(all(c[0][0] == "Busy" for c in mb.showwarning.call_args_list))
            # Nothing happened: the undo was not applied and no file dialog opened.
            self.assertEqual(len(sess.undo_stack), 1)
            self.assertNotEqual(fm.params["slope"].value, 9.0)
            fd.askopenfilename.assert_not_called()
            fd2.askopenfilename.assert_not_called()
        finally:
            self.app._fit_thread = None

        # With no fit running the same undo goes through.
        self.app._undo_param_edit()
        self.assertEqual(fm.params["slope"].value, 9.0)


class BackgroundRunnerTests(GuiTestBase):
    """Slow single fits and global fits share one background runner: the
    panel locks while the job runs, results land on the Tk thread, an
    abort drops the result, and an exception becomes a dialog."""

    def _pump_until_done(self, timeout_s=10.0):
        import time
        deadline = time.time() + timeout_s
        while self.app._fit_thread is not None and time.time() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.root.update()
        self.assertIsNone(self.app._fit_thread, "background job did not finish")

    def test_job_result_is_delivered_and_panel_unlocked(self):
        self._add_fitted_series(run=False)
        seen = []
        self.assertTrue(self.app._run_in_background(lambda: 42, seen.append))
        self.assertEqual(self.app.fit_panel._fit_btn["text"], "Abort")
        self._pump_until_done()
        self.assertEqual(seen, [42])
        self.assertEqual(self.app.fit_panel._fit_btn["text"], "Fit")

    def test_job_error_is_shown_and_abort_is_silent(self):
        self._add_fitted_series(run=False)

        def boom():
            raise RuntimeError("no good")
        with mock.patch("curvelab.app_fit_handlers.messagebox") as mb:
            self.app._run_in_background(boom, lambda r: None, error_title="Global Fit Error")
            self._pump_until_done()
        mb.showerror.assert_called_once()
        self.assertEqual(mb.showerror.call_args[0][0], "Global Fit Error")

        seen = []
        with mock.patch("curvelab.app_fit_handlers.messagebox") as mb:
            self.app._run_in_background(lambda: self.app._fit_abort.set() or 1, seen.append)
            self._pump_until_done()
        self.assertEqual(seen, [])                 # aborted: result dropped
        mb.showerror.assert_not_called()

    def test_global_fit_runs_on_the_background_thread(self):
        rec_a, sess_a = self._add_fitted_series(sid="ds::x::y", label="A", dataset="ds", run=False)
        rec_b, _ = self._add_fitted_series(sid="ds2::x::y", label="B", dataset="ds2", run=False)
        self.app._active_series_id = "ds::x::y"
        captured = {}
        with mock.patch("curvelab.app_fit_handlers.GlobalFitDialog",
                        side_effect=lambda *a, **kw: captured.update(kw)):
            self.app._on_global_fit()
        with mock.patch("curvelab.app_fit_handlers.messagebox"):
            captured["on_fit"](["ds::x::y", "ds2::x::y"], {"slope"})
            self.assertIsNotNone(self.app._fit_thread)       # ran in the background
            self._pump_until_done()
        self.assertIsNotNone(sess_a.result)
        self.assertIsNotNone(rec_b.fit_sessions["Fit 1"].result)
        self.assertEqual(sess_a.result.params["slope"]["value"],
                         rec_b.fit_sessions["Fit 1"].result.params["slope"]["value"])
        self.assertIn("ds::x::y::Fit 1", self.app.plot_mgr._fit_lines)


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


class ParamTableEditabilityTests(GuiTestBase):
    """Nothing in the parameter table said which cells could be edited.
    Editable headings carry a pencil and a line under the table explains
    them; derived (expr) rows are grayed and refuse Value/Vary edits, which
    the expression would override; Vary toggles on double-click instead of
    asking for typed Yes/No."""

    EDITABLE = ("value", "min", "max", "vary", "expr")
    COLUMNS = ("value", "init", "stderr", "min", "max", "vary", "expr")

    def _params(self):
        inf = float("inf")
        return {
            "sigma": {"value": 0.7, "init_value": 0.7, "stderr": None,
                      "min": 0.0, "max": inf, "vary": True, "expr": ""},
            "fwhm": {"value": 1.65, "init_value": 1.65, "stderr": None,
                     "min": -inf, "max": inf, "vary": False,
                     "expr": "2.3548200*sigma"},
        }

    def _panel(self, edits=None):
        panel = self.app.fit_results
        if edits is not None:
            panel._on_param_edited = lambda n, f, v: edits.append((n, f, v))
        panel.set_params(self._params())
        self.root.update()
        return panel

    def _double_click(self, panel, name, column):
        tree = panel.param_tree
        item = next(i for i in tree.get_children() if tree.item(i, "text") == name)
        col_id = f"#{self.COLUMNS.index(column) + 1}"
        bbox = tree.bbox(item, col_id)
        panel._on_double_click(mock.Mock(x=bbox[0] + bbox[2] // 2,
                                         y=bbox[1] + bbox[3] // 2))
        self.root.update()
        return item

    def test_editable_headings_are_marked(self):
        tree = self.app.fit_results.param_tree
        for col in self.COLUMNS:
            text = tree.heading(col, "text")
            if col in self.EDITABLE:
                self.assertIn("✎", text, col)
            else:
                self.assertNotIn("✎", text, col)

    def test_hint_line_explains_editing(self):
        hint = self.app.fit_results.hint_var.get()
        self.assertIn("Double-click", hint)
        self.assertIn("start", hint)       # Value sets the next fit's start

    def test_derived_row_is_grayed(self):
        panel = self._panel()
        tree = panel.param_tree
        tags = {tree.item(i, "text"): tree.item(i, "tags") for i in tree.get_children()}
        self.assertIn("derived", tags["fwhm"])
        self.assertNotIn("derived", tags["sigma"])
        self.assertTrue(tree.tag_configure("derived", "foreground"))

    def test_derived_row_refuses_value_and_vary(self):
        edits = []
        panel = self._panel(edits)
        for col in ("value", "vary"):
            self._double_click(panel, "fwhm", col)
            self.assertIsNone(panel._editing_entry, col)
            self.assertIn("expression", panel.hint_var.get().lower())
        self.assertEqual(edits, [])

    def test_derived_row_expr_still_editable(self):
        # Editing the expression is how a derived parameter is untied.
        panel = self._panel()
        self._double_click(panel, "fwhm", "expr")
        self.assertIsNotNone(panel._editing_entry)
        panel._editing_entry.event_generate("<Escape>")
        self.root.update()

    def test_vary_toggles_on_double_click(self):
        edits = []
        panel = self._panel(edits)
        item = self._double_click(panel, "sigma", "vary")
        self.assertIsNone(panel._editing_entry)      # no typing needed
        self.assertEqual(panel.param_tree.set(item, "vary"), "No")
        self._double_click(panel, "sigma", "vary")
        self.assertEqual(panel.param_tree.set(item, "vary"), "Yes")
        self.assertEqual(edits, [("sigma", "vary", "No"), ("sigma", "vary", "Yes")])

    def test_vary_toggle_reaches_fit_manager(self):
        rec, sess = self._add_fitted_series(run=False)
        panel = self.app.fit_results
        self.app._refresh_param_display()
        self.root.update()
        self._double_click(panel, "slope", "vary")
        self.assertFalse(sess.fit_manager.params["slope"].vary)

    def test_vary_toggle_ignored_while_locked(self):
        edits = []
        panel = self._panel(edits)
        panel.set_locked(True)
        item = self._double_click(panel, "sigma", "vary")
        self.assertEqual(panel.param_tree.set(item, "vary"), "Yes")
        self.assertEqual(edits, [])


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


class EmptyPlotWarningTests(GuiTestBase):
    """Clearing the plot by removing the last series is deliberate, so it must
    not warn; pressing Plot with nothing to draw still should."""

    def _one_series(self):
        dp = self.app.data_panel
        self.app.data_mgr.add_dataframe(
            "ds", pd.DataFrame({"x": np.linspace(0, 10, 20),
                                "y": np.linspace(0, 10, 20) * 2}))
        dp.dataset_var.set("ds")
        dp.x_var.set("x"); dp.y_var.set("y"); dp.label_var.set("s1")
        dp._add_series()
        return dp

    def test_removing_last_series_does_not_warn(self):
        dp = self._one_series()
        dp.series_listbox.selection_set(0)
        with mock.patch("curvelab.app_plotting.messagebox.showwarning") as warn:
            dp._remove_series()
        warn.assert_not_called()
        self.assertEqual(len(self.app._series_records), 0)

    def test_plot_button_still_warns_with_no_series(self):
        self._one_series()
        self.app.data_panel._series_items.clear()
        with mock.patch("curvelab.app_plotting.messagebox.showwarning") as warn:
            self.app.data_panel._plot()
        warn.assert_called_once()
        self.assertEqual(warn.call_args[0][0], "No Series")


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


class FitLineStyleGuiTests(GuiTestBase):
    """The fit line was always solid and 2 points wide, in a color picked
    by session order, with no way to change it. A Style button in the Fit
    panel sets the active session's color, line style and width."""

    def _fit_line(self, sid="ds::x::y", session="Fit 1"):
        return self.app.plot_mgr._fit_lines[f"{sid}::{session}"][0]

    def test_fit_line_drawn_with_session_style(self):
        from matplotlib.colors import to_hex
        rec, sess = self._add_fitted_series()
        sess.color = "xkcd:black"
        sess.linestyle = "--"
        sess.linewidth = 1.0
        self.app._show_fit_on_plot("ds::x::y", sess, rec)
        line = self._fit_line()
        self.assertEqual(to_hex(line.get_color()), to_hex("xkcd:black"))
        self.assertEqual(line.get_linestyle(), "--")
        self.assertEqual(line.get_linewidth(), 1.0)

    def test_set_session_style_updates_session_and_plot(self):
        rec, sess = self._add_fitted_series()
        self.app._show_fit_on_plot("ds::x::y", sess, rec)
        self.app._set_session_style(color="xkcd:black", linestyle=":", linewidth=3.0)
        self.assertEqual((sess.color, sess.linestyle, sess.linewidth),
                         ("xkcd:black", ":", 3.0))
        line = self._fit_line()
        self.assertEqual(line.get_linestyle(), ":")
        self.assertEqual(line.get_linewidth(), 3.0)

    def test_set_session_style_rejects_bad_width(self):
        rec, sess = self._add_fitted_series()
        with mock.patch("curvelab.app_series.messagebox") as mb:
            self.app._set_session_style(color=sess.color, linestyle="-", linewidth=-1.0)
        mb.showwarning.assert_called_once()
        self.assertEqual(sess.linewidth, 2.0)

    def test_style_button_locked_during_fit(self):
        self._add_fitted_series(run=False)
        btn = self.app.fit_panel._style_btn
        self.app._set_fit_running(True)
        self.assertEqual(str(btn["state"]), "disabled")
        self.app._set_fit_running(False)
        self.assertEqual(str(btn["state"]), "normal")


class BatchFitSourceAndSetupTests(GuiTestBase):
    """Batch Fit re-fitted the source series from a fresh guess, replacing
    the fit the user had set up; skipped the setup check that Fit does;
    ran an ordinary fit when ODR was selected; and fitted hidden series."""

    def _add_series(self, n=3, xerr=False, hidden=()):
        from curvelab.session import SeriesRecord, FitSession
        rng = np.random.default_rng(1)
        for i in range(n):
            x = np.linspace(0, 10, 30)
            y = (1.5 + i) * x + 2.0 + rng.normal(0, 0.2, 30)
            ds = f"ds{i}"
            self.app.data_mgr.add_dataframe(ds, pd.DataFrame({"x": x, "y": y}))
            sid = f"{ds}::x::y"
            rec = SeriesRecord(
                x=x, y=y, yerr=np.full(30, 0.2),
                xerr=np.full(30, 0.05) if xerr else None, dataset_name=ds,
                style={"dataset": ds, "x": "x", "y": "y", "label": f"s{i}"},
                visible=i not in hidden,
            )
            self.app._series_records[sid] = rec
        src = self.app._series_records["ds0::x::y"]
        sess = FitSession(name="Fit 1", color="C0")
        src.fit_sessions["Fit 1"] = sess
        src.active_session_name = "Fit 1"
        self.app._active_series_id = "ds0::x::y"
        fm = sess.fit_manager
        fm.add_component("Linear")
        fm.auto_guess(src.x, src.y)
        fm.set_param("intercept", value=1.0)          # a start the user chose
        fm.set_param_hint("intercept", value=1.0)
        sess.result = fm.run_fit(src.x, src.y, weight_mode="No weights")
        return src, sess

    def _batch_fit(self, method="least_squares"):
        self.app.fit_panel.method_var.set(method)
        with mock.patch("curvelab.app.messagebox") as app_mb, \
             mock.patch("curvelab.app_fit_handlers.messagebox") as fit_mb, \
             mock.patch("curvelab.app_fit_handlers.ModelComparisonDialog") as dlg:
            self.app._on_batch_fit()
        return app_mb, fit_mb, dlg

    def _target_result(self, i):
        sess = self.app._series_records[f"ds{i}::x::y"].fit_sessions.get("Fit 1")
        return None if sess is None else sess.result

    def test_source_series_is_left_alone(self):
        src, sess = self._add_series()
        result = sess.result
        start = dict(sess.fit_manager._start_values)
        self._batch_fit()
        self.assertIs(sess.result, result)
        self.assertEqual(sess.fit_manager._start_values, start)
        self.assertIsNotNone(self._target_result(1))
        self.assertIsNotNone(self._target_result(2))

    def test_source_still_listed_in_summary(self):
        self._add_series()
        _, _, dlg = self._batch_fit()
        rows = dlg.call_args[0][1]
        self.assertEqual(len(rows), 3)
        self.assertTrue(any(r["session"].startswith("s0") for r in rows))

    def test_hidden_series_is_skipped(self):
        self._add_series(hidden=(2,))
        _, _, dlg = self._batch_fit()
        self.assertIsNotNone(self._target_result(1))
        self.assertIsNone(self._target_result(2))
        rows = dlg.call_args[0][1]
        self.assertFalse(any(r["session"].startswith("s2") for r in rows))

    def test_setup_error_refused_once_before_fitting(self):
        # differential_evolution needs finite bounds; Linear has none.
        self._add_series()
        app_mb, fit_mb, dlg = self._batch_fit("differential_evolution")
        fit_mb.showwarning.assert_called_once()
        self.assertEqual(fit_mb.showwarning.call_args[0][0], "Cannot Fit")
        app_mb.showwarning.assert_not_called()
        dlg.assert_not_called()
        self.assertIsNone(self._target_result(1))

    def test_odr_selected_runs_odr(self):
        from curvelab.fit_manager import FitManager
        self._add_series(xerr=True)
        with mock.patch.object(FitManager, "run_odr", autospec=True,
                               side_effect=FitManager.run_odr) as odr, \
             mock.patch.object(FitManager, "run_fit", autospec=True,
                               side_effect=FitManager.run_fit) as fit:
            self._batch_fit("odr")
        self.assertEqual(odr.call_count, 2)            # the two targets
        fit.assert_not_called()
        self.assertIsNotNone(self._target_result(1))


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
        for label in ("s1", "s2"):
            self.assertIn(label, body)
        self.assertNotIn("s0", body)     # the source series is not refitted

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


class WorkspaceLoadWarningTests(GuiTestBase):
    """Series whose columns are gone and fits that cannot be reconstructed
    used to vanish silently on load. They are now collected and reported
    once, and the rest of the workspace still loads."""

    def test_missing_column_and_failed_refit_are_reported_once(self):
        x = np.linspace(0, 10, 20)
        self.app.data_mgr.add_dataframe("ds", pd.DataFrame({"x": x, "y": 2 * x}))
        mapping = {"ds": "ds"}
        ws = {"series": {
            "ds::x::y": {"dataset_name": "ds", "visible": True, "fit_sessions": {},
                         "style": {"dataset": "ds", "x": "x", "y": "y", "yerr": "", "xerr": "",
                                   "label": "good", "marker": "o", "linestyle": "None", "color": ""}},
            "ds::x::gone": {"dataset_name": "ds", "visible": True, "fit_sessions": {},
                            "style": {"dataset": "ds", "x": "x", "y": "gone", "yerr": "", "xerr": "",
                                      "label": "stale", "marker": "o", "linestyle": "None", "color": ""}},
        }}
        problems: list[str] = []
        self.app._load_workspace_series(ws, mapping, problems)
        self.assertEqual(list(self.app._series_records), ["ds::x::y"])
        self.assertEqual(len(problems), 1)
        self.assertIn("'stale' skipped", problems[0])
        self.assertIn("No column 'gone'", problems[0])

        # A session whose refit blows up is reported, not passed over.
        rec = self.app._series_records["ds::x::y"]
        sess = rec.ensure_session("fit1")
        sess.fit_manager.add_component("Linear")
        sess.result = mock.Mock()
        with mock.patch.object(sess.fit_manager, "refit_from_result",
                               side_effect=RuntimeError("boom")):
            self.app._load_workspace_refit(problems)
        self.assertEqual(len(problems), 2)
        self.assertIn("session 'fit1'", problems[1])
        self.assertIn("boom", problems[1])

        with mock.patch("curvelab.app.messagebox") as mb:
            self.app._report_collected_warnings("Workspace Load Warnings", problems)
        mb.showwarning.assert_called_once()
        self.assertEqual(mb.showwarning.call_args[0][0], "Workspace Load Warnings")


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

        objective_kws = self.app._get_fit_options().objective_kws
        self.assertEqual(objective_kws, {"loss": "cauchy", "f_scale": 3.0})

        fp.method_var.set("nelder"); fp._on_method_changed()
        fp.objective_var.set("Cauchy log-pdf"); fp._on_objective_changed()
        objective_kws = self.app._get_fit_options().objective_kws
        self.assertIn("reduce_fcn", objective_kws)

    def test_bad_f_scale_falls_back_to_default(self):
        self._add_fitted_series(run=False)
        fp = self.app.fit_panel
        fp.method_var.set("least_squares"); fp._on_method_changed()
        fp.objective_var.set("Cauchy"); fp._on_objective_changed()
        fp.f_scale_var.set("not a number")
        objective_kws = self.app._get_fit_options().objective_kws
        self.assertEqual(objective_kws["f_scale"], 1.0)


class DerivativeIntegralDialogTests(GuiTestBase):
    """The dialog opens on a fitted composite model. This is the only path
    that computes component areas, and it crashed on NumPy 2 until the
    numerics moved into analysis_tools."""

    def test_dialog_opens_with_component_areas(self):
        import tkinter as tk
        self._add_fitted_series(run=False)
        sess = self.app._active_session
        fm = sess.fit_manager
        fm.add_component("Constant")          # Linear + Constant -> component curves
        rec = self.app._active_record
        sess.result = fm.run_fit(rec.x, rec.y, weight_mode="No weights")
        self.assertTrue(sess.result.component_curves)

        before = set(self.app.winfo_children())
        self.app._show_derivative_integral()
        new = [w for w in self.app.winfo_children()
               if w not in before and isinstance(w, tk.Toplevel)]
        self.assertEqual(len(new), 1)
        # First child is the info frame holding the area labels.
        labels = new[0].winfo_children()[0].winfo_children()
        texts = [str(w.cget("text")) for w in labels]
        self.assertTrue(any(t.startswith("Total area") for t in texts))
        self.assertEqual(sum(t.startswith("  ") for t in texts), 2)
        new[0].destroy()


class ModelComparisonComparabilityTests(GuiTestBase):
    """Model Comparison counted fixed and derived parameters as k and ranked
    AICs from fits to different data, weights or objectives; the F-test did
    the same. Rows now carry a comparison key and only comparable rows are
    ranked or tested."""

    def _fit(self, fm, x, y, **kw):
        return fm.run_fit(x, y, weight_mode="No weights", **kw)

    def test_row_counts_varying_parameters_only(self):
        from curvelab.fit_manager import FitManager
        from curvelab.ui_dialogs_analysis import comparison_row
        x = np.linspace(-5, 5, 80)
        y = 3.0 * np.exp(-x ** 2 / 2)
        fm = FitManager()
        fm.add_component("Gaussian")               # amplitude, center, sigma, fwhm, height
        fm.set_param_hint("center", value=0.0, vary=False)
        fm.auto_guess(x, y)
        row = comparison_row("s", fm.model_description(), self._fit(fm, x, y))
        self.assertEqual(row["n_params"], 2)       # amplitude and sigma
        self.assertIsNotNone(row["key"])

    def test_incomparable_rows_are_marked_and_not_ranked(self):
        from curvelab.fit_manager import FitManager, objective_kwargs
        from curvelab.ui_dialogs_analysis import comparison_row, ModelComparisonDialog
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + np.random.default_rng(0).normal(0, 0.1, 50)
        rows = []
        for label, xr, kw in (("full", slice(None), {}),
                              ("half range", slice(0, 25), {}),
                              ("robust", slice(None), {"method": "least_squares",
                                                       "objective_kws": objective_kwargs(
                                                           "least_squares", "Cauchy", 1.0)})):
            fm = FitManager()
            fm.add_component("Linear")
            fm.auto_guess(x[xr], y[xr])
            rows.append(comparison_row(label, "Linear", self._fit(fm, x[xr], y[xr], **kw)))
        self.assertNotEqual(rows[0]["key"], rows[1]["key"])   # different points
        self.assertNotEqual(rows[0]["key"], rows[2]["key"])   # different objective
        self.assertIsNone(rows[2]["aic"])

        dlg = ModelComparisonDialog(self.app, rows)
        try:
            self.assertIn("\u2020 fitted to different data", dlg.note)
            self.assertIn("robust loss", dlg.note)
            tree = next(w for w in dlg.winfo_children()[0].winfo_children()
                        if w.winfo_class() == "Treeview")
            items = [tree.item(i) for i in tree.get_children()]
            self.assertEqual(list(items[0]["tags"]), ["best"])    # only comparable row
            self.assertTrue(items[1]["text"].endswith("\u2020"))
            self.assertNotIn("best", items[1]["tags"])
        finally:
            dlg.destroy()

    def test_batch_summary_does_not_rank_series(self):
        from curvelab.ui_dialogs_analysis import ModelComparisonDialog
        rows = [{"session": "A / fit", "model": "Linear", "n_params": 2, "chisqr": 1.0,
                 "redchi": 0.1, "aic": -10.0, "bic": -9.0, "key": ("a",)},
                {"session": "B / fit", "model": "Linear", "n_params": 2, "chisqr": 1.0,
                 "redchi": 0.1, "aic": -20.0, "bic": -19.0, "key": ("b",)}]
        dlg = ModelComparisonDialog(self.app, rows, highlight_best=False)
        try:
            self.assertIn("not compared across rows", dlg.note)
            tree = next(w for w in dlg.winfo_children()[0].winfo_children()
                        if w.winfo_class() == "Treeview")
            self.assertTrue(all("best" not in tree.item(i)["tags"] for i in tree.get_children()))
        finally:
            dlg.destroy()


class EmceeSettingsTests(GuiTestBase):
    """emcee had no controls: lmfit's defaults (burn 0, no seed) applied and
    the handler set is_weighted from the yerr column. Settings now live on
    the app, are edited in a dialog, reach run_fit as fit_kws, and survive
    a workspace round trip."""

    def test_settings_dialog_applies_and_validates(self):
        from curvelab.ui_dialogs_analysis import EmceeSettingsDialog
        from curvelab.fit_manager import DEFAULT_EMCEE_SETTINGS
        self.assertEqual(self.app._emcee_settings, DEFAULT_EMCEE_SETTINGS)
        applied = {}
        dlg = EmceeSettingsDialog(self.app, dict(self.app._emcee_settings), on_apply=applied.update)
        dlg._vars["burn"].set("5000")                # burn >= steps
        dlg._apply()
        self.assertIn("Burn-in must be smaller", dlg._status_var.get())
        dlg._vars["burn"].set("50"); dlg._vars["steps"].set("400"); dlg._vars["seed"].set("")
        dlg._apply()
        self.assertEqual(applied, {"nwalkers": 100, "steps": 400, "burn": 50, "thin": 1, "seed": None})

    def test_settings_reach_run_fit_and_round_trip(self):
        self._add_fitted_series(run=False)
        self.app._emcee_settings = {"steps": 123, "nwalkers": 10, "burn": 7, "thin": 2, "seed": 3}
        fp = self.app.fit_panel
        fp.method_var.set("emcee"); fp._on_method_changed()
        sess = self.app._active_session
        with mock.patch.object(sess.fit_manager, "run_fit") as run, \
                mock.patch("curvelab.app_fit_handlers.messagebox"):
            self.app._on_fit()
            deadline = 200
            while self.app._fit_thread is not None and deadline:
                self.root.update(); deadline -= 1
        self.assertEqual(run.call_args.kwargs["fit_kws"],
                         {"steps": 123, "nwalkers": 10, "burn": 7, "thin": 2, "seed": 3})
        self.assertNotIn("is_weighted", run.call_args.kwargs["fit_kws"])

        ws = {"plot_controls": {"emcee_settings": {"steps": 55, "seed": None}}}
        self.app._load_workspace_plot_controls(ws)
        self.assertEqual(self.app._emcee_settings["steps"], 55)
        self.assertIsNone(self.app._emcee_settings["seed"])
        self.assertEqual(self.app._emcee_settings["nwalkers"], 100)   # default filled in


class PlotFontTests(GuiTestBase):
    """set_font used get_*ticklabels(), which returns only the ticks in use.
    Matplotlib keeps spare Tick objects from earlier views and reuses them
    when the limits change, so a tick that came back after a font change
    (the 1.0 at the top of the axis) kept the old size."""

    def test_spare_ticks_follow_the_font_size(self):
        pm = self.app.plot_mgr
        pm.ax.set_ylim(0, 3); pm.canvas.draw()        # seven ticks exist
        pm.ax.set_ylim(0, 0.8); pm.canvas.draw()      # five in use, two spare
        pm.set_font("DejaVu Sans", 16)
        pm.ax.set_ylim(0, 1.0); pm.canvas.draw()      # the sixth is a spare
        sizes = {t.get_text(): t.get_fontsize() for t in pm.ax.yaxis.get_ticklabels()}
        self.assertEqual(set(sizes.values()), {16.0}, sizes)
        # And a tick created afterwards.
        pm.ax.set_ylim(0, 5); pm.canvas.draw()
        self.assertEqual({t.get_fontsize() for t in pm.ax.yaxis.get_ticklabels()}, {16.0})

    def test_residual_label_and_annotation_follow_the_font(self):
        pm = self.app.plot_mgr
        pm.set_font("DejaVu Sans", 20)
        self.assertEqual(pm.ax_resid.yaxis.label.get_fontsize(), 18.0)
        pm.plot_residuals("k", np.array([0.0, 1.0]), np.array([0.1, -0.1]), color="C0")
        self.assertEqual(pm.ax_resid.yaxis.label.get_fontsize(), 18.0)


class ContourDialogTests(GuiTestBase):
    """The contour dialog used to fill a sigma grid with no levels and no
    colorbar, so 1/2/3-sigma regions could not be read off. It now draws
    labeled joint-sigma contours over a delta-chi2 map with a colorbar."""

    def test_compute_draws_labeled_contours_and_colorbar(self):
        rec, sess = self._add_fitted_series(run=True)
        self.app._active_series_id = "ds::x::y"
        import tkinter as tk
        before = set(self.app.winfo_children())
        self.app._show_confidence_contours()
        dlg = next(w for w in self.app.winfo_children()
                   if w not in before and isinstance(w, tk.Toplevel))
        try:
            dlg._grid_var.set(5)
            dlg._compute()
            self.assertEqual(dlg._status_var.get(), "Done.")
            self.assertIsNotNone(dlg._colorbar)
            self.assertEqual(dlg._colorbar.ax.get_ylabel(), "\u0394\u03c7\u00b2")
            labels = [t.get_text() for t in dlg._ax.texts]      # clabel texts
            self.assertTrue(any(l.startswith("1") for l in labels), labels)
            self.assertIn("Joint confidence regions", dlg._ax.get_title())
        finally:
            dlg.destroy()


class BadMaxNfevTests(GuiTestBase):
    """A Max nfev that is not a positive whole number used to raise from
    int() after the async path had already flipped the panel into Abort
    state, leaving every fit control disabled with no fit running."""

    def _fit_with_max_nfev(self, text):
        self._add_fitted_series(run=False)
        fp = self.app.fit_panel
        fp.method_var.set("nelder"); fp._on_method_changed()   # async path
        fp.max_nfev_var.set(text)
        with mock.patch("curvelab.app_fit_handlers.messagebox") as mb:
            self.app._on_fit()
        return fp, mb

    def test_non_numeric_max_nfev_warns_and_leaves_fit_button_usable(self):
        fp, mb = self._fit_with_max_nfev("abc")
        mb.showwarning.assert_called_once()
        self.assertEqual(mb.showwarning.call_args[0][0], "Invalid Fit Option")
        self.assertEqual(fp._fit_btn["text"], "Fit")
        self.assertEqual(str(fp._add_comp_btn.cget("state")), "normal")
        self.assertIsNone(self.app._fit_thread)

    def test_zero_max_nfev_is_rejected(self):
        fp, mb = self._fit_with_max_nfev("0")
        mb.showwarning.assert_called_once()
        self.assertEqual(fp._fit_btn["text"], "Fit")

    def test_get_fit_options_raises_for_bad_max_nfev(self):
        self._add_fitted_series(run=False)
        self.app.fit_panel.max_nfev_var.set("12.5")
        with self.assertRaises(ValueError):
            self.app._get_fit_options()
        self.app.fit_panel.max_nfev_var.set(" 250 ")
        self.assertEqual(self.app._get_fit_options().max_nfev, 250)


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
                                "reduce_fcn": "Cauchy log-pdf"}}
        self.app._load_workspace_plot_controls(ws)
        self.assertEqual(self.app.fit_panel.method_var.get(), "nelder")
        self.assertEqual(self.app.fit_panel.objective_var.get(), "Cauchy log-pdf")

    def test_removed_objective_label_falls_back_to_default(self):
        # "Neg. entropy" no longer exists; a workspace that saved it must
        # load with the default objective rather than fail.
        ws = {"plot_controls": {"fit_method": "nelder",
                                "objective": "Neg. entropy"}}
        self.app._load_workspace_plot_controls(ws)
        self.assertEqual(self.app.fit_panel.objective_var.get(), "Chi-square (default)")


class UseResultAsStartTests(GuiTestBase):
    """The 'Use as Start' button seeds the next fit from the current result,
    the opt-in way to chain a refinement now that fits restart from the guess."""

    def test_button_makes_the_next_fit_start_from_the_result(self):
        rec, sess = self._add_fitted_series(run=False)
        fm = sess.fit_manager
        # A deliberately poor start, so the guess and the result differ
        # (the Linear auto-guess would already land on the optimum).
        fm.set_param("slope", value=1.0)
        sess.result = fm.run_fit(rec.x, rec.y, weight_mode="No weights")
        guess_slope = sess.result.init_params["slope"]
        fitted_slope = sess.result.params["slope"]["value"]
        self.assertEqual(guess_slope, 1.0)
        self.assertNotAlmostEqual(guess_slope, fitted_slope, places=3)

        # Without the button the next fit restarts from the guess...
        again = fm.run_fit(rec.x, rec.y, weight_mode="No weights")
        self.assertAlmostEqual(again.init_params["slope"], guess_slope)

        # ...with it, from the fitted result.
        self.app._on_use_result_as_start()
        chained = fm.run_fit(rec.x, rec.y, weight_mode="No weights")
        self.assertAlmostEqual(chained.init_params["slope"], fitted_slope, places=8)

    def test_button_needs_a_result(self):
        rec, sess = self._add_fitted_series(run=False)  # no result yet
        with mock.patch("curvelab.app.messagebox.showwarning") as warn:
            self.app._on_use_result_as_start()
        warn.assert_called_once()


class NoUncertaintyWarningTests(GuiTestBase):
    """A fit that produced no uncertainties (singular covariance) warns the
    user, since the curve can look good while parameters are meaningless."""

    def test_warns_when_errorbars_missing(self):
        rec, sess = self._add_fitted_series(run=False)
        # Fabricate a completed result flagged as having no uncertainties.
        x = np.linspace(0, 1, 10)
        from curvelab.session import FitResult
        sess.result = FitResult(
            x_dense=x, y_fit_dense=x, x_data=x, y_data=x, y_fit_data=x,
            yerr_data=None, params={}, report="", errorbars=False)

        with mock.patch("curvelab.app_fit_handlers.messagebox.showwarning") as warn:
            self.app._post_fit_update(sess, rec, "ds::x::y")

        warn.assert_called_once()
        self.assertIn("uncertaint", warn.call_args[0][1].lower())

    def test_not_converged_and_at_bound_share_one_dialog(self):
        rec, sess = self._add_fitted_series(run=False)
        x = np.linspace(0, 1, 10)
        from curvelab.session import FitResult
        from curvelab.fit_manager import param_info
        sess.result = FitResult(
            x_dense=x, y_fit_dense=x, x_data=x, y_data=x, y_fit_data=x,
            yerr_data=None, report="",
            params={"slope": param_info(1.5, 0.1, 0.0, 1.5, True)},
            success=False, message="Fit aborted: number of function evaluations > 4",
            nfev=4, errorbars=True)
        with mock.patch("curvelab.app_fit_handlers.messagebox.showwarning") as warn:
            self.app._post_fit_update(sess, rec, "ds::x::y")
        warn.assert_called_once()
        text = warn.call_args[0][1]
        self.assertIn("did not converge after 4", text)
        self.assertIn("slope (at max)", text)
        # The table marks the bound in the StdErr column.
        tree = self.app.fit_results.param_tree
        row = tree.item(tree.get_children()[0])
        self.assertIn("(at max)", row["values"][2])

    def test_no_warning_for_normal_fit(self):
        rec, sess = self._add_fitted_series(run=True)  # ordinary Linear fit
        self.assertTrue(sess.result.errorbars)
        with mock.patch("curvelab.app_fit_handlers.messagebox.showwarning") as warn:
            self.app._post_fit_update(sess, rec, "ds::x::y")
        warn.assert_not_called()


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
