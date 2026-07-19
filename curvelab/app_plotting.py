"""Plot orchestration for CurveLabApp.

Mixed into CurveLabApp; these methods draw series, fit curves, and
residuals through the PlotManager, and handle plot-level interaction
(point exclusion, visibility toggles, coordinate readout). They rely on
the coordinator's state, panels, and refresh helpers, so they are not
usable standalone.
"""

from tkinter import messagebox

import numpy as np

from .plot_manager import SeriesStyle
from .session import (
    FitSession, SeriesRecord,
    make_series_id as _make_series_id,
    make_session_key as _make_session_key,
)


class PlottingMixin:
    """Series/fit/residual drawing and plot interaction."""

    def _iter_visible_results(self):
        """Yield (sid, rec, sess) for every visible session holding a result."""
        for sid, rec in self._series_records.items():
            for sess in rec.fit_sessions.values():
                if sess.result is not None and sess.visible:
                    yield sid, rec, sess

    # --- Residuals / confidence band helpers ---

    def _plot_residuals_for_session(self, skey: str, sess: FitSession, rec: SeriesRecord):
        """Compute and plot residuals for one session."""
        result = sess.result
        if result is None:
            return
        if self.plot_controls.weighted_resid_var.get():
            residuals = result.weighted_residuals()
        else:
            residuals = result.residuals()
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

    def _show_fit_on_plot(self, sid: str, sess: FitSession, rec: SeriesRecord) -> str:
        """Clear and redraw one session's fit curve, plus residuals when the
        residuals panel is shown. Returns the session key of the artists."""
        skey = _make_session_key(sid, sess.name)
        label = f"{rec.style.get('label', sid)} — {sess.name}"
        self.plot_mgr.clear_fit_session(skey)
        self._plot_fit_for_session(skey, sess, label)
        if self.plot_controls.residuals_var.get():
            self._plot_residuals_for_session(skey, sess, rec)
        return skey

    # --- Plot callback ---

    def _on_plot(self, series_list: list[dict]):
        if not self.data_mgr.is_loaded:
            messagebox.showwarning("No Data", "Load a data file first.")
            return

        self.plot_mgr.clear_all()

        if not series_list:
            messagebox.showwarning("No Series", "Add at least one series.")

        new_records: dict[str, SeriesRecord] = {}
        latest_new_sid: str | None = None

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
                    latest_new_sid = sid

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

        for sid, rec, sess in self._iter_visible_results():
            self._show_fit_on_plot(sid, sess, rec)

        # Switch to the most recently added series so new sessions target it
        if latest_new_sid is not None:
            self._active_series_id = latest_new_sid
        elif self._active_series_id not in self._series_records:
            self._active_series_id = next(iter(self._series_records), None)

        self._refresh_series_ui()

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
            for sess in rec.fit_sessions.values():
                if sess.result is not None and sess.visible:
                    self._show_fit_on_plot(sid, sess, rec)

        # Restore residuals visibility state
        self.plot_mgr.set_residuals_visible(show_resid)

        # Respect data visibility toggle
        if not self.plot_controls.data_var.get():
            self.plot_mgr.set_data_visible(False)

    # --- Plot interaction ---

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
        click_display = ax.transData.transform((event.xdata, event.ydata))
        for sid, rec in self._series_records.items():
            if not rec.visible or len(rec.x) == 0:
                continue
            # Transform all of this series' points to display coords in one
            # call instead of once per point -- click_display is constant
            # per click, so it's computed outside both loops.
            pts_display = ax.transData.transform(np.column_stack((rec.x, rec.y)))
            dists = np.hypot(
                pts_display[:, 0] - click_display[0],
                pts_display[:, 1] - click_display[1],
            )
            i = int(np.argmin(dists))
            dist = dists[i]
            if dist < best_dist:
                best_dist = dist
                best_sid = sid
                best_idx = i

        # Only toggle if click is within 10 pixels of a point
        if best_sid is None or best_dist > self._CLICK_HIT_RADIUS_PX:
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
        """Clear point exclusions from the active series."""
        rec = self._active_record
        if rec is None:
            return
        rec.mask = None
        self._replot_all_series()
        self.plot_mgr.canvas.draw()
        self.parent.title("CurveLab")

    def _clear_all_exclusions(self):
        """Clear point exclusions from all series."""
        for rec in self._series_records.values():
            rec.mask = None
        self._replot_all_series()
        self.plot_mgr.canvas.draw()
        self.parent.title("CurveLab")

    def _on_residuals_toggled(self, show: bool):
        self.plot_mgr.set_residuals_visible(show)
        if show:
            # Plot residuals for all sessions that have results
            for sid, rec, sess in self._iter_visible_results():
                skey = _make_session_key(sid, sess.name)
                self._plot_residuals_for_session(skey, sess, rec)
        else:
            self.plot_mgr.clear_all_residuals()

    def _on_weighted_resid_toggled(self, _weighted: bool):
        """Re-plot residuals when weighted/raw toggle changes."""
        if not self.plot_controls.residuals_var.get():
            return
        # Clear and re-plot all residuals
        self.plot_mgr.clear_all_residuals()
        for sid, rec, sess in self._iter_visible_results():
            skey = _make_session_key(sid, sess.name)
            self._plot_residuals_for_session(skey, sess, rec)
        self.plot_mgr.canvas.draw_idle()

    def _on_confidence_band_toggled(self, show: bool):
        xlim = self.plot_mgr.ax.get_xlim()
        ylim = self.plot_mgr.ax.get_ylim()
        self._replot_all_series()
        self.plot_mgr.ax.set_xlim(xlim)
        self.plot_mgr.ax.set_ylim(ylim)
        self.plot_mgr.canvas.draw_idle()

    def _on_axis_labels(self, xlabel: str, ylabel: str):
        self.plot_mgr.set_axis_labels(xlabel, ylabel)

    def _on_title(self, title: str):
        self.plot_mgr.set_title(title)

    def _on_axis_limits(self, xmin: str, xmax: str, ymin: str, ymax: str):
        """Parse the axis-limit entries and pin the plot limits.
        Empty fields mean automatic; unparseable input warns and is ignored."""
        values = []
        for name, text in (("X min", xmin), ("X max", xmax),
                           ("Y min", ymin), ("Y max", ymax)):
            text = text.strip()
            if not text:
                values.append(None)
                continue
            try:
                values.append(float(text))
            except ValueError:
                messagebox.showwarning(
                    "Invalid Axis Limit",
                    f"Could not parse {name} ('{text}') as a number.",
                )
                return
        self.plot_mgr.set_axis_limits(*values)
