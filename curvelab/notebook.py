"""ipywidgets-based interactive UI for Jupyter notebooks."""

from __future__ import annotations

import tempfile
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.gridspec import GridSpec

from .data_manager import DataManager
from .fit_manager import FitManager, FitResult
from .models import MODEL_NAMES
from .session import COMPONENT_COLORS, FIT_COLORS, FitSession, SeriesRecord


def _make_series_id(dataset: str, x_col: str, y_col: str) -> str:
    return f"{dataset}::{x_col}::{y_col}"


def _make_session_key(series_id: str, session_name: str) -> str:
    return f"{series_id}::{session_name}"


class CurveLabWidget(widgets.VBox):
    """Interactive curve fitting widget for Jupyter notebooks.

    Usage::

        %matplotlib widget
        from curvelab.notebook import CurveLabWidget
        CurveLabWidget()
    """

    def __init__(self, figsize=(8, 5), **kwargs):
        super().__init__(**kwargs)

        # Core state
        self._data_mgr = DataManager()
        self._series_records: dict[str, SeriesRecord] = {}
        self._active_series_id: str | None = None
        self._session_counter: int = 0

        # Plot artist tracking
        self._series_artists: list = []
        self._fit_artists: dict[str, list] = {}
        self._residual_artists: dict[str, list] = {}
        self._confidence_artists: dict[str, list] = {}
        self._component_artists: dict[str, list] = {}

        # Build figure
        self._figsize = figsize
        self._build_figure()

        # Build UI
        self._build_widgets()
        self._assemble_layout()

    # ------------------------------------------------------------------
    # Helper properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    def _build_figure(self):
        self._fig, _ = plt.subplots(figsize=self._figsize)
        plt.close(self._fig)  # Prevent duplicate display

        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05, figure=self._fig)
        self._fig.clear()
        self._ax = self._fig.add_subplot(gs[0])
        self._ax_resid = self._fig.add_subplot(gs[1], sharex=self._ax)
        self._ax_resid.set_visible(False)
        self._ax.tick_params(labelbottom=True)

        self._plot_output = widgets.Output()
        with self._plot_output:
            display(self._fig.canvas)

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_widgets(self):
        self._build_data_tab()
        self._build_fit_tab()
        self._build_results_tab()
        self._build_plot_controls()
        self._build_status_bar()

    def _build_data_tab(self):
        # File upload
        self._file_upload = widgets.FileUpload(
            accept=".csv,.tsv,.xlsx,.xls,.json,.parquet",
            multiple=False,
            description="Upload",
        )
        self._file_upload.observe(self._on_file_upload, names="value")

        # Path entry
        self._path_text = widgets.Text(
            placeholder="File path...",
            layout=widgets.Layout(width="240px"),
        )
        self._load_path_btn = widgets.Button(
            description="Load", layout=widgets.Layout(width="60px")
        )
        self._load_path_btn.on_click(self._on_load_path)
        path_row = widgets.HBox([self._path_text, self._load_path_btn])

        # Dataset dropdown
        self._dataset_dd = widgets.Dropdown(
            description="Dataset:", options=[], layout=widgets.Layout(width="auto")
        )
        self._dataset_dd.observe(self._on_dataset_change, names="value")

        # Column dropdowns
        self._x_dd = widgets.Dropdown(description="X:", options=[])
        self._y_dd = widgets.Dropdown(description="Y:", options=[])
        self._yerr_dd = widgets.Dropdown(
            description="Y err:", options=["(none)"], value="(none)"
        )

        # Plot button
        self._plot_btn = widgets.Button(
            description="Plot", button_style="primary",
            layout=widgets.Layout(width="100%"),
        )
        self._plot_btn.on_click(self._on_plot_click)

        self._data_tab = widgets.VBox([
            widgets.HTML("<b>Load Data</b>"),
            self._file_upload,
            path_row,
            self._dataset_dd,
            self._x_dd,
            self._y_dd,
            self._yerr_dd,
            self._plot_btn,
        ])

    def _build_fit_tab(self):
        # Series dropdown
        self._series_dd = widgets.Dropdown(
            description="Series:", options=[]
        )
        self._series_dd.observe(self._on_series_change, names="value")

        # Session management
        self._session_select = widgets.Select(
            description="Sessions:",
            options=[],
            rows=3,
            layout=widgets.Layout(width="auto"),
        )
        self._session_select.observe(self._on_session_change, names="value")

        self._new_sess_btn = widgets.Button(
            description="New", layout=widgets.Layout(width="60px")
        )
        self._new_sess_btn.on_click(self._on_new_session)
        self._del_sess_btn = widgets.Button(
            description="Delete", layout=widgets.Layout(width="60px")
        )
        self._del_sess_btn.on_click(self._on_delete_session)
        sess_btns = widgets.HBox([self._new_sess_btn, self._del_sess_btn])

        # Model selection
        self._model_dd = widgets.Dropdown(
            description="Model:", options=MODEL_NAMES, value=MODEL_NAMES[0]
        )
        self._model_dd.observe(self._on_model_change, names="value")

        self._operator_dd = widgets.Dropdown(
            description="Operator:", options=["+", "*"], value="+"
        )

        self._expr_text = widgets.Text(
            description="Expr:",
            placeholder="e.g. a*exp(-b*x)+c",
            layout=widgets.Layout(display="none"),
        )

        # Component management
        self._add_comp_btn = widgets.Button(
            description="Add Component", layout=widgets.Layout(width="140px")
        )
        self._add_comp_btn.on_click(self._on_add_component)
        self._rm_comp_btn = widgets.Button(
            description="Remove", layout=widgets.Layout(width="80px")
        )
        self._rm_comp_btn.on_click(self._on_remove_component)
        comp_btns = widgets.HBox([self._add_comp_btn, self._rm_comp_btn])

        self._comp_select = widgets.Select(
            description="Components:",
            options=[],
            rows=3,
            layout=widgets.Layout(width="auto"),
        )

        # Action buttons
        self._autoguess_btn = widgets.Button(
            description="Auto Guess", layout=widgets.Layout(width="100px")
        )
        self._autoguess_btn.on_click(self._on_auto_guess)
        self._fit_btn = widgets.Button(
            description="Fit", button_style="success",
            layout=widgets.Layout(width="80px"),
        )
        self._fit_btn.on_click(self._on_fit)
        self._clear_btn = widgets.Button(
            description="Clear", layout=widgets.Layout(width="80px")
        )
        self._clear_btn.on_click(self._on_clear_fit)
        action_row = widgets.HBox([self._autoguess_btn, self._fit_btn, self._clear_btn])

        self._fit_tab = widgets.VBox([
            self._series_dd,
            self._session_select,
            sess_btns,
            widgets.HTML("<hr style='margin:4px 0'>"),
            self._model_dd,
            self._operator_dd,
            self._expr_text,
            comp_btns,
            self._comp_select,
            widgets.HTML("<hr style='margin:4px 0'>"),
            action_row,
        ])

    def _build_results_tab(self):
        # Parameter table (HTML)
        self._param_html = widgets.HTML(value="<i>No fit results yet.</i>")

        # Parameter editor
        self._param_dd = widgets.Dropdown(description="Param:", options=[])
        self._param_value = widgets.FloatText(description="Value:", value=0.0)
        self._param_min = widgets.FloatText(description="Min:", value=float("-inf"))
        self._param_max = widgets.FloatText(description="Max:", value=float("inf"))
        self._param_vary = widgets.Checkbox(description="Vary", value=True)
        self._param_apply_btn = widgets.Button(
            description="Apply", button_style="info",
            layout=widgets.Layout(width="80px"),
        )
        self._param_apply_btn.on_click(self._on_param_apply)
        self._param_dd.observe(self._on_param_selected, names="value")

        param_editor = widgets.VBox([
            widgets.HTML("<b>Edit Parameter</b>"),
            self._param_dd,
            self._param_value,
            widgets.HBox([self._param_min, self._param_max]),
            widgets.HBox([self._param_vary, self._param_apply_btn]),
        ])

        # Fit report
        self._report_html = widgets.HTML(value="")

        self._results_tab = widgets.VBox([
            self._param_html,
            widgets.HTML("<hr style='margin:4px 0'>"),
            param_editor,
            widgets.HTML("<hr style='margin:4px 0'>"),
            self._report_html,
        ])

    def _build_plot_controls(self):
        self._grid_cb = widgets.Checkbox(description="Grid", value=False,
                                         layout=widgets.Layout(width="auto"))
        self._grid_cb.observe(lambda c: self._toggle_grid(c["new"]), names="value")

        self._legend_cb = widgets.Checkbox(description="Legend", value=True,
                                           layout=widgets.Layout(width="auto"))
        self._legend_cb.observe(lambda c: self._toggle_legend(c["new"]), names="value")

        self._resid_cb = widgets.Checkbox(description="Residuals", value=False,
                                          layout=widgets.Layout(width="auto"))
        self._resid_cb.observe(lambda c: self._toggle_residuals(c["new"]), names="value")

        self._band_cb = widgets.Checkbox(description="Conf. Band", value=False,
                                         layout=widgets.Layout(width="auto"))
        self._band_cb.observe(lambda c: self._toggle_confidence_band(c["new"]), names="value")

        self._plot_controls_box = widgets.HBox([
            self._grid_cb, self._legend_cb, self._resid_cb, self._band_cb,
        ])

    def _build_status_bar(self):
        self._status = widgets.HTML(value="<i>Ready.</i>")

    # ------------------------------------------------------------------
    # Layout assembly
    # ------------------------------------------------------------------

    def _assemble_layout(self):
        tabs = widgets.Tab(children=[self._data_tab, self._fit_tab, self._results_tab])
        tabs.set_title(0, "Data")
        tabs.set_title(1, "Fit")
        tabs.set_title(2, "Results")

        left = widgets.VBox(
            [tabs],
            layout=widgets.Layout(
                width="380px", min_width="300px",
                overflow_y="auto",
            ),
        )

        right = widgets.VBox(
            [self._plot_output, self._plot_controls_box, self._status],
            layout=widgets.Layout(flex="1"),
        )

        self.children = [widgets.HBox([left, right])]

    # ------------------------------------------------------------------
    # Status helper
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, error: bool = False):
        color = "red" if error else "inherit"
        self._status.value = f"<span style='color:{color}'>{msg}</span>"

    # ------------------------------------------------------------------
    # Data callbacks
    # ------------------------------------------------------------------

    def _on_file_upload(self, change):
        uploaded = change["new"]
        if not uploaded:
            return
        # FileUpload v2 returns a tuple of dicts
        for item in uploaded:
            name = item.get("name", "upload.csv")
            content = item.get("content", b"")
            suffix = Path(name).suffix or ".csv"
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix="curvelab_"
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                ds_name, columns = self._data_mgr.load(tmp_path)
                self._update_dataset_dropdown(select=ds_name)
                self._set_columns(columns)
                self._set_status(f"Loaded '{name}' as '{ds_name}'.")
            except Exception as e:
                self._set_status(f"Load error: {e}", error=True)
        # Reset uploader so same file can be re-uploaded
        self._file_upload.value = ()

    def _on_load_path(self, _btn):
        path = self._path_text.value.strip()
        if not path:
            self._set_status("Enter a file path.", error=True)
            return
        try:
            ds_name, columns = self._data_mgr.load(path)
            self._update_dataset_dropdown(select=ds_name)
            self._set_columns(columns)
            self._set_status(f"Loaded '{ds_name}'.")
        except Exception as e:
            self._set_status(f"Load error: {e}", error=True)

    def _on_dataset_change(self, change):
        name = change["new"]
        if name is None:
            return
        columns = self._data_mgr.column_names(name)
        self._set_columns(columns)

    def _update_dataset_dropdown(self, select=None):
        names = self._data_mgr.dataset_names
        self._dataset_dd.options = names
        if select and select in names:
            self._dataset_dd.value = select

    def _set_columns(self, columns: list[str]):
        self._x_dd.options = columns
        self._y_dd.options = columns
        self._yerr_dd.options = ["(none)"] + columns
        self._yerr_dd.value = "(none)"
        if len(columns) >= 2:
            self._x_dd.value = columns[0]
            self._y_dd.value = columns[1]

    def _on_plot_click(self, _btn):
        if not self._data_mgr.is_loaded:
            self._set_status("Load a data file first.", error=True)
            return
        dataset = self._dataset_dd.value
        x_col = self._x_dd.value
        y_col = self._y_dd.value
        if not dataset or not x_col or not y_col:
            self._set_status("Select dataset and columns.", error=True)
            return

        yerr_col = self._yerr_dd.value
        yerr_col = yerr_col if yerr_col != "(none)" else None

        try:
            x = self._data_mgr.get_column(dataset, x_col)
            y = self._data_mgr.get_column(dataset, y_col)
            yerr = self._data_mgr.get_column(dataset, yerr_col) if yerr_col else None
        except Exception as e:
            self._set_status(f"Data error: {e}", error=True)
            return

        sid = _make_series_id(dataset, x_col, y_col)

        if sid in self._series_records:
            rec = self._series_records[sid]
            rec.x, rec.y, rec.yerr = x, y, yerr
        else:
            rec = SeriesRecord(
                x=x, y=y, yerr=yerr,
                style={"label": f"{dataset}: {y_col} vs {x_col}"},
                dataset_name=dataset,
            )
            self._series_records[sid] = rec

        if self._active_series_id is None:
            self._active_series_id = sid

        self._replot_all()
        self._sync_series_dropdown()
        self._sync_session_list()
        self._set_status(f"Plotted {y_col} vs {x_col} ({len(x)} points).")

    # ------------------------------------------------------------------
    # Series / Session callbacks
    # ------------------------------------------------------------------

    def _on_series_change(self, change):
        sid = change["new"]
        if sid is None or sid not in self._series_records:
            return
        self._active_series_id = sid
        self._sync_session_list()
        self._load_session_into_ui()

    def _on_session_change(self, change):
        name = change["new"]
        rec = self._active_record
        if rec is None or name is None or name not in rec.fit_sessions:
            return
        rec.active_session_name = name
        self._load_session_into_ui()

    def _on_new_session(self, _btn):
        rec = self._active_record
        if rec is None:
            self._set_status("Plot a series first.", error=True)
            return
        self._session_counter += 1
        name = f"Fit {self._session_counter}"
        color = FIT_COLORS[len(rec.fit_sessions) % len(FIT_COLORS)]
        sess = FitSession(name=name, color=color)
        rec.fit_sessions[name] = sess
        rec.active_session_name = name
        self._sync_session_list()
        self._load_session_into_ui()
        self._set_status(f"Created session '{name}'.")

    def _on_delete_session(self, _btn):
        rec = self._active_record
        if rec is None:
            return
        name = self._session_select.value
        if name is None or name not in rec.fit_sessions:
            return

        skey = _make_session_key(self._active_series_id, name)
        self._clear_fit_artists(skey)
        del rec.fit_sessions[name]

        if rec.active_session_name == name:
            rec.active_session_name = next(iter(rec.fit_sessions), None)

        self._sync_session_list()
        self._load_session_into_ui()
        self._redraw()
        self._set_status(f"Deleted session '{name}'.")

    # ------------------------------------------------------------------
    # Model / Component callbacks
    # ------------------------------------------------------------------

    def _on_model_change(self, change):
        if change["new"] == "Expression":
            self._expr_text.layout.display = None  # show
        else:
            self._expr_text.layout.display = "none"  # hide

    def _on_add_component(self, _btn):
        fm = self._active_fit_mgr
        if fm is None:
            self._set_status("Create a fit session first.", error=True)
            return
        model_name = self._model_dd.value
        operator = self._operator_dd.value
        expression = self._expr_text.value if model_name == "Expression" else ""
        fm.add_component(model_name, operator=operator, expression=expression)
        self._sync_component_list()
        self._set_status(f"Added {model_name} component.")

    def _on_remove_component(self, _btn):
        fm = self._active_fit_mgr
        if fm is None:
            return
        idx = self._comp_select.index
        if idx is None:
            return
        fm.remove_component(idx)
        self._sync_component_list()
        self._set_status("Removed component.")

    # ------------------------------------------------------------------
    # Fitting callbacks
    # ------------------------------------------------------------------

    def _get_fit_data(self, rec: SeriesRecord):
        """Return (x, y, yerr) cleaned of NaN/inf and sorted by x."""
        x = rec.x.copy()
        y = rec.y.copy()
        yerr = rec.yerr.copy() if rec.yerr is not None else None

        # Filter NaN / inf
        finite_mask = np.isfinite(x) & np.isfinite(y)
        if yerr is not None:
            finite_mask &= np.isfinite(yerr)
        n_dropped = int((~finite_mask).sum())
        if n_dropped > 0:
            x, y = x[finite_mask], y[finite_mask]
            yerr = yerr[finite_mask] if yerr is not None else None

        # Sort by x
        order = np.argsort(x)
        x, y = x[order], y[order]
        yerr = yerr[order] if yerr is not None else None

        return x, y, yerr

    def _on_auto_guess(self, _btn):
        rec = self._active_record
        fm = self._active_fit_mgr
        if rec is None or fm is None:
            self._set_status("Create a fit session first.", error=True)
            return
        if not fm.components:
            self._set_status("Add at least one model component.", error=True)
            return
        try:
            x, y, _ = self._get_fit_data(rec)
            params = fm.auto_guess(x, y)
            params_info = {}
            for name, par in params.items():
                params_info[name] = {
                    "value": par.value, "stderr": None,
                    "min": par.min, "max": par.max, "vary": par.vary,
                }
            self._refresh_param_table(params_info)
            self._set_status("Parameters auto-guessed.")
        except Exception as e:
            self._set_status(f"Guess error: {e}", error=True)

    def _on_fit(self, _btn):
        rec = self._active_record
        sess = self._active_session
        if rec is None or sess is None:
            self._set_status("Create a fit session first.", error=True)
            return
        fm = sess.fit_manager
        if not fm.components:
            self._set_status("Add at least one model component.", error=True)
            return

        try:
            x, y, yerr = self._get_fit_data(rec)
            result = fm.run_fit(x, y, yerr=yerr)
            sess.result = result

            skey = _make_session_key(self._active_series_id, sess.name)
            label = f"{rec.style.get('label', self._active_series_id)} \u2014 {sess.name}"

            self._clear_fit_artists(skey)
            self._plot_fit_curve(skey, sess, label)
            if self._resid_cb.value:
                self._plot_residuals(skey, sess, rec)
            if self._band_cb.value:
                self._plot_confidence_band(skey, sess)

            self._refresh_param_table(result.params)
            self._report_html.value = (
                "<details><summary><b>Fit Report</b></summary>"
                f"<pre style='font-size:11px'>{result.report}</pre></details>"
            )
            self._redraw()
            self._set_status("Fit complete.")
        except Exception as e:
            self._set_status(f"Fit error: {e}", error=True)

    def _on_clear_fit(self, _btn):
        sess = self._active_session
        if sess is None:
            return
        skey = self._active_session_key()
        if skey:
            self._clear_fit_artists(skey)
        sess.fit_manager.clear_components()
        sess.result = None
        self._sync_component_list()
        self._param_html.value = "<i>No fit results yet.</i>"
        self._report_html.value = ""
        self._param_dd.options = []
        self._redraw()
        self._set_status("Fit cleared.")

    # ------------------------------------------------------------------
    # Results / Parameter editing
    # ------------------------------------------------------------------

    def _refresh_param_table(self, params: dict[str, dict]):
        self._param_html.value = self._render_param_html(params)
        names = list(params.keys())
        self._param_dd.options = names
        if names:
            self._param_dd.value = names[0]
            self._on_param_selected({"new": names[0]})

    def _render_param_html(self, params: dict[str, dict]) -> str:
        rows = []
        for name, info in params.items():
            val = f"{info['value']:.6g}"
            stderr = f"\u00b1 {info['stderr']:.4g}" if info["stderr"] is not None else ""
            lo = f"{info['min']:.4g}" if np.isfinite(info["min"]) else "-inf"
            hi = f"{info['max']:.4g}" if np.isfinite(info["max"]) else "inf"
            vary = "yes" if info["vary"] else "fixed"
            rows.append(
                f"<tr><td><b>{name}</b></td><td>{val}</td>"
                f"<td>{stderr}</td><td>[{lo}, {hi}]</td><td>{vary}</td></tr>"
            )
        header = (
            "<tr style='background:#eee'><th>Parameter</th><th>Value</th>"
            "<th>Stderr</th><th>Bounds</th><th>Vary</th></tr>"
        )
        return (
            "<table style='font-size:12px; border-collapse:collapse; width:100%'>"
            f"{header}{''.join(rows)}</table>"
        )

    def _on_param_selected(self, change):
        name = change["new"]
        if name is None:
            return
        fm = self._active_fit_mgr
        if fm is None or fm.params is None or name not in fm.params:
            return
        par = fm.params[name]
        self._param_value.value = par.value
        self._param_min.value = par.min if np.isfinite(par.min) else -1e308
        self._param_max.value = par.max if np.isfinite(par.max) else 1e308
        self._param_vary.value = par.vary

    def _on_param_apply(self, _btn):
        fm = self._active_fit_mgr
        if fm is None or fm.params is None:
            return
        name = self._param_dd.value
        if name is None or name not in fm.params:
            return
        val = self._param_value.value
        lo = self._param_min.value
        hi = self._param_max.value
        vary = self._param_vary.value
        # Treat extreme floats as inf
        if lo <= -1e307:
            lo = float("-inf")
        if hi >= 1e307:
            hi = float("inf")
        fm.set_param(name, value=val, min=lo, max=hi, vary=vary)
        # Refresh display from live params
        params_info = {}
        for pname, par in fm.params.items():
            params_info[pname] = {
                "value": par.value, "stderr": par.stderr,
                "min": par.min, "max": par.max, "vary": par.vary,
            }
        self._refresh_param_table(params_info)
        self._set_status(f"Updated parameter '{name}'.")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_series(self, rec: SeriesRecord, sid: str):
        label = rec.style.get("label", sid)
        kwargs = {"label": label, "fmt": "o", "markersize": 4, "capsize": 2}
        if rec.yerr is not None:
            artists = self._ax.errorbar(rec.x, rec.y, yerr=rec.yerr, **kwargs)
        else:
            artists = self._ax.errorbar(rec.x, rec.y, **kwargs)
        # errorbar returns an ErrorbarContainer; store it as a list
        self._series_artists.append(artists)

    def _plot_fit_curve(self, skey: str, sess: FitSession, label: str):
        result = sess.result
        if result is None:
            return
        artists = []
        line, = self._ax.plot(
            result.x_dense, result.y_fit_dense,
            color=sess.color, linewidth=1.5, label=label,
        )
        artists.append(line)

        # Component curves
        if result.component_curves:
            for i, (comp_key, curve) in enumerate(result.component_curves.items()):
                c_color = COMPONENT_COLORS[i % len(COMPONENT_COLORS)]
                cline, = self._ax.plot(
                    result.x_dense, curve, "--",
                    color=c_color, linewidth=1, alpha=0.7,
                )
                artists.append(cline)

        self._fit_artists[skey] = artists

    def _plot_residuals(self, skey: str, sess: FitSession, rec: SeriesRecord):
        result = sess.result
        if result is None:
            return
        residuals = result.y_data - result.y_fit_data
        if result.yerr_data is not None:
            residuals = residuals / result.yerr_data
        artists = []
        line, = self._ax_resid.plot(
            result.x_data, residuals, "o",
            color=sess.color, markersize=3,
        )
        artists.append(line)
        self._residual_artists[skey] = artists

    def _plot_confidence_band(self, skey: str, sess: FitSession):
        result = sess.result
        if result is None or result.y_uncertainty is None:
            return
        fill = self._ax.fill_between(
            result.x_dense,
            result.y_fit_dense - result.y_uncertainty,
            result.y_fit_dense + result.y_uncertainty,
            color=sess.color, alpha=0.15,
        )
        self._confidence_artists.setdefault(skey, []).append(fill)

    def _clear_fit_artists(self, skey: str):
        for artists_dict in (
            self._fit_artists,
            self._residual_artists,
            self._confidence_artists,
        ):
            for artist in artists_dict.pop(skey, []):
                artist.remove()

    def _replot_all(self):
        """Clear and redraw everything."""
        self._ax.cla()
        self._ax_resid.cla()
        self._series_artists.clear()
        self._fit_artists.clear()
        self._residual_artists.clear()
        self._confidence_artists.clear()

        show_resid = self._resid_cb.value
        show_band = self._band_cb.value

        for sid, rec in self._series_records.items():
            self._plot_series(rec, sid)
            for sess_name, sess in rec.fit_sessions.items():
                if sess.result is not None and sess.visible:
                    skey = _make_session_key(sid, sess_name)
                    label = f"{rec.style.get('label', sid)} \u2014 {sess_name}"
                    self._plot_fit_curve(skey, sess, label)
                    if show_resid:
                        self._plot_residuals(skey, sess, rec)
                    if show_band:
                        self._plot_confidence_band(skey, sess)

        self._ax_resid.set_visible(show_resid)
        if show_resid:
            self._ax_resid.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            self._ax_resid.set_ylabel("Residuals")
            self._ax.tick_params(labelbottom=not show_resid)

        if self._grid_cb.value:
            self._ax.grid(True, alpha=0.3)
        if self._legend_cb.value and self._ax.get_legend_handles_labels()[1]:
            self._ax.legend(fontsize=8)

        self._fig.tight_layout()
        self._redraw()

    def _redraw(self):
        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Plot toggle callbacks
    # ------------------------------------------------------------------

    def _toggle_grid(self, show: bool):
        self._ax.grid(show, alpha=0.3)
        if not show:
            self._ax.grid(False)
        self._redraw()

    def _toggle_legend(self, show: bool):
        if show:
            handles, labels = self._ax.get_legend_handles_labels()
            if labels:
                self._ax.legend(fontsize=8)
        else:
            leg = self._ax.get_legend()
            if leg:
                leg.remove()
        self._redraw()

    def _toggle_residuals(self, show: bool):
        if show:
            # Plot residuals for all sessions with results
            for sid, rec in self._series_records.items():
                for sess_name, sess in rec.fit_sessions.items():
                    if sess.result is not None and sess.visible:
                        skey = _make_session_key(sid, sess_name)
                        if skey not in self._residual_artists:
                            self._plot_residuals(skey, sess, rec)
            self._ax_resid.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            self._ax_resid.set_ylabel("Residuals")
        else:
            for artists in self._residual_artists.values():
                for a in artists:
                    a.remove()
            self._residual_artists.clear()
            self._ax_resid.cla()

        self._ax_resid.set_visible(show)
        self._ax.tick_params(labelbottom=not show)
        self._fig.tight_layout()
        self._redraw()

    def _toggle_confidence_band(self, show: bool):
        if show:
            for sid, rec in self._series_records.items():
                for sess_name, sess in rec.fit_sessions.items():
                    if sess.result is not None and sess.visible:
                        skey = _make_session_key(sid, sess_name)
                        if skey not in self._confidence_artists:
                            self._plot_confidence_band(skey, sess)
        else:
            for artists in self._confidence_artists.values():
                for a in artists:
                    a.remove()
            self._confidence_artists.clear()
        self._redraw()

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def _sync_series_dropdown(self):
        ids = list(self._series_records.keys())
        self._series_dd.options = ids
        if self._active_series_id and self._active_series_id in ids:
            self._series_dd.value = self._active_series_id

    def _sync_session_list(self):
        rec = self._active_record
        if rec is None:
            self._session_select.options = []
            return
        names = list(rec.fit_sessions.keys())
        self._session_select.options = names
        if rec.active_session_name and rec.active_session_name in names:
            self._session_select.value = rec.active_session_name

    def _sync_component_list(self):
        fm = self._active_fit_mgr
        if fm is None:
            self._comp_select.options = []
            return
        labels = []
        for i, c in enumerate(fm.components):
            if c.name == "Expression" and c.expression:
                display_str = f"Expression: {c.expression}"
                if c.prefix:
                    display_str = f"{display_str} ({c.prefix})"
            else:
                display_str = f"{c.name} ({c.prefix})" if c.prefix else c.name
            if i > 0:
                display_str = f"{c.operator} {display_str}"
            labels.append(display_str)
        self._comp_select.options = labels

    def _load_session_into_ui(self):
        sess = self._active_session
        if sess is None:
            self._comp_select.options = []
            self._param_html.value = "<i>No fit results yet.</i>"
            self._report_html.value = ""
            self._param_dd.options = []
            return

        self._sync_component_list()

        if sess.result is not None:
            self._refresh_param_table(sess.result.params)
            self._report_html.value = (
                "<details><summary><b>Fit Report</b></summary>"
                f"<pre style='font-size:11px'>{sess.result.report}</pre></details>"
            )
        else:
            self._param_html.value = "<i>No fit results yet.</i>"
            self._report_html.value = ""
            self._param_dd.options = []
            # Show current params if model exists
            fm = sess.fit_manager
            if fm.params is not None:
                params_info = {}
                for name, par in fm.params.items():
                    params_info[name] = {
                        "value": par.value, "stderr": None,
                        "min": par.min, "max": par.max, "vary": par.vary,
                    }
                self._refresh_param_table(params_info)
