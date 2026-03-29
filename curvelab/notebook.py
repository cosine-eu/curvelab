"""ipywidgets-based interactive UI for Jupyter notebooks."""

from __future__ import annotations

import csv
import json
import tempfile
import warnings
from io import StringIO
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.gridspec import GridSpec

from .data_manager import DataManager
from .fit_manager import FitManager, FitResult, REDUCE_FUNCTIONS, WEIGHT_MODES
from .models import MODEL_NAMES
from .preprocessing import prepare_fit_data
from .session import (
    COMPONENT_COLORS, FIT_COLORS, FitSession, ParamEdit, SeriesRecord,
)
from .workspace import (
    WorkspaceEncoder, encode_value, decode_workspace,
    serialize_series_records, deserialize_series_record,
)


def _make_series_id(dataset: str, x_col: str, y_col: str) -> str:
    return f"{dataset}::{x_col}::{y_col}"


def _make_session_key(series_id: str, session_name: str) -> str:
    return f"{series_id}::{session_name}"


# Methods that may be slow — still run synchronously in notebook
_SLOW_METHODS = {"emcee", "differential_evolution", "basinhopping",
                 "dual_annealing", "shgo", "ampgo"}


class CurveLabWidget(widgets.VBox):
    """Interactive curve fitting widget for Jupyter notebooks.

    Usage::

        %matplotlib widget
        from curvelab.notebook import CurveLabWidget
        CurveLabWidget()
    """

    def __init__(self, figsize=(9, 5), show_warnings=False, **kwargs):
        super().__init__(**kwargs)

        # Suppress noisy lmfit/scipy runtime warnings by default
        self._show_warnings = show_warnings
        if not show_warnings:
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning,
                module=r"(lmfit|scipy|uncertainties)\.",
            )

        # Core state
        self._data_mgr = DataManager()
        self._series_records: dict[str, SeriesRecord] = {}
        self._active_series_id: str | None = None
        self._session_counter: int = 0

        # Plot artist tracking
        self._series_artists: dict[str, list] = {}
        self._excluded_artists: dict[str, list] = {}
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

        # Coordinate readout on mouse motion
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)
        self._fig.canvas.mpl_connect("button_press_event", self._on_plot_click)

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
        self._build_analysis_tab()
        self._build_plot_controls()
        self._build_status_bar()

    def _build_data_tab(self):
        # File upload
        self._file_upload = widgets.FileUpload(
            accept=".csv,.tsv,.xlsx,.xls,.ods,.json,.parquet,.h5,.hdf5,.hdf,.sqlite,.db",
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

        # Paste data
        self._paste_text = widgets.Textarea(
            placeholder="Paste tabular data here...",
            layout=widgets.Layout(width="100%", height="60px"),
        )
        self._paste_btn = widgets.Button(
            description="Load Pasted", layout=widgets.Layout(width="100px")
        )
        self._paste_btn.on_click(self._on_paste_data)

        # Dataset dropdown
        self._dataset_dd = widgets.Dropdown(
            description="Dataset:", options=[], layout=widgets.Layout(width="auto")
        )
        self._dataset_dd.observe(self._on_dataset_change, names="value")

        # Remove dataset button
        self._remove_ds_btn = widgets.Button(
            description="Remove Dataset", layout=widgets.Layout(width="120px")
        )
        self._remove_ds_btn.on_click(self._on_remove_dataset)

        # Column dropdowns
        self._x_dd = widgets.Dropdown(description="X:", options=[])
        self._y_dd = widgets.Dropdown(description="Y:", options=[])
        self._yerr_dd = widgets.Dropdown(
            description="Y err:", options=["(none)"], value="(none)"
        )
        self._xerr_dd = widgets.Dropdown(
            description="X err:", options=["(none)"], value="(none)"
        )

        # Plot button
        self._plot_btn = widgets.Button(
            description="Plot", button_style="primary",
            layout=widgets.Layout(width="100%"),
        )
        self._plot_btn.on_click(self._on_plot_click_btn)

        self._data_tab = widgets.VBox([
            widgets.HTML("<b>Load Data</b>"),
            self._file_upload,
            path_row,
            self._paste_text,
            self._paste_btn,
            self._dataset_dd,
            self._remove_ds_btn,
            self._x_dd,
            self._y_dd,
            self._yerr_dd,
            self._xerr_dd,
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
        self._rename_sess_btn = widgets.Button(
            description="Rename", layout=widgets.Layout(width="60px")
        )
        self._rename_sess_btn.on_click(self._on_rename_session)
        sess_btns = widgets.HBox([
            self._new_sess_btn, self._del_sess_btn, self._rename_sess_btn,
        ])

        # Model selection
        self._model_dd = widgets.Dropdown(
            description="Model:", options=MODEL_NAMES, value=MODEL_NAMES[0]
        )
        self._model_dd.observe(self._on_model_change, names="value")

        self._operator_dd = widgets.Dropdown(
            description="Operator:", options=["+", "*", "-", "/"], value="+"
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

        # Fitting method
        self._method_dd = widgets.Dropdown(
            description="Method:",
            options=[
                "leastsq", "least_squares", "nelder", "powell",
                "cobyla", "lbfgsb",
                "differential_evolution", "basinhopping",
                "dual_annealing", "shgo", "ampgo",
                "brute", "emcee",
            ],
            value="leastsq",
        )

        # Reduce function
        self._reduce_dd = widgets.Dropdown(
            description="Reduce:",
            options=list(REDUCE_FUNCTIONS.keys()),
            value="Chi-square (default)",
        )

        # Weight mode
        self._weight_dd = widgets.Dropdown(
            description="Weights:",
            options=WEIGHT_MODES,
            value="1/yerr (default)",
        )

        # Max nfev
        self._max_nfev_text = widgets.Text(
            description="Max nfev:",
            value="",
            placeholder="(unlimited)",
            layout=widgets.Layout(width="200px"),
        )

        # Scale covariance
        self._scale_covar_cb = widgets.Checkbox(
            description="Scale covariance",
            value=True,
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
        self._batch_fit_btn = widgets.Button(
            description="Batch Fit", layout=widgets.Layout(width="90px")
        )
        self._batch_fit_btn.on_click(self._on_batch_fit)
        self._clear_btn = widgets.Button(
            description="Clear", layout=widgets.Layout(width="80px")
        )
        self._clear_btn.on_click(self._on_clear_fit)
        action_row = widgets.HBox([
            self._autoguess_btn, self._fit_btn, self._batch_fit_btn, self._clear_btn,
        ])

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
            self._method_dd,
            self._reduce_dd,
            self._weight_dd,
            self._max_nfev_text,
            self._scale_covar_cb,
            widgets.HTML("<hr style='margin:4px 0'>"),
            action_row,
        ])

    def _build_results_tab(self):
        # GOF summary
        self._gof_html = widgets.HTML(value="")

        # Parameter table (HTML)
        self._param_html = widgets.HTML(value="<i>No fit results yet.</i>")

        # Parameter editor
        self._param_dd = widgets.Dropdown(description="Param:", options=[])
        self._param_value = widgets.FloatText(description="Value:", value=0.0)
        self._param_min = widgets.FloatText(description="Min:", value=float("-inf"))
        self._param_max = widgets.FloatText(description="Max:", value=float("inf"))
        self._param_vary = widgets.Checkbox(description="Vary", value=True)
        self._param_expr = widgets.Text(
            description="Expr:", placeholder="e.g. g1_center * 2",
            layout=widgets.Layout(width="auto"),
        )
        self._param_apply_btn = widgets.Button(
            description="Apply", button_style="info",
            layout=widgets.Layout(width="80px"),
        )
        self._param_apply_btn.on_click(self._on_param_apply)
        self._param_dd.observe(self._on_param_selected, names="value")

        # Undo / Redo
        self._undo_btn = widgets.Button(
            description="Undo", layout=widgets.Layout(width="70px")
        )
        self._undo_btn.on_click(self._on_undo)
        self._redo_btn = widgets.Button(
            description="Redo", layout=widgets.Layout(width="70px")
        )
        self._redo_btn.on_click(self._on_redo)

        param_editor = widgets.VBox([
            widgets.HTML("<b>Edit Parameter</b>"),
            self._param_dd,
            self._param_value,
            widgets.HBox([self._param_min, self._param_max]),
            self._param_expr,
            widgets.HBox([self._param_vary, self._param_apply_btn,
                          self._undo_btn, self._redo_btn]),
        ])

        # Fit report
        self._report_html = widgets.HTML(value="")

        self._results_tab = widgets.VBox([
            self._gof_html,
            self._param_html,
            widgets.HTML("<hr style='margin:4px 0'>"),
            param_editor,
            widgets.HTML("<hr style='margin:4px 0'>"),
            self._report_html,
        ])

    def _build_analysis_tab(self):
        """Build the Analysis tab with buttons for analysis features."""
        btn_layout = widgets.Layout(width="100%")

        self._model_comparison_btn = widgets.Button(
            description="Model Comparison", layout=btn_layout
        )
        self._model_comparison_btn.on_click(self._on_model_comparison)

        self._export_params_btn = widgets.Button(
            description="Export Parameters (CSV)", layout=btn_layout
        )
        self._export_params_btn.on_click(self._on_export_params)

        self._export_report_btn = widgets.Button(
            description="Export Fit Report", layout=btn_layout
        )
        self._export_report_btn.on_click(self._on_export_report)

        self._export_curve_btn = widgets.Button(
            description="Export Curve Data (CSV)", layout=btn_layout
        )
        self._export_curve_btn.on_click(self._on_export_curve)

        self._save_plot_btn = widgets.Button(
            description="Save Plot (PNG)", layout=btn_layout
        )
        self._save_plot_btn.on_click(self._on_save_plot)

        # Workspace
        self._save_ws_path = widgets.Text(
            placeholder="workspace.clw",
            layout=widgets.Layout(width="200px"),
        )
        self._save_ws_btn = widgets.Button(
            description="Save Workspace", layout=widgets.Layout(width="130px")
        )
        self._save_ws_btn.on_click(self._on_save_workspace)
        self._load_ws_btn = widgets.Button(
            description="Load Workspace", layout=widgets.Layout(width="130px")
        )
        self._load_ws_btn.on_click(self._on_load_workspace)

        # Point exclusion
        self._exclude_cb = widgets.Checkbox(
            description="Click to exclude/include points",
            value=False, layout=widgets.Layout(width="auto"),
        )
        self._clear_excl_btn = widgets.Button(
            description="Clear All Exclusions", layout=btn_layout
        )
        self._clear_excl_btn.on_click(self._on_clear_exclusions)

        # Export path for file-based exports
        self._export_path = widgets.Text(
            placeholder="Export file path...",
            layout=widgets.Layout(width="100%"),
        )

        self._analysis_tab = widgets.VBox([
            widgets.HTML("<b>Analysis & Export</b>"),
            self._model_comparison_btn,
            widgets.HTML("<hr style='margin:4px 0'>"),
            widgets.HTML("<b>Export</b>"),
            self._export_path,
            self._export_params_btn,
            self._export_report_btn,
            self._export_curve_btn,
            self._save_plot_btn,
            widgets.HTML("<hr style='margin:4px 0'>"),
            widgets.HTML("<b>Workspace</b>"),
            widgets.HBox([self._save_ws_path, self._save_ws_btn, self._load_ws_btn]),
            widgets.HTML("<hr style='margin:4px 0'>"),
            widgets.HTML("<b>Point Exclusion</b>"),
            self._exclude_cb,
            self._clear_excl_btn,
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

        self._components_cb = widgets.Checkbox(description="Components", value=True,
                                               layout=widgets.Layout(width="auto"))
        self._components_cb.observe(lambda c: self._replot_all(), names="value")

        # Axis scale
        self._xscale_dd = widgets.Dropdown(
            description="X scale:", options=["linear", "log"],
            value="linear", layout=widgets.Layout(width="150px"),
        )
        self._xscale_dd.observe(self._on_scale_change, names="value")
        self._yscale_dd = widgets.Dropdown(
            description="Y scale:", options=["linear", "log"],
            value="linear", layout=widgets.Layout(width="150px"),
        )
        self._yscale_dd.observe(self._on_scale_change, names="value")

        # Axis labels
        self._xlabel = widgets.Text(
            description="X label:", value="",
            layout=widgets.Layout(width="200px"),
        )
        self._xlabel.observe(self._on_axis_labels_change, names="value")
        self._ylabel = widgets.Text(
            description="Y label:", value="",
            layout=widgets.Layout(width="200px"),
        )
        self._ylabel.observe(self._on_axis_labels_change, names="value")

        # Fit range
        self._fit_xmin = widgets.Text(
            description="Fit xmin:", value="",
            placeholder="(auto)", layout=widgets.Layout(width="150px"),
        )
        self._fit_xmax = widgets.Text(
            description="Fit xmax:", value="",
            placeholder="(auto)", layout=widgets.Layout(width="150px"),
        )

        # Band sigma
        self._band_sigma = widgets.IntSlider(
            description="Band σ:", value=1, min=1, max=3,
            layout=widgets.Layout(width="200px"),
        )

        row1 = widgets.HBox([
            self._grid_cb, self._legend_cb, self._resid_cb,
            self._band_cb, self._components_cb,
        ])
        row2 = widgets.HBox([
            self._xscale_dd, self._yscale_dd,
        ])
        row3 = widgets.HBox([self._xlabel, self._ylabel])
        row4 = widgets.HBox([self._fit_xmin, self._fit_xmax, self._band_sigma])

        self._plot_controls_box = widgets.VBox([row1, row2, row3, row4])

    def _build_status_bar(self):
        self._status = widgets.HTML(value="<i>Ready.</i>")
        self._coord_label = widgets.HTML(value="")

    # ------------------------------------------------------------------
    # Layout assembly
    # ------------------------------------------------------------------

    def _assemble_layout(self):
        tabs = widgets.Tab(children=[
            self._data_tab, self._fit_tab, self._results_tab, self._analysis_tab,
        ])
        tabs.set_title(0, "Data")
        tabs.set_title(1, "Fit")
        tabs.set_title(2, "Results")
        tabs.set_title(3, "Analysis")

        left = widgets.VBox(
            [tabs],
            layout=widgets.Layout(
                width="400px", min_width="340px",
                overflow_y="auto",
            ),
        )

        right = widgets.VBox(
            [self._plot_output, self._plot_controls_box,
             widgets.HBox([self._status, self._coord_label])],
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

    def _on_paste_data(self, _btn):
        text = self._paste_text.value.strip()
        if not text:
            self._set_status("Paste some data first.", error=True)
            return
        try:
            ds_name, columns = self._data_mgr.load_from_text(text)
            self._update_dataset_dropdown(select=ds_name)
            self._set_columns(columns)
            self._set_status(f"Loaded pasted data as '{ds_name}'.")
            self._paste_text.value = ""
        except Exception as e:
            self._set_status(f"Paste error: {e}", error=True)

    def _on_dataset_change(self, change):
        name = change["new"]
        if name is None:
            return
        columns = self._data_mgr.column_names(name)
        self._set_columns(columns)

    def _on_remove_dataset(self, _btn):
        name = self._dataset_dd.value
        if not name:
            return
        # Remove any series using this dataset
        to_remove = [sid for sid, rec in self._series_records.items()
                     if rec.dataset_name == name]
        for sid in to_remove:
            for sess_name in list(self._series_records[sid].fit_sessions):
                skey = _make_session_key(sid, sess_name)
                self._clear_fit_artists(skey)
            del self._series_records[sid]
        if self._active_series_id in to_remove:
            self._active_series_id = next(iter(self._series_records), None)

        self._data_mgr.remove_dataset(name)
        self._update_dataset_dropdown()
        self._sync_series_dropdown()
        self._replot_all()
        self._set_status(f"Removed dataset '{name}'.")

    def _update_dataset_dropdown(self, select=None):
        names = self._data_mgr.dataset_names
        self._dataset_dd.options = names
        if select and select in names:
            self._dataset_dd.value = select

    def _set_columns(self, columns: list[str]):
        self._x_dd.options = columns
        self._y_dd.options = columns
        self._yerr_dd.options = ["(none)"] + columns
        self._xerr_dd.options = ["(none)"] + columns
        self._yerr_dd.value = "(none)"
        self._xerr_dd.value = "(none)"
        if len(columns) >= 2:
            self._x_dd.value = columns[0]
            self._y_dd.value = columns[1]

    def _on_plot_click_btn(self, _btn):
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
        xerr_col = self._xerr_dd.value
        xerr_col = xerr_col if xerr_col != "(none)" else None

        try:
            x = self._data_mgr.get_column(dataset, x_col)
            y = self._data_mgr.get_column(dataset, y_col)
            yerr = self._data_mgr.get_column(dataset, yerr_col) if yerr_col else None
            xerr = self._data_mgr.get_column(dataset, xerr_col) if xerr_col else None
        except Exception as e:
            self._set_status(f"Data error: {e}", error=True)
            return

        sid = _make_series_id(dataset, x_col, y_col)

        if sid in self._series_records:
            rec = self._series_records[sid]
            rec.x, rec.y, rec.yerr, rec.xerr = x, y, yerr, xerr
        else:
            rec = SeriesRecord(
                x=x, y=y, yerr=yerr, xerr=xerr,
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

    def _on_rename_session(self, _btn):
        rec = self._active_record
        if rec is None:
            return
        old_name = self._session_select.value
        if old_name is None or old_name not in rec.fit_sessions:
            return
        # Use a simple dialog approach — prompt inline
        new_name = f"{old_name} (renamed)"
        # Try to get a better name — we'll just increment
        base = old_name.rstrip("0123456789 ")
        count = len(rec.fit_sessions) + 1
        new_name = f"{base} {count}"

        if new_name in rec.fit_sessions:
            self._set_status("Name conflict.", error=True)
            return

        sess = rec.fit_sessions.pop(old_name)
        sess.name = new_name
        rec.fit_sessions[new_name] = sess

        old_skey = _make_session_key(self._active_series_id, old_name)
        new_skey = _make_session_key(self._active_series_id, new_name)
        for artists_dict in (self._fit_artists, self._residual_artists,
                             self._confidence_artists):
            if old_skey in artists_dict:
                artists_dict[new_skey] = artists_dict.pop(old_skey)

        if rec.active_session_name == old_name:
            rec.active_session_name = new_name

        self._sync_session_list()
        self._set_status(f"Renamed '{old_name}' to '{new_name}'.")

    # ------------------------------------------------------------------
    # Model / Component callbacks
    # ------------------------------------------------------------------

    def _on_model_change(self, change):
        if change["new"] == "Expression":
            self._expr_text.layout.display = None
        else:
            self._expr_text.layout.display = "none"

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
        """Return (x, y, yerr, xerr) using preprocessing module."""
        x_range = None
        xmin_str = self._fit_xmin.value.strip()
        xmax_str = self._fit_xmax.value.strip()
        if xmin_str or xmax_str:
            try:
                xmin = float(xmin_str) if xmin_str else -np.inf
                xmax = float(xmax_str) if xmax_str else np.inf
                x_range = (xmin, xmax)
            except ValueError:
                pass

        x, y, yerr, xerr, warnings = prepare_fit_data(rec, x_range=x_range)
        for w in warnings:
            self._set_status(w)
        return x, y, yerr, xerr

    def _get_fit_options(self):
        """Read fit options from UI."""
        reduce_fcn = REDUCE_FUNCTIONS.get(self._reduce_dd.value)
        weight_mode = self._weight_dd.value
        max_nfev_str = self._max_nfev_text.value.strip()
        max_nfev = int(max_nfev_str) if max_nfev_str else None
        band_sigma = self._band_sigma.value
        scale_covar = self._scale_covar_cb.value
        return reduce_fcn, weight_mode, max_nfev, band_sigma, scale_covar

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
            x, y, _, _ = self._get_fit_data(rec)
            params = fm.auto_guess(x, y)
            params_info = {}
            for name, par in params.items():
                params_info[name] = {
                    "value": par.value, "stderr": None,
                    "min": par.min, "max": par.max, "vary": par.vary,
                    "expr": par.expr or "",
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

        method = self._method_dd.value

        # Brute validation
        if method == "brute" and fm.params is not None:
            for name, par in fm.params.items():
                if par.vary and (par.min == float("-inf") or par.max == float("inf")):
                    self._set_status(
                        f"Brute requires finite bounds for '{name}'.", error=True
                    )
                    return

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
            self._post_fit_update(sess, rec)
        except Exception as e:
            self._set_status(f"Fit error: {e}", error=True)

    def _post_fit_update(self, sess: FitSession, rec: SeriesRecord):
        """Update UI after a successful fit."""
        result = sess.result
        skey = _make_session_key(self._active_series_id, sess.name)
        label = f"{rec.style.get('label', self._active_series_id)} \u2014 {sess.name}"

        self._clear_fit_artists(skey)
        self._plot_fit_curve(skey, sess, label)
        if self._resid_cb.value:
            self._plot_residuals(skey, sess, rec)
        if self._band_cb.value:
            self._plot_confidence_band(skey, sess)

        self._refresh_param_table(result.params)
        self._update_gof(result)
        self._report_html.value = (
            "<details><summary><b>Fit Report</b></summary>"
            f"<pre style='font-size:11px'>{result.report}</pre></details>"
        )
        self._redraw()
        self._set_status("Fit complete.")

    def _update_gof(self, result: FitResult):
        """Update GOF summary display."""
        gof = result.gof
        parts = []
        for key in ("chi-squared", "reduced chi-squared", "R-squared", "AIC", "BIC"):
            val = gof.get(key)
            if val is not None:
                parts.append(f"<b>{key}:</b> {val:.4g}")
        self._gof_html.value = (
            "<div style='font-size:11px; background:#f8f8f8; padding:4px; "
            "border-radius:3px; margin-bottom:4px'>"
            + " &nbsp;|&nbsp; ".join(parts) + "</div>"
        )

    def _on_batch_fit(self, _btn):
        """Apply the active session's model to all plotted series."""
        rec = self._active_record
        sess = self._active_session
        if rec is None or sess is None:
            self._set_status("Create a fit session first.", error=True)
            return
        source_fm = sess.fit_manager
        if not source_fm.components:
            self._set_status("Add at least one model component.", error=True)
            return

        session_name = sess.name
        method = self._method_dd.value
        reduce_fcn, weight_mode, max_nfev, band_sigma, scale_covar = self._get_fit_options()
        summary_rows = []

        for sid, target_rec in self._series_records.items():
            if session_name not in target_rec.fit_sessions:
                color = FIT_COLORS[len(target_rec.fit_sessions) % len(FIT_COLORS)]
                target_rec.fit_sessions[session_name] = FitSession(
                    name=session_name, color=color,
                )
                if target_rec.active_session_name is None:
                    target_rec.active_session_name = session_name

            target_sess = target_rec.fit_sessions[session_name]
            target_fm = target_sess.fit_manager
            source_fm.clone_components_to(target_fm)

            try:
                x, y, yerr, xerr = self._get_fit_data(target_rec)
                target_fm.auto_guess(x, y)
                result = target_fm.run_fit(
                    x, y, yerr=yerr, xerr=xerr, method=method,
                    reduce_fcn=reduce_fcn, weight_mode=weight_mode,
                    max_nfev=max_nfev, band_sigma=band_sigma,
                    scale_covar=scale_covar,
                )
                target_sess.result = result

                skey = _make_session_key(sid, session_name)
                series_label = target_rec.style.get("label", sid)
                label = f"{series_label} \u2014 {session_name}"
                self._clear_fit_artists(skey)
                self._plot_fit_curve(skey, target_sess, label)
                if self._resid_cb.value:
                    self._plot_residuals(skey, target_sess, target_rec)

                gof = result.gof
                summary_rows.append(
                    f"{series_label}: \u03c7\u00b2r={gof.get('reduced chi-squared', 0):.4g}, "
                    f"AIC={gof.get('AIC', 0):.1f}"
                )
            except Exception as e:
                series_label = target_rec.style.get("label", sid)
                summary_rows.append(f"{series_label}: ERROR - {e}")

        self._sync_session_list()
        self._load_session_into_ui()
        self._redraw()
        self._set_status("Batch fit complete. " + "; ".join(summary_rows))

    def _on_clear_fit(self, _btn):
        sess = self._active_session
        if sess is None:
            return
        skey = self._active_session_key()
        if skey:
            self._clear_fit_artists(skey)
        sess.fit_manager.clear_components()
        sess.result = None
        sess.undo_stack.clear()
        sess.redo_stack.clear()
        self._sync_component_list()
        self._param_html.value = "<i>No fit results yet.</i>"
        self._report_html.value = ""
        self._gof_html.value = ""
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
            stderr = f"\u00b1 {info['stderr']:.4g}" if info.get("stderr") is not None else ""
            lo = f"{info['min']:.4g}" if np.isfinite(info["min"]) else "-inf"
            hi = f"{info['max']:.4g}" if np.isfinite(info["max"]) else "inf"
            vary = "yes" if info["vary"] else "fixed"
            expr = info.get("expr", "")
            expr_col = f"<td>{expr}</td>" if expr else "<td></td>"
            rows.append(
                f"<tr><td><b>{name}</b></td><td>{val}</td>"
                f"<td>{stderr}</td><td>[{lo}, {hi}]</td><td>{vary}</td>{expr_col}</tr>"
            )
        header = (
            "<tr style='background:#eee'><th>Parameter</th><th>Value</th>"
            "<th>Stderr</th><th>Bounds</th><th>Vary</th><th>Expr</th></tr>"
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
        self._param_expr.value = par.expr or ""

    def _on_param_apply(self, _btn):
        fm = self._active_fit_mgr
        sess = self._active_session
        if fm is None or fm.params is None or sess is None:
            return
        name = self._param_dd.value
        if name is None or name not in fm.params:
            return

        par = fm.params[name]
        val = self._param_value.value
        lo = self._param_min.value
        hi = self._param_max.value
        vary = self._param_vary.value
        expr = self._param_expr.value.strip()

        # Treat extreme floats as inf
        if lo <= -1e307:
            lo = float("-inf")
        if hi >= 1e307:
            hi = float("inf")

        # Record undo
        old_values = {
            "value": par.value, "min": par.min, "max": par.max,
            "vary": par.vary, "expr": par.expr or "",
        }

        if expr:
            fm.set_param(name, expr=expr)
        else:
            fm.set_param(name, value=val, min=lo, max=hi, vary=vary, expr="")

        # Push to undo stack
        edit = ParamEdit(param_name=name, field="value",
                         old_value=old_values, new_value=val)
        sess.undo_stack.append(edit)
        sess.redo_stack.clear()

        # Persist as hint
        fm.set_param_hint(name, value=val, min=lo, max=hi, vary=vary)

        self._refresh_param_display()
        self._set_status(f"Updated parameter '{name}'.")

    def _refresh_param_display(self):
        """Refresh the parameter table from live FitManager params."""
        fm = self._active_fit_mgr
        if fm is None or fm.params is None:
            return
        params_info = {}
        for name, par in fm.params.items():
            params_info[name] = {
                "value": par.value,
                "stderr": par.stderr,
                "min": par.min,
                "max": par.max,
                "vary": par.vary,
                "expr": par.expr or "",
            }
        self._refresh_param_table(params_info)

    def _on_undo(self, _btn):
        sess = self._active_session
        fm = self._active_fit_mgr
        if sess is None or fm is None or not sess.undo_stack:
            return
        edit = sess.undo_stack.pop()
        if isinstance(edit.old_value, dict):
            fm.set_param(edit.param_name, **{k: v for k, v in edit.old_value.items()
                                              if k != "expr"})
            if edit.old_value.get("expr"):
                fm.set_param(edit.param_name, expr=edit.old_value["expr"])
        else:
            fm.set_param(edit.param_name, **{edit.field: edit.old_value})
        sess.redo_stack.append(edit)
        self._refresh_param_display()

    def _on_redo(self, _btn):
        sess = self._active_session
        fm = self._active_fit_mgr
        if sess is None or fm is None or not sess.redo_stack:
            return
        edit = sess.redo_stack.pop()
        fm.set_param(edit.param_name, **{edit.field: edit.new_value})
        sess.undo_stack.append(edit)
        self._refresh_param_display()

    # ------------------------------------------------------------------
    # Analysis callbacks
    # ------------------------------------------------------------------

    def _on_model_comparison(self, _btn):
        rec = self._active_record
        if rec is None:
            self._set_status("Select a series first.", error=True)
            return
        rows = []
        for sess_name, sess in rec.fit_sessions.items():
            if sess.result is None:
                continue
            model_desc = " + ".join(
                (f"{c.operator} " if i > 0 else "") + c.name
                for i, c in enumerate(sess.fit_manager.components)
            )
            gof = sess.result.gof
            rows.append(
                f"<tr><td>{sess_name}</td><td>{model_desc}</td>"
                f"<td>{len(sess.result.params)}</td>"
                f"<td>{gof.get('chi-squared', 0):.4g}</td>"
                f"<td>{gof.get('reduced chi-squared', 0):.4g}</td>"
                f"<td>{gof.get('AIC', 0):.2f}</td>"
                f"<td>{gof.get('BIC', 0):.2f}</td></tr>"
            )
        if not rows:
            self._set_status("No completed fits to compare.", error=True)
            return
        header = (
            "<tr style='background:#eee'><th>Session</th><th>Model</th>"
            "<th>Params</th><th>\u03c7\u00b2</th><th>\u03c7\u00b2r</th>"
            "<th>AIC</th><th>BIC</th></tr>"
        )
        html = (
            "<h3>Model Comparison</h3>"
            "<table style='font-size:12px; border-collapse:collapse; width:100%'>"
            f"{header}{''.join(rows)}</table>"
        )
        self._report_html.value = html

    # ------------------------------------------------------------------
    # Export callbacks
    # ------------------------------------------------------------------

    def _on_export_params(self, _btn):
        sess = self._active_session
        if sess is None or sess.result is None:
            self._set_status("Run a fit first.", error=True)
            return
        filepath = self._export_path.value.strip()
        if not filepath:
            filepath = "parameters.csv"
        try:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "value", "stderr", "min", "max", "vary", "expr"])
                for name, info in sess.result.params.items():
                    writer.writerow([
                        name, info["value"], info["stderr"],
                        info["min"], info["max"], info["vary"],
                        info.get("expr", ""),
                    ])
            self._set_status(f"Parameters exported to '{filepath}'.")
        except Exception as e:
            self._set_status(f"Export error: {e}", error=True)

    def _on_export_report(self, _btn):
        sess = self._active_session
        if sess is None or sess.result is None:
            self._set_status("Run a fit first.", error=True)
            return
        filepath = self._export_path.value.strip()
        if not filepath:
            filepath = "fit_report.txt"
        try:
            with open(filepath, "w") as f:
                f.write(sess.result.report)
            self._set_status(f"Report exported to '{filepath}'.")
        except Exception as e:
            self._set_status(f"Export error: {e}", error=True)

    def _on_export_curve(self, _btn):
        sess = self._active_session
        if sess is None or sess.result is None:
            self._set_status("Run a fit first.", error=True)
            return
        filepath = self._export_path.value.strip()
        if not filepath:
            filepath = "curve_data.csv"
        try:
            result = sess.result
            dense_data = {"x": result.x_dense, "y_fit": result.y_fit_dense}
            if result.y_uncertainty is not None:
                dense_data["y_uncertainty"] = result.y_uncertainty
            for comp_name, comp_curve in result.component_curves.items():
                dense_data[f"component_{comp_name.rstrip('_')}"] = comp_curve

            residuals = result.y_data - result.y_fit_data
            resid_data = {
                "x": result.x_data,
                "y_data": result.y_data,
                "y_fit": result.y_fit_data,
                "residuals": residuals,
            }
            if result.yerr_data is not None:
                safe_yerr = np.maximum(np.abs(result.yerr_data), 1e-12)
                resid_data["weighted_residuals"] = residuals / safe_yerr

            df_curve = pd.DataFrame(dense_data)
            df_resid = pd.DataFrame(resid_data)
            sep = "\t" if filepath.endswith(".tsv") else ","

            with open(filepath, "w", newline="") as f:
                f.write("# Fit curve (dense grid)\n")
                df_curve.to_csv(f, sep=sep, index=False)
                f.write("\n# Data points and residuals\n")
                df_resid.to_csv(f, sep=sep, index=False)

            self._set_status(f"Curve data exported to '{filepath}'.")
        except Exception as e:
            self._set_status(f"Export error: {e}", error=True)

    def _on_save_plot(self, _btn):
        filepath = self._export_path.value.strip()
        if not filepath:
            filepath = "plot.png"
        try:
            self._fig.savefig(filepath, dpi=150, bbox_inches="tight")
            self._set_status(f"Plot saved to '{filepath}'.")
        except Exception as e:
            self._set_status(f"Save error: {e}", error=True)

    # ------------------------------------------------------------------
    # Workspace persistence
    # ------------------------------------------------------------------

    def _serialize_workspace(self) -> dict:
        data_filepaths = {
            name: str(path) for name, path in self._data_mgr.filepaths.items()
        }
        series = serialize_series_records(self._series_records)
        return {
            "version": 1,
            "data_filepaths": data_filepaths,
            "table_names": self._data_mgr.table_names,
            "series": series,
            "active_series_id": self._active_series_id,
            "plot_controls": {
                "xscale": self._xscale_dd.value,
                "yscale": self._yscale_dd.value,
                "grid": self._grid_cb.value,
                "legend": self._legend_cb.value,
                "residuals": self._resid_cb.value,
                "confidence_band": self._band_cb.value,
                "fit_xmin": self._fit_xmin.value,
                "fit_xmax": self._fit_xmax.value,
                "fit_method": self._method_dd.value,
                "reduce_fcn": self._reduce_dd.value,
                "weight_mode": self._weight_dd.value,
                "max_nfev": self._max_nfev_text.value,
                "xlabel": self._xlabel.value,
                "ylabel": self._ylabel.value,
            },
        }

    def _on_save_workspace(self, _btn):
        filepath = self._save_ws_path.value.strip()
        if not filepath:
            filepath = "workspace.clw"
        try:
            workspace = self._serialize_workspace()
            with open(filepath, "w") as f:
                json.dump(workspace, f, cls=WorkspaceEncoder, indent=2)
            self._set_status(f"Workspace saved to '{filepath}'.")
        except Exception as e:
            self._set_status(f"Save error: {e}", error=True)

    def _on_load_workspace(self, _btn):
        filepath = self._save_ws_path.value.strip()
        if not filepath:
            self._set_status("Enter workspace file path.", error=True)
            return
        try:
            with open(filepath, "r") as f:
                ws = json.load(f, object_hook=decode_workspace)
        except Exception as e:
            self._set_status(f"Load error: {e}", error=True)
            return

        # Reload datasets
        self._data_mgr = DataManager()
        dataset_name_map = {}
        loaded_sqlite_files: set[str] = set()
        saved_table_names = ws.get("table_names", {})

        for name, fpath in ws.get("data_filepaths", {}).items():
            try:
                fpath_str = str(fpath)
                ext = Path(fpath).suffix.lower()
                if ext in (".sqlite", ".db"):
                    if fpath_str in loaded_sqlite_files:
                        table = saved_table_names.get(name, "")
                        for ds_name, tbl in self._data_mgr.table_names.items():
                            if tbl == table and str(self._data_mgr.filepaths.get(ds_name)) == fpath_str:
                                dataset_name_map[name] = ds_name
                                break
                        continue
                    loaded_sqlite_files.add(fpath_str)
                    self._data_mgr.load(fpath)
                    for old_name, old_fpath in ws.get("data_filepaths", {}).items():
                        if str(old_fpath) == fpath_str:
                            table = saved_table_names.get(old_name, "")
                            for ds_name, tbl in self._data_mgr.table_names.items():
                                if tbl == table and str(self._data_mgr.filepaths.get(ds_name)) == fpath_str:
                                    dataset_name_map[old_name] = ds_name
                                    break
                else:
                    new_name, _ = self._data_mgr.load(fpath)
                    dataset_name_map[name] = new_name
            except Exception:
                pass

        # Rebuild series records
        self._series_records.clear()
        self._fit_artists.clear()
        self._residual_artists.clear()
        self._confidence_artists.clear()

        for sid, sdata in ws.get("series", {}).items():
            parts = sid.split("::")
            if len(parts) < 3:
                continue
            old_ds = parts[0]
            x_col, y_col = parts[1], parts[2]
            ds_name = dataset_name_map.get(old_ds, old_ds)
            try:
                x = self._data_mgr.get_column(ds_name, x_col)
                y = self._data_mgr.get_column(ds_name, y_col)
            except Exception:
                continue
            yerr_col = sdata.get("style", {}).get("yerr_col")
            xerr_col = sdata.get("style", {}).get("xerr_col")
            yerr = self._data_mgr.get_column(ds_name, yerr_col) if yerr_col else None
            xerr = self._data_mgr.get_column(ds_name, xerr_col) if xerr_col else None

            rec = deserialize_series_record(sdata, x, y, yerr, xerr, ds_name)
            new_sid = _make_series_id(ds_name, x_col, y_col)
            self._series_records[new_sid] = rec

        self._active_series_id = ws.get("active_series_id")
        if self._active_series_id not in self._series_records:
            self._active_series_id = next(iter(self._series_records), None)

        # Restore plot controls
        pc = ws.get("plot_controls", {})
        self._xscale_dd.value = pc.get("xscale", "linear")
        self._yscale_dd.value = pc.get("yscale", "linear")
        self._grid_cb.value = pc.get("grid", False)
        self._legend_cb.value = pc.get("legend", True)
        self._resid_cb.value = pc.get("residuals", False)
        self._band_cb.value = pc.get("confidence_band", False)
        self._fit_xmin.value = pc.get("fit_xmin", "")
        self._fit_xmax.value = pc.get("fit_xmax", "")
        self._method_dd.value = pc.get("fit_method", "leastsq")
        self._reduce_dd.value = pc.get("reduce_fcn", "Chi-square (default)")
        self._weight_dd.value = pc.get("weight_mode", "1/yerr (default)")
        self._max_nfev_text.value = pc.get("max_nfev", "")
        self._xlabel.value = pc.get("xlabel", "")
        self._ylabel.value = pc.get("ylabel", "")

        self._update_dataset_dropdown()
        self._sync_series_dropdown()
        self._replot_all()
        self._sync_session_list()
        self._load_session_into_ui()
        self._set_status(f"Workspace loaded from '{filepath}'.")

    # ------------------------------------------------------------------
    # Point exclusion
    # ------------------------------------------------------------------

    def _on_plot_click(self, event):
        """Handle click on plot — toggle point exclusion when enabled."""
        if not self._exclude_cb.value:
            return
        if event.inaxes != self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        best_dist = float("inf")
        best_sid = None
        best_idx = None

        ax = self._ax
        for sid, rec in self._series_records.items():
            if not rec.visible:
                continue
            for i in range(len(rec.x)):
                dx_display = ax.transData.transform((rec.x[i], rec.y[i]))
                click_display = ax.transData.transform((event.xdata, event.ydata))
                dist = ((dx_display[0] - click_display[0]) ** 2 +
                        (dx_display[1] - click_display[1]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_sid = sid
                    best_idx = i

        if best_sid is None or best_dist > 10:
            return

        rec = self._series_records[best_sid]
        if rec.mask is None:
            rec.mask = np.ones(len(rec.x), dtype=bool)
        rec.mask[best_idx] = not rec.mask[best_idx]

        n_excluded = int((~rec.mask).sum())
        self._replot_all()
        self._set_status(f"{n_excluded} point(s) excluded.")

    def _on_clear_exclusions(self, _btn):
        rec = self._active_record
        if rec is None:
            return
        rec.mask = None
        self._replot_all()
        self._set_status("All exclusions cleared.")

    # ------------------------------------------------------------------
    # Mouse motion (coordinate readout)
    # ------------------------------------------------------------------

    def _on_mouse_motion(self, event):
        if event.inaxes is not None and event.xdata is not None:
            self._coord_label.value = (
                f"<span style='font-size:11px; color:#666'>"
                f"x={event.xdata:.6g}  y={event.ydata:.6g}</span>"
            )
        else:
            self._coord_label.value = ""

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_series(self, rec: SeriesRecord, sid: str):
        if not rec.visible:
            return
        label = rec.style.get("label", sid)

        # Apply mask
        x, y = rec.x, rec.y
        yerr = rec.yerr
        xerr = rec.xerr
        mask = rec.mask

        if mask is not None:
            x_inc, y_inc = x[mask], y[mask]
            yerr_inc = yerr[mask] if yerr is not None else None
            xerr_inc = xerr[mask] if xerr is not None else None
            x_exc, y_exc = x[~mask], y[~mask]
        else:
            x_inc, y_inc = x, y
            yerr_inc = yerr
            xerr_inc = xerr
            x_exc, y_exc = np.array([]), np.array([])

        kwargs = {"label": label, "fmt": "o", "markersize": 4, "capsize": 2}
        artists = self._ax.errorbar(
            x_inc, y_inc, yerr=yerr_inc, xerr=xerr_inc, **kwargs
        )
        self._series_artists[sid] = [artists]

        # Plot excluded points as gray
        if len(x_exc) > 0:
            exc_artists = self._ax.plot(
                x_exc, y_exc, "o", color="gray", markersize=3, alpha=0.4,
            )
            self._excluded_artists[sid] = exc_artists

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
        if result.component_curves and self._components_cb.value:
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
        self._excluded_artists.clear()
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
        else:
            self._ax.tick_params(labelbottom=True)

        # Axis scales
        self._ax.set_xscale(self._xscale_dd.value)
        self._ax.set_yscale(self._yscale_dd.value)

        # Axis labels
        if self._xlabel.value:
            self._ax.set_xlabel(self._xlabel.value)
        if self._ylabel.value:
            self._ax.set_ylabel(self._ylabel.value)

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

    def _on_scale_change(self, _change):
        self._ax.set_xscale(self._xscale_dd.value)
        self._ax.set_yscale(self._yscale_dd.value)
        self._redraw()

    def _on_axis_labels_change(self, _change):
        self._ax.set_xlabel(self._xlabel.value)
        self._ax.set_ylabel(self._ylabel.value)
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
            self._gof_html.value = ""
            self._param_dd.options = []
            return

        self._sync_component_list()

        if sess.result is not None:
            self._refresh_param_table(sess.result.params)
            self._update_gof(sess.result)
            self._report_html.value = (
                "<details><summary><b>Fit Report</b></summary>"
                f"<pre style='font-size:11px'>{sess.result.report}</pre></details>"
            )
        else:
            self._param_html.value = "<i>No fit results yet.</i>"
            self._report_html.value = ""
            self._gof_html.value = ""
            self._param_dd.options = []
            # Show current params if model exists
            fm = sess.fit_manager
            if fm.params is not None:
                params_info = {}
                for name, par in fm.params.items():
                    params_info[name] = {
                        "value": par.value, "stderr": None,
                        "min": par.min, "max": par.max, "vary": par.vary,
                        "expr": par.expr or "",
                    }
                self._refresh_param_table(params_info)
