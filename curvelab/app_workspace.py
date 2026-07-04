"""Workspace persistence and export handlers for CurveLabApp.

Mixed into CurveLabApp; these methods rely on the coordinator's managers,
panels, and sync helpers, so they are not usable standalone.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tkinter import messagebox, filedialog

from .data_manager import DataManager
from .fit_manager import MIN_ERROR
from .workspace import (
    WorkspaceEncoder, decode_workspace,
    serialize_series_records, deserialize_series_record,
)


class WorkspaceMixin:
    """Save/load .clw workspaces and export parameters, reports, curves, plots."""

    # --- Export ---

    def _export_params(self):
        sess = self._require_fit_result()
        if sess is None:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            title="Export Parameters",
        )
        if not filepath:
            return
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "value", "stderr", "min", "max", "vary"])
            for name, info in sess.result.params.items():
                writer.writerow([
                    name, info["value"], info["stderr"],
                    info["min"], info["max"], info["vary"],
                ])

    def _export_report(self):
        sess = self._require_fit_result()
        if sess is None:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")],
            title="Export Fit Report",
        )
        if not filepath:
            return
        with open(filepath, "w") as f:
            f.write(sess.result.report)

    def _export_curve_data(self):
        """Export fit curve, residuals, and component curves as CSV."""
        sess = self._require_fit_result()
        if sess is None:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("TSV", "*.tsv"), ("All files", "*.*")],
            title="Export Curve Data",
        )
        if not filepath:
            return

        result = sess.result

        # Build dense curve data (fit curve + components)
        dense_data = {"x": result.x_dense, "y_fit": result.y_fit_dense}
        if result.y_uncertainty is not None:
            dense_data["y_uncertainty"] = result.y_uncertainty
        for comp_name, comp_curve in result.component_curves.items():
            dense_data[f"component_{comp_name.rstrip('_')}"] = comp_curve

        # Build data-point residuals
        residuals = result.y_data - result.y_fit_data
        weighted_residuals = None
        if result.yerr_data is not None:
            safe_yerr = np.maximum(np.abs(result.yerr_data), MIN_ERROR)
            weighted_residuals = residuals / safe_yerr

        try:
            # Sheet 1: dense fit curve
            df_curve = pd.DataFrame(dense_data)

            # Sheet 2: data-point residuals
            resid_data = {
                "x": result.x_data,
                "y_data": result.y_data,
                "y_fit": result.y_fit_data,
                "residuals": residuals,
            }
            if weighted_residuals is not None:
                resid_data["weighted_residuals"] = weighted_residuals
            df_resid = pd.DataFrame(resid_data)

            if filepath.endswith(".tsv"):
                sep = "\t"
            else:
                sep = ","

            # Write both tables separated by a blank line
            with open(filepath, "w", newline="") as f:
                f.write("# Fit curve (dense grid)\n")
                df_curve.to_csv(f, sep=sep, index=False)
                f.write("\n# Data points and residuals\n")
                df_resid.to_csv(f, sep=sep, index=False)

        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _save_plot(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"), ("PDF", "*.pdf"),
                ("SVG", "*.svg"), ("All files", "*.*"),
            ],
            title="Save Plot",
        )
        if not filepath:
            return
        try:
            self.plot_mgr.fig.savefig(filepath, dpi=150, bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    # --- Workspace persistence ---

    def _serialize_workspace(self) -> dict:
        """Build a workspace dict from current app state."""
        data_filepaths = {
            name: str(path) for name, path in self.data_mgr.filepaths.items()
        }

        series = serialize_series_records(self._series_records)

        return {
            "version": 1,
            "data_filepaths": data_filepaths,
            "table_names": self.data_mgr.table_names,
            "series": series,
            "active_series_id": self._active_series_id,
            "plot_controls": {
                "xscale": self.plot_controls.xscale_var.get(),
                "yscale": self.plot_controls.yscale_var.get(),
                "grid": self.plot_controls.grid_var.get(),
                "equal": self.plot_controls.equal_var.get(),
                "legend": self.plot_controls.legend_var.get(),
                "show_params": self.plot_controls.show_params_var.get(),
                "fit_visible": self.plot_controls.fit_visible_var.get(),
                "fit_xmin": self.plot_controls.fit_xmin_var.get(),
                "fit_xmax": self.plot_controls.fit_xmax_var.get(),
                "residuals": self.plot_controls.residuals_var.get(),
                "confidence_band": self.plot_controls.confidence_band_var.get(),
                "fit_method": self.fit_panel.method_var.get(),
                "reduce_fcn": self.fit_panel.reduce_var.get(),
                "weight_mode": self.fit_panel.weight_var.get(),
                "max_nfev": self.fit_panel.max_nfev_var.get(),
                "xlabel": self.plot_controls.xlabel_var.get(),
                "ylabel": self.plot_controls.ylabel_var.get(),
            },
            "fonts": {
                "ui_family": self._ui_family,
                "ui_size": self._ui_size,
                "plot_family": self._plot_family,
                "plot_size": self._plot_size,
            },
        }

    def _save_workspace(self, event=None):
        """Save current workspace to a .clw JSON file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".clw",
            filetypes=[("CurveLab Workspace", "*.clw"), ("All files", "*.*")],
            title="Save Workspace",
        )
        if not filepath:
            return
        try:
            workspace = self._serialize_workspace()
            with open(filepath, "w") as f:
                json.dump(workspace, f, cls=WorkspaceEncoder, indent=2)
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _load_workspace(self, event=None):
        """Load a workspace from a .clw JSON file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("CurveLab Workspace", "*.clw"), ("All files", "*.*")],
            title="Load Workspace",
        )
        if not filepath:
            return
        try:
            with open(filepath, "r") as f:
                ws = json.load(f, object_hook=decode_workspace)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return

        dataset_name_map = self._load_workspace_datasets(ws)
        self._load_workspace_series(ws, dataset_name_map)

        # Restore active series
        self._active_series_id = ws.get("active_series_id")
        if self._active_series_id not in self._series_records:
            self._active_series_id = next(iter(self._series_records), None)

        self._load_workspace_plot_controls(ws)
        self._load_workspace_fonts(ws)
        self._load_workspace_refit()

        # Replot everything and sync UI
        self._replot_all_series()
        self._sync_series_combo()
        self._sync_session_list()
        self._load_session_into_ui()

    def _load_workspace_datasets(self, ws) -> dict:
        """Reload datasets from saved filepaths; return {old name -> new name}."""
        self.data_mgr = DataManager()
        dataset_name_map = {}  # old name -> new name (may differ on reload)
        loaded_sqlite_files: set[str] = set()  # avoid loading same .sqlite/.db twice
        saved_table_names = ws.get("table_names", {})
        for name, fpath in ws.get("data_filepaths", {}).items():
            try:
                fpath_str = str(fpath)
                ext = Path(fpath).suffix.lower()
                if ext in (".sqlite", ".db"):
                    if fpath_str in loaded_sqlite_files:
                        # Already loaded — find the matching dataset by table name
                        table = saved_table_names.get(name, "")
                        for ds_name, tbl in self.data_mgr.table_names.items():
                            if tbl == table and str(self.data_mgr.filepaths.get(ds_name)) == fpath_str:
                                dataset_name_map[name] = ds_name
                                break
                        continue
                    loaded_sqlite_files.add(fpath_str)
                    self.data_mgr.load(fpath)
                    # Map all old names for this file to their new dataset names
                    for old_name, old_fpath in ws.get("data_filepaths", {}).items():
                        if str(old_fpath) == fpath_str:
                            table = saved_table_names.get(old_name, "")
                            for ds_name, tbl in self.data_mgr.table_names.items():
                                if tbl == table and str(self.data_mgr.filepaths.get(ds_name)) == fpath_str:
                                    dataset_name_map[old_name] = ds_name
                                    break
                else:
                    new_name, columns = self.data_mgr.load(fpath)
                    dataset_name_map[name] = new_name
            except Exception as e:
                messagebox.showwarning(
                    "Missing File",
                    f"Could not reload '{fpath}':\n{e}\n\nSeries from this dataset will be skipped.",
                )

        self.data_panel.set_datasets(self.data_mgr.dataset_names)
        if self.data_mgr.dataset_names:
            first = self.data_mgr.dataset_names[0]
            self.data_panel.set_columns(self.data_mgr.column_names(first))
        return dataset_name_map

    def _load_workspace_series(self, ws, dataset_name_map):
        """Reconstruct SeriesRecord objects and populate the DataPanel series list."""
        self._series_records.clear()
        self._simulated_counter = 0
        self.data_panel.clear_series_entries()
        for sid, sdata in ws.get("series", {}).items():
            ds_name = sdata.get("dataset_name", "")
            actual_ds = dataset_name_map.get(ds_name)
            if actual_ds is None:
                continue

            style = sdata.get("style", {})
            try:
                x = self.data_mgr.get_column(actual_ds, style.get("x", ""))
                y = self.data_mgr.get_column(actual_ds, style.get("y", ""))
                yerr = self.data_mgr.get_column(actual_ds, style.get("yerr")) if style.get("yerr") else None
                xerr = self.data_mgr.get_column(actual_ds, style.get("xerr")) if style.get("xerr") else None
            except Exception:
                continue

            rec = deserialize_series_record(sdata, x, y, yerr, xerr, actual_ds)
            self._series_records[sid] = rec

            # Add to DataPanel so _on_plot sees it
            self.data_panel.add_series_entry(style)

    def _load_workspace_plot_controls(self, ws):
        """Restore plot-control widget values and apply them to the plot."""
        pc = ws.get("plot_controls", {})
        self.plot_controls.xscale_var.set(pc.get("xscale", "linear"))
        self.plot_controls.yscale_var.set(pc.get("yscale", "linear"))
        self.plot_controls.grid_var.set(pc.get("grid", True))
        self.plot_controls.equal_var.set(pc.get("equal", False))
        self.plot_controls.legend_var.set(pc.get("legend", True))
        self.plot_controls.show_params_var.set(pc.get("show_params", False))
        self.plot_controls.fit_visible_var.set(pc.get("fit_visible", False))
        self.plot_controls.fit_xmin_var.set(pc.get("fit_xmin", ""))
        self.plot_controls.fit_xmax_var.set(pc.get("fit_xmax", ""))
        self.plot_controls.residuals_var.set(pc.get("residuals", False))
        self.plot_controls.confidence_band_var.set(pc.get("confidence_band", False))
        self.fit_panel.method_var.set(pc.get("fit_method", "least_squares"))
        self.fit_panel.reduce_var.set(pc.get("reduce_fcn", "Chi-square (default)"))
        self.fit_panel.weight_var.set(pc.get("weight_mode", "1/yerr (default)"))
        self.fit_panel.max_nfev_var.set(pc.get("max_nfev", ""))
        self.plot_controls.xlabel_var.set(pc.get("xlabel", ""))
        self.plot_controls.ylabel_var.set(pc.get("ylabel", ""))

        # Apply plot control states
        self.plot_mgr.set_xscale(pc.get("xscale", "linear"))
        self.plot_mgr.set_yscale(pc.get("yscale", "linear"))
        self.plot_mgr.set_grid(pc.get("grid", True))
        self.plot_mgr.set_equal_aspect(pc.get("equal", False))
        self.plot_mgr.set_legend(pc.get("legend", True))
        self.plot_mgr.set_axis_labels(pc.get("xlabel", ""), pc.get("ylabel", ""))

    def _load_workspace_fonts(self, ws):
        """Restore saved UI/plot fonts."""
        fonts = ws.get("fonts", {})
        if fonts:
            self._apply_fonts(
                ui_family=fonts.get("ui_family", self._ui_family),
                ui_size=fonts.get("ui_size", self._ui_size),
                plot_family=fonts.get("plot_family", self._plot_family),
                plot_size=fonts.get("plot_size", self._plot_size),
            )

    def _load_workspace_refit(self):
        """Reconstruct lmfit ModelResult for each session so analysis tools work."""
        for rec in self._series_records.values():
            for sess in rec.fit_sessions.values():
                if sess.result is not None and sess.fit_manager.model is not None:
                    try:
                        sess.fit_manager.refit_from_result(sess.result)
                    except Exception:
                        pass  # Analysis tools will show "run a fit first"
