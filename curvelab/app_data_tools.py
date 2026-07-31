"""Data-tool dialog handlers for CurveLabApp.

Mixed into CurveLabApp; these methods rely on the coordinator's session
accessors, managers (data_mgr, plot_mgr), and panels (data_panel), so they are
not usable standalone.
"""

import numpy as np
import pandas as pd
from tkinter import messagebox

from .ui_dialogs_data import (
    SimulateDataDialog, ColumnCalculatorDialog,
    EvaluateModelDialog, FindPeaksDialog, DerivativeIntegralDialog,
    SmoothOutlierDialog,
)


class DataToolsMixin:
    """Menu/button handlers for data generation and manipulation dialogs."""

    def _evaluate_model(self):
        """Evaluate the fitted model at user-specified x values."""
        sess = self._require_fit_result()
        if sess is None:
            return
        EvaluateModelDialog(self, sess.fit_manager)

    def _find_peaks(self):
        """Auto-detect peaks in active series and add Gaussian components."""
        sess = self._require_session()
        if sess is None:
            return
        rec = self._active_record
        fm = sess.fit_manager
        x, y, yerr, xerr = self._get_fit_data(rec)
        if len(x) < 3:
            messagebox.showwarning("Insufficient Data", "Need at least 3 data points.")
            return
        FindPeaksDialog(self, x, y, fm, on_done=self._update_component_list)

    def _smooth_outlier_dialog(self):
        rec = self._require_series()
        if rec is None:
            return
        SmoothOutlierDialog(
            self,
            x=rec.x.copy(),
            y=rec.y.copy(),
            yerr=rec.yerr.copy() if rec.yerr is not None else None,
            mask=rec.mask.copy() if rec.mask is not None else None,
            ax=self.plot_mgr.ax,
            canvas=self.plot_mgr.canvas,
            on_apply_mask=self._apply_smooth_mask,
            on_export_series=self._export_smooth_series,
        )

    def _apply_smooth_mask(self, new_mask):
        rec = self._active_record
        if rec is None:
            return
        rec.mask = new_mask
        self._replot_all_series()
        self.plot_mgr.canvas.draw_idle()

    def _export_smooth_series(self, x, y, label_suffix):
        rec = self._active_record
        ds_base = rec.dataset_name if rec else "data"
        df = pd.DataFrame({"x": x, "y": y})
        self._register_series_dataframe(
            f"{ds_base} ({label_suffix})", df, linestyle="-"
        )
        self._on_plot(self.data_panel.series_list)

    def _show_derivative_integral(self):
        """Show derivative and integral of the fitted curve."""
        sess = self._require_fit_result()
        if sess is None:
            return
        DerivativeIntegralDialog(self, sess.fit_manager, sess.result)

    def _on_simulate_data(self):
        sess = self._require_session()
        if sess is None:
            return
        fm = sess.fit_manager
        if not self._require_model(fm):
            return

        # Default x-range from active series or plot limits
        rec = self._active_record
        if rec is not None and len(rec.x) > 0:
            x_min, x_max = float(rec.x.min()), float(rec.x.max())
        else:
            x_min, x_max = 0.0, 10.0

        SimulateDataDialog(
            self,
            x_min=x_min,
            x_max=x_max,
            n_points=200,
            on_generate=self._generate_simulated_data,
        )

    def _generate_simulated_data(self, x_min, x_max, n_points, noise_cfg):
        fm = self._active_fit_mgr
        x = np.linspace(x_min, x_max, n_points)
        xerr = None

        # X-jitter
        if noise_cfg.get("jitter"):
            sigma_x = noise_cfg["jitter_sigma"]
            x = x + np.random.normal(0, sigma_x, size=n_points)
            xerr = np.full(n_points, sigma_x)

        y = fm.model.eval(fm.params, x=x)

        variance = np.zeros(n_points)

        # Poisson noise
        if noise_cfg.get("poisson"):
            scale = noise_cfg["poisson_scale"]
            poisson_sigma = scale * np.sqrt(np.abs(y))
            y = y + poisson_sigma * np.random.normal(0, 1, size=n_points)
            variance += poisson_sigma ** 2

        # Gaussian noise
        if noise_cfg.get("gaussian"):
            sigma = noise_cfg["gaussian_sigma"]
            y = y + np.random.normal(0, sigma, size=n_points)
            variance += sigma ** 2

        yerr = np.sqrt(variance) if variance.any() else None

        # Sort by x (jitter may reorder)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        if yerr is not None:
            yerr = yerr[order]
        if xerr is not None:
            xerr = xerr[order]

        # Check if active series is an empty placeholder — reuse its dataset
        cur_rec = self._active_record
        reuse_placeholder = (
            cur_rec is not None
            and len(cur_rec.x) == 0
            and cur_rec.dataset_name in self.data_mgr.datasets
        )

        if reuse_placeholder:
            ds_name = cur_rec.dataset_name
            data = {"x": x, "y": y}
            if yerr is not None:
                data["yerr"] = yerr
            if xerr is not None:
                data["xerr"] = xerr
            self.data_mgr.datasets[ds_name] = pd.DataFrame(data)
            # Update column combos for this dataset
            self.data_panel.set_columns(self.data_mgr.column_names(ds_name))
            # Update the series entry in DataPanel to reflect new error columns
            for item in self.data_panel._series_items:
                if item["dataset"] == ds_name:
                    item["yerr"] = "yerr" if yerr is not None else ""
                    item["xerr"] = "xerr" if xerr is not None else ""
                    break
        else:
            self._simulated_counter += 1
            data = {"x": x, "y": y}
            if yerr is not None:
                data["yerr"] = yerr
            if xerr is not None:
                data["xerr"] = xerr
            self._register_series_dataframe(
                f"Simulated {self._simulated_counter}", pd.DataFrame(data),
                yerr="yerr" if yerr is not None else "",
                xerr="xerr" if xerr is not None else "",
            )

        self._on_plot(self.data_panel.series_list)

    def _on_column_calc(self):
        dataset = self.data_panel.dataset_var.get()
        if not dataset:
            messagebox.showwarning("No Dataset", "Load a dataset first.")
            return
        columns = self.data_mgr.column_names(dataset)

        _SAFE_NAMES = {
            "abs": np.abs, "sqrt": np.sqrt, "log": np.log, "log2": np.log2,
            "log10": np.log10, "exp": np.exp, "sin": np.sin, "cos": np.cos,
            "tan": np.tan, "arcsin": np.arcsin, "arccos": np.arccos,
            "arctan": np.arctan, "arctan2": np.arctan2,
            "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
            "floor": np.floor, "ceil": np.ceil, "round": np.round_,
            "sign": np.sign, "clip": np.clip,
            "pi": np.pi, "e": np.e, "inf": np.inf, "nan": np.nan,
            "diff": np.diff, "cumsum": np.cumsum,
            "mean": np.mean, "std": np.std, "min": np.min, "max": np.max,
            "where": np.where, "isnan": np.isnan, "isinf": np.isinf,
        }

        def on_apply(name, expr, preview_only):
            if not expr:
                raise ValueError("Enter an expression.")
            from asteval import Interpreter
            aeval = Interpreter()
            df = self.data_mgr.datasets[dataset]
            for k, v in _SAFE_NAMES.items():
                aeval.symtable[k] = v
            for col in df.columns:
                aeval.symtable[col] = df[col].to_numpy(dtype=float)
            result = aeval(expr)
            if aeval.error:
                raise ValueError(aeval.error[0].get_error()[1])
            result = np.asarray(result, dtype=float)
            if result.ndim == 0:
                result = np.full(len(df), result)
            if len(result) != len(df):
                raise ValueError(
                    f"Result has {len(result)} values, expected {len(df)}. "
                    f"(Functions like diff reduce length by 1.)"
                )
            if preview_only:
                return result
            df[name] = result
            # Refresh column dropdowns
            self.data_panel.set_columns(list(df.columns))
            return result

        ColumnCalculatorDialog(self, columns=columns, on_apply=on_apply)
