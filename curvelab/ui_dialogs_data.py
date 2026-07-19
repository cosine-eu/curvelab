"""Data manipulation dialog classes for CurveLab."""

import tkinter as tk
from tkinter import ttk, messagebox

from .ui_common import BaseDialog


class SimulateDataDialog(BaseDialog):
    """Dialog for generating synthetic data from the current model."""

    def __init__(self, parent, x_min=0.0, x_max=10.0, n_points=200, on_generate=None):
        super().__init__(parent, "Simulate Data", resizable=(False, False))
        self._on_generate = on_generate

        # --- X Range ---
        range_frame = ttk.LabelFrame(self, text="X Range", padding=5)
        range_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        for col, (label, default) in enumerate([
            ("x min", x_min), ("x max", x_max), ("N points", n_points),
        ]):
            ttk.Label(range_frame, text=label).grid(row=0, column=col * 2, padx=(5, 2))
            var = tk.StringVar(value=str(default))
            ttk.Entry(range_frame, textvariable=var, width=10).grid(
                row=0, column=col * 2 + 1, padx=(0, 5)
            )
            if col == 0:
                self._xmin_var = var
            elif col == 1:
                self._xmax_var = var
            else:
                self._npts_var = var

        # --- Noise ---
        noise_frame = ttk.LabelFrame(self, text="Noise", padding=5)
        noise_frame.pack(fill=tk.X, padx=10, pady=5)

        self._gauss_on = tk.BooleanVar(value=False)
        self._gauss_sigma = tk.StringVar(value="1.0")
        self._poisson_on = tk.BooleanVar(value=False)
        self._poisson_scale = tk.StringVar(value="1.0")
        self._jitter_on = tk.BooleanVar(value=False)
        self._jitter_sigma = tk.StringVar(value="0.1")

        for row, (var_on, var_mag, label, mag_label) in enumerate([
            (self._gauss_on, self._gauss_sigma, "Gaussian noise", "sigma"),
            (self._poisson_on, self._poisson_scale, "Poisson noise", "scale"),
            (self._jitter_on, self._jitter_sigma, "X-jitter", "sigma_x"),
        ]):
            ttk.Checkbutton(noise_frame, text=label, variable=var_on).grid(
                row=row, column=0, sticky=tk.W, padx=(5, 10)
            )
            ttk.Label(noise_frame, text=mag_label).grid(row=row, column=1, padx=(5, 2))
            ttk.Entry(noise_frame, textvariable=var_mag, width=8).grid(
                row=row, column=2, padx=(0, 5)
            )

        # --- Buttons ---
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self._status_var = tk.StringVar(value="")
        ttk.Label(btn_frame, textvariable=self._status_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(btn_frame, text="Generate", command=self._do_generate).pack(
            side=tk.RIGHT, padx=(5, 0)
        )
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)

    def _do_generate(self):
        try:
            x_min = float(self._xmin_var.get())
            x_max = float(self._xmax_var.get())
            n_points = int(self._npts_var.get())
        except ValueError:
            self._status_var.set("Invalid x-range or N.")
            return
        if x_min >= x_max or n_points < 2:
            self._status_var.set("Need x_min < x_max and N >= 2.")
            return

        try:
            noise_cfg = {
                "gaussian": self._gauss_on.get(),
                "gaussian_sigma": float(self._gauss_sigma.get()) if self._gauss_on.get() else 0,
                "poisson": self._poisson_on.get(),
                "poisson_scale": float(self._poisson_scale.get()) if self._poisson_on.get() else 0,
                "jitter": self._jitter_on.get(),
                "jitter_sigma": float(self._jitter_sigma.get()) if self._jitter_on.get() else 0,
            }
        except ValueError:
            self._status_var.set("Invalid noise parameter.")
            return

        if self._on_generate:
            try:
                self._on_generate(x_min, x_max, n_points, noise_cfg)
                self.destroy()
            except Exception as e:
                self._status_var.set(f"Error: {e}")


class SmoothOutlierDialog(BaseDialog):
    """Smooth data, detect outliers, export smoothed/baseline-subtracted series."""

    _METHODS = ["Savitzky-Golay", "Moving Average", "Median Filter", "Gaussian Filter"]

    def __init__(self, parent, x, y, yerr, mask, ax, canvas,
                 on_apply_mask=None, on_export_series=None):
        super().__init__(parent, "Smooth / Outlier Detection", size="420x380")
        self._x = x
        self._y = y
        self._yerr = yerr
        self._orig_mask = mask  # may be None
        self._ax = ax
        self._canvas = canvas
        self._on_apply_mask = on_apply_mask
        self._on_export_series = on_export_series
        self._smooth_line = None
        self._outlier_scatter = None
        self._debounce_id = None
        self._y_smooth = None
        self._outlier_mask = None  # True = inlier

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.after(200, self._update_preview)

    def _build_ui(self):
        import numpy as np
        n = len(self._x)

        # --- Smoothing ---
        sf = ttk.LabelFrame(self, text="Smoothing", padding=5)
        sf.pack(fill=tk.X, padx=10, pady=5)

        row0 = ttk.Frame(sf)
        row0.pack(fill=tk.X)
        ttk.Label(row0, text="Method:").pack(side=tk.LEFT)
        self._method_var = tk.StringVar(value=self._METHODS[0])
        ttk.Combobox(row0, textvariable=self._method_var, values=self._METHODS,
                     state="readonly", width=18).pack(side=tk.LEFT, padx=5)
        self._method_var.trace_add("write", self._on_method_change)

        row1 = ttk.Frame(sf)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Window:").pack(side=tk.LEFT)
        max_win = min(n // 2 * 2 + 1, 201)  # odd, capped
        self._window_var = tk.IntVar(value=min(11, max_win))
        self._window_scale = tk.Scale(row1, variable=self._window_var, from_=3,
                                       to=max(3, max_win), orient=tk.HORIZONTAL,
                                       resolution=2, command=self._on_slider)
        self._window_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._param2_frame = ttk.Frame(sf)
        self._param2_frame.pack(fill=tk.X, pady=2)
        self._param2_label = ttk.Label(self._param2_frame, text="Order:")
        self._param2_label.pack(side=tk.LEFT)
        self._param2_var = tk.IntVar(value=3)
        self._param2_scale = tk.Scale(self._param2_frame, variable=self._param2_var,
                                       from_=1, to=7, orient=tk.HORIZONTAL,
                                       command=self._on_slider)
        self._param2_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._show_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sf, text="Show smooth overlay", variable=self._show_var,
                        command=self._schedule_update).pack(anchor=tk.W)

        # --- Outlier Detection ---
        of = ttk.LabelFrame(self, text="Outlier Detection", padding=5)
        of.pack(fill=tk.X, padx=10, pady=5)

        self._outlier_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(of, text="Enable", variable=self._outlier_var,
                        command=self._schedule_update).pack(anchor=tk.W)

        row_sigma = ttk.Frame(of)
        row_sigma.pack(fill=tk.X, pady=2)
        ttk.Label(row_sigma, text="Sigma:").pack(side=tk.LEFT)
        self._sigma_var = tk.DoubleVar(value=3.0)
        tk.Scale(row_sigma, variable=self._sigma_var, from_=1.0, to=10.0,
                 orient=tk.HORIZONTAL, resolution=0.1,
                 command=self._on_slider).pack(side=tk.LEFT, fill=tk.X, expand=True)

        row_iter = ttk.Frame(of)
        row_iter.pack(fill=tk.X, pady=2)
        ttk.Label(row_iter, text="Iterations:").pack(side=tk.LEFT)
        self._iter_var = tk.IntVar(value=1)
        tk.Scale(row_iter, variable=self._iter_var, from_=1, to=5,
                 orient=tk.HORIZONTAL, command=self._on_slider).pack(
                     side=tk.LEFT, fill=tk.X, expand=True)

        self._status_var = tk.StringVar(value="")
        ttk.Label(of, textvariable=self._status_var,
                  font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(2, 0))

        # --- Actions ---
        af = ttk.Frame(self)
        af.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(af, text="Apply Exclusions", command=self._apply_exclusions).pack(
            side=tk.LEFT, padx=3)
        ttk.Button(af, text="Export Smoothed", command=self._export_smoothed).pack(
            side=tk.LEFT, padx=3)
        ttk.Button(af, text="Export Baseline-Sub.", command=self._export_baseline_sub).pack(
            side=tk.LEFT, padx=3)
        ttk.Button(af, text="Close", command=self.destroy).pack(side=tk.RIGHT, padx=3)

    def _on_method_change(self, *_args):
        method = self._method_var.get()
        if method == "Savitzky-Golay":
            self._param2_label.config(text="Order:")
            self._param2_scale.config(from_=1, to=7, resolution=1)
            self._param2_var.set(3)
            self._param2_frame.pack(fill=tk.X, pady=2)
            self._window_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._window_scale.master.pack(fill=tk.X, pady=2)
        elif method == "Gaussian Filter":
            self._param2_label.config(text="Sigma:")
            self._param2_scale.config(from_=1, to=50, resolution=1)
            self._param2_var.set(5)
            self._param2_frame.pack(fill=tk.X, pady=2)
            # Hide window for Gaussian (uses sigma param instead)
            self._window_scale.master.pack_forget()
        else:
            self._param2_frame.pack_forget()
            self._window_scale.master.pack(fill=tk.X, pady=2)
        self._schedule_update()

    def _on_slider(self, *_args):
        self._schedule_update()

    def _schedule_update(self):
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(100, self._update_preview)

    def _compute_smooth(self):
        return self._compute_smooth_on(self._y.copy())

    def _compute_outliers(self, y_smooth):
        import numpy as np
        n = len(self._y)
        inlier = np.ones(n, dtype=bool)
        threshold = self._sigma_var.get()
        n_iter = self._iter_var.get()

        for _ in range(n_iter):
            residuals = self._y - y_smooth
            med = np.median(residuals[inlier]) if inlier.any() else 0.0
            mad = np.median(np.abs(residuals[inlier] - med)) if inlier.any() else 1.0
            sigma_est = 1.4826 * mad if mad > 0 else 1.0
            inlier = np.abs(residuals - med) < threshold * sigma_est
            # Re-smooth on inliers for next iteration
            if not inlier.all() and inlier.sum() >= 3:
                from scipy.interpolate import interp1d
                f = interp1d(self._x[inlier], self._y[inlier], kind="linear",
                             fill_value="extrapolate")
                y_interp = f(self._x)
                y_smooth = self._compute_smooth_on(y_interp)
        return inlier

    def _compute_smooth_on(self, y):
        """Smooth a given y array with current settings (for iterative outlier rejection)."""
        import numpy as np
        from scipy.signal import savgol_filter, medfilt
        from scipy.ndimage import uniform_filter1d, gaussian_filter1d

        method = self._method_var.get()
        if len(y) < 3:
            return y  # Too few points for any windowed smoothing method

        window = self._window_var.get()
        if window % 2 == 0:
            window += 1
        # len(y) >= 3 here, so this max is always >= 3 and the floor below
        # never has to push window past the data length.
        window = min(window, len(y) - 1 if len(y) % 2 == 0 else len(y))
        if window < 3:
            window = 3

        if method == "Savitzky-Golay":
            order = min(self._param2_var.get(), window - 1)
            return savgol_filter(y, window, order)
        elif method == "Moving Average":
            return uniform_filter1d(y, size=window)
        elif method == "Median Filter":
            return medfilt(y, kernel_size=window)
        elif method == "Gaussian Filter":
            sigma = max(1, self._param2_var.get())
            return gaussian_filter1d(y, sigma=sigma)
        return y

    def _update_preview(self):
        import numpy as np
        self._debounce_id = None

        # Remove old artists
        if self._smooth_line is not None:
            self._smooth_line.remove()
            self._smooth_line = None
        if self._outlier_scatter is not None:
            self._outlier_scatter.remove()
            self._outlier_scatter = None

        try:
            self._y_smooth = self._compute_smooth()
        except Exception as e:
            self._status_var.set(f"Smoothing error: {e}")
            self._canvas.draw_idle()
            return

        if self._show_var.get():
            self._smooth_line, = self._ax.plot(
                self._x, self._y_smooth, "--", color="orange", linewidth=1.5,
                label="_smooth_preview", zorder=5)

        if self._outlier_var.get():
            try:
                self._outlier_mask = self._compute_outliers(self._y_smooth)
            except Exception as e:
                self._status_var.set(f"Outlier detection error: {e}")
                self._canvas.draw_idle()
                return
            outliers = ~self._outlier_mask
            n_out = int(outliers.sum())
            self._status_var.set(
                f"Outliers: {n_out} / {len(self._y)} ({100*n_out/len(self._y):.1f}%)")
            if n_out > 0:
                self._outlier_scatter = self._ax.scatter(
                    self._x[outliers], self._y[outliers],
                    s=60, facecolors="none", edgecolors="red", linewidths=1.5,
                    zorder=6, label="_outlier_preview")
        else:
            self._outlier_mask = None
            self._status_var.set("")

        self._canvas.draw_idle()

    def _apply_exclusions(self):
        import numpy as np
        from tkinter import messagebox
        if self._outlier_mask is None:
            messagebox.showinfo("No Outliers", "Enable outlier detection first.", parent=self)
            return
        if self._outlier_mask.all():
            messagebox.showinfo("No Outliers", "No outliers detected with current settings.",
                                parent=self)
            return
        if not self._outlier_mask.any():
            messagebox.showwarning("All Excluded",
                                   "All points flagged as outliers. Lower the sigma threshold.",
                                   parent=self)
            return
        # AND-merge with existing mask
        if self._orig_mask is not None:
            combined = self._orig_mask & self._outlier_mask
        else:
            combined = self._outlier_mask.copy()
        if self._on_apply_mask:
            self._on_apply_mask(combined)
        self.destroy()

    def _export_smoothed(self):
        if self._y_smooth is None:
            return
        if self._on_export_series:
            self._on_export_series(self._x, self._y_smooth, "Smoothed")

    def _export_baseline_sub(self):
        if self._y_smooth is None:
            return
        if self._on_export_series:
            self._on_export_series(self._x, self._y - self._y_smooth, "Baseline-subtracted")

    def destroy(self):
        if self._smooth_line is not None:
            self._smooth_line.remove()
            self._smooth_line = None
        if self._outlier_scatter is not None:
            self._outlier_scatter.remove()
            self._outlier_scatter = None
        self._canvas.draw_idle()
        super().destroy()


class DerivativeIntegralDialog(BaseDialog):
    """Plot derivative and integral of the fitted curve."""

    def __init__(self, parent, fit_manager, result):
        super().__init__(parent, "Derivative / Integral", size="700x500")

        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        x = result.x_dense
        y = result.y_fit_dense

        # Numerical derivative
        dydx = np.gradient(y, x)

        # Numerical integral (cumulative)
        from scipy.integrate import cumulative_trapezoid
        integral = cumulative_trapezoid(y, x, initial=0)

        # Summary
        total_area = integral[-1] if len(integral) > 0 else 0.0

        info_frame = ttk.Frame(self)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(info_frame, text=f"Total area (integral): {total_area:.6g}",
                  font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)

        # Component areas if available
        if result.component_curves:
            for name, y_comp in result.component_curves.items():
                area = np.trapz(y_comp, x)
                ttk.Label(info_frame, text=f"  {name}: {area:.6g}").pack(anchor=tk.W)

        fig = Figure(figsize=(7, 4))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(x, dydx, "r-", linewidth=1)
        ax1.set_ylabel("dy/dx")
        ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax1.set_title("Derivative")

        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        ax2.plot(x, integral, "b-", linewidth=1)
        ax2.set_ylabel("∫y dx")
        ax2.set_xlabel("x")
        ax2.set_title("Cumulative Integral")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


class FindPeaksDialog(BaseDialog):
    """Auto-detect peaks and add Gaussian components."""

    def __init__(self, parent, x, y, fit_manager, on_done=None):
        super().__init__(parent, "Find Peaks", size="500x420")
        self._x = x
        self._y = y
        self._fm = fit_manager
        self._on_done = on_done
        self._peaks = []

        # Controls
        ctrl = ttk.Frame(self)
        ctrl.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(ctrl, text="Prominence:").pack(side=tk.LEFT)
        self._prominence_var = tk.StringVar(value="")
        ttk.Entry(ctrl, textvariable=self._prominence_var, width=8).pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, text="Min distance (pts):").pack(side=tk.LEFT, padx=(10, 0))
        self._distance_var = tk.StringVar(value="")
        ttk.Entry(ctrl, textvariable=self._distance_var, width=6).pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, text="Model:").pack(side=tk.LEFT, padx=(10, 0))
        self._model_var = tk.StringVar(value="Gaussian")
        ttk.Combobox(ctrl, textvariable=self._model_var, width=12,
                     values=["Gaussian", "Lorentzian", "Voigt", "PseudoVoigt"],
                     state="readonly").pack(side=tk.LEFT, padx=5)

        btn_row = ttk.Frame(self)
        btn_row.pack(pady=5)
        ttk.Button(btn_row, text="Detect", command=self._detect).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Add to Model", command=self._add_to_model).pack(side=tk.LEFT, padx=5)

        # Results tree
        cols = ("center", "amplitude", "width")
        self._tree = ttk.Treeview(self, columns=cols, show="headings", height=8)
        self._tree.heading("center", text="Center (x)")
        self._tree.heading("amplitude", text="Amplitude (y)")
        self._tree.heading("width", text="Est. Width")
        for c in cols:
            self._tree.column(c, width=130)
        self._tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Auto-detect on open with defaults
        self.after(100, self._detect)

    def _detect(self):
        from scipy.signal import find_peaks, peak_widths
        import numpy as np

        kwargs = {}
        prom = self._prominence_var.get().strip()
        dist = self._distance_var.get().strip()
        try:
            if prom:
                kwargs["prominence"] = float(prom)
            else:
                # Auto-prominence: 10% of data range
                yrange = np.ptp(self._y)
                if yrange > 0:
                    kwargs["prominence"] = yrange * 0.1
            if dist:
                kwargs["distance"] = int(dist)
        except ValueError:
            messagebox.showwarning(
                "Invalid Input", "Prominence and min distance must be numbers."
            )
            return

        indices, properties = find_peaks(self._y, **kwargs)
        # Estimate widths
        if len(indices) > 0:
            widths_result = peak_widths(self._y, indices, rel_height=0.5)
            widths_pts = widths_result[0]
            dx = np.median(np.diff(self._x)) if len(self._x) > 1 else 1.0
            widths_x = widths_pts * abs(dx)
        else:
            widths_x = []

        self._peaks = []
        self._tree.delete(*self._tree.get_children())
        for i, idx in enumerate(indices):
            cx, cy = self._x[idx], self._y[idx]
            w = widths_x[i] if i < len(widths_x) else 0.0
            self._peaks.append((cx, cy, w))
            self._tree.insert("", tk.END, values=(
                f"{cx:.6g}", f"{cy:.6g}", f"{w:.6g}"))

    def _add_to_model(self):
        if not self._peaks:
            return
        import numpy as np
        model_name = self._model_var.get()
        for cx, cy, w in self._peaks:
            self._fm.add_component(model_name, operator="+")
            # Set initial guesses for the last added component
            prefix = self._fm.components[-1].prefix
            sigma = w / 2.355 if w > 0 else abs(cx) * 0.01 or 0.1  # FWHM to sigma
            self._fm.set_param(f"{prefix}center", value=cx)
            self._fm.set_param(f"{prefix}amplitude", value=cy * sigma * (2 * np.pi) ** 0.5)
            self._fm.set_param(f"{prefix}sigma", value=sigma)
        if self._on_done:
            self._on_done()
        self.destroy()


class EvaluateModelDialog(BaseDialog):
    """Evaluate fitted model at user-specified x values."""

    def __init__(self, parent, fit_manager):
        super().__init__(parent, "Evaluate Model", size="500x400")
        self._fm = fit_manager

        ttk.Label(self, text="Enter x values (comma or space separated, or start:stop:npoints):").pack(
            anchor=tk.W, padx=10, pady=(10, 0))
        self._x_entry = ttk.Entry(self, width=60)
        self._x_entry.pack(padx=10, pady=5, fill=tk.X)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Evaluate", command=self._evaluate).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Copy", command=self._copy).pack(side=tk.LEFT, padx=5)

        self._result_text = tk.Text(self, height=18, wrap=tk.NONE, state=tk.DISABLED,
                                    font=("TkFixedFont", 10))
        scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._result_text.yview)
        self._result_text.configure(yscrollcommand=scroll.set)
        self._result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0, 10))
        scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=(0, 10))

    def _parse_x(self):
        import numpy as np
        text = self._x_entry.get().strip()
        if not text:
            return None
        # Range syntax: start:stop:npoints
        if text.count(":") == 2:
            parts = text.split(":")
            return np.linspace(float(parts[0]), float(parts[1]), int(parts[2]))
        # Comma or space separated
        text = text.replace(",", " ")
        return np.array([float(v) for v in text.split()])

    def _evaluate(self):
        import numpy as np
        try:
            x = self._parse_x()
            if x is None or len(x) == 0:
                return
            y = self._fm.evaluate(x)
            lines = [f"{'x':>16s}  {'y':>16s}"]
            lines.append("-" * 34)
            for xi, yi in zip(x, y):
                lines.append(f"{xi:16.8g}  {yi:16.8g}")
            self._last_text = "\n".join(lines)
            self._result_text.config(state=tk.NORMAL)
            self._result_text.delete("1.0", tk.END)
            self._result_text.insert("1.0", self._last_text)
            self._result_text.config(state=tk.DISABLED)
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Evaluate Error", str(e), parent=self)

    def _copy(self):
        if hasattr(self, "_last_text"):
            self.clipboard_clear()
            self.clipboard_append(self._last_text)


class ColumnCalculatorDialog(BaseDialog):
    """Dialog for creating new columns from expressions on existing columns."""

    def __init__(self, parent, columns: list[str], on_apply=None):
        super().__init__(parent, "Column Calculator", resizable=(True, False))
        self._columns = list(columns)
        self._on_apply = on_apply

        # --- Column name ---
        name_frame = ttk.Frame(self)
        name_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Label(name_frame, text="New column name:").pack(side=tk.LEFT)
        self._name_var = tk.StringVar()
        ttk.Entry(name_frame, textvariable=self._name_var, width=20).pack(
            side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True
        )

        # --- Expression ---
        expr_frame = ttk.LabelFrame(self, text="Expression", padding=5)
        expr_frame.pack(fill=tk.X, padx=10, pady=5)
        self._expr_var = tk.StringVar()
        ttk.Entry(expr_frame, textvariable=self._expr_var, width=50).pack(
            fill=tk.X, pady=(0, 5)
        )

        col_text = ", ".join(columns) if columns else "(no columns)"
        ttk.Label(expr_frame, text=f"Columns: {col_text}",
                  wraplength=400, justify=tk.LEFT).pack(anchor=tk.W)
        ttk.Label(expr_frame,
                  text="Functions: abs, sqrt, log, log10, exp, sin, cos, tan, "
                       "diff, cumsum, mean, std, where, clip, pi, e",
                  wraplength=400, justify=tk.LEFT,
                  foreground="gray").pack(anchor=tk.W)
        ttk.Label(expr_frame,
                  text="Examples: log(intensity), col_0 / col_1, "
                       "sqrt(x**2 + y**2)",
                  wraplength=400, justify=tk.LEFT,
                  foreground="gray").pack(anchor=tk.W)

        # --- Preview ---
        preview_frame = ttk.LabelFrame(self, text="Preview (first 10 values)", padding=5)
        preview_frame.pack(fill=tk.X, padx=10, pady=5)
        self._preview_text = tk.Text(preview_frame, height=3, state=tk.DISABLED,
                                     wrap=tk.WORD)
        self._preview_text.pack(fill=tk.X)

        # --- Buttons ---
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        self._status_var = tk.StringVar(value="")
        ttk.Label(btn_frame, textvariable=self._status_var,
                  foreground="red").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="Preview", command=self._preview).pack(
            side=tk.RIGHT, padx=(5, 0)
        )
        ttk.Button(btn_frame, text="Apply", command=self._apply).pack(
            side=tk.RIGHT, padx=(5, 0)
        )
        ttk.Button(btn_frame, text="Close", command=self.destroy).pack(
            side=tk.RIGHT
        )

    def _preview(self):
        self._do_eval(preview_only=True)

    def _apply(self):
        self._do_eval(preview_only=False)

    def _do_eval(self, preview_only: bool):
        name = self._name_var.get().strip()
        if not name and not preview_only:
            self._status_var.set("Enter a column name.")
            return
        if self._on_apply is None:
            return
        try:
            result = self._on_apply(name, self._expr_var.get().strip(), preview_only)
            if preview_only and result is not None:
                preview = ", ".join(f"{v:.6g}" for v in result[:10])
                if len(result) > 10:
                    preview += f", ... ({len(result)} values)"
                self._preview_text.config(state=tk.NORMAL)
                self._preview_text.delete("1.0", tk.END)
                self._preview_text.insert("1.0", preview)
                self._preview_text.config(state=tk.DISABLED)
                self._status_var.set("")
            elif not preview_only:
                self._status_var.set("")
                if name not in self._columns:
                    self._columns.append(name)
                messagebox.showinfo("Column Calculator",
                                    f"Column '{name}' created.", parent=self)
        except Exception as e:
            self._status_var.set(str(e))
