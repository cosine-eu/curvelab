"""Core tkinter panel widgets: DataPanel, PlotControlPanel, FitPanel, FitResultsPanel."""

import tkinter as tk
from tkinter import ttk, filedialog, simpledialog

from .fit_manager import REDUCE_FUNCTIONS, WEIGHT_MODES
from .models import MODEL_NAMES
from .ui_common import set_readonly_text

# Marker choices for the style dropdown
MARKERS = ["o", "s", "^", "v", "D", "x", "+", ".", "*", "h"]
LINESTYLES = ["None", "-", "--", "-.", ":"]
COLORS = [
    "", "xkcd:blue", "xkcd:red", "xkcd:green", "xkcd:orange", "xkcd:purple",
    "xkcd:brown", "xkcd:black", "xkcd:cyan", "xkcd:magenta", "xkcd:teal",
    "xkcd:sky blue", "xkcd:olive", "xkcd:coral", "xkcd:lavender",
]


class DataPanel(ttk.LabelFrame):
    """File loading, dataset selection, column selection, series list, and style controls."""

    def __init__(
        self,
        parent,
        on_load=None,
        on_plot=None,
        on_dataset_selected=None,
        on_remove_dataset=None,
        on_toggle_series_visible=None,
        on_column_calc=None,
        on_confirm_remove_series=None,
    ):
        super().__init__(parent, text="Data", padding=5)
        self._on_load = on_load
        self._on_plot = on_plot
        self._on_dataset_selected = on_dataset_selected
        self._on_remove_dataset = on_remove_dataset
        self._on_toggle_series_visible = on_toggle_series_visible
        self._on_column_calc = on_column_calc
        self._on_confirm_remove_series = on_confirm_remove_series
        self._series_items = []  # list of dicts describing each series

        self._build_ui()

    def _build_ui(self):
        # --- Load button + Column calc ---
        load_frame = ttk.Frame(self)
        load_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(load_frame, text="Load File...", command=self._load_file).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2)
        )
        ttk.Button(load_frame, text="Column Calc...", command=self._column_calc).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0)
        )

        # --- Dataset selector ---
        ds_frame = ttk.Frame(self)
        ds_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(ds_frame, text="Dataset:").pack(side=tk.LEFT)
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(
            ds_frame, textvariable=self.dataset_var, state="readonly", width=15
        )
        self.dataset_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.dataset_combo.bind("<<ComboboxSelected>>", self._dataset_selected)

        ttk.Button(ds_frame, text="Remove", command=self._remove_dataset, width=7).pack(
            side=tk.LEFT
        )

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # --- Column selection ---
        col_frame = ttk.Frame(self)
        col_frame.pack(fill=tk.X)

        ttk.Label(col_frame, text="X:").grid(row=0, column=0, sticky=tk.W)
        self.x_var = tk.StringVar()
        self.x_combo = ttk.Combobox(
            col_frame, textvariable=self.x_var, state="readonly", width=15
        )
        self.x_combo.grid(row=0, column=1, sticky=tk.EW, padx=2)

        ttk.Label(col_frame, text="Y:").grid(row=1, column=0, sticky=tk.W)
        self.y_var = tk.StringVar()
        self.y_combo = ttk.Combobox(
            col_frame, textvariable=self.y_var, state="readonly", width=15
        )
        self.y_combo.grid(row=1, column=1, sticky=tk.EW, padx=2)

        ttk.Label(col_frame, text="Y err:").grid(row=2, column=0, sticky=tk.W)
        self.yerr_var = tk.StringVar()
        self.yerr_combo = ttk.Combobox(
            col_frame, textvariable=self.yerr_var, state="readonly", width=15
        )
        self.yerr_combo.grid(row=2, column=1, sticky=tk.EW, padx=2)

        ttk.Label(col_frame, text="X err:").grid(row=3, column=0, sticky=tk.W)
        self.xerr_var = tk.StringVar()
        self.xerr_combo = ttk.Combobox(
            col_frame, textvariable=self.xerr_var, state="readonly", width=15
        )
        self.xerr_combo.grid(row=3, column=1, sticky=tk.EW, padx=2)

        col_frame.columnconfigure(1, weight=1)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # --- Style controls ---
        style_frame = ttk.Frame(self)
        style_frame.pack(fill=tk.X)

        ttk.Label(style_frame, text="Marker:").grid(row=0, column=0, sticky=tk.W)
        self.marker_var = tk.StringVar(value="o")
        ttk.Combobox(
            style_frame,
            textvariable=self.marker_var,
            values=MARKERS,
            state="readonly",
            width=5,
        ).grid(row=0, column=1, padx=2)

        ttk.Label(style_frame, text="Line:").grid(row=0, column=2, sticky=tk.W)
        self.line_var = tk.StringVar(value="None")
        ttk.Combobox(
            style_frame,
            textvariable=self.line_var,
            values=LINESTYLES,
            state="readonly",
            width=5,
        ).grid(row=0, column=3, padx=2)

        ttk.Label(style_frame, text="Color:").grid(row=1, column=0, sticky=tk.W)
        self.color_var = tk.StringVar(value="")
        ttk.Combobox(
            style_frame,
            textvariable=self.color_var,
            values=COLORS,
            state="readonly",
            width=8,
        ).grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=2)

        ttk.Label(style_frame, text="Label:").grid(row=2, column=0, sticky=tk.W)
        self.label_var = tk.StringVar()
        ttk.Entry(style_frame, textvariable=self.label_var, width=15).grid(
            row=2, column=1, columnspan=3, sticky=tk.EW, padx=2
        )

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # --- Add Series / Remove / Plot buttons ---
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Add Series", command=self._add_series).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2)
        )
        ttk.Button(btn_frame, text="Remove", command=self._remove_series).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )
        ttk.Button(btn_frame, text="Show/Hide", command=self._toggle_series_visible).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )
        ttk.Button(btn_frame, text="Plot", command=self._plot).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )

        ttk.Button(btn_frame, text="Update Style", command=self._update_series_style).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0)
        )

        # --- Series list ---
        self.series_listbox = tk.Listbox(self, height=4, exportselection=False)
        self.series_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.series_listbox.bind("<<ListboxSelect>>", self._on_select_series)

    def _column_calc(self):
        if self._on_column_calc:
            self._on_column_calc()

    def _load_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[
                ("All supported", "*.csv *.tsv *.txt *.dat *.xlsx *.xls *.ods *.json *.parquet *.h5 *.hdf5 *.hdf *.sqlite *.db"),
                ("CSV", "*.csv"),
                ("TSV", "*.tsv"),
                ("Text", "*.txt *.dat"),
                ("Excel / ODS", "*.xlsx *.xls *.ods"),
                ("JSON", "*.json"),
                ("Parquet", "*.parquet"),
                ("HDF5", "*.h5 *.hdf5 *.hdf"),
                ("SQLite", "*.sqlite *.db"),
                ("All files", "*.*"),
            ]
        )
        if filepath and self._on_load:
            self._on_load(filepath)

    def _dataset_selected(self, event=None):
        if self._on_dataset_selected:
            self._on_dataset_selected(self.dataset_var.get())

    def _remove_dataset(self):
        name = self.dataset_var.get()
        if name and self._on_remove_dataset:
            self._on_remove_dataset(name)

    def remove_series_for_dataset(self, dataset_name: str):
        """Remove all series entries that belong to the given dataset."""
        indices = [
            i for i, info in enumerate(self._series_items)
            if info.get("dataset") == dataset_name
        ]
        for i in reversed(indices):
            self._series_items.pop(i)
            self.series_listbox.delete(i)

    def set_datasets(self, names: list[str], select: str = ""):
        """Update the dataset dropdown."""
        self.dataset_combo["values"] = names
        if select and select in names:
            self.dataset_var.set(select)
        elif names:
            self.dataset_var.set(names[0])
        else:
            self.dataset_var.set("")

    def set_columns(self, columns: list[str]):
        """Populate dropdowns with column names."""
        err_columns = [""] + columns
        # Preserve current selections if still valid
        old_x = self.x_var.get()
        old_y = self.y_var.get()
        old_yerr = self.yerr_var.get()
        old_xerr = self.xerr_var.get()

        self.x_combo["values"] = columns
        self.y_combo["values"] = columns
        self.yerr_combo["values"] = err_columns
        self.xerr_combo["values"] = err_columns

        if old_x in columns:
            self.x_var.set(old_x)
        elif columns:
            self.x_var.set(columns[0])
        else:
            self.x_var.set("")

        if old_y in columns:
            self.y_var.set(old_y)
        elif columns:
            self.y_var.set(columns[1] if len(columns) > 1 else columns[0])
        else:
            self.y_var.set("")

        self.yerr_var.set(old_yerr if old_yerr in err_columns else "")
        self.xerr_var.set(old_xerr if old_xerr in err_columns else "")

    def _add_series(self):
        if not self.x_var.get() or not self.y_var.get():
            return
        dataset = self.dataset_var.get()
        if not dataset:
            return
        series_info = {
            "dataset": dataset,
            "x": self.x_var.get(),
            "y": self.y_var.get(),
            "yerr": self.yerr_var.get() or None,
            "xerr": self.xerr_var.get() or None,
            "marker": self.marker_var.get(),
            "linestyle": self.line_var.get(),
            "color": self.color_var.get(),
            "label": self.label_var.get() or self.y_var.get(),
        }
        self._series_items.append(series_info)
        self.series_listbox.insert(
            tk.END, f"{dataset}::{series_info['x']} vs {series_info['y']}"
        )
        self.series_listbox.selection_clear(0, tk.END)
        self.series_listbox.selection_set(tk.END)

    def _on_select_series(self, event=None):
        sel = self.series_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        info = self._series_items[idx]
        self.marker_var.set(info.get("marker", "o"))
        self.line_var.set(info.get("linestyle", "None"))
        self.color_var.set(info.get("color", ""))
        self.label_var.set(info.get("label", ""))

    def _update_series_style(self):
        sel = self.series_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        info = self._series_items[idx]
        info["marker"] = self.marker_var.get()
        info["linestyle"] = self.line_var.get()
        info["color"] = self.color_var.get()
        info["label"] = self.label_var.get() or info.get("y", "")
        self.series_listbox.delete(idx)
        self.series_listbox.insert(idx, f"{info['dataset']}::{info['x']} vs {info['y']}")
        self.series_listbox.selection_set(idx)
        if self._on_plot:
            self._on_plot(self._series_items)

    def _remove_series(self):
        sel = self.series_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        # Removing a series discards its fit sessions, so let the app veto.
        if self._on_confirm_remove_series and not self._on_confirm_remove_series(
            self._series_items[idx]
        ):
            return
        self.series_listbox.delete(idx)
        self._series_items.pop(idx)
        if self._on_plot:
            self._on_plot(self._series_items)

    def _plot(self):
        if self._on_plot:
            self._on_plot(self._series_items)

    def _toggle_series_visible(self):
        sel = self.series_listbox.curselection()
        if sel and self._on_toggle_series_visible:
            self._on_toggle_series_visible(sel[0])

    def set_series_visibility(self, visibility: list[bool]):
        """Refresh listbox labels to show [hidden] prefix for hidden series."""
        sel = self.series_listbox.curselection()
        for i, (info, vis) in enumerate(zip(self._series_items, visibility)):
            label = f"{info['dataset']}::{info['x']} vs {info['y']}"
            if not vis:
                label = f"[hidden] {label}"
            self.series_listbox.delete(i)
            self.series_listbox.insert(i, label)
        if sel:
            self.series_listbox.selection_set(sel[0])

    def add_series_entry(self, series_info: dict):
        """Programmatically add a series entry (same effect as the Add Series button)."""
        self._series_items.append(series_info)
        self.series_listbox.insert(
            tk.END,
            f"{series_info['dataset']}::{series_info['x']} vs {series_info['y']}",
        )

    def clear_series_entries(self):
        """Remove all series entries from the list."""
        self._series_items.clear()
        self.series_listbox.delete(0, tk.END)

    @property
    def series_list(self) -> list[dict]:
        return list(self._series_items)


class PlotControlPanel(ttk.Frame):
    """Scale, grid, equal axes, legend controls."""

    def __init__(
        self,
        parent,
        on_xscale=None,
        on_yscale=None,
        on_grid=None,
        on_equal=None,
        on_legend=None,
        on_show_params_toggled=None,
        on_residuals_toggled=None,
        on_confidence_band_toggled=None,
        on_axis_labels=None,
        on_data_toggled=None,
        on_weighted_resid_toggled=None,
        on_title=None,
        on_axis_limits=None,
    ):
        super().__init__(parent, padding=5)
        self._on_xscale = on_xscale
        self._on_yscale = on_yscale
        self._on_grid = on_grid
        self._on_equal = on_equal
        self._on_legend = on_legend
        self._on_show_params_toggled = on_show_params_toggled
        self._on_residuals_toggled = on_residuals_toggled
        self._on_confidence_band_toggled = on_confidence_band_toggled
        self._on_axis_labels = on_axis_labels
        self._on_data_toggled = on_data_toggled
        self._on_weighted_resid_toggled = on_weighted_resid_toggled
        self._on_title = on_title
        self._on_axis_limits = on_axis_limits

        self._build_ui()

    def _build_ui(self):
        # --- Row 1: Scale, toggles ---
        row1 = ttk.Frame(self)
        row1.pack(fill=tk.X)

        ttk.Label(row1, text="X:").pack(side=tk.LEFT)
        self.xscale_var = tk.StringVar(value="linear")
        xscale = ttk.Combobox(
            row1,
            textvariable=self.xscale_var,
            values=["linear", "log"],
            state="readonly",
            width=6,
        )
        xscale.pack(side=tk.LEFT, padx=(0, 10))
        xscale.bind("<<ComboboxSelected>>",
                    lambda e: self._fire(self._on_xscale, self.xscale_var))

        ttk.Label(row1, text="Y:").pack(side=tk.LEFT)
        self.yscale_var = tk.StringVar(value="linear")
        yscale = ttk.Combobox(
            row1,
            textvariable=self.yscale_var,
            values=["linear", "log"],
            state="readonly",
            width=6,
        )
        yscale.pack(side=tk.LEFT, padx=(0, 10))
        yscale.bind("<<ComboboxSelected>>",
                    lambda e: self._fire(self._on_yscale, self.yscale_var))

        self.data_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row1, text="Data", variable=self.data_var,
            command=lambda: self._fire(self._on_data_toggled, self.data_var),
        ).pack(side=tk.LEFT, padx=5)

        self.grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row1, text="Grid", variable=self.grid_var,
            command=lambda: self._fire(self._on_grid, self.grid_var),
        ).pack(side=tk.LEFT, padx=5)

        self.equal_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1, text="Equal Axes", variable=self.equal_var,
            command=lambda: self._fire(self._on_equal, self.equal_var),
        ).pack(side=tk.LEFT, padx=5)

        self.legend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row1, text="Legend", variable=self.legend_var,
            command=lambda: self._fire(self._on_legend, self.legend_var),
        ).pack(side=tk.LEFT, padx=5)

        self.show_params_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1, text="Params", variable=self.show_params_var,
            command=lambda: self._fire(self._on_show_params_toggled, self.show_params_var),
        ).pack(side=tk.LEFT, padx=5)

        self.fit_visible_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1, text="Fit visible range", variable=self.fit_visible_var,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="x:").pack(side=tk.LEFT, padx=(5, 0))
        self.fit_xmin_var = tk.StringVar(value="")
        ttk.Entry(row1, textvariable=self.fit_xmin_var, width=8).pack(side=tk.LEFT, padx=1)
        ttk.Label(row1, text="–").pack(side=tk.LEFT)
        self.fit_xmax_var = tk.StringVar(value="")
        ttk.Entry(row1, textvariable=self.fit_xmax_var, width=8).pack(side=tk.LEFT, padx=1)

        self.residuals_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1, text="Residuals", variable=self.residuals_var,
            command=lambda: self._fire(self._on_residuals_toggled, self.residuals_var),
        ).pack(side=tk.LEFT, padx=5)

        self.confidence_band_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1, text="Conf. band", variable=self.confidence_band_var,
            command=lambda: self._fire(self._on_confidence_band_toggled,
                                       self.confidence_band_var),
        ).pack(side=tk.LEFT, padx=5)

        self.band_sigma_var = tk.StringVar(value="1")
        ttk.Combobox(
            row1, textvariable=self.band_sigma_var,
            values=["1", "2", "3"], state="readonly", width=3,
        ).pack(side=tk.LEFT)
        ttk.Label(row1, text="\u03c3").pack(side=tk.LEFT, padx=(0, 5))

        self.weighted_resid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row1, text="Wt. resid", variable=self.weighted_resid_var,
            command=lambda: self._fire(self._on_weighted_resid_toggled,
                                       self.weighted_resid_var),
        ).pack(side=tk.LEFT, padx=5)

        self.exclude_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1, text="Exclude pts", variable=self.exclude_var,
        ).pack(side=tk.LEFT, padx=5)

        # --- Row 2: Title, axis labels, axis limits ---
        row2 = ttk.Frame(self)
        row2.pack(fill=tk.X, pady=(2, 0))

        self.title_var = tk.StringVar()
        self.xlabel_var = tk.StringVar()
        self.ylabel_var = tk.StringVar()
        self.xmin_var = tk.StringVar()
        self.xmax_var = tk.StringVar()
        self.ymin_var = tk.StringVar()
        self.ymax_var = tk.StringVar()

        fire_title = lambda e: self._fire(self._on_title, self.title_var)
        fire_labels = lambda e: self._fire(
            self._on_axis_labels, self.xlabel_var, self.ylabel_var)
        fire_limits = lambda e: self._fire(
            self._on_axis_limits,
            self.xmin_var, self.xmax_var, self.ymin_var, self.ymax_var)

        def entry(label, var, width, fire, sep="-"):
            ttk.Label(row2, text=label).pack(side=tk.LEFT)
            e = ttk.Entry(row2, textvariable=var, width=width)
            e.pack(side=tk.LEFT, padx=(0, 10 if sep is None else 0))
            e.bind("<Return>", fire)
            e.bind("<FocusOut>", fire)
            return e

        entry("Title:", self.title_var, 18, fire_title, sep=None)
        entry("X Label:", self.xlabel_var, 12, fire_labels, sep=None)
        entry("Y Label:", self.ylabel_var, 12, fire_labels, sep=None)

        entry("X Range:", self.xmin_var, 7, fire_limits)
        ttk.Label(row2, text="–").pack(side=tk.LEFT)
        entry("", self.xmax_var, 7, fire_limits, sep=None)
        entry("Y Range:", self.ymin_var, 7, fire_limits)
        ttk.Label(row2, text="–").pack(side=tk.LEFT)
        entry("", self.ymax_var, 7, fire_limits, sep=None)

    def _fire(self, callback, *tk_vars):
        """Invoke callback (if set) with the current values of the given tk variables."""
        if callback:
            callback(*(v.get() for v in tk_vars))


class FitPanel(ttk.LabelFrame):
    """Model selection, component management, parameter editing, fit controls."""

    def __init__(
        self,
        parent,
        on_add_component=None,
        on_remove_component=None,
        on_edit_expression=None,
        on_auto_guess=None,
        on_fit=None,
        on_clear_fit=None,
        on_series_selected=None,
        on_session_selected=None,
        on_new_session=None,
        on_rename_session=None,
        on_delete_session=None,
        on_batch_fit=None,
        on_toggle_session_visible=None,
        on_toggle_series_visible=None,
        on_abort=None,
    ):
        super().__init__(parent, text="Fit", padding=5)
        self._on_add_component = on_add_component
        self._on_remove_component = on_remove_component
        self._on_edit_expression = on_edit_expression
        self._on_auto_guess = on_auto_guess
        self._on_fit = on_fit
        self._on_clear_fit = on_clear_fit
        self._on_series_selected = on_series_selected
        self._on_session_selected = on_session_selected
        self._on_new_session = on_new_session
        self._on_rename_session = on_rename_session
        self._on_delete_session = on_delete_session
        self._on_batch_fit = on_batch_fit
        self._on_toggle_session_visible = on_toggle_session_visible
        self._on_toggle_series_visible = on_toggle_series_visible
        self._on_abort = on_abort
        self._session_names: list[str] = []
        self._model_locked = False

        self._build_ui()

    def _build_ui(self):
        # --- Series selector ---
        series_frame = ttk.Frame(self)
        series_frame.pack(fill=tk.X, pady=(0, 3))

        ttk.Label(series_frame, text="Series:").pack(side=tk.LEFT)
        self.series_var = tk.StringVar()
        self.series_combo = ttk.Combobox(
            series_frame, textvariable=self.series_var, state="readonly", width=20
        )
        self.series_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.series_combo.bind("<<ComboboxSelected>>", self._series_selected)
        ttk.Button(
            series_frame, text="Show/Hide All", width=12,
            command=self._toggle_series_visible,
        ).pack(side=tk.LEFT, padx=(2, 0))

        # --- Session controls ---
        sess_label_frame = ttk.Frame(self)
        sess_label_frame.pack(fill=tk.X)
        self._sess_label = ttk.Label(sess_label_frame, text="Fit Sessions:")
        self._sess_label.pack(side=tk.LEFT)

        self.session_listbox = tk.Listbox(self, height=3, exportselection=False)
        self.session_listbox.pack(fill=tk.X, pady=2)
        self.session_listbox.bind("<<ListboxSelect>>", self._session_selected)

        sess_btn_frame = ttk.Frame(self)
        sess_btn_frame.pack(fill=tk.X, pady=(0, 3))
        self._new_session_btn = ttk.Button(
            sess_btn_frame, text="New", command=self._new_session
        )
        self._new_session_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        self._rename_session_btn = ttk.Button(
            sess_btn_frame, text="Rename", command=self._rename_session
        )
        self._rename_session_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self._delete_session_btn = ttk.Button(
            sess_btn_frame, text="Delete", command=self._delete_session
        )
        self._delete_session_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(sess_btn_frame, text="Show/Hide", command=self._toggle_session_visible).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0)
        )

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # --- Model selection ---
        model_frame = ttk.Frame(self)
        model_frame.pack(fill=tk.X)

        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=MODEL_NAMES[0])
        ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=MODEL_NAMES,
            state="readonly",
            width=18,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # --- Operator + Component buttons ---
        op_frame = ttk.Frame(self)
        op_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(op_frame, text="Op:").pack(side=tk.LEFT)
        self.operator_var = tk.StringVar(value="+")
        ttk.Combobox(
            op_frame,
            textvariable=self.operator_var,
            values=["+", "*", "-", "/"],
            state="readonly",
            width=3,
        ).pack(side=tk.LEFT, padx=2)

        comp_btn_frame = ttk.Frame(self)
        comp_btn_frame.pack(fill=tk.X, pady=3)
        self._add_comp_btn = ttk.Button(
            comp_btn_frame, text="Add Component", command=self._add_component
        )
        self._add_comp_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        self._remove_comp_btn = ttk.Button(
            comp_btn_frame, text="Remove", command=self._remove_component
        )
        self._remove_comp_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # --- Component list ---
        self.comp_listbox = tk.Listbox(self, height=3)
        self.comp_listbox.pack(fill=tk.X, pady=3)
        self.comp_listbox.bind("<Double-1>", self._edit_expression)

        # --- Method selection ---
        method_frame = ttk.Frame(self)
        method_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="least_squares")
        ttk.Combobox(
            method_frame,
            textvariable=self.method_var,
            values=[
                "leastsq", "least_squares", "nelder", "powell",
                "cobyla", "lbfgsb",
                "differential_evolution", "basinhopping",
                "dual_annealing", "shgo", "ampgo",
                "brute", "emcee", "odr",
            ],
            state="readonly",
            width=20,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # --- Reduce function ---
        reduce_frame = ttk.Frame(self)
        reduce_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(reduce_frame, text="Reduce:").pack(side=tk.LEFT)
        self.reduce_var = tk.StringVar(value="Chi-square (default)")
        ttk.Combobox(
            reduce_frame,
            textvariable=self.reduce_var,
            values=list(REDUCE_FUNCTIONS.keys()),
            state="readonly",
            width=20,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # --- Weights ---
        weight_frame = ttk.Frame(self)
        weight_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(weight_frame, text="Weights:").pack(side=tk.LEFT)
        self.weight_var = tk.StringVar(value="1/yerr (default)")
        ttk.Combobox(
            weight_frame,
            textvariable=self.weight_var,
            values=WEIGHT_MODES,
            state="readonly",
            width=20,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # --- Max nfev ---
        nfev_frame = ttk.Frame(self)
        nfev_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(nfev_frame, text="Max nfev:").pack(side=tk.LEFT)
        self.max_nfev_var = tk.StringVar(value="")
        ttk.Entry(nfev_frame, textvariable=self.max_nfev_var, width=10).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Label(nfev_frame, text="(empty = unlimited)").pack(side=tk.LEFT, padx=2)

        # --- Scale covariance ---
        self.scale_covar_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self, text="Scale covariance by reduced \u03c7\u00b2 (assume model is correct)",
            variable=self.scale_covar_var,
        ).pack(anchor=tk.W, pady=(3, 0))

        # --- Fit buttons ---
        fit_btn_frame = ttk.Frame(self)
        fit_btn_frame.pack(fill=tk.X, pady=3)
        self._auto_guess_btn = ttk.Button(
            fit_btn_frame, text="Auto Guess", command=self._auto_guess
        )
        self._auto_guess_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        self._fit_btn = ttk.Button(fit_btn_frame, text="Fit", command=self._fit)
        self._fit_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self._batch_fit_btn = ttk.Button(
            fit_btn_frame, text="Batch Fit", command=self._batch_fit
        )
        self._batch_fit_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self._clear_fit_btn = ttk.Button(
            fit_btn_frame, text="Clear Model", command=self._clear_fit
        )
        self._clear_fit_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

    def _series_selected(self, event=None):
        if self._on_series_selected:
            self._on_series_selected(self.series_var.get())

    def _session_selected(self, event=None):
        sel = self.session_listbox.curselection()
        if sel and self._on_session_selected:
            name = self._session_names[sel[0]]
            self._on_session_selected(name)

    def _new_session(self):
        # Generate a default name like "Fit 1", "Fit 2", ...
        existing = set(self._session_names)
        n = 1
        while f"Fit {n}" in existing:
            n += 1
        default = f"Fit {n}"
        name = simpledialog.askstring(
            "New Fit Session", "Session name:", initialvalue=default, parent=self
        )
        if name and self._on_new_session:
            self._on_new_session(name.strip())

    def _rename_session(self):
        sel = self.session_listbox.curselection()
        if not sel:
            return
        old_name = self._session_names[sel[0]]
        new_name = simpledialog.askstring(
            "Rename Session", "New name:", initialvalue=old_name, parent=self
        )
        if new_name and new_name.strip() != old_name and self._on_rename_session:
            self._on_rename_session(old_name, new_name.strip())

    def _delete_session(self):
        sel = self.session_listbox.curselection()
        if sel and self._on_delete_session:
            name = self._session_names[sel[0]]
            self._on_delete_session(name)

    def _toggle_session_visible(self):
        sel = self.session_listbox.curselection()
        if sel and self._on_toggle_session_visible:
            name = self._session_names[sel[0]]
            self._on_toggle_session_visible(name)

    def _toggle_series_visible(self):
        if self._on_toggle_series_visible:
            self._on_toggle_series_visible()

    def set_series_list(self, series_ids: list[str], select: str = ""):
        """Update the series combobox."""
        self.series_combo["values"] = series_ids
        if select and select in series_ids:
            self.series_var.set(select)
        elif series_ids:
            self.series_var.set(series_ids[0])
        else:
            self.series_var.set("")

    def set_sessions(self, session_names: list[str], select: str = "",
                     visibility: dict[str, bool] | None = None,
                     series_label: str = ""):
        """Update the session listbox. visibility maps name -> visible flag."""
        if series_label:
            self._sess_label.config(text=f"Sessions for: {series_label}")
        else:
            self._sess_label.config(text="Fit Sessions:")
        self._session_names = list(session_names)
        self.session_listbox.delete(0, tk.END)
        for name in session_names:
            if visibility is not None and not visibility.get(name, True):
                label = f"[hidden] {name}"
            else:
                label = name
            self.session_listbox.insert(tk.END, label)
        if select and select in session_names:
            idx = session_names.index(select)
            self.session_listbox.selection_set(idx)
            self.session_listbox.see(idx)

    def _add_component(self):
        name = self.model_var.get()
        if not name or not self._on_add_component:
            return
        expression = ""
        if name == "Expression":
            expression = simpledialog.askstring(
                "Expression Model",
                "Enter math expression (use x as independent variable):",
                parent=self,
            )
            if not expression:
                return
        elif name == "Spline":
            n_knots = simpledialog.askinteger(
                "Spline Model",
                "Number of knots (4\u2013100):",
                initialvalue=8,
                minvalue=4,
                maxvalue=100,
                parent=self,
            )
            if n_knots is None:
                return
            expression = f"knots:{n_knots}"
        self._on_add_component(name, self.operator_var.get(), expression=expression)

    def _remove_component(self):
        sel = self.comp_listbox.curselection()
        if sel and self._on_remove_component:
            self._on_remove_component(sel[0])

    def _edit_expression(self, event=None):
        if self._model_locked:
            return
        sel = self.comp_listbox.curselection()
        if not sel or not self._on_edit_expression:
            return
        idx = sel[0]
        label = self.comp_listbox.get(idx)
        if "Expression:" not in label:
            return
        # Extract current expression from label
        current = label.split("Expression:", 1)[1].strip()
        # Strip trailing prefix like " (expression1_)" or leading operator
        if " (" in current:
            current = current[:current.rfind(" (")]
        new_expr = simpledialog.askstring(
            "Edit Expression",
            "Expression (use x as independent variable):",
            initialvalue=current,
            parent=self,
        )
        if new_expr and new_expr != current:
            self._on_edit_expression(idx, new_expr)

    def _auto_guess(self):
        if self._on_auto_guess:
            self._on_auto_guess()

    def _fit(self):
        if self._on_fit:
            self._on_fit()

    def _batch_fit(self):
        if self._on_batch_fit:
            self._on_batch_fit()

    def _clear_fit(self):
        if self._on_clear_fit:
            self._on_clear_fit()

    def set_fitting_state(self, fitting: bool):
        """Toggle the Fit button between Fit and Abort modes, and lock out
        every control that mutates the model or parameters while a
        background fit thread is reading/writing them."""
        if fitting:
            self._fit_btn.configure(text="Abort", command=self._abort)
        else:
            self._fit_btn.configure(text="Fit", command=self._fit)

        self._model_locked = fitting
        state = tk.DISABLED if fitting else tk.NORMAL
        for btn in (
            self._add_comp_btn,
            self._remove_comp_btn,
            self._auto_guess_btn,
            self._batch_fit_btn,
            self._clear_fit_btn,
            self._new_session_btn,
            self._rename_session_btn,
            self._delete_session_btn,
        ):
            btn.configure(state=state)

    def _abort(self):
        if self._on_abort:
            self._on_abort()

    def set_components(self, components: list[str]):
        """Update the component listbox."""
        self.comp_listbox.delete(0, tk.END)
        for c in components:
            self.comp_listbox.insert(tk.END, c)


class FitResultsPanel(ttk.LabelFrame):
    """Parameter table and fit report display."""

    def __init__(self, parent, on_param_edited=None):
        super().__init__(parent, text="Fit Results", padding=5)
        self._on_param_edited = on_param_edited
        self._editing_entry = None
        self._editing_done = True  # no edit in progress
        self._locked = False
        self._build_ui()

    def set_locked(self, locked: bool):
        """Disable cell editing while a background fit is mutating params."""
        self._locked = locked
        if locked and self._editing_entry is not None:
            # Mark done first so the <FocusOut> that destroy() triggers
            # doesn't re-enter commit()/cancel() on a destroyed widget.
            self._editing_done = True
            self._editing_entry.destroy()
            self._editing_entry = None

    def _build_ui(self):
        # --- Goodness-of-fit summary ---
        self.gof_var = tk.StringVar(value="")
        self.gof_label = ttk.Label(self, textvariable=self.gof_var,
                                   font=("TkDefaultFont", 9, "bold"))
        self.gof_label.pack(anchor=tk.W, pady=(0, 3))

        # --- Parameter Treeview ---
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("value", "init", "stderr", "min", "max", "vary", "expr")
        self.param_tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=6
        )
        self.param_tree.heading("value", text="Value")
        self.param_tree.heading("init", text="Initial")
        self.param_tree.heading("stderr", text="StdErr")
        self.param_tree.heading("min", text="Min")
        self.param_tree.heading("max", text="Max")
        self.param_tree.heading("vary", text="Vary")
        self.param_tree.heading("expr", text="Expr")

        self.param_tree.column("value", width=80)
        self.param_tree.column("init", width=70)
        self.param_tree.column("stderr", width=80)
        self.param_tree.column("min", width=60)
        self.param_tree.column("max", width=60)
        self.param_tree.column("vary", width=40)
        self.param_tree.column("expr", width=120)

        # Add Name as a "tree" column
        self.param_tree["show"] = ("tree", "headings")
        self.param_tree.column("#0", width=200, stretch=False)
        self.param_tree.heading("#0", text="Name")

        # Alternating row colors
        self.param_tree.tag_configure("even", background="#f0f0f0")
        self.param_tree.tag_configure("odd", background="#ffffff")

        tree_scroll = ttk.Scrollbar(
            tree_frame, orient=tk.VERTICAL, command=self.param_tree.yview
        )
        self.param_tree.configure(yscrollcommand=tree_scroll.set)
        self.param_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Double-click to edit
        self.param_tree.bind("<Double-1>", self._on_double_click)

        # --- Copy button ---
        ttk.Button(self, text="Copy Table", command=self._copy_table).pack(
            anchor=tk.E, pady=(2, 0))

        # --- Fit report ---
        ttk.Label(self, text="Fit Report:").pack(anchor=tk.W, pady=(5, 0))
        report_frame = ttk.Frame(self)
        report_frame.pack(fill=tk.BOTH, expand=True)
        self.report_text = tk.Text(report_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        report_scroll = ttk.Scrollbar(
            report_frame, orient=tk.VERTICAL, command=self.report_text.yview
        )
        self.report_text.configure(yscrollcommand=report_scroll.set)
        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        report_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def set_params(self, params: dict[str, dict]):
        """Populate parameter table from param info dicts."""
        self.param_tree.delete(*self.param_tree.get_children())
        for i, (name, info) in enumerate(params.items()):
            val = f"{info['value']:.6g}" if info["value"] is not None else ""
            init_val = f"{info['init_value']:.6g}" if info.get("init_value") is not None else ""
            stderr = f"{info['stderr']:.6g}" if info.get("stderr") is not None else ""
            mn = f"{info['min']:.6g}" if info["min"] not in (None, float("-inf")) else "-inf"
            mx = f"{info['max']:.6g}" if info["max"] not in (None, float("inf")) else "inf"
            vary = "Yes" if info.get("vary", True) else "No"
            expr = info.get("expr") or ""
            tag = "even" if i % 2 == 0 else "odd"
            self.param_tree.insert("", tk.END, text=name, values=(val, init_val, stderr, mn, mx, vary, expr), tags=(tag,))

    def set_gof(self, gof: dict | None):
        """Update the goodness-of-fit summary line."""
        if not gof:
            self.gof_var.set("")
            return
        parts = []
        if "chi-squared" in gof and gof["chi-squared"] is not None:
            parts.append(f"\u03c7\u00b2={gof['chi-squared']:.4g}")
        if "reduced chi-squared" in gof and gof["reduced chi-squared"] is not None:
            parts.append(f"\u03c7\u00b2/\u03bd={gof['reduced chi-squared']:.4g}")
        if "R-squared" in gof and gof["R-squared"] is not None:
            parts.append(f"R\u00b2={gof['R-squared']:.6f}")
        if "AIC" in gof and gof["AIC"] is not None:
            parts.append(f"AIC={gof['AIC']:.4g}")
        if "BIC" in gof and gof["BIC"] is not None:
            parts.append(f"BIC={gof['BIC']:.4g}")
        self.gof_var.set("  ".join(parts))

    def set_report(self, report: str):
        set_readonly_text(self.report_text, report)

    def _copy_table(self):
        """Copy parameter table to clipboard as tab-separated text."""
        rows = self.param_tree.get_children()
        if not rows:
            return
        header = "Name\tValue\tInitial\tStdErr\tMin\tMax\tVary\tExpr"
        lines = [header]
        for item in rows:
            name = self.param_tree.item(item, "text")
            vals = self.param_tree.item(item, "values")
            lines.append(f"{name}\t" + "\t".join(str(v) for v in vals))
        # Append GOF summary if present
        gof_text = self.gof_var.get()
        if gof_text:
            lines.append("")
            lines.append(gof_text)
        text = "\n".join(lines)
        self.clipboard_clear()
        self.clipboard_append(text)

    def clear(self):
        self.gof_var.set("")
        self.param_tree.delete(*self.param_tree.get_children())
        set_readonly_text(self.report_text, "")

    def _on_double_click(self, event):
        """Handle double-click to edit a cell in the parameter treeview."""
        if self._locked:
            return
        region = self.param_tree.identify_region(event.x, event.y)
        if region != "cell":
            return

        item = self.param_tree.identify_row(event.y)
        column = self.param_tree.identify_column(event.x)
        if not item or not column:
            return

        # column is like "#1", "#2", etc.
        col_idx = int(column.replace("#", "")) - 1
        col_names = ("value", "init", "stderr", "min", "max", "vary", "expr")
        if col_idx < 0 or col_idx >= len(col_names):
            return

        col_name = col_names[col_idx]
        # Only allow editing value, min, max, vary, expr
        if col_name not in ("value", "min", "max", "vary", "expr"):
            return

        # Get cell bbox
        bbox = self.param_tree.bbox(item, column)
        if not bbox:
            return

        current_val = self.param_tree.set(item, column)
        param_name = self.param_tree.item(item, "text")

        # Create entry overlay
        if self._editing_entry:
            self._editing_entry.destroy()

        entry = ttk.Entry(self.param_tree, width=10)
        entry.place(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
        entry.insert(0, current_val)
        entry.select_range(0, tk.END)
        entry.focus_set()
        self._editing_entry = entry
        self._editing_done = False

        def commit(e=None):
            # Destroying the Entry below triggers a deferred <FocusOut> on
            # it, which would otherwise re-enter commit()/cancel() on an
            # already-destroyed widget and raise TclError. Guard with
            # self._editing_done (shared with set_locked()) instead of a
            # closure-local flag, since set_locked() can also destroy this
            # entry from outside these closures.
            if self._editing_done:
                return
            self._editing_done = True
            new_val = entry.get()
            entry.destroy()
            self._editing_entry = None
            self.param_tree.set(item, column, new_val)
            if self._on_param_edited:
                self._on_param_edited(param_name, col_name, new_val)

        def cancel(e=None):
            if self._editing_done:
                return
            self._editing_done = True
            entry.destroy()
            self._editing_entry = None

        entry.bind("<Return>", commit)
        entry.bind("<Escape>", cancel)
        entry.bind("<FocusOut>", commit)


