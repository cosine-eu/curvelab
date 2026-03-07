"""All tkinter panel widgets: DataPanel, PlotControlPanel, FitPanel, FontDialog."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import tkinter.font as tkfont

from .fit_manager import REDUCE_FUNCTIONS, WEIGHT_MODES
from .models import MODEL_NAMES

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
        on_add_series=None,
        on_plot=None,
        on_dataset_selected=None,
        on_remove_dataset=None,
        on_toggle_series_visible=None,
        on_column_calc=None,
    ):
        super().__init__(parent, text="Data", padding=5)
        self._on_load = on_load
        self._on_add_series = on_add_series
        self._on_plot = on_plot
        self._on_dataset_selected = on_dataset_selected
        self._on_remove_dataset = on_remove_dataset
        self._on_toggle_series_visible = on_toggle_series_visible
        self._on_column_calc = on_column_calc
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
                ("All supported", "*.csv *.tsv *.txt *.dat *.xlsx *.xls *.json *.parquet *.sqlite *.db"),
                ("CSV", "*.csv"),
                ("TSV", "*.tsv"),
                ("Text", "*.txt *.dat"),
                ("Excel", "*.xlsx *.xls"),
                ("JSON", "*.json"),
                ("Parquet", "*.parquet"),
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

    def set_columns(self, columns: list[str], filename: str = ""):
        """Populate dropdowns with column names."""
        err_columns = [""] + columns
        self.x_combo["values"] = columns
        self.y_combo["values"] = columns
        self.yerr_combo["values"] = err_columns
        self.xerr_combo["values"] = err_columns
        if columns:
            self.x_var.set(columns[0])
            self.y_var.set(columns[1] if len(columns) > 1 else columns[0])
        else:
            self.x_var.set("")
            self.y_var.set("")
        self.yerr_var.set("")
        self.xerr_var.set("")

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
        if self._on_add_series:
            self._on_add_series(series_info)

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
        if sel:
            idx = sel[0]
            self.series_listbox.delete(idx)
            self._series_items.pop(idx)

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
        xscale.bind("<<ComboboxSelected>>", lambda e: self._fire_xscale())

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
        yscale.bind("<<ComboboxSelected>>", lambda e: self._fire_yscale())

        self.data_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row1, text="Data", variable=self.data_var, command=self._fire_data
        ).pack(side=tk.LEFT, padx=5)

        self.grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row1, text="Grid", variable=self.grid_var, command=self._fire_grid
        ).pack(side=tk.LEFT, padx=5)

        self.equal_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1, text="Equal Axes", variable=self.equal_var, command=self._fire_equal
        ).pack(side=tk.LEFT, padx=5)

        self.legend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row1, text="Legend", variable=self.legend_var, command=self._fire_legend
        ).pack(side=tk.LEFT, padx=5)

        self.show_params_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1, text="Params", variable=self.show_params_var,
            command=self._fire_show_params,
        ).pack(side=tk.LEFT, padx=5)

        self.fit_visible_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1, text="Fit visible range", variable=self.fit_visible_var,
        ).pack(side=tk.LEFT, padx=5)

        self.residuals_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1, text="Residuals", variable=self.residuals_var,
            command=self._fire_residuals,
        ).pack(side=tk.LEFT, padx=5)

        self.confidence_band_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1, text="Conf. band", variable=self.confidence_band_var,
            command=self._fire_confidence_band,
        ).pack(side=tk.LEFT, padx=5)

        self.band_sigma_var = tk.StringVar(value="1")
        ttk.Combobox(
            row1, textvariable=self.band_sigma_var,
            values=["1", "2", "3"], state="readonly", width=3,
        ).pack(side=tk.LEFT)
        ttk.Label(row1, text="\u03c3").pack(side=tk.LEFT, padx=(0, 5))

        self.exclude_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1, text="Exclude pts", variable=self.exclude_var,
        ).pack(side=tk.LEFT, padx=5)

        # --- Row 2: Axis labels ---
        row2 = ttk.Frame(self)
        row2.pack(fill=tk.X, pady=(2, 0))

        self.xlabel_var = tk.StringVar()
        self.ylabel_var = tk.StringVar()

        ttk.Label(row2, text="X Label:").pack(side=tk.LEFT)
        xlabel_entry = ttk.Entry(row2, textvariable=self.xlabel_var, width=15)
        xlabel_entry.pack(side=tk.LEFT, padx=(0, 10))
        xlabel_entry.bind("<Return>", lambda e: self._fire_axis_labels())
        xlabel_entry.bind("<FocusOut>", lambda e: self._fire_axis_labels())

        ttk.Label(row2, text="Y Label:").pack(side=tk.LEFT)
        ylabel_entry = ttk.Entry(row2, textvariable=self.ylabel_var, width=15)
        ylabel_entry.pack(side=tk.LEFT, padx=(0, 10))
        ylabel_entry.bind("<Return>", lambda e: self._fire_axis_labels())
        ylabel_entry.bind("<FocusOut>", lambda e: self._fire_axis_labels())

    def _fire_xscale(self):
        if self._on_xscale:
            self._on_xscale(self.xscale_var.get())

    def _fire_yscale(self):
        if self._on_yscale:
            self._on_yscale(self.yscale_var.get())

    def _fire_data(self):
        if self._on_data_toggled:
            self._on_data_toggled(self.data_var.get())

    def _fire_grid(self):
        if self._on_grid:
            self._on_grid(self.grid_var.get())

    def _fire_equal(self):
        if self._on_equal:
            self._on_equal(self.equal_var.get())

    def _fire_legend(self):
        if self._on_legend:
            self._on_legend(self.legend_var.get())

    def _fire_show_params(self):
        if self._on_show_params_toggled:
            self._on_show_params_toggled(self.show_params_var.get())

    def _fire_residuals(self):
        if self._on_residuals_toggled:
            self._on_residuals_toggled(self.residuals_var.get())

    def _fire_confidence_band(self):
        if self._on_confidence_band_toggled:
            self._on_confidence_band_toggled(self.confidence_band_var.get())

    def _fire_axis_labels(self):
        if self._on_axis_labels:
            self._on_axis_labels(self.xlabel_var.get(), self.ylabel_var.get())


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
        on_param_changed=None,
        on_series_selected=None,
        on_session_selected=None,
        on_new_session=None,
        on_rename_session=None,
        on_delete_session=None,
        on_batch_fit=None,
        on_toggle_session_visible=None,
        on_abort=None,
    ):
        super().__init__(parent, text="Fit", padding=5)
        self._on_add_component = on_add_component
        self._on_remove_component = on_remove_component
        self._on_edit_expression = on_edit_expression
        self._on_auto_guess = on_auto_guess
        self._on_fit = on_fit
        self._on_clear_fit = on_clear_fit
        self._on_param_changed = on_param_changed
        self._on_series_selected = on_series_selected
        self._on_session_selected = on_session_selected
        self._on_new_session = on_new_session
        self._on_rename_session = on_rename_session
        self._on_delete_session = on_delete_session
        self._on_batch_fit = on_batch_fit
        self._on_toggle_session_visible = on_toggle_session_visible
        self._on_abort = on_abort
        self._session_names: list[str] = []

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

        # --- Session controls ---
        sess_label_frame = ttk.Frame(self)
        sess_label_frame.pack(fill=tk.X)
        ttk.Label(sess_label_frame, text="Fit Sessions:").pack(side=tk.LEFT)

        self.session_listbox = tk.Listbox(self, height=3, exportselection=False)
        self.session_listbox.pack(fill=tk.X, pady=2)
        self.session_listbox.bind("<<ListboxSelect>>", self._session_selected)

        sess_btn_frame = ttk.Frame(self)
        sess_btn_frame.pack(fill=tk.X, pady=(0, 3))
        ttk.Button(sess_btn_frame, text="New", command=self._new_session).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2)
        )
        ttk.Button(sess_btn_frame, text="Rename", command=self._rename_session).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )
        ttk.Button(sess_btn_frame, text="Delete", command=self._delete_session).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )
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
        ttk.Button(
            comp_btn_frame, text="Add Component", command=self._add_component
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(
            comp_btn_frame, text="Remove", command=self._remove_component
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # --- Component list ---
        self.comp_listbox = tk.Listbox(self, height=3)
        self.comp_listbox.pack(fill=tk.X, pady=3)
        self.comp_listbox.bind("<Double-1>", self._edit_expression)

        # --- Method selection ---
        method_frame = ttk.Frame(self)
        method_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="leastsq")
        ttk.Combobox(
            method_frame,
            textvariable=self.method_var,
            values=[
                "leastsq", "least_squares", "nelder", "powell",
                "cobyla", "lbfgsb",
                "differential_evolution", "basinhopping",
                "dual_annealing", "shgo", "ampgo",
                "brute", "emcee",
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
        ttk.Button(fit_btn_frame, text="Auto Guess", command=self._auto_guess).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2)
        )
        self._fit_btn = ttk.Button(fit_btn_frame, text="Fit", command=self._fit)
        self._fit_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(fit_btn_frame, text="Batch Fit", command=self._batch_fit).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )
        ttk.Button(fit_btn_frame, text="Clear", command=self._clear_fit).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0)
        )

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
                     visibility: dict[str, bool] | None = None):
        """Update the session listbox. visibility maps name -> visible flag."""
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
        """Toggle the Fit button between Fit and Abort modes."""
        if fitting:
            self._fit_btn.configure(text="Abort", command=self._abort)
        else:
            self._fit_btn.configure(text="Fit", command=self._fit)

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
        self._build_ui()

    def _build_ui(self):
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

    def set_report(self, report: str):
        self.report_text.config(state=tk.NORMAL)
        self.report_text.delete("1.0", tk.END)
        self.report_text.insert("1.0", report)
        self.report_text.config(state=tk.DISABLED)

    def clear(self):
        self.param_tree.delete(*self.param_tree.get_children())
        self.report_text.config(state=tk.NORMAL)
        self.report_text.delete("1.0", tk.END)
        self.report_text.config(state=tk.DISABLED)

    def _on_double_click(self, event):
        """Handle double-click to edit a cell in the parameter treeview."""
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

        cancelled = False

        def commit(e=None):
            if cancelled:
                return
            new_val = entry.get()
            entry.destroy()
            self._editing_entry = None
            self.param_tree.set(item, column, new_val)
            if self._on_param_edited:
                self._on_param_edited(param_name, col_name, new_val)

        def cancel(e=None):
            nonlocal cancelled
            cancelled = True
            entry.destroy()
            self._editing_entry = None

        entry.bind("<Return>", commit)
        entry.bind("<Escape>", cancel)
        entry.bind("<FocusOut>", commit)


class FontDialog(tk.Toplevel):
    """Dialog for setting UI and plot fonts."""

    SIZES = [7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 24]

    def __init__(self, parent, ui_family="", ui_size=10, plot_family="", plot_size=10, on_apply=None):
        super().__init__(parent)
        self.title("Font Settings")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._on_apply = on_apply
        families = sorted(tkfont.families())

        # --- UI Font ---
        ui_frame = ttk.LabelFrame(self, text="UI Font", padding=10)
        ui_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        ttk.Label(ui_frame, text="Family:").grid(row=0, column=0, sticky=tk.W)
        self.ui_family_var = tk.StringVar(value=ui_family)
        ui_fam = ttk.Combobox(ui_frame, textvariable=self.ui_family_var, values=families, width=25)
        ui_fam.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(ui_frame, text="Size:").grid(row=1, column=0, sticky=tk.W)
        self.ui_size_var = tk.IntVar(value=ui_size)
        ttk.Combobox(
            ui_frame, textvariable=self.ui_size_var, values=self.SIZES, width=5
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # --- Plot Font ---
        plot_frame = ttk.LabelFrame(self, text="Plot Font", padding=10)
        plot_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(plot_frame, text="Family:").grid(row=0, column=0, sticky=tk.W)
        self.plot_family_var = tk.StringVar(value=plot_family)
        plot_fam = ttk.Combobox(plot_frame, textvariable=self.plot_family_var, values=families, width=25)
        plot_fam.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(plot_frame, text="Size:").grid(row=1, column=0, sticky=tk.W)
        self.plot_size_var = tk.IntVar(value=plot_size)
        ttk.Combobox(
            plot_frame, textvariable=self.plot_size_var, values=self.SIZES, width=5
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # --- Buttons ---
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Apply", command=self._apply).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)

    def _apply(self):
        if self._on_apply:
            self._on_apply(
                ui_family=self.ui_family_var.get(),
                ui_size=self.ui_size_var.get(),
                plot_family=self.plot_family_var.get(),
                plot_size=self.plot_size_var.get(),
            )
        self.destroy()


class ModelComparisonDialog(tk.Toplevel):
    """Side-by-side comparison of fit sessions: AIC, BIC, reduced chi-squared."""

    def __init__(self, parent, rows: list[dict]):
        """rows: list of dicts with keys session, model, n_params, chisqr, redchi, aic, bic."""
        super().__init__(parent)
        self.title("Model Comparison")
        self.resizable(True, True)
        self.transient(parent)
        self.geometry("700x300")

        columns = ("model", "n_params", "chisqr", "redchi", "aic", "bic")
        tree = ttk.Treeview(self, columns=columns, show=("tree", "headings"), height=10)
        tree.column("#0", width=120, stretch=False)
        tree.heading("#0", text="Session")
        tree.heading("model", text="Model")
        tree.column("model", width=150)
        tree.heading("n_params", text="N params")
        tree.column("n_params", width=70)
        tree.heading("chisqr", text="\u03c7\u00b2")
        tree.column("chisqr", width=90)
        tree.heading("redchi", text="\u03c7\u00b2/\u03bd")
        tree.column("redchi", width=90)
        tree.heading("aic", text="AIC")
        tree.column("aic", width=90)
        tree.heading("bic", text="BIC")
        tree.column("bic", width=90)

        tree.tag_configure("best_aic", background="#d4edda")
        tree.tag_configure("even", background="#f0f0f0")
        tree.tag_configure("odd", background="#ffffff")

        # Find best AIC for highlighting
        aic_vals = [r["aic"] for r in rows if r["aic"] is not None]
        best_aic = min(aic_vals) if aic_vals else None

        for i, r in enumerate(rows):
            tags = []
            if best_aic is not None and r["aic"] == best_aic:
                tags.append("best_aic")
            else:
                tags.append("even" if i % 2 == 0 else "odd")
            tree.insert(
                "", tk.END, text=r["session"],
                values=(
                    r["model"],
                    r["n_params"],
                    f"{r['chisqr']:.4g}" if r["chisqr"] is not None else "",
                    f"{r['redchi']:.4g}" if r["redchi"] is not None else "",
                    f"{r['aic']:.4g}" if r["aic"] is not None else "",
                    f"{r['bic']:.4g}" if r["bic"] is not None else "",
                ),
                tags=tuple(tags),
            )

        scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 10))


class FTestDialog(tk.Toplevel):
    """F-test for nested model comparison between two fit sessions."""

    def __init__(self, parent, sessions: dict[str, dict]):
        """sessions: {name: {n_params, chisqr, n_data}}."""
        super().__init__(parent)
        self.title("F-Test for Nested Models")
        self.resizable(False, False)
        self.transient(parent)
        self._sessions = sessions
        names = list(sessions.keys())

        # Session selectors
        sel_frame = ttk.LabelFrame(self, text="Select two sessions (simpler vs more complex)", padding=5)
        sel_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(sel_frame, text="Reduced model (fewer params):").grid(row=0, column=0, sticky=tk.W)
        self._reduced_var = tk.StringVar(value=names[0] if names else "")
        ttk.Combobox(sel_frame, textvariable=self._reduced_var,
                     values=names, state="readonly", width=25).grid(row=0, column=1, padx=5)

        ttk.Label(sel_frame, text="Full model (more params):").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self._full_var = tk.StringVar(value=names[1] if len(names) > 1 else "")
        ttk.Combobox(sel_frame, textvariable=self._full_var,
                     values=names, state="readonly", width=25).grid(row=1, column=1, padx=5, pady=(5, 0))

        ttk.Button(self, text="Compute", command=self._compute).pack(pady=5)

        self._result_text = tk.Text(self, height=10, width=60, wrap=tk.WORD,
                                     font=("Courier", 10), state=tk.DISABLED)
        self._result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 10))

    def _compute(self):
        import scipy.stats as stats

        r_name = self._reduced_var.get()
        f_name = self._full_var.get()
        if r_name == f_name:
            self._show_result("Select two different sessions.")
            return
        if r_name not in self._sessions or f_name not in self._sessions:
            self._show_result("Select valid sessions.")
            return

        r = self._sessions[r_name]
        f = self._sessions[f_name]

        p1, p2 = r["n_params"], f["n_params"]
        chi1, chi2 = r["chisqr"], f["chisqr"]
        n = r["n_data"]

        # Ensure reduced model has fewer params
        if p1 >= p2:
            self._show_result(
                f"Reduced model ({r_name}) has {p1} params, "
                f"full model ({f_name}) has {p2} params.\n\n"
                f"The reduced model must have fewer parameters than the full model."
            )
            return

        if chi2 >= chi1:
            self._show_result(
                f"Full model has equal or worse \u03c7\u00b2 ({chi2:.4g}) "
                f"than reduced model ({chi1:.4g}).\n\n"
                f"The extra parameters do not improve the fit."
            )
            return

        df1 = p2 - p1  # extra parameters
        df2 = n - p2    # residual DOF of full model

        if df2 <= 0:
            self._show_result("Not enough data points for this comparison.")
            return

        f_stat = ((chi1 - chi2) / df1) / (chi2 / df2)
        p_value = stats.f.sf(f_stat, df1, df2)

        lines = [
            f"Reduced model: {r_name}",
            f"  Parameters: {p1},  \u03c7\u00b2 = {chi1:.6g}",
            f"",
            f"Full model: {f_name}",
            f"  Parameters: {p2},  \u03c7\u00b2 = {chi2:.6g}",
            f"",
            f"Extra parameters: {df1}",
            f"Residual DOF:     {df2}",
            f"",
            f"F-statistic: {f_stat:.4f}",
            f"p-value:     {p_value:.6g}",
            f"",
        ]
        if p_value < 0.01:
            lines.append("The extra parameters significantly improve the fit (p < 0.01).")
        elif p_value < 0.05:
            lines.append("The extra parameters marginally improve the fit (p < 0.05).")
        else:
            lines.append("The extra parameters do NOT significantly improve the fit.")
            lines.append("The simpler model is preferred.")

        self._show_result("\n".join(lines))

    def _show_result(self, text: str):
        self._result_text.config(state=tk.NORMAL)
        self._result_text.delete("1.0", tk.END)
        self._result_text.insert("1.0", text)
        self._result_text.config(state=tk.DISABLED)


class ConfidenceIntervalDialog(tk.Toplevel):
    """Display confidence interval report in monospace text."""

    def __init__(self, parent, ci_text: str):
        super().__init__(parent)
        self.title("Confidence Intervals")
        self.resizable(True, True)
        self.transient(parent)
        self.geometry("600x400")

        text = tk.Text(self, wrap=tk.NONE, font=("Courier", 10))
        text.insert("1.0", ci_text)
        text.config(state=tk.DISABLED)

        xscroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=text.xview)
        yscroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=text.yview)
        text.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)

        text.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=(10, 0))
        yscroll.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=(10, 0))
        xscroll.grid(row=1, column=0, sticky="ew", padx=(10, 0))

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        ttk.Button(self, text="Close", command=self.destroy).grid(
            row=2, column=0, columnspan=2, pady=10
        )


class CorrelationMatrixDialog(tk.Toplevel):
    """Display parameter correlation matrix in a Treeview."""

    def __init__(self, parent, correlations: dict[str, dict[str, float]]):
        super().__init__(parent)
        self.title("Correlation Matrix")
        self.resizable(True, True)
        self.transient(parent)
        self.geometry("700x400")

        # Collect all parameter names
        all_params = list(correlations.keys())
        if not all_params:
            ttk.Label(self, text="No correlations available.").pack(padx=20, pady=20)
            ttk.Button(self, text="Close", command=self.destroy).pack(pady=10)
            return

        tree = ttk.Treeview(self, columns=all_params, show=("tree", "headings"))
        tree.column("#0", width=120, stretch=False)
        tree.heading("#0", text="Parameter")

        for p in all_params:
            tree.heading(p, text=p)
            tree.column(p, width=80)

        tree.tag_configure("even", background="#f0f0f0")
        tree.tag_configure("odd", background="#ffffff")

        for i, name in enumerate(all_params):
            vals = []
            for other in all_params:
                if name == other:
                    vals.append("1.000")
                elif other in correlations.get(name, {}):
                    vals.append(f"{correlations[name][other]:.3f}")
                else:
                    vals.append("")
            tag = "even" if i % 2 == 0 else "odd"
            tree.insert("", tk.END, text=name, values=tuple(vals), tags=(tag,))

        scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 10))


class BruteCandidatesDialog(tk.Toplevel):
    """Display brute-force candidates with option to load one."""

    def __init__(self, parent, candidates: list[dict], on_select=None):
        super().__init__(parent)
        self.title("Brute-Force Candidates")
        self.resizable(True, True)
        self.transient(parent)
        self.geometry("700x400")
        self._on_select = on_select
        self._candidates = candidates

        if not candidates:
            ttk.Label(self, text="No candidates available.").pack(padx=20, pady=20)
            ttk.Button(self, text="Close", command=self.destroy).pack(pady=10)
            return

        # Get parameter names from first candidate
        param_names = list(candidates[0]["params"].keys())
        columns = ("score",) + tuple(param_names)

        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        self._tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=15
        )
        self._tree.heading("score", text="Score")
        self._tree.column("score", width=100)
        for p in param_names:
            self._tree.heading(p, text=p)
            self._tree.column(p, width=90)

        self._tree.tag_configure("even", background="#f0f0f0")
        self._tree.tag_configure("odd", background="#ffffff")
        self._tree.tag_configure("best", background="#d4edda")

        for i, cand in enumerate(candidates):
            vals = [f"{cand['score']:.6g}"]
            for p in param_names:
                vals.append(f"{cand['params'][p]:.6g}")
            tag = "best" if i == 0 else ("even" if i % 2 == 0 else "odd")
            self._tree.insert("", tk.END, values=tuple(vals), tags=(tag,))

        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Load Selected", command=self._load_selected).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(btn_frame, text="Close", command=self.destroy).pack(side=tk.LEFT)

    def _load_selected(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = self._tree.index(sel[0])
        if self._on_select and idx < len(self._candidates):
            self._on_select(self._candidates[idx]["params"])
            self.destroy()


class EmceeSummaryDialog(tk.Toplevel):
    """Display emcee MCMC summary statistics."""

    def __init__(self, parent, flatchain, params_info: dict):
        super().__init__(parent)
        self.title("MCMC (emcee) Summary")
        self.resizable(True, True)
        self.transient(parent)
        self.geometry("600x400")

        text = tk.Text(self, wrap=tk.NONE, font=("Courier", 10))

        try:
            import pandas as pd
            if isinstance(flatchain, pd.DataFrame):
                lines = [f"{'Parameter':<20s} {'Median':>12s} {'Mean':>12s} "
                         f"{'Std':>12s} {'2.5%':>12s} {'97.5%':>12s}"]
                lines.append("-" * 80)
                for col in flatchain.columns:
                    data = flatchain[col]
                    lines.append(
                        f"{col:<20s} {data.median():>12.6g} {data.mean():>12.6g} "
                        f"{data.std():>12.6g} {data.quantile(0.025):>12.6g} "
                        f"{data.quantile(0.975):>12.6g}"
                    )
                text.insert("1.0", "\n".join(lines))
            else:
                text.insert("1.0", "Flatchain data not available as DataFrame.")
        except Exception as e:
            text.insert("1.0", f"Error processing emcee results: {e}")

        text.config(state=tk.DISABLED)

        xscroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=text.xview)
        yscroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=text.yview)
        text.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)

        text.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=(10, 0))
        yscroll.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=(10, 0))
        xscroll.grid(row=1, column=0, sticky="ew", padx=(10, 0))

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        ttk.Button(self, text="Close", command=self.destroy).grid(
            row=2, column=0, columnspan=2, pady=10
        )


class DiagnosticPlotsDialog(tk.Toplevel):
    """2x2 diagnostic plot grid: residuals vs fitted, Q-Q, scale-location, ACF."""

    def __init__(self, parent, fit_result):
        super().__init__(parent)
        self.title("Fit Diagnostic Plots")
        self.resizable(True, True)
        self.transient(parent)
        self.geometry("800x600")

        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        import scipy.stats as stats

        r = fit_result
        residuals = r.y_data - r.y_fit_data
        fitted = r.y_fit_data

        # Standardize residuals
        std_resid = residuals - residuals.mean()
        s = residuals.std()
        if s > 0:
            std_resid = std_resid / s

        fig = Figure(figsize=(8, 6))

        # 1. Residuals vs Fitted
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.scatter(fitted, residuals, s=12, alpha=0.7)
        ax1.axhline(0, color="red", linestyle="--", linewidth=0.8)
        ax1.set_xlabel("Fitted values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs Fitted")

        # 2. Normal Q-Q
        ax2 = fig.add_subplot(2, 2, 2)
        stats.probplot(std_resid, plot=ax2)
        ax2.set_title("Normal Q-Q")

        # 3. Scale-Location
        ax3 = fig.add_subplot(2, 2, 3)
        sqrt_abs_resid = np.sqrt(np.abs(std_resid))
        ax3.scatter(fitted, sqrt_abs_resid, s=12, alpha=0.7)
        ax3.set_xlabel("Fitted values")
        ax3.set_ylabel("\u221a|Standardized residuals|")
        ax3.set_title("Scale-Location")

        # 4. Autocorrelation
        ax4 = fig.add_subplot(2, 2, 4)
        n = len(residuals)
        max_lag = min(20, n - 1)
        mean_r = residuals.mean()
        var_r = np.sum((residuals - mean_r) ** 2)
        acf = []
        for lag in range(max_lag + 1):
            c = np.sum((residuals[:n - lag] - mean_r) * (residuals[lag:] - mean_r))
            acf.append(c / var_r if var_r > 0 else 0.0)
        lags = np.arange(max_lag + 1)
        ax4.bar(lags, acf, width=0.4, color="steelblue")
        ci = 1.96 / np.sqrt(n)
        ax4.axhline(ci, color="red", linestyle="--", linewidth=0.8)
        ax4.axhline(-ci, color="red", linestyle="--", linewidth=0.8)
        ax4.axhline(0, color="black", linewidth=0.5)
        ax4.set_xlabel("Lag")
        ax4.set_ylabel("ACF")
        ax4.set_title("Autocorrelation")

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Residual statistics summary ---
        stats_lines = []

        # Durbin-Watson statistic
        diff_resid = np.diff(residuals)
        ss_resid = np.sum(residuals ** 2)
        if ss_resid > 0:
            dw = np.sum(diff_resid ** 2) / ss_resid
            if dw < 1.5:
                dw_interp = "positive autocorrelation (model may be systematically wrong)"
            elif dw > 2.5:
                dw_interp = "negative autocorrelation"
            else:
                dw_interp = "no significant autocorrelation"
            stats_lines.append(f"Durbin-Watson: {dw:.4f} — {dw_interp}")

        # Runs test (sign changes in residuals)
        signs = np.sign(residuals)
        signs = signs[signs != 0]  # drop zeros
        if len(signs) >= 10:
            n_pos = int(np.sum(signs > 0))
            n_neg = int(np.sum(signs < 0))
            n_total = n_pos + n_neg
            runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
            # Expected runs and variance under H0 (random sequence)
            expected = 1 + 2 * n_pos * n_neg / n_total
            var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n_total)) / (
                n_total ** 2 * (n_total - 1)
            )
            if var_runs > 0:
                z_runs = (runs - expected) / np.sqrt(var_runs)
                p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
                if p_runs < 0.05:
                    runs_interp = "non-random pattern (systematic misfit)"
                else:
                    runs_interp = "consistent with random residuals"
                stats_lines.append(
                    f"Runs test: {runs} runs (expected {expected:.1f}), "
                    f"z = {z_runs:.3f}, p = {p_runs:.4f} — {runs_interp}"
                )

        if stats_lines:
            stats_frame = ttk.LabelFrame(self, text="Residual Statistics", padding=5)
            stats_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
            for line in stats_lines:
                ttk.Label(stats_frame, text=line, wraplength=750,
                          justify=tk.LEFT).pack(anchor=tk.W)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=5)


class ConfidenceContourDialog(tk.Toplevel):
    """Interactive 2D confidence contour plot using lmfit.conf_interval2d."""

    def __init__(self, parent, last_result, vary_params: list[str]):
        super().__init__(parent)
        self.title("2D Confidence Contours")
        self.resizable(True, True)
        self.transient(parent)
        self.geometry("700x600")

        self._last_result = last_result
        self._vary_params = vary_params

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        # Controls frame
        ctrl = ttk.Frame(self, padding=5)
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="Param X:").pack(side=tk.LEFT)
        self._x_var = tk.StringVar(value=vary_params[0])
        ttk.Combobox(
            ctrl, textvariable=self._x_var, values=vary_params,
            state="readonly", width=15,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="Param Y:").pack(side=tk.LEFT, padx=(10, 0))
        self._y_var = tk.StringVar(value=vary_params[1] if len(vary_params) > 1 else vary_params[0])
        ttk.Combobox(
            ctrl, textvariable=self._y_var, values=vary_params,
            state="readonly", width=15,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="Grid:").pack(side=tk.LEFT, padx=(10, 0))
        self._grid_var = tk.IntVar(value=10)
        ttk.Spinbox(
            ctrl, textvariable=self._grid_var, from_=5, to=50, width=4,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(ctrl, text="Compute", command=self._compute).pack(side=tk.LEFT, padx=(10, 0))

        # Status
        self._status_var = tk.StringVar(value="Select parameters and click Compute.")
        ttk.Label(self, textvariable=self._status_var).pack(fill=tk.X, padx=10)

        # Plot area
        self._fig = Figure(figsize=(6, 5))
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=5)

    def _compute(self):
        x_name = self._x_var.get()
        y_name = self._y_var.get()
        if x_name == y_name:
            self._status_var.set("Select two different parameters.")
            return
        nx = ny = self._grid_var.get()
        self._status_var.set("Computing...")
        self.update_idletasks()

        try:
            from lmfit import conf_interval2d
            x_arr, y_arr, grid = conf_interval2d(
                self._last_result, self._last_result,
                x_name, y_name, nx=nx, ny=ny,
            )
            self._ax.clear()
            self._ax.contourf(x_arr, y_arr, grid, cmap="coolwarm")
            self._ax.set_xlabel(x_name)
            self._ax.set_ylabel(y_name)
            self._ax.set_title(f"Confidence: {x_name} vs {y_name}")
            self._fig.tight_layout()
            self._canvas.draw()
            self._status_var.set("Done.")
        except Exception as e:
            self._status_var.set(f"Error: {e}")


class GlobalFitDialog(tk.Toplevel):
    """Dialog for global fitting across multiple series with shared parameters."""

    def __init__(self, parent, series_info: list[dict], base_param_names: list[str],
                 on_fit=None):
        """
        series_info: list of dicts with keys 'id' and 'label'.
        base_param_names: list of parameter names from the model.
        on_fit: callback(selected_series_ids, shared_param_names).
        """
        super().__init__(parent)
        self.title("Global Fit")
        self.resizable(True, True)
        self.transient(parent)
        self.geometry("500x500")
        self._on_fit = on_fit

        # --- Series selection ---
        series_frame = ttk.LabelFrame(self, text="Series", padding=5)
        series_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        self._series_vars = {}
        for info in series_info:
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(series_frame, text=info["label"], variable=var).pack(
                anchor=tk.W
            )
            self._series_vars[info["id"]] = var

        # --- Parameter sharing ---
        param_frame = ttk.LabelFrame(self, text="Shared Parameters", padding=5)
        param_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        canvas = tk.Canvas(param_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(param_frame, orient=tk.VERTICAL, command=canvas.yview)
        inner_frame = ttk.Frame(canvas)

        inner_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._param_vars = {}
        for name in base_param_names:
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(inner_frame, text=name, variable=var).pack(anchor=tk.W)
            self._param_vars[name] = var

        # --- Buttons ---
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self._status_var = tk.StringVar(value="")
        ttk.Label(btn_frame, textvariable=self._status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(btn_frame, text="Fit", command=self._do_fit).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)

    def _do_fit(self):
        selected_ids = [sid for sid, var in self._series_vars.items() if var.get()]
        shared = {name for name, var in self._param_vars.items() if var.get()}
        if len(selected_ids) < 2:
            self._status_var.set("Select at least 2 series.")
            return
        self._status_var.set("Fitting...")
        self.update_idletasks()
        if self._on_fit:
            try:
                self._on_fit(selected_ids, shared)
                self.destroy()
            except Exception as e:
                self._status_var.set(f"Error: {e}")


class UncertaintyPropagationDialog(tk.Toplevel):
    """Evaluate expressions with propagated uncertainties using ufloats."""

    def __init__(self, parent, uvars: dict):
        """uvars: dict mapping param name to ufloat (from lmfit result.uvars)."""
        super().__init__(parent)
        self.title("Uncertainty Propagation")
        self.resizable(True, True)
        self.transient(parent)
        self.geometry("550x400")
        self._uvars = uvars

        # --- Available variables ---
        var_frame = ttk.LabelFrame(self, text="Available Variables", padding=5)
        var_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        var_list = tk.Listbox(var_frame, height=8)
        var_scroll = ttk.Scrollbar(var_frame, orient=tk.VERTICAL, command=var_list.yview)
        var_list.configure(yscrollcommand=var_scroll.set)
        var_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        var_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        for name, uval in uvars.items():
            try:
                var_list.insert(tk.END, f"{name} = {uval.nominal_value:.6g} \u00b1 {uval.std_dev:.6g}")
            except AttributeError:
                var_list.insert(tk.END, f"{name} = {uval}")

        # --- Expression entry ---
        expr_frame = ttk.Frame(self, padding=5)
        expr_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(expr_frame, text="Expression:").pack(side=tk.LEFT)
        self._expr_var = tk.StringVar()
        expr_entry = ttk.Entry(expr_frame, textvariable=self._expr_var, width=40)
        expr_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        expr_entry.bind("<Return>", lambda e: self._evaluate())

        ttk.Button(expr_frame, text="Evaluate", command=self._evaluate).pack(side=tk.LEFT)

        # --- Result ---
        self._result_var = tk.StringVar(value="Enter an expression using parameter names above.")
        ttk.Label(self, textvariable=self._result_var, wraplength=500, justify=tk.LEFT).pack(
            fill=tk.X, padx=10, pady=5
        )

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 10))

    def _evaluate(self):
        expr = self._expr_var.get().strip()
        if not expr:
            self._result_var.set("Enter an expression.")
            return
        try:
            from uncertainties import umath
            # Build namespace with uvars and umath functions
            ns = dict(self._uvars)
            for fname in dir(umath):
                if not fname.startswith("_"):
                    ns[fname] = getattr(umath, fname)
            result = eval(expr, {"__builtins__": {}}, ns)
            try:
                self._result_var.set(
                    f"{expr} = {result.nominal_value:.6g} \u00b1 {result.std_dev:.6g}"
                )
            except AttributeError:
                self._result_var.set(f"{expr} = {result}")
        except Exception as e:
            self._result_var.set(f"Error: {e}")


class SimulateDataDialog(tk.Toplevel):
    """Dialog for generating synthetic data from the current model."""

    def __init__(self, parent, x_min=0.0, x_max=10.0, n_points=200, on_generate=None):
        super().__init__(parent)
        self.title("Simulate Data")
        self.resizable(False, False)
        self.transient(parent)
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

        noise_cfg = {
            "gaussian": self._gauss_on.get(),
            "gaussian_sigma": float(self._gauss_sigma.get()) if self._gauss_on.get() else 0,
            "poisson": self._poisson_on.get(),
            "poisson_scale": float(self._poisson_scale.get()) if self._poisson_on.get() else 0,
            "jitter": self._jitter_on.get(),
            "jitter_sigma": float(self._jitter_sigma.get()) if self._jitter_on.get() else 0,
        }

        if self._on_generate:
            try:
                self._on_generate(x_min, x_max, n_points, noise_cfg)
                self.destroy()
            except Exception as e:
                self._status_var.set(f"Error: {e}")


class ColumnCalculatorDialog(tk.Toplevel):
    """Dialog for creating new columns from expressions on existing columns."""

    def __init__(self, parent, columns: list[str], on_apply=None):
        super().__init__(parent)
        self.title("Column Calculator")
        self.resizable(True, False)
        self.transient(parent)
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
