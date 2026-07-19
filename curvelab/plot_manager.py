"""Matplotlib figure management and all drawing logic."""

from dataclasses import dataclass

import numpy as np
import matplotlib
from .session import COMPONENT_COLORS
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

matplotlib.use("TkAgg")


@dataclass
class SeriesStyle:
    """Visual style for a data series."""

    marker: str = "o"
    linestyle: str = "None"
    color: str = ""  # Empty = auto
    markersize: float = 4.0
    label: str = ""


class PlotManager:
    """Owns the matplotlib Figure and Axes. All drawing goes through here."""

    def __init__(self, parent_frame):
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self._gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        self.ax = self.fig.add_subplot(self._gs[0])
        self.ax_resid = self.fig.add_subplot(self._gs[1], sharex=self.ax)
        self.ax_resid.set_visible(False)
        self.ax_resid.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent_frame)
        self.toolbar.update()

        self._series_lines = []  # Track plotted data series
        self._fit_lines: dict[str, list] = {}  # session_key -> list of lines
        self._annotations: dict[str, object] = {}  # session_key -> Text
        self._residual_lines: dict[str, list] = {}  # session_key -> artists on ax_resid
        self._show_legend = True
        self._show_grid = True
        self._residuals_visible = False
        self._xlabel = ""
        self._ylabel = ""

        self.ax.grid(self._show_grid)
        self.ax_resid.grid(self._show_grid)

    def get_canvas_widget(self):
        return self.canvas.get_tk_widget()

    def clear_all(self):
        """Clear everything from the axes."""
        # Save state before clear (ax.clear() resets to linear and removes labels)
        xscale = self.ax.get_xscale()
        yscale = self.ax.get_yscale()
        xlabel = self._xlabel
        ylabel = self._ylabel
        self.ax.clear()
        self.ax_resid.clear()
        # Re-establish shared X axis (ax.clear() breaks the link)
        self.ax_resid.sharex(self.ax)
        self.ax_resid.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        self._series_lines.clear()
        self._fit_lines.clear()
        self._annotations.clear()
        self._residual_lines.clear()
        self.ax.set_xscale(xscale)
        self.ax.set_yscale(yscale)
        self._xlabel = xlabel
        self._ylabel = ylabel
        if xlabel:
            self.ax.set_xlabel(xlabel)
        if ylabel:
            self.ax.set_ylabel(ylabel)
        self.ax.grid(self._show_grid)
        self.ax_resid.grid(self._show_grid)
        self._apply_tick_visibility()
        self.canvas.draw_idle()

    def plot_series(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray | None = None,
        xerr: np.ndarray | None = None,
        style: SeriesStyle | None = None,
    ):
        """Plot a data series with optional error bars."""
        style = style or SeriesStyle()
        kwargs = {
            "marker": style.marker,
            "linestyle": style.linestyle,
            "markersize": style.markersize,
            "label": style.label or None,
        }
        if style.color:
            kwargs["color"] = style.color

        if yerr is not None or xerr is not None:
            kwargs["capsize"] = 3
            container = self.ax.errorbar(x, y, yerr=yerr, xerr=xerr, **kwargs)
            self._series_lines.append(container)
        else:
            (line,) = self.ax.plot(x, y, **kwargs)
            self._series_lines.append(line)

        self._update_legend()
        self.canvas.draw_idle()

    def plot_fit(
        self,
        x_dense: np.ndarray,
        y_dense: np.ndarray,
        label: str = "Fit",
        color: str = "red",
        component_curves: dict[str, np.ndarray] | None = None,
        session_key: str = "",
        y_uncertainty: np.ndarray | None = None,
        show_band: bool = False,
    ):
        """Plot a fit curve and optional component curves.

        If session_key is provided, stores lines under that key (does NOT
        auto-clear).  Caller is responsible for clearing via
        clear_fit_session() before re-plotting.
        """
        lines: list = []

        (line,) = self.ax.plot(
            x_dense, y_dense, color=color, linewidth=2, label=label
        )
        lines.append(line)

        if show_band and y_uncertainty is not None:
            band = self.ax.fill_between(
                x_dense,
                y_dense - y_uncertainty,
                y_dense + y_uncertainty,
                alpha=0.2,
                color=color,
            )
            lines.append(band)

        if component_curves:
            for i, (comp_name, y_comp) in enumerate(component_curves.items()):
                comp_color = COMPONENT_COLORS[i % len(COMPONENT_COLORS)]
                (cl,) = self.ax.plot(
                    x_dense,
                    y_comp,
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                    color=comp_color,
                    label=comp_name.rstrip("_"),
                )
                lines.append(cl)

        if session_key:
            self._fit_lines[session_key] = lines
        else:
            self._fit_lines[""] = lines

        self._update_legend()
        self.canvas.draw_idle()

    def plot_residuals(self, session_key: str, x: np.ndarray, residuals: np.ndarray,
                       color: str = "red", markersize: float = 3.0):
        """Plot residual points on the residuals axes."""
        self.clear_residuals(session_key)
        artists = []
        (line,) = self.ax_resid.plot(
            x, residuals, marker="o", linestyle="None",
            color=color, markersize=markersize,
        )
        artists.append(line)
        self._residual_lines[session_key] = artists
        self.ax_resid.set_ylabel("Residuals", fontsize=8)
        self.canvas.draw_idle()

    def clear_residuals(self, session_key: str):
        """Remove residual artists for one session."""
        artists = self._residual_lines.pop(session_key, [])
        for a in artists:
            a.remove()
        self.canvas.draw_idle()

    def clear_all_residuals(self):
        """Remove all residual artists."""
        for artists in self._residual_lines.values():
            for a in artists:
                a.remove()
        self._residual_lines.clear()
        self.canvas.draw_idle()

    def set_data_visible(self, visible: bool):
        """Show or hide data series artists (points/error bars)."""
        from matplotlib.container import ErrorbarContainer
        for artist in self._series_lines:
            if isinstance(artist, ErrorbarContainer):
                for child in artist.get_children():
                    child.set_visible(visible)
            else:
                artist.set_visible(visible)
        self.canvas.draw_idle()

    def set_residuals_visible(self, visible: bool):
        """Show or hide the residuals subplot."""
        self._residuals_visible = visible
        self.ax_resid.set_visible(visible)
        self._apply_tick_visibility()
        self.canvas.draw_idle()

    def _apply_tick_visibility(self):
        """Manage x-tick labels: hide on main when residuals visible."""
        if self._residuals_visible:
            self.ax.tick_params(labelbottom=False)
        else:
            self.ax.tick_params(labelbottom=True)

    def clear_fit_session(self, session_key: str):
        """Remove only the fit lines + annotation + residuals for one session."""
        for line in self._fit_lines.pop(session_key, []):
            line.remove()
        self.remove_annotation(session_key)
        self.clear_residuals(session_key)
        self._update_legend()
        self.canvas.draw_idle()

    def rename_session_key(self, old_key: str, new_key: str):
        """Re-key a session's artists under a new session key."""
        for d in (self._fit_lines, self._annotations, self._residual_lines):
            if old_key in d:
                d[new_key] = d.pop(old_key)

    def clear_all_fits(self):
        """Remove all fit lines, annotations, and residuals."""
        keys = set(self._fit_lines) | set(self._annotations) | set(self._residual_lines)
        for key in keys:
            self.clear_fit_session(key)

    def annotate_params(
        self,
        params: dict[str, dict],
        gof: dict[str, float] | None = None,
        session_key: str = "",
    ):
        """Show parameter values and goodness-of-fit as a text box on the plot."""
        self.remove_annotation(session_key=session_key)
        lines = []
        if gof:
            for stat_name, stat_val in gof.items():
                lines.append(f"{stat_name} = {stat_val:.4g}")
            lines.append("")  # blank separator
        for name, info in params.items():
            val = info["value"]
            err = info["stderr"]
            if err is not None:
                lines.append(f"{name} = {val:.4g} \u00b1 {err:.4g}")
            else:
                lines.append(f"{name} = {val:.4g}")
        text = "\n".join(lines)

        # Offset annotations vertically based on how many already exist
        y_offset = 0.98 - 0.02 * len(self._annotations)

        annotation = self.ax.text(
            0.02,
            y_offset,
            text,
            transform=self.ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )
        self._annotations[session_key] = annotation
        self.canvas.draw_idle()

    def remove_annotation(self, session_key: str = ""):
        ann = self._annotations.pop(session_key, None)
        if ann is not None:
            ann.remove()
            self.canvas.draw_idle()

    def set_xscale(self, scale: str):
        self.ax.set_xscale(scale)
        self.canvas.draw_idle()

    def set_yscale(self, scale: str):
        self.ax.set_yscale(scale)
        self.canvas.draw_idle()

    def set_axis_labels(self, xlabel: str, ylabel: str):
        self._xlabel = xlabel
        self._ylabel = ylabel
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.canvas.draw_idle()

    def set_grid(self, enabled: bool):
        self._show_grid = enabled
        self.ax.grid(enabled)
        self.ax_resid.grid(enabled)
        self.canvas.draw_idle()

    def set_equal_aspect(self, enabled: bool):
        if enabled:
            # Save original limits before equal aspect modifies them
            self._saved_xlim = self.ax.get_xlim()
            self._saved_ylim = self.ax.get_ylim()
            self.ax.set_aspect("equal", adjustable="datalim")
        else:
            self.ax.set_aspect("auto")
            # Restore original limits
            if hasattr(self, "_saved_xlim"):
                self.ax.set_xlim(self._saved_xlim)
                self.ax.set_ylim(self._saved_ylim)
        # Residuals axis always uses auto aspect
        self.ax_resid.set_aspect("auto")
        self.canvas.draw_idle()

    def set_legend(self, enabled: bool):
        self._show_legend = enabled
        self._update_legend()
        self.canvas.draw_idle()

    def set_font(self, family: str, size: int):
        """Apply font to all plot text elements."""
        matplotlib.rcParams.update({
            "font.family": family,
            "font.size": size,
        })
        for ax in (self.ax, self.ax_resid):
            ax.title.set_fontfamily(family)
            ax.title.set_fontsize(size)
            ax.xaxis.label.set_fontfamily(family)
            ax.xaxis.label.set_fontsize(size)
            ax.yaxis.label.set_fontfamily(family)
            ax.yaxis.label.set_fontsize(size)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily(family)
                label.set_fontsize(size)
        legend = self.ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontfamily(family)
                text.set_fontsize(size)
        for ann in self._annotations.values():
            ann.set_fontfamily(family)
            ann.set_fontsize(max(size - 2, 6))
        self.canvas.draw_idle()

    def _update_legend(self):
        legend = self.ax.get_legend()
        if self._show_legend:
            handles, labels = self.ax.get_legend_handles_labels()
            if handles:
                self.ax.legend()
            elif legend:
                legend.remove()
        elif legend:
            legend.remove()
