"""Shared Tkinter building blocks for CurveLab dialogs."""

import tkinter as tk


class BaseDialog(tk.Toplevel):
    """Toplevel with the common dialog setup: title, transient parent,
    optional fixed size, resizability, and modality."""

    def __init__(self, parent, title: str, size: str | None = None,
                 resizable: tuple[bool, bool] = (True, True), modal: bool = False):
        super().__init__(parent)
        self.title(title)
        self.resizable(*resizable)
        self.transient(parent)
        if size:
            self.geometry(size)
        if modal:
            self.grab_set()


# Alternating row tints for Treeview tables, plus the highlight used for a
# "best" row (lowest AIC, top brute-force candidate).
ROW_TINTS = {"even": "#f0f0f0", "odd": "#ffffff", "best": "#d4edda"}


def configure_row_tags(tree):
    """Apply the shared row tints to a Treeview."""
    for tag, color in ROW_TINTS.items():
        tree.tag_configure(tag, background=color)


def row_tag(index: int) -> str:
    """Alternating row tag for a zero-based row index."""
    return "even" if index % 2 == 0 else "odd"


def set_readonly_text(text: tk.Text, content: str):
    """Replace the content of a disabled (read-only) Text widget."""
    text.config(state=tk.NORMAL)
    text.delete("1.0", tk.END)
    if content:
        text.insert("1.0", content)
    text.config(state=tk.DISABLED)
