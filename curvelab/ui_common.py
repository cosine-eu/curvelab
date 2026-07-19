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
