"""Workspace JSON encoder/decoder utilities (no GUI imports)."""

from __future__ import annotations

import json
import math

import numpy as np

from .session import SeriesRecord, FitSession, FitResult
from .fit_manager import FitManager


class WorkspaceEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and special float values."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if math.isinf(v):
                return "Infinity" if v > 0 else "-Infinity"
            if math.isnan(v):
                return None
            return v
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def encode_value(v):
    """Recursively encode special float values in nested structures."""
    if isinstance(v, float):
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        if math.isnan(v):
            return None
        return v
    if isinstance(v, np.ndarray):
        return {"__ndarray__": v.tolist()}
    if isinstance(v, dict):
        return {k: encode_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [encode_value(item) for item in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return encode_value(float(v))
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def decode_workspace(obj):
    """JSON object_hook that restores ndarray and special floats."""
    if "__ndarray__" in obj:
        return np.array(obj["__ndarray__"])
    # Restore string-encoded infinities wherever they appear: encode_value
    # turns any inf float into these markers, not just param min/max bounds
    # (e.g. gof["reduced chi-squared"] can be inf when dof <= 0).
    for key, val in obj.items():
        if isinstance(val, str):
            if val == "Infinity":
                obj[key] = float("inf")
            elif val == "-Infinity":
                obj[key] = float("-inf")
    return obj


def serialize_series_records(
    series_records: dict[str, SeriesRecord],
) -> dict:
    """Serialize all series records and their fit sessions to a dict."""
    series = {}
    for sid, rec in series_records.items():
        fit_sessions = {}
        for sess_name, sess in rec.fit_sessions.items():
            sess_data = {
                "name": sess.name,
                "color": sess.color,
                "visible": sess.visible,
                **sess.fit_manager.serialize(),
            }
            if sess.result is not None:
                r = sess.result
                sess_data["result"] = encode_value({
                    "x_dense": r.x_dense,
                    "y_fit_dense": r.y_fit_dense,
                    "x_data": r.x_data,
                    "y_data": r.y_data,
                    "y_fit_data": r.y_fit_data,
                    "yerr_data": r.yerr_data,
                    "y_uncertainty": r.y_uncertainty,
                    "component_curves": {
                        k: v for k, v in r.component_curves.items()
                    },
                    "params": r.params,
                    "gof": r.gof,
                    "report": r.report,
                    "candidates": r.candidates,
                    "init_params": r.init_params,
                })
            fit_sessions[sess_name] = sess_data

        sdata = {
            "dataset_name": rec.dataset_name,
            "style": rec.style,
            "visible": rec.visible,
            "fit_sessions": fit_sessions,
            "active_session_name": rec.active_session_name,
        }
        if rec.mask is not None:
            sdata["mask"] = encode_value(rec.mask)
        series[sid] = sdata

    return series


def _to_array(v):
    """Convert lists back to numpy arrays."""
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, list):
        return np.array(v)
    return v


def deserialize_series_record(
    sdata: dict,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None,
    xerr: np.ndarray | None,
    dataset_name: str,
) -> SeriesRecord:
    """Reconstruct a SeriesRecord from workspace data and loaded arrays."""
    saved_mask = sdata.get("mask")
    if saved_mask is not None:
        saved_mask = np.asarray(saved_mask, dtype=bool)

    rec = SeriesRecord(
        x=x, y=y, yerr=yerr, xerr=xerr,
        style=sdata.get("style", {}),
        dataset_name=dataset_name,
        visible=sdata.get("visible", True),
        mask=saved_mask,
        active_session_name=sdata.get("active_session_name"),
    )

    for sess_name, sess_data in sdata.get("fit_sessions", {}).items():
        fm = FitManager.deserialize(sess_data)

        sess = FitSession(
            name=sess_data.get("name", sess_name),
            fit_manager=fm,
            color=sess_data.get("color", ""),
            visible=sess_data.get("visible", True),
        )

        rdata = sess_data.get("result")
        if rdata is not None:
            component_curves = {
                k: _to_array(v) for k, v in rdata.get("component_curves", {}).items()
            }
            sess.result = FitResult(
                x_dense=_to_array(rdata.get("x_dense", [])),
                y_fit_dense=_to_array(rdata.get("y_fit_dense", [])),
                x_data=_to_array(rdata.get("x_data", [])),
                y_data=_to_array(rdata.get("y_data", [])),
                y_fit_data=_to_array(rdata.get("y_fit_data", [])),
                yerr_data=_to_array(rdata["yerr_data"]) if rdata.get("yerr_data") is not None else None,
                y_uncertainty=_to_array(rdata["y_uncertainty"]) if rdata.get("y_uncertainty") is not None else None,
                component_curves=component_curves,
                params=rdata.get("params", {}),
                gof=rdata.get("gof", {}),
                report=rdata.get("report", ""),
                candidates=rdata.get("candidates"),
                init_params=rdata.get("init_params"),
            )

        rec.fit_sessions[sess_name] = sess

    return rec
