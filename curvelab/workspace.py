"""Workspace JSON encoder/decoder utilities (no GUI imports)."""

import json
import math

import numpy as np


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
    # Restore string-encoded infinities in param dicts
    for key in ("min", "max"):
        if key in obj and isinstance(obj[key], str):
            if obj[key] == "Infinity":
                obj[key] = float("inf")
            elif obj[key] == "-Infinity":
                obj[key] = float("-inf")
    return obj
