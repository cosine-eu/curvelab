# CurveLab TODO

Parked feature ideas, with enough context to resume the discussion.

## Model plugin system

Let users register their own lmfit models without touching CurveLab code.
The architecture is ready: everything works off `MODEL_REGISTRY` and the
factory contract `factory(prefix=...) -> lmfit.Model` (models.py).

Sketch:

- Plugin = a `.py` file in a well-known user directory exposing a
  `CURVELAB_MODELS = {"Display Name": factory}` dict. Values may be a
  `lmfit.Model` subclass (can provide `guess()`) or a plain function with
  an `x` argument and default parameter values (auto-wrapped in
  `Model(func, prefix=...)`).
- Loader (`plugins.py`, GUI-free): scan the directory at startup, import,
  validate, merge via `register_model(name, factory)`. A broken plugin
  reports its error and is skipped — never crashes startup.
- UI: refresh model combobox after load; menu entries "Open Model Folder"
  and "Reload Model Plugins".
- Workspace: loading a workspace that references a missing plugin model
  must warn clearly ("model X comes from a plugin that isn't installed"),
  not KeyError. Optionally record the plugin filename in the workspace.

Decisions still open:

1. Plugin directory: `~/.config/curvelab/models/` (platformdirs) vs
   `~/.curvelab/models/`.
2. Discovery: explicit `CURVELAB_MODELS` dict (recommended) vs
   auto-registering everything found in the file.
3. Name collisions with built-ins: refuse with warning (recommended) vs
   allow override.
4. Trust: plugins are arbitrary Python (unlike sandboxed Expression
   components) — load only from the user's own plugin directory.

Later extension: setuptools entry points so pip-installable packages can
contribute models. Verify the notebook widget picks up plugin models when
it is reworked.

Estimated effort: loader + registry hook + tests ~half a day; menu polish
and workspace messaging a bit more.

## 2D models

lmfit's engine is ready: `Model(func, independent_vars=["x", "y"])` keeps
parameters, hints, prefixes, composition operators, weights, all fit
methods, and CIs working. What lmfit does NOT provide: 2D lineshapes
(only `Gaussian2dModel`), 2D auto-guess (moment-based guessing is easy to
write), any 2D visualization, and `eval_uncertainty` for multiple
independent vars only in recent versions (check against our minimum).

Options, in increasing ambition:

- **Option 0 (exists today):** `run_global_fit` with shared parameters
  covers "2D" data that is really a family of 1D curves (spectra vs
  temperature, scans vs angle).
- **Option 1 (pragmatic):** a contained 2D Fit tool — separate window
  that loads grid data (CSV x/y/z, npy, TIFF/FITS), composes from a small
  2D registry (Gaussian2D, rotated Gaussian, Moffat/Airy, plane/poly
  background), fits, and shows data/fit/residual image panels. Own small
  `Fit2DManager` + `Result2D`; reuses the registry pattern, parameter
  table, and plugin contract. Main 1D app untouched. Good fit for
  beam-spot / PSF characterization.
- **Option 2 (large):** full 2D integration — 2D series as first-class
  citizens through data manager, sessions, workspace, PlotManager image
  mode, region masking, batch over image stacks. Only if 2D becomes a
  primary workflow, and after the notebook/controller consolidation.

Design the plugin loader with 2D in mind: let plugins declare
`independent_vars` so the same `CURVELAB_MODELS` contract carries 2D
models.

Deciding question: what does the 2D data look like — detector images on a
regular grid, scattered (x, y, z) points, or families of 1D curves?

## Workspace (.clw) format hardening

Current state: single JSON file; data saved by reference (file paths only),
fit results embedded in full (arrays/params/gof/report); live lmfit result
reconstructed by refit-on-load; numpy/inf/nan handled by custom
encoder/decoder. A `"version": 1` field is written but never read —
compatibility today comes only from defensive `.get(key, default)` reads.

Improvements, in suggested order:

1. **Warn at save time** when a dataset has no reloadable file path
   (pasted or simulated data) — today it is silently dropped on the next
   load. Sharpest edge in the current format.
2. **Check `version` on load**: warn on files written by a newer CurveLab;
   define a bump policy (bump on any semantic change to the structure).
   Optionally also record the app version string for diagnostics.
3. **Write strict JSON**: `encode_value` turns inf into `"Infinity"`
   strings, but NaN inside a saved array reaches `json.dump` through
   `ndarray.tolist()` and is written as a bare `NaN` token (allow_nan
   defaults to True). Python reads it back fine; other JSON parsers
   reject it. Fix by encoding NaN the same way as inf (a marker string,
   decoded back on load) and passing `allow_nan=False` so any remaining
   case fails loudly instead of silently producing invalid JSON.
4. **Optional "embed data in workspace"**: store datasets inline (the
   ndarray machinery already exists) for self-contained, shareable files
   at the cost of size. Could auto-embed datasets that lack a file path,
   which would also fix item 1 properly.
