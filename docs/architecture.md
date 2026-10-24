# CurveLab Architecture

How the code is organized, for contributors. The diagrams are Mermaid and
render on GitHub. They describe the code as of version 0.12.0.

## Modules

Arrows point from a module to the modules it imports. Dashed arrows are
imports done inside a function, to avoid a cycle or a heavy dependency.

```mermaid
flowchart TD
    subgraph entry[Entry points]
        main[__main__]
        init[__init__]
    end

    subgraph desktop[Desktop frontend - Tkinter]
        app[app.py<br/>CurveLabApp]
        subgraph mixins[Mixins]
            fit_h[app_fit_handlers]
            ana_h[app_analysis_handlers]
            data_t[app_data_tools]
            plotting[app_plotting]
            series[app_series]
            ws_mix[app_workspace]
            menu[app_menu]
        end
        panels[ui_panels]
        dlg_a[ui_dialogs_analysis]
        dlg_d[ui_dialogs_data]
        common[ui_common]
    end

    nb[notebook.py<br/>CurveLabWidget]

    subgraph core[Core - GUI-agnostic]
        fm[fit_manager]
        models[models]
        session[session]
        dm[data_manager]
        pm[plot_manager]
        ws[workspace]
        pre[preprocessing]
        at[analysis_tools]
    end

    main --> app
    init --> app
    init -.-> nb
    app --> mixins
    app --> panels & dm & fm & pm & pre & session
    fit_h --> fm & dlg_a
    ana_h --> fm & dlg_a
    data_t --> at & dlg_d
    plotting --> pm & session
    series --> session
    ws_mix --> dm & fm & ws
    panels --> fm & models & common
    dlg_a --> at & fm & common
    dlg_d --> at & common
    nb --> dm & fm & models & pre & session & ws
    fm --> models & session
    session -.-> fm
    pm --> session
    pre --> session
    ws --> fm & session
```

The core modules never import a frontend, so the fitting engine can be used
from a script. `__init__` imports `app` unconditionally, so `import curvelab`
loads Tkinter even in a notebook.

## Data model

One `SeriesRecord` per plotted x/y pair, keyed by
`series_id = "dataset::x_col::y_col"`. Each series holds named fit sessions;
each session owns its own `FitManager` and the last `FitResult`.

```mermaid
classDiagram
    direction LR
    class DataManager {
        +load(path)
        +load_from_text(text)
        +add_dataframe(name, df)
        +get_column(dataset, column)
        +dataset_names()
    }
    class SeriesRecord {
        +x, y, yerr, xerr
        +style: dict
        +dataset_name: str
        +visible: bool
        +mask: ndarray
        +fit_sessions: dict
        +active_session_name: str
        +active_session()
        +ensure_session(name)
    }
    class FitSession {
        +name: str
        +fit_manager: FitManager
        +result: FitResult
        +color: str
        +visible: bool
        +undo_stack: list
        +redo_stack: list
    }
    class ParamEdit {
        +param_name: str
        +field: str
        +old_value
        +new_value
    }
    class FitResult {
        +params: dict
        +gof: dict
        +report: str
        +x_dense, y_fit_dense
        +component_curves: dict
        +y_uncertainty
        +fit_weights
        +fit_options: dict
        +success, message, nfev
        +mcmc: dict
        +comparison_key()
        +residuals()
        +weighted_residuals()
    }
    class FitManager {
        +components: list
        -_model: lmfit.Model
        -_params: lmfit.Parameters
        -_param_hints: dict
        -_start_values: dict
        -_last_result: ModelResult
    }

    DataManager ..> SeriesRecord : columns become
    SeriesRecord "1" *-- "*" FitSession
    FitSession "1" *-- "1" FitManager
    FitSession "1" o-- "0..1" FitResult
    FitSession "1" *-- "*" ParamEdit : undo/redo
    FitManager ..> FitResult : run_fit returns
```

## FitManager

The fitting engine wraps lmfit. Components are combined with `+ - * /`
into one composite model, with prefixed parameter names
(`gaussian1_center`). Parameter edits are stored twice: on the live
`Parameters`, and as hints that survive a model rebuild.

```mermaid
classDiagram
    class FitManager {
        +add_component(model_name, operator)
        +remove_component(index)
        +edit_expression(index, expr)
        +clear_components()
        +clone_components_to(target)
        +model_description()
        +params
        +set_param(name, ...)
        +set_param_hint(name, ...)
        +set_start_value(name, value)
        +capture_start_values()
        +auto_guess(x, y)
        +run_fit(x, y, ...) FitResult
        +refit_from_result(result)
        +run_global_fit(...)
        +run_odr(x, y, ...)
        +run_bootstrap(...)
        +compute_confidence_intervals()
        +compute_ci_profiles()
        +compute_ci_contour()
        +get_covariance_matrix()
        +get_correlations()
        +serialize() dict
        +deserialize(data)$
    }
    class FitOptions {
        +run_fit_kwargs() dict
    }
    class MethodCaps
    class FitComponent
    FitManager *-- "*" FitComponent
    FitManager ..> MethodCaps : METHOD_CAPS
    FitOptions ..> FitManager : run_fit kwargs
```

## Desktop application

`CurveLabApp` is a `ttk.Frame` assembled from mixins, one per area of the
UI. The mixins share state through `self`; they are split by topic, not
isolated from each other.

```mermaid
classDiagram
    class CurveLabApp
    CurveLabApp --|> MenuMixin
    CurveLabApp --|> SeriesSessionMixin
    CurveLabApp --|> PlottingMixin
    CurveLabApp --|> AnalysisHandlersMixin
    CurveLabApp --|> DataToolsMixin
    CurveLabApp --|> WorkspaceMixin
    CurveLabApp --|> FitHandlersMixin
    CurveLabApp --|> Frame
    CurveLabApp *-- DataManager
    CurveLabApp *-- PlotManager
    CurveLabApp *-- DataPanel
    CurveLabApp *-- PlotControlPanel
    CurveLabApp *-- FitPanel
    CurveLabApp *-- FitResultsPanel
    CurveLabApp o-- "*" SeriesRecord : _series_records
```

## Running a fit

Fast methods run on the Tk thread. The methods in `SLOW_METHODS` (emcee,
brute, the global optimizers) run on a background thread: the panel locks,
the Tk thread polls the thread, and Abort sets an event that an lmfit
`iter_cb` checks. ODR takes its own path through `run_odr`.

```mermaid
sequenceDiagram
    actor User
    participant App as CurveLabApp
    participant Thread as fit thread
    participant FM as FitManager
    participant lmfit
    participant UI as plot and panels

    User->>App: Fit
    App->>App: validate_fit_setup, read FitOptions
    alt fast method
        App->>FM: run_fit(x, y, yerr, xerr, method, options)
    else slow method
        App->>App: lock panel, Fit becomes Abort
        App->>Thread: start job
        Thread->>FM: run_fit(..., iter_cb)
        loop every 100 ms
            App->>Thread: is_alive?
        end
    end
    FM->>FM: reset to start values
    FM->>FM: prepare_fit_data (weights, drop sigma <= 0)
    FM->>lmfit: Model.fit(y, params, weights, method)
    lmfit-->>FM: ModelResult
    FM->>FM: keep _last_result, copy fitted params
    FM-->>App: FitResult
    App->>App: session.result = result, unlock
    App->>UI: plot fit, residuals, parameter table, report
    App->>User: warnings (no uncertainties, at a bound)
```

## Batch Fit

Batch Fit applies the active session's model to every plotted series, on
the Tk thread, with no abort.

```mermaid
sequenceDiagram
    actor User
    participant App as CurveLabApp
    participant Src as source FitManager
    participant Tgt as target FitManager
    participant UI as plot and summary

    User->>App: Batch Fit
    loop each plotted series (source included)
        App->>App: ensure_session(same name)
        App->>Src: clone_components_to(target)
        Src->>Tgt: components and parameter hints
        App->>Tgt: auto_guess(x, y)
        App->>Tgt: run_fit(x, y, ...)
        Tgt-->>App: FitResult
        App->>UI: plot fit
    end
    App->>UI: summary table, collected warnings
```

## Fit lifecycle

The states of a session's fit as the desktop app sees them.

```mermaid
stateDiagram-v2
    [*] --> NoModel
    NoModel --> Ready : add component
    Ready --> Ready : edit parameter, Auto Guess
    Ready --> Running : Fit (slow method)
    Ready --> Fitted : Fit (fast method)
    Running --> Fitted : thread finishes
    Running --> Aborted : Abort
    Running --> Ready : error dialog
    Aborted --> Ready
    Fitted --> Ready : edit parameter
    Fitted --> Fitted : Fit again, Use as Start
    Ready --> NoModel : Clear Model
    Fitted --> NoModel : Clear Model
```

While a fit is Running, the controls that change the model or parameters
are disabled, and menu items and shortcuts refuse with a message. An
aborted fit is not shown, but it still leaves its parameters in the
`FitManager` (a known issue).
