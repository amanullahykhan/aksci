# AK-SCI — Full Documentation

Version `0.1.0` · Author: Amanullah Khan

This is the complete reference for every public class, method, and behavior in AK-SCI, organized by category. For a quick overview and install instructions, see [README.md](README.md).

## Table of Contents

1. [Installation & Setup](#1-installation--setup)
2. [ErrorResolver — Error Diagnosis](#2-errorresolver--error-diagnosis)
3. [The Local Diagnostic Model](#3-the-local-diagnostic-model)
4. [Auto-fix Rules (Opt-in)](#4-auto-fix-rules-opt-in)
5. [MicroPipeline — Streaming](#5-micropipeline--streaming)
6. [UnifiedFrame, ml & stats](#6-unifiedframe--ml--stats)
7. [AIClient — Optional Cloud AI-Assist](#7-aiclient--optional-cloud-ai-assist)
8. [FAQ / Troubleshooting](#8-faq--troubleshooting)

---

## 1. Installation & Setup

```bash
pip install aksci                # core — works fully offline
pip install "aksci[ai]"          # + cloud AI-assist (adds the `anthropic` package)
pip install "aksci[dev]"         # + pytest, build (for contributors)
```

**Requirements:** Python 3.10, 3.11, or 3.12.

**Core dependencies** (installed automatically): `numpy`, `pandas`, `polars`, `pyarrow`, `scipy`, `scikit-learn`, `matplotlib`, `joblib`. These are version-ranged, not pinned, so AK-SCI fits into an existing project's environment without forcing exact versions.

**Import surface:**

```python
from aksci import (
    ErrorResolver,        # error diagnosis + auto-fix
    MicroPipeline,         # chunked streaming
    BoundedBuffer,         # fixed-capacity FIFO buffer
    UnifiedFrame,          # pandas/Polars-agnostic frame
    ml,                     # ML shortcuts
    stats,                  # stats shortcuts
    GradientDescentResult,  # return type of ml.gradient_descent
    AIClient,               # optional cloud AI-assist
)
```

---

## 2. ErrorResolver — Error Diagnosis

`aksci.ErrorResolver` is the central error-handling class. It never silently rewrites your code — by default it **explains and re-raises**. Every other behavior (swallowing an exception, auto-fixing a value) is something you explicitly opt into.

### Constructor

```python
ErrorResolver(ai_client: Optional[AIClient] = None, auto_fix: bool = False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ai_client` | `AIClient \| None` | `None` | If provided and configured with an API key, `.explain()` adds a deeper cloud-generated explanation after the local diagnosis. Optional — everything works without it. |
| `auto_fix` | `bool` | `False` | If `True`, `safe_run` will attempt the whitelisted auto-fix rules before falling back to `reraise`/`None`. See [§4](#4-auto-fix-rules-opt-in). |

### `.diagnose(exc) -> Diagnosis`

Classifies an exception using the local offline model. Returns a `Diagnosis` dataclass:

```python
@dataclass
class Diagnosis:
    category: str        # e.g. "missing_column"
    confidence: float     # 0.0–1.0
    explanation: str      # human-readable fix guidance
    suggestion: str       # same text as explanation
```

```python
try:
    df["revenu"]
except KeyError as exc:
    diag = resolver.diagnose(exc)
    print(diag.category, diag.confidence)
    # "missing_column" 0.84
```

### `.explain(exc) -> str`

Builds the full, human-readable diagnostic report (what gets printed by `safe_run` and the global hook). Includes the local diagnosis, and — if an `AIClient` was supplied and is available — an additional AI-generated explanation and suggested fix code.

```python
print(resolver.explain(exc))
# [AKSCI] Caught KeyError: 'price'
#   Category   : missing_column (confidence 84%)
#   Likely fix : This column name does not exist in the DataFrame...
```

### `.safe_run(func=None, *, reraise=True, auto_fix_context=None)`

The main decorator. Wraps a function so any exception raised inside it is diagnosed before deciding what happens next.

| Parameter | Type | Default | Behavior |
|---|---|---|---|
| `reraise` | `bool` | `True` | `True`: print the diagnosis, then re-raise (you still see the real traceback). `False`: print the diagnosis, swallow the exception, return `None` instead. |
| `auto_fix_context` | `Callable[..., dict] \| None` | `None` | A function taking the same arguments as the decorated function, returning a context `dict` for the auto-fix rules. Only used if the resolver was built with `auto_fix=True`. |

```python
resolver = ErrorResolver()

@resolver.safe_run()                    # reraise=True (default) — diagnosis + real traceback
def risky():
    return {"a": 1}["b"]

@resolver.safe_run(reraise=False)       # swallow, return None instead
def optional_step():
    return {"a": 1}["b"]
```

**Important limitation to know before using auto-fix here:** `safe_run` replaces the function's *entire return value* if a fix is applied — it cannot resume execution partway through a function that failed on its second or third risky line. Design decorated functions to do **one risky thing** (`df[col]`, `a / b`, `float(x)`) rather than several steps. For a multi-step function, split it into small single-purpose helpers and decorate each one. See [`examples/lead_scoring_pipeline.py`](examples/lead_scoring_pipeline.py) for this pattern applied to a realistic script.

### `.install_global_hook()`

Routes **all uncaught exceptions in the process** through AK-SCI's diagnostic report, printed before Python's normal traceback. Good for a top-level script or long-running automation job where you want every crash to start with a category and a suggested fix.

```python
resolver = ErrorResolver()
resolver.install_global_hook()
# From here on, any uncaught exception anywhere in the process prints
# the AK-SCI diagnosis first, then the normal traceback.
```

### `.suggest_column(missing, available) -> str | None` (static method)

Fuzzy-matches a mistyped column/key name against a list of real ones (cutoff `0.6`, via `difflib.get_close_matches`). Returns the closest match or `None` — it never modifies data on its own; you decide whether to use the suggestion.

```python
ErrorResolver.suggest_column("pric", ["price", "quantity", "customer_id"])
# -> "price"
```

### `.try_auto_fix(exc, context=None) -> AutoFixResult`

Runs the auto-fix rules directly against an exception, independent of `safe_run` and independent of the `auto_fix` constructor flag (that flag only controls whether `safe_run` calls this *automatically*). Always returns an `AutoFixResult` — check `.applied`. See [§4](#4-auto-fix-rules-opt-in).

---

## 3. The Local Diagnostic Model

`aksci.ai_core.diagnostics.LocalDiagnosticModel` is the offline classifier `ErrorResolver` uses under the hood. You normally interact with it through `ErrorResolver`, but it can be used directly:

```python
from aksci.ai_core.diagnostics import LocalDiagnosticModel

model = LocalDiagnosticModel()
diagnosis = model.diagnose("KeyError", "'price'")
print(diagnosis.category, diagnosis.confidence)
```

### How it works

- **Architecture:** TF-IDF vectorizer (1–2 grams) → Logistic Regression (`C=15`, tuned for this dataset — see the comment in `diagnostics.py` for the cross-validation reasoning).
- **Training data:** 134 hand-written examples across 16 categories (`src/aksci/ai_core/_training_data.py`).
- **Training time:** trains once, in-process, in a fraction of a second.
- **Caching:** the trained model is cached to disk at `src/aksci/ai_core/_diagnostic_model_v<N>.joblib`, where `<N>` is the training-set size — so upgrading AK-SCI with a larger/different training set automatically invalidates any stale cache instead of silently reusing old predictions.
- **No network calls.** Training and inference are both fully local.

### The 16 categories

| Category | Fires on | Example |
|---|---|---|
| `missing_column` | KeyError on a DataFrame column | `KeyError: 'price'` |
| `key_error_generic` | KeyError on a plain dict | `KeyError: 'api_key'` |
| `type_mismatch` | Incompatible types in an operation | `TypeError: unsupported operand type(s) for +: 'int' and 'str'` |
| `shape_mismatch` | numpy/sklearn array or matrix shape mismatch | `ValueError: shapes (3,4) and (3,) not aligned` |
| `division_by_zero` | Dividing by zero (scalar or array) | `ZeroDivisionError: division by zero` |
| `missing_module` | Package not installed / bad import | `ModuleNotFoundError: No module named 'requests'` |
| `index_out_of_range` | List/array/DataFrame positional index out of range | `IndexError: list index out of range` |
| `attribute_error` | Object missing an attribute/method (often `None`) | `AttributeError: 'NoneType' object has no attribute 'get'` |
| `null_values` | NaN/None/inf where clean numeric data expected | `ValueError: Input contains NaN` |
| `value_error_generic` | Malformed value passed to a function | `ValueError: could not convert string to float: 'N/A'` |
| `file_not_found` | Missing file/path on disk | `FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'` |
| `permission_denied` | OS-level permission errors on files/sockets | `PermissionError: [Errno 13] Permission denied` |
| `json_decode_error` | Malformed JSON parsing | `JSONDecodeError: Expecting value: line 1 column 1` |
| `connection_error` | Network/API connectivity failures | `ConnectionError: ...Max retries exceeded` |
| `memory_error` | Out-of-memory conditions | `MemoryError: Unable to allocate 4.00 GiB...` |
| `timeout_error` | Operation exceeded a time limit | `TimeoutError: [Errno 110] Connection timed out` |

### Known limitation (stated honestly)

Cross-validation shows overall held-out accuracy around **80%**. The two categories that are genuinely hardest to separate are `missing_column` and `key_error_generic` — both are `KeyError:` under the hood, and short error text alone doesn't always carry enough signal to tell "a DataFrame column" from "a plain dict key" apart. This is an inherent limit of classifying short technical strings, not a bug — treat the confidence score as a real signal (lower confidence = the model is genuinely less sure), not a formality.

### Extending the training data

If you add examples to `_training_data.py`, add a category, you should also:
1. Add a matching `FIX_TEMPLATES` entry in `diagnostics.py` for any new category.
2. Run `pytest tests/ -v` — `test_all_fix_templates_have_matching_category` will fail if you forget step 1.

---

## 4. Auto-fix Rules (Opt-in)

`aksci.error_handler.autofix` contains four whitelisted, conservative recovery rules. **Nothing here runs unless you explicitly opt in** via `ErrorResolver(auto_fix=True)` — and even then, only for `safe_run`-decorated functions that also supply `auto_fix_context`.

### Design principles

- **Whitelisted** — each rule only fires for the specific exception type it was written for, never as a generic catch-all.
- **Conservative** — if a rule can't be confident, it declines (`returns None`) rather than guessing.
- **Logged** — `try_fix` always returns an `AutoFixResult`, even when nothing changed; every applied fix prints to stderr.
- **Non-destructive** — rules recover a *value*; they never mutate source files, delete data, or hide the original exception from the diagnostic report.

### The `AutoFixResult` type

```python
@dataclass
class AutoFixResult:
    applied: bool             # did a rule fire?
    rule_name: str             # which rule (or "none")
    original_error: str        # str(exc)
    recovered_value: Any       # the value to use instead, if applied
    description: str           # human-readable summary of what happened
```

### The four rules

#### `missing_column_fuzzy_match`
- **Fires on:** `KeyError`
- **Needs in context:** `available_columns` (list of real column names), `frame` (the object to pull the matched column from)
- **Does:** fuzzy-matches the missing key against `available_columns` (cutoff `0.6`, same threshold as `ErrorResolver.suggest_column`) and returns `frame[best_match]`.

```python
resolver = ErrorResolver(auto_fix=True)
df = pd.DataFrame({"price": [10, 20], "qty": [1, 2]})

@resolver.safe_run(
    reraise=False,
    auto_fix_context=lambda row: {"available_columns": list(df.columns), "frame": df},
)
def get_price(row):
    return df["prise"]  # typo

get_price(None)
# stderr: [AKSCI auto-fix] missing_column_fuzzy_match: Column 'prise' not found; used closest match 'price' instead.
# -> returns df["price"]
```

#### `null_values_fillna`
- **Fires on:** `ValueError` whose message mentions "nan", "infinity", or "inf"
- **Needs in context:** `frame` (a DataFrame/Series/array), optional `fill_value` (default `0`)
- **Does:** calls `frame.fillna(fill_value)` (or `np.nan_to_num` if `frame` doesn't have `.fillna`) and returns the filled result.

```python
@resolver.safe_run(reraise=False, auto_fix_context=lambda df: {"frame": df, "fill_value": 0})
def fit(df):
    model.fit(df[["x"]], y)  # raises ValueError: Input contains NaN
    return df

# -> returns df with NaN filled to 0
```

#### `division_by_zero_guard`
- **Fires on:** `ZeroDivisionError`
- **Needs in context:** `default` (the fallback value to return)
- **Does:** returns `default` instead of raising.

```python
@resolver.safe_run(reraise=False, auto_fix_context=lambda a, b: {"default": 0.0})
def safe_divide(a, b):
    return a / b

safe_divide(10, 0)  # -> 0.0
```

#### `type_coercion`
- **Fires on:** `TypeError` or `ValueError`
- **Needs in context:** `raw_value` (the value to coerce), `target_type` (`int` or `float`)
- **Does:** strips whitespace and coerces `raw_value` to `target_type`.

```python
@resolver.safe_run(reraise=False, auto_fix_context=lambda s: {"raw_value": s, "target_type": float})
def parse_price(s):
    return float(s)

parse_price(" 19.99 ")  # -> 19.99
```

### Calling auto-fix directly (without `safe_run`)

```python
from aksci.error_handler import autofix

try:
    df["revenu"]
except KeyError as exc:
    result = autofix.try_fix(exc, {"available_columns": list(df.columns), "frame": df})
    if result.applied:
        print(result.description)
        value = result.recovered_value
```

### The critical limitation — read this before using auto-fix

`safe_run` replaces a decorated function's **entire return value**, not a single line inside it. If `score_lead(row)` does four risky operations and the third one fails, auto-fix can't "patch" that one line and let the function keep going — the whole call is abandoned, and the recovered value becomes the *entire* return value of `score_lead`.

**The fix:** wrap small, single-purpose helper functions — one column lookup, one division, one type coercion — rather than one large multi-step function. See [`examples/lead_scoring_pipeline.py`](examples/lead_scoring_pipeline.py) for a complete, realistic script built this way.

---

## 5. MicroPipeline — Streaming

`aksci.MicroPipeline` and `aksci.BoundedBuffer` handle memory-bounded, chunked data flow — the core idea is deliberately simple: **never materialize more than `chunk_size` rows in memory between stages.**

This is honest about its scope: it's a single-process, generator-based pipeline. It reduces memory pressure and keeps the code simple; it does **not** parallelize CPU-bound work across cores by itself. For that, wrap a CPU-bound, independent stage with `concurrent.futures.ProcessPoolExecutor`.

### `MicroPipeline(chunk_size=10_000)`

```python
pipeline = MicroPipeline(chunk_size=50_000)
pipeline.add_stage("drop_nulls", lambda df: df.dropna())
pipeline.add_stage("add_margin", lambda df: df.assign(margin=df["revenue"] - df["cost"]))

for processed_chunk in pipeline.run_csv("sales.csv"):
    processed_chunk.to_parquet("out.parquet", engine="pyarrow", append=True)
```

| Method | Description |
|---|---|
| `.add_stage(name, func) -> MicroPipeline` | Appends a named transformation stage. Chainable (`pipeline.add_stage(...).add_stage(...)`). |
| `.run(source: Iterable) -> Iterator` | Runs an in-memory iterable through all stages, lazily (generator-based). |
| `.run_csv(path, **read_csv_kwargs) -> Iterator[pd.DataFrame]` | Streams a CSV via pandas' native `chunksize` reader — at any moment, only one chunk is in memory. Extra kwargs pass through to `pd.read_csv`. |
| `.run_polars_lazy(path) -> pl.LazyFrame` | Uses Polars' lazy/streaming engine instead — stages become `.map_batches` transformations, evaluated only on `.collect(streaming=True)`. |

### `BoundedBuffer(maxsize=1000)`

A fixed-capacity FIFO buffer — pushing past `maxsize` drops the oldest item rather than growing unboundedly, guaranteeing the buffer itself never exceeds `maxsize` items regardless of total throughput.

```python
buf = BoundedBuffer(maxsize=100)
buf.push(item)
len(buf)          # current size
list(buf.drain()) # yields and removes all items, oldest first
```

---

## 6. UnifiedFrame, ml & stats

`aksci.unified_api` gives one consistent API over pandas and Polars for common operations, plus simplified shortcuts over scikit-learn and SciPy.

### `UnifiedFrame`

Wraps a pandas **or** Polars DataFrame and exposes the same operations for both.

```python
uf = UnifiedFrame(df)              # df can be pandas.DataFrame or polars.DataFrame
uf.backend                          # "pandas" or "polars"
uf.shape                            # (rows, cols)
uf.standardize(columns=None)        # z-score standardize numeric columns -> new UnifiedFrame
uf.dropna()                         # drop rows with any nulls -> new UnifiedFrame
uf.describe()                       # delegates to the backend's .describe()
uf.to_pandas()                      # convert to pandas.DataFrame (no-op if already pandas)
uf.to_polars()                      # convert to polars.DataFrame (no-op if already polars)
```

`standardize()` and `dropna()` return a **new** `UnifiedFrame` — they don't mutate in place.

### `ml`

Not a replacement for scikit-learn — a thin, teaching-friendly layer over it, plus one from-scratch algorithm.

| Method | Description |
|---|---|
| `ml.linear_regression(X, y)` | Fits `sklearn.linear_model.LinearRegression`. Returns the fitted model (`.coef_`, `.intercept_`, `.predict()` all work normally). |
| `ml.gradient_descent(X, y, learning_rate=0.01, epochs=500)` | Batch gradient descent written from scratch in explicit NumPy — every update step visible, no hidden optimizer. Returns a `GradientDescentResult`. |

```python
@dataclass
class GradientDescentResult:
    weights: np.ndarray
    bias: float
    loss_history: List[float]   # loss at each epoch, useful for plotting convergence
```

### `stats`

| Method | Description |
|---|---|
| `stats.ttest(a, b)` | Independent two-sample t-test. Returns SciPy's `TtestResult`. |
| `stats.correlation(a, b)` | Pearson correlation. Returns `{"r": float, "p_value": float}`. |

---

## 7. AIClient — Optional Cloud AI-Assist

Everything above works **fully offline**. `AIClient` is a strictly opt-in layer for deeper, natural-language explanations on top of the local diagnosis — it never runs automatically and never activates without an API key.

```python
from aksci import ErrorResolver, AIClient

client = AIClient(api_key="sk-ant-...")          # or set AKSCI_ANTHROPIC_API_KEY env var
resolver = ErrorResolver(ai_client=client)

resolver.explain(some_exception)
# Local diagnosis, plus an "AI insight" section and optional suggested fix code.
```

| Constructor param | Default | Description |
|---|---|---|
| `api_key` | `None` | Falls back to `AKSCI_ANTHROPIC_API_KEY` env var if omitted. |
| `model` | `"claude-sonnet-4-6"` | Model string used for AI-assist requests. |

`client.available` is `True` if a key is configured (does not verify the key is valid). Calling `.diagnose_error(...)` without a key raises `AIAssistUnavailable` — it never fails silently or falls back to fabricated output.

Requires the `anthropic` package: `pip install "aksci[ai]"`.

---

## 8. FAQ / Troubleshooting

**Does AK-SCI send my code or data anywhere?**
No, not by default. The local diagnostic model and everything in `error_handler`, `micro_pipeline`, and `unified_api` run entirely offline. The only thing that ever makes a network call is `AIClient`, and only if you explicitly construct one with an API key.

**Why is my confidence score low for some errors?**
Some error categories are genuinely hard to tell apart from short text alone — see the [known limitation](#known-limitation-stated-honestly) above. Low confidence is the model being honest about ambiguity, not a bug.

**`safe_run(auto_fix_context=...)` didn't fix my error — why?**
Check three things: (1) is `ErrorResolver(auto_fix=True)` actually set — auto-fix is off by default; (2) does your `auto_fix_context` return the exact keys the specific rule needs (see [§4](#4-auto-fix-rules-opt-in)); (3) is your decorated function doing more than one risky operation — auto-fix replaces the whole return value, so it works best on single-purpose helpers.

**Can I add my own auto-fix rules?**
Not via a public plugin API yet (0.1.0) — but `AUTO_FIX_RULES` in `autofix.py` is a plain list of `(name, function)` pairs following a documented signature (`(exc, context) -> AutoFixResult | None`). You can fork the module and add your own, following the same conservative/whitelisted/logged principles.

**Does `MicroPipeline` parallelize across CPU cores?**
No — it's single-process and generator-based, focused on memory, not CPU parallelism. Wrap a CPU-bound stage with `concurrent.futures.ProcessPoolExecutor` if you need that.

**What Python versions are supported?**
3.10, 3.11, 3.12 — enforced in CI (`.github/workflows/ci.yml`).
