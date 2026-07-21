# AK-SCI

**A unified, AI-assisted data science toolkit.**
Author: Amanullah Khan

AK-SCI does four concrete things:

1. **Diagnoses runtime errors** using a small, embedded, offline machine-learning
   model (TF-IDF + Logistic Regression, trained on 130+ examples across 16
   categories of common Python / data-science / automation errors) — no API
   key or internet connection needed. An optional cloud AI-assist mode (needs
   your own API key) adds deeper natural-language explanations for trickier
   cases.
2. **Optionally auto-fixes a narrow, whitelisted set of safe recoveries** —
   fuzzy-matching a mistyped column name, filling a blank numeric field,
   guarding a division by zero, coercing a string to a number — entirely
   opt-in and always logged. See [Auto-fix](#auto-fix-opt-in) below.
3. **Streams large datasets through memory-bounded chunks**, so a pipeline
   built with `MicroPipeline` can process CSV files far bigger than available
   RAM without loading the whole thing at once.
4. **Wraps pandas, Polars, NumPy, SciPy, and scikit-learn behind one
   consistent API** (`UnifiedFrame`, `ml`, `stats`), so common operations —
   standardizing a dataframe, fitting a regression, running a t-test — read
   the same way no matter which backend your data happens to be in.

### What this library is *not*
It doesn't change how Python itself works (the GIL, dynamic typing, etc. are
unaffected — no pure-Python library can do that), and it doesn't silently
rewrite your code. The error handler explains and suggests by default; it
only returns a fallback value if you explicitly opt in with `reraise=False`,
and it only *auto-fixes* anything if you explicitly opt in with
`auto_fix=True` **and** supply the context a specific rule needs.

## Install

```bash
pip install -e .            # from this directory, for development
pip install aksci[ai]       # add cloud AI-assist (needs anthropic + an API key)
pip install aksci[dev]      # add pytest for running the test suite
```

### Publishing (PyPI / TestPyPI)

The package already builds cleanly with the standard tooling:

```bash
pip install build twine
python -m build              # produces dist/aksci-X.Y.Z.tar.gz and .whl
twine check dist/*           # validates metadata before uploading

# Try it on TestPyPI first:
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ aksci

# Once verified, upload for real:
twine upload dist/*
```

You'll need a PyPI (and/or TestPyPI) account and an API token — generate one
under Account Settings → API tokens on each site, then use `__token__` as the
username and the token as the password when `twine upload` prompts you (or
put them in `~/.pypirc`).

### GitHub

A CI workflow is already set up at `.github/workflows/ci.yml` — it runs the
test suite on Python 3.10/3.11/3.12 on every push/PR to `main`, then builds
and metadata-checks the distribution. To publish the repo:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/aksci.git
git push -u origin main
```

`.gitignore` and `LICENSE` (MIT, matching `pyproject.toml`) are already in
place, so the repo is push-ready as-is.

## Quick start

```python
from aksci import ErrorResolver, UnifiedFrame, MicroPipeline, ml, stats
import pandas as pd

# 1. Error diagnosis
resolver = ErrorResolver()

@resolver.safe_run(reraise=False)
def load_price(row):
    return row["price"]

load_price({"prise": 19.99})
# stderr:
# [AKSCI] Caught KeyError: 'price'
#   Category   : missing_column (confidence 84%)
#   Likely fix : This column name does not exist in the DataFrame...

# 2. Unified frame API (works the same on pandas or Polars)
df = pd.DataFrame({"revenue": [100, 200, 300], "cost": [60, 90, 150]})
standardized = UnifiedFrame(df).standardize().to_pandas()

# 3. Simplified ML, including a from-scratch gradient descent for learning
model = ml.linear_regression(X, y)                       # scikit-learn under the hood
gd = ml.gradient_descent(X, y, learning_rate=0.1, epochs=300)  # hand-rolled, explicit

# 4. Memory-bounded streaming for large files
pipeline = MicroPipeline(chunk_size=50_000)
pipeline.add_stage("drop_nulls", lambda chunk: chunk.dropna())
for processed_chunk in pipeline.run_csv("big_file.csv"):
    ...  # only one chunk in memory at a time
```

Run `python examples/demo.py` for a full, working walkthrough of every module.

Run `python examples/lead_scoring_pipeline.py` for a realistic, messier
example: a CRM-style CSV export with a renamed column, blank fields, and a
zero-division case, scored end-to-end using `install_global_hook()` plus
`auto_fix=True` on small, single-purpose helper functions.

## The local diagnostic model

The offline classifier is trained on 130+ hand-written examples spanning 16
categories: `missing_column`, `key_error_generic`, `type_mismatch`,
`shape_mismatch`, `division_by_zero`, `missing_module`, `index_out_of_range`,
`attribute_error`, `null_values`, `value_error_generic`, `file_not_found`,
`permission_denied`, `json_decode_error`, `connection_error`, `memory_error`,
and `timeout_error` — covering both data-science errors (pandas/numpy/
sklearn) and general automation/API-script errors (file I/O, network,
JSON parsing).

It trains once in-process (a fraction of a second) and caches itself to
disk; the cache filename is tied to the size of the bundled training set,
so upgrading aksci automatically invalidates any stale cached model instead
of silently reusing old predictions.

## Auto-fix (opt-in)

`ErrorResolver(auto_fix=True)` turns on a small, whitelisted set of safe
recovery rules for any `safe_run`-decorated function that supplies
`auto_fix_context`. Every rule is conservative — if it can't be confident,
it declines rather than guessing — and every applied fix is printed to
stderr, never silent.

```python
resolver = ErrorResolver(auto_fix=True)

@resolver.safe_run(reraise=False, auto_fix_context=lambda a, b: {"default": 0.0})
def safe_divide(a, b):
    return a / b

safe_divide(10, 0)
# stderr:
# [AKSCI] Caught ZeroDivisionError: division by zero
#   ...
#   [AKSCI auto-fix] division_by_zero_guard: Division by zero; used fallback value 0.0 instead.
# -> returns 0.0
```

The four rules, and what each needs in `auto_fix_context`:

| Rule | Fires on | Needs in context | Does |
|---|---|---|---|
| `missing_column_fuzzy_match` | `KeyError` | `available_columns`, `frame` | Fuzzy-matches the missing key against real column names (same 0.6 cutoff as `suggest_column`) and returns that column's value |
| `null_values_fillna` | `ValueError` mentioning NaN/inf | `frame`, optional `fill_value` (default 0) | Calls `.fillna()` (or `np.nan_to_num`) on `frame` |
| `division_by_zero_guard` | `ZeroDivisionError` | `default` | Returns `default` instead of raising |
| `type_coercion` | `TypeError`/`ValueError` | `raw_value`, `target_type` (`int`/`float`) | Coerces `raw_value` to the target type |

**Important:** auto-fix replaces a function's *entire return value* for that
call — it can't resume execution partway through a function that failed on
its second or third risky line. Wrap small, single-purpose helpers (one
column lookup, one division) rather than a multi-step function; see
`examples/lead_scoring_pipeline.py` for the pattern in a realistic script.

You can also call auto-fix directly without `safe_run`:

```python
from aksci.error_handler import autofix

try:
    df["revenu"]
except KeyError as exc:
    result = autofix.try_fix(exc, {"available_columns": list(df.columns), "frame": df})
    if result.applied:
        print(result.description)
```

## Enabling cloud AI-assist (optional)

```python
from aksci import ErrorResolver, AIClient

resolver = ErrorResolver(ai_client=AIClient(api_key="sk-ant-..."))
# or set the AKSCI_ANTHROPIC_API_KEY environment variable instead of passing api_key=
```

Without a key, `ErrorResolver` still works — it just uses the offline model only.

## Project layout

```
aksci/
├── .github/workflows/ci.yml    # test matrix (3.10-3.12) + build check
├── .gitignore
├── LICENSE                      # MIT
├── pyproject.toml
├── README.md
├── src/aksci/
│   ├── __init__.py             # public API surface
│   ├── ai_core/                 # the "AI" layer
│   │   ├── prompts.py           # hardcoded system prompts for cloud AI-assist
│   │   ├── client.py            # optional cloud AI-assist client (opt-in)
│   │   ├── diagnostics.py       # embedded offline ML classifier
│   │   └── _training_data.py    # bundled training examples (130+, 16 categories)
│   ├── error_handler/
│   │   ├── resolver.py          # ErrorResolver, safe_run decorator, global hook
│   │   └── autofix.py           # whitelisted, opt-in auto-fix rules
│   ├── micro_pipeline/
│   │   └── buffer.py            # BoundedBuffer, Stage, MicroPipeline
│   └── unified_api/
│       └── facade.py            # UnifiedFrame, ml, stats
├── tests/
│   └── test_basic.py            # 29 tests, all passing — covers real behavior,
│                                 # not just imports (e.g. gradient descent
│                                 # converging to known weights, auto-fix rules
│                                 # firing/declining correctly)
└── examples/
    ├── demo.py                  # runnable end-to-end walkthrough of every module
    └── lead_scoring_pipeline.py # realistic messy-CSV automation example
```

## Running the tests

```bash
pip install -e .[dev]
pytest tests/ -v
```

## Extending into a larger tool ecosystem

Each subpackage (`ai_core`, `error_handler`, `micro_pipeline`, `unified_api`)
is independent and importable on its own — e.g. a larger coding-assistant
environment could import just `aksci.error_handler.ErrorResolver` and reuse
its `explain()` / `diagnose()` methods against exceptions it catches itself,
without pulling in the rest of the package. Similarly,
`aksci.error_handler.autofix.try_fix()` can be called directly against any
caught exception, independent of `safe_run`.
