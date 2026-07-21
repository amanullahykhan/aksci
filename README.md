<<<<<<< HEAD
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
=======
<div align="center">

# AK-SCI

**A unified, AI-assisted data science toolkit for Python.**

Offline error diagnosis · opt-in auto-fix · memory-bounded streaming · one API over pandas, Polars, NumPy, SciPy & scikit-learn

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/test_basic.py)

[Quick Start](#quick-start) · [Documentation](DOCS.md) · [Examples](examples/) · [Contributing](#contributing)

</div>

---

## What is AK-SCI?

AK-SCI is a Python library built around one idea: **the most common failures in a data-science or automation script are recognizable, and a lot of the debugging cycle can be shortened without pretending the computer understands your code.**

It does four concrete things:

1. **Diagnoses runtime errors offline.** A small, embedded machine-learning classifier — trained on 134 examples across 16 categories of common Python and data-science errors — tells you *what kind* of error you hit and *what to do about it*, in under a second, with no internet connection or API key required.
2. **Optionally auto-fixes a narrow, whitelisted set of safe recoveries** — a mistyped column name, a blank numeric field, a division by zero, a string that should be a number — entirely opt-in, and every fix is logged so nothing happens silently.
3. **Streams large datasets through memory-bounded chunks**, so a pipeline can process a CSV far bigger than available RAM without loading the whole file at once.
4. **Wraps pandas, Polars, NumPy, SciPy, and scikit-learn behind one consistent API**, so common operations read the same way regardless of which backend your data happens to be in.

Full category-by-category reference: **[DOCS.md](DOCS.md)**

---

## Why AK-SCI

If you've written a data pipeline, a scraping script, or a small automation tool, you've hit this loop: something throws `KeyError: 'price'`, you stare at the traceback, you remember it's because the column got renamed upstream, you fix it, you move on. AK-SCI shortens that loop:

```
[AKSCI] Caught KeyError: 'price'
  Category   : missing_column (confidence 84%)
  Likely fix : This column name does not exist in the DataFrame. Check spelling
               and case, run `df.columns` to see what's actually available, or
               the column may need to be created before you can use it.
```

That's the default, always-on behavior — it explains, it never rewrites your code unless you explicitly ask it to (see [Auto-fix](#auto-fix-opt-in)).

---
>>>>>>> 09ef8c4e139cb28b067cbb981b77ff3e0e25afe5

## Install

```bash
<<<<<<< HEAD
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
=======
pip install aksci                # core library
pip install "aksci[ai]"          # + optional cloud AI-assist (needs an Anthropic API key)
pip install "aksci[dev]"         # + pytest, for running the test suite
```

> **Status:** AK-SCI is in active early development (`0.1.0`, Alpha). The API may change between minor versions until `1.0`.

---

## Quick Start
>>>>>>> 09ef8c4e139cb28b067cbb981b77ff3e0e25afe5

```python
from aksci import ErrorResolver, UnifiedFrame, MicroPipeline, ml, stats
import pandas as pd

<<<<<<< HEAD
# 1. Error diagnosis
=======
# 1. Offline error diagnosis — no API key needed
>>>>>>> 09ef8c4e139cb28b067cbb981b77ff3e0e25afe5
resolver = ErrorResolver()

@resolver.safe_run(reraise=False)
def load_price(row):
    return row["price"]

load_price({"prise": 19.99})
# stderr:
<<<<<<< HEAD
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
=======
#   [AKSCI] Caught KeyError: 'price'
#     Category   : missing_column (confidence 84%)
#     Likely fix : This column name does not exist in the DataFrame...

# 2. One API over pandas or Polars
df = pd.DataFrame({"revenue": [100, 200, 300], "cost": [60, 90, 150]})
standardized = UnifiedFrame(df).standardize().to_pandas()

# 3. Simplified, well-documented ML
model = ml.linear_regression(X, y)                             # scikit-learn under the hood
gd = ml.gradient_descent(X, y, learning_rate=0.1, epochs=300)  # hand-rolled, every step visible

# 4. Memory-bounded streaming for files larger than RAM
>>>>>>> 09ef8c4e139cb28b067cbb981b77ff3e0e25afe5
pipeline = MicroPipeline(chunk_size=50_000)
pipeline.add_stage("drop_nulls", lambda chunk: chunk.dropna())
for processed_chunk in pipeline.run_csv("big_file.csv"):
    ...  # only one chunk in memory at a time
```

<<<<<<< HEAD
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
=======
Run the full walkthrough:

```bash
python examples/demo.py                     # every module, end to end
python examples/lead_scoring_pipeline.py     # realistic messy-CSV automation example
```

---

## Auto-fix (opt-in)

`ErrorResolver(auto_fix=True)` turns on four whitelisted, conservative recovery rules. Each one declines rather than guesses if it can't be confident, and every applied fix prints to stderr — nothing happens silently.
>>>>>>> 09ef8c4e139cb28b067cbb981b77ff3e0e25afe5

```python
resolver = ErrorResolver(auto_fix=True)

@resolver.safe_run(reraise=False, auto_fix_context=lambda a, b: {"default": 0.0})
def safe_divide(a, b):
    return a / b

safe_divide(10, 0)
<<<<<<< HEAD
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
=======
# -> logs the fix, returns 0.0 instead of raising
```

See **[DOCS.md § Auto-fix Rules](DOCS.md#4-auto-fix-rules-opt-in)** for what each rule needs and its exact limits.

---

## Documentation

| Section | What's covered |
|---|---|
| **[DOCS.md](DOCS.md)** | Full categorized reference: every class, method, parameter, and return value |
| [§ ErrorResolver](DOCS.md#2-errorresolver--error-diagnosis) | `safe_run`, `explain`, `diagnose`, `install_global_hook`, `suggest_column` |
| [§ Auto-fix Rules](DOCS.md#4-auto-fix-rules-opt-in) | All 4 rules, required context keys, worked examples |
| [§ Local Diagnostic Model](DOCS.md#3-the-local-diagnostic-model) | How the offline classifier works, all 16 categories |
| [§ MicroPipeline](DOCS.md#5-micropipeline--streaming) | Chunked processing for large files |
| [§ UnifiedFrame](DOCS.md#6-unifiedframe--ml--stats) | pandas/Polars-agnostic API, `ml`, `stats` |
| [§ AIClient](DOCS.md#7-aiclient--optional-cloud-ai-assist) | Optional cloud AI-assist setup |
| [§ FAQ / Troubleshooting](DOCS.md#8-faq--troubleshooting) | Common questions |

---

## Project Layout

```
aksci/
├── .github/workflows/ci.yml     # CI: tests on Python 3.10-3.12 + build check
├── LICENSE                      # MIT
├── README.md
├── DOCS.md                      # full categorized reference
├── pyproject.toml
├── src/aksci/
│   ├── ai_core/                 # local offline classifier + optional cloud client
│   ├── error_handler/           # ErrorResolver, safe_run, auto-fix rules
│   ├── micro_pipeline/          # BoundedBuffer, Stage, MicroPipeline
│   └── unified_api/             # UnifiedFrame, ml, stats
├── tests/                       # pytest suite
└── examples/                    # runnable, realistic walkthroughs
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Contributing

Issues and pull requests are welcome. Before opening a PR:

1. Run `pytest tests/ -v` — all tests must pass.
2. If you add a new error category to `_training_data.py`, add a matching entry to `FIX_TEMPLATES` in `diagnostics.py` (there's a test that enforces this).
3. Keep new auto-fix rules conservative: decline rather than guess, and always log what happened.

## License

MIT © 2026 [Amanullah Khan](#author) — see [LICENSE](LICENSE).

You're free to use, modify, and distribute this project, including commercially, as long as the original copyright notice and license text are kept intact in copies and substantial portions of the software. If you build on AK-SCI, a credit/link back is appreciated but not required beyond what the license already asks for.

## Author

**Amanullah Khan**
IT Department, Alkhidmat Foundation Ghotki
Web development · SEO · digital content · automation tooling

- GitHub: [https://github.com/amanullahykhan](https://github.com/amanullahykhan)
- Email: [amanullahykhan@gmail.com](mailto:amanullahykhan@gmail.com)

---

<div align="center">
<sub>Built as a learning-by-doing project — offline-first, dependency-transparent, and honest about what it does and doesn't do.</sub>
</div>
>>>>>>> 09ef8c4e139cb28b067cbb981b77ff3e0e25afe5
