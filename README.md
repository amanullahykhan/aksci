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

## Install

```bash
pip install aksci                # core library
pip install "aksci[ai]"          # + optional cloud AI-assist (needs an Anthropic API key)
pip install "aksci[dev]"         # + pytest, for running the test suite
```

> **Status:** AK-SCI is in active early development (`0.1.0`, Alpha). The API may change between minor versions until `1.0`.

---

## Quick Start

```python
from aksci import ErrorResolver, UnifiedFrame, MicroPipeline, ml, stats
import pandas as pd

# 1. Offline error diagnosis — no API key needed
resolver = ErrorResolver()

@resolver.safe_run(reraise=False)
def load_price(row):
    return row["price"]

load_price({"prise": 19.99})
# stderr:
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
pipeline = MicroPipeline(chunk_size=50_000)
pipeline.add_stage("drop_nulls", lambda chunk: chunk.dropna())
for processed_chunk in pipeline.run_csv("big_file.csv"):
    ...  # only one chunk in memory at a time
```

Run the full walkthrough:

```bash
python examples/demo.py                     # every module, end to end
python examples/lead_scoring_pipeline.py     # realistic messy-CSV automation example
```

---

## Auto-fix (opt-in)

`ErrorResolver(auto_fix=True)` turns on four whitelisted, conservative recovery rules. Each one declines rather than guesses if it can't be confident, and every applied fix prints to stderr — nothing happens silently.

```python
resolver = ErrorResolver(auto_fix=True)

@resolver.safe_run(reraise=False, auto_fix_context=lambda a, b: {"default": 0.0})
def safe_divide(a, b):
    return a / b

safe_divide(10, 0)
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
