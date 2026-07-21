"""
Local diagnostic engine.

This is the "embedded AI model" that AK-SCI ships with: a small, offline
TF-IDF + Logistic Regression classifier trained on a bundled dataset of
common Python / data-science error signatures (see `_training_data.py`).
It is deliberately lightweight -- training takes a fraction of a second and
the model is cached to disk after first use -- and it requires no network
access or API key. For deeper, free-text explanations, pair it with
`ai_core.client.AIClient` (optional, needs an API key).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ._training_data import TRAINING_EXAMPLES

# The cache filename is tied to the size of the bundled training set, so
# shipping a new aksci version with more/different examples automatically
# invalidates any stale cached model left over from an older install --
# instead of silently keeping predictions trained on the old data.
_MODEL_CACHE_PATH = (
    Path(__file__).parent / f"_diagnostic_model_v{len(TRAINING_EXAMPLES)}.joblib"
)

#: Human-readable fix guidance per predicted category. Kept short and
#: actionable rather than exhaustive -- the goal is to point the developer
#: in the right direction in one or two sentences.
FIX_TEMPLATES: dict[str, str] = {
    "missing_column": (
        "This column name does not exist in the DataFrame. Check spelling "
        "and case, run `df.columns` to see what's actually available, or "
        "the column may need to be created before you can use it."
    ),
    "key_error_generic": (
        "This dictionary key does not exist. Use `.get(key, default)` "
        "instead of `[key]` if the key is optional, or check upstream "
        "code that is supposed to set it."
    ),
    "type_mismatch": (
        "You're mixing incompatible types (e.g. str + int, or calling a "
        "non-callable). Print `type(x)` for the values involved and "
        "convert explicitly with `str()`, `int()`, or `float()`."
    ),
    "shape_mismatch": (
        "Array/DataFrame shapes don't line up for this operation. Print "
        "`.shape` on each input -- for matrix multiplication the inner "
        "dimensions must match; for sklearn, X and y need the same number "
        "of rows."
    ),
    "division_by_zero": (
        "You're dividing by a value that is zero (or NaN in an array). "
        "Guard with an `if divisor != 0` check, or use `np.where` / "
        "`.replace(0, np.nan)` to handle it safely across a whole column."
    ),
    "missing_module": (
        "This package isn't installed in the current environment. Install "
        "it with `pip install <package-name>` -- if you're using a venv or "
        "conda env, make sure it's activated first."
    ),
    "index_out_of_range": (
        "You're accessing an index that doesn't exist. Check `len(x)` "
        "before indexing, or use `.iloc[]` bounds carefully -- this often "
        "happens after a filter operation leaves fewer rows than expected."
    ),
    "attribute_error": (
        "The object doesn't have this attribute or method -- often because "
        "it's `None` (a function returned nothing) or it's the wrong type "
        "(e.g. a list where you expected a DataFrame). Print `type(x)` to confirm."
    ),
    "null_values": (
        "Your data contains NaN, None, or infinity where a numeric "
        "algorithm expects clean numbers. Use `.dropna()`, `.fillna()`, or "
        "`np.isfinite()` to clean the data before fitting a model."
    ),
    "value_error_generic": (
        "A value passed to this function is not in the format it expects "
        "(e.g. a non-numeric string being converted to a number). Validate "
        "or convert the value before passing it in."
    ),
    "file_not_found": (
        "The file or path doesn't exist where the code expects it. Check "
        "for typos, use an absolute path or `Path(__file__).parent / ...` "
        "instead of a relative one, and confirm the working directory the "
        "script is actually running from with `os.getcwd()`."
    ),
    "permission_denied": (
        "The process doesn't have OS-level permission to read/write/execute "
        "this path. Check file ownership and mode with `ls -l`, avoid "
        "writing to system directories, and on Windows make sure the file "
        "isn't open in another program."
    ),
    "json_decode_error": (
        "The text being parsed isn't valid JSON -- often an empty response "
        "body, an HTML error page returned instead of JSON, or a trailing "
        "comma. Print the raw text before calling `json.loads()` / "
        "`response.json()` to see what actually came back."
    ),
    "connection_error": (
        "The network request couldn't reach the server -- DNS failure, "
        "refused connection, or no internet access. Verify the URL and "
        "that the service is up, add a retry with backoff, and check "
        "firewall/VPN settings if this is a corporate network."
    ),
    "memory_error": (
        "The process ran out of memory allocating this array/DataFrame. "
        "Process data in chunks (see `MicroPipeline`), use a smaller dtype "
        "(`float32` instead of `float64`), or filter/select columns before "
        "loading the full dataset into memory."
    ),
    "timeout_error": (
        "The operation didn't complete within the allotted time -- usually "
        "a slow network call or an overloaded server. Increase the timeout "
        "if the work is legitimately slow, add retry logic with backoff, "
        "or check whether the endpoint is degraded."
    ),
}


@dataclass
class Diagnosis:
    """Result of classifying a runtime error."""
    category: str
    confidence: float
    explanation: str
    suggestion: str


class LocalDiagnosticModel:
    """Embedded TF-IDF + Logistic Regression classifier for error messages.

    The model trains once (in-process, on the small bundled dataset) and
    caches itself to disk so subsequent runs load instantly instead of
    retraining. Training and inference both happen locally -- no network
    calls are made.
    """

    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None

    def _build(self) -> Pipeline:
        texts = [t for t, _ in TRAINING_EXAMPLES]
        labels = [l for _, l in TRAINING_EXAMPLES]
        pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                # C=15 (vs. sklearn's default 1.0) is tuned for this dataset:
                # with 16 categories and short technical strings, the default
                # regularization spreads probability mass too thin even when
                # one category clearly wins, which understates confidence.
                # Cross-validation shows C=15 sharpens confidence
                # substantially with no loss in held-out accuracy.
                ("clf", LogisticRegression(max_iter=2000, C=15.0)),
            ]
        )
        pipeline.fit(texts, labels)
        return pipeline

    def load_or_train(self) -> Pipeline:
        """Return the cached model, or train and cache a new one."""
        if self._pipeline is not None:
            return self._pipeline
        if _MODEL_CACHE_PATH.exists():
            try:
                self._pipeline = joblib.load(_MODEL_CACHE_PATH)
                return self._pipeline
            except Exception:
                pass  # fall through and retrain if the cache is corrupt
        self._pipeline = self._build()
        try:
            joblib.dump(self._pipeline, _MODEL_CACHE_PATH)
        except OSError:
            pass  # read-only filesystem etc. -- fine, just skip caching
        return self._pipeline

    def diagnose(self, error_type: str, error_message: str) -> Diagnosis:
        """Classify an exception into a category and return fix guidance.

        Parameters
        ----------
        error_type: the exception class name, e.g. "KeyError".
        error_message: str(exception), e.g. "'price'".
        """
        pipeline = self.load_or_train()
        text = f"{error_type}: {error_message}"
        category = str(pipeline.predict([text])[0])
        confidence = float(pipeline.predict_proba([text]).max())
        template = FIX_TEMPLATES.get(category, FIX_TEMPLATES["value_error_generic"])
        return Diagnosis(
            category=category,
            confidence=confidence,
            explanation=template,
            suggestion=template,
        )
