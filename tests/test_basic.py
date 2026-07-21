"""Smoke tests covering every core module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aksci import ErrorResolver, MicroPipeline, UnifiedFrame, ml, stats
from aksci.ai_core.diagnostics import LocalDiagnosticModel


def test_unified_frame_standardize_pandas() -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    result = UnifiedFrame(df).standardize().to_pandas()
    assert abs(result["a"].mean()) < 1e-9
    assert abs(result["a"].std(ddof=0) - 1.0) < 1e-9


def test_unified_frame_backend_conversion() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    uf = UnifiedFrame(df)
    assert uf.backend == "pandas"
    polars_df = uf.to_polars()
    uf2 = UnifiedFrame(polars_df)
    assert uf2.backend == "polars"
    assert uf2.to_pandas()["x"].tolist() == [1, 2, 3]


def test_gradient_descent_converges_to_known_line() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1))
    true_w, true_b = 3.0, 1.5
    y = true_w * X[:, 0] + true_b + rng.normal(scale=0.01, size=200)

    result = ml.gradient_descent(X, y, learning_rate=0.1, epochs=300)

    assert abs(result.weights[0] - true_w) < 0.1
    assert abs(result.bias - true_b) < 0.1
    assert result.loss_history[-1] < result.loss_history[0]


def test_linear_regression_matches_gradient_descent() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 2))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5

    sk_model = ml.linear_regression(X, y)
    gd_result = ml.gradient_descent(X, y, learning_rate=0.1, epochs=1000)

    np.testing.assert_allclose(sk_model.coef_, gd_result.weights, atol=0.1)


def test_stats_correlation_and_ttest() -> None:
    a = [1, 2, 3, 4, 5]
    b = [2, 4, 6, 8, 10]
    corr = stats.correlation(a, b)
    assert corr["r"] > 0.99

    result = stats.ttest(a, b)
    assert hasattr(result, "pvalue")


def test_local_diagnostic_model_classifies_key_error() -> None:
    model = LocalDiagnosticModel()
    diagnosis = model.diagnose("KeyError", "'total_sales'")
    assert diagnosis.category in {"missing_column", "key_error_generic"}
    assert 0.0 <= diagnosis.confidence <= 1.0


def test_local_diagnostic_model_confidence_is_reasonably_high() -> None:
    """Guards against silent regressions in training data / hyperparameters
    that would make the classifier's confidence scores useless again."""
    model = LocalDiagnosticModel()
    diagnosis = model.diagnose("KeyError", "'price'")
    assert diagnosis.confidence > 0.6


@pytest.mark.parametrize(
    "error_type,error_message,expected_category",
    [
        ("FileNotFoundError", "[Errno 2] No such file or directory: 'data.csv'", "file_not_found"),
        ("PermissionError", "[Errno 13] Permission denied: 'output.csv'", "permission_denied"),
        ("JSONDecodeError", "Expecting value: line 1 column 1 (char 0)", "json_decode_error"),
        (
            "ConnectionError",
            "HTTPSConnectionPool(host='api.example.com', port=443): Max retries exceeded",
            "connection_error",
        ),
        (
            "MemoryError",
            "Unable to allocate 6.00 GiB for an array with shape (800000, 900)",
            "memory_error",
        ),
        ("TimeoutError", "[Errno 110] Connection timed out", "timeout_error"),
        ("ZeroDivisionError", "division by zero", "division_by_zero"),
        ("AttributeError", "'NoneType' object has no attribute 'get'", "attribute_error"),
        ("ModuleNotFoundError", "No module named 'requests'", "missing_module"),
    ],
)
def test_local_diagnostic_model_classifies_new_categories(
    error_type: str, error_message: str, expected_category: str
) -> None:
    model = LocalDiagnosticModel()
    diagnosis = model.diagnose(error_type, error_message)
    assert diagnosis.category == expected_category
    assert diagnosis.confidence > 0.5


def test_all_fix_templates_have_matching_category() -> None:
    """Every category referenced in the training data must have a
    FIX_TEMPLATES entry, or diagnose() would silently fall back to a
    generic, possibly-wrong suggestion."""
    from aksci.ai_core._training_data import TRAINING_EXAMPLES
    from aksci.ai_core.diagnostics import FIX_TEMPLATES

    categories_in_training_data = {label for _, label in TRAINING_EXAMPLES}
    assert categories_in_training_data <= set(FIX_TEMPLATES.keys())


def test_error_resolver_safe_run_swallows_and_reports() -> None:
    resolver = ErrorResolver()

    @resolver.safe_run(reraise=False)
    def boom() -> int:
        data = {"x": 1}
        return data["y"]  # type: ignore[index]

    assert boom() is None


def test_error_resolver_safe_run_reraises_by_default() -> None:
    resolver = ErrorResolver()

    @resolver.safe_run()
    def boom() -> int:
        return 1 / 0

    try:
        boom()
        assert False, "expected ZeroDivisionError to propagate"
    except ZeroDivisionError:
        pass


def test_suggest_column_fuzzy_match() -> None:
    resolver = ErrorResolver()
    match = resolver.suggest_column("pric", ["price", "quantity", "customer_id"])
    assert match == "price"


def test_autofix_disabled_by_default_even_with_context() -> None:
    """auto_fix=False (the default) must never apply a fix, even if a
    context that *would* satisfy a rule is supplied -- opt-in is only
    honored when the resolver was explicitly built with auto_fix=True."""
    resolver = ErrorResolver()  # auto_fix defaults to False

    @resolver.safe_run(reraise=False, auto_fix_context=lambda a, b: {"default": 99})
    def divide(a: float, b: float) -> float:
        return a / b

    assert divide(10, 0) is None


def test_autofix_division_by_zero_guard() -> None:
    resolver = ErrorResolver(auto_fix=True)

    @resolver.safe_run(reraise=False, auto_fix_context=lambda a, b: {"default": 0.0})
    def divide(a: float, b: float) -> float:
        return a / b

    assert divide(10, 0) == 0.0


def test_autofix_missing_column_fuzzy_match() -> None:
    resolver = ErrorResolver(auto_fix=True)
    df = pd.DataFrame({"price": [10, 20], "qty": [1, 2]})

    @resolver.safe_run(
        reraise=False,
        auto_fix_context=lambda row: {"available_columns": list(df.columns), "frame": df},
    )
    def get_price(row):
        return df["prise"]

    result = get_price(None)
    assert result is not None
    assert result.tolist() == [10, 20]


def test_autofix_type_coercion() -> None:
    resolver = ErrorResolver(auto_fix=True)

    @resolver.safe_run(
        reraise=False,
        auto_fix_context=lambda s: {"raw_value": s, "target_type": float},
    )
    def parse_price(s: str) -> float:
        return float(s)

    assert parse_price(" 19.99 ") == 19.99


def test_autofix_null_values_fillna() -> None:
    resolver = ErrorResolver(auto_fix=True)
    from sklearn.linear_model import LinearRegression

    df_with_nan = pd.DataFrame({"x": [1.0, float("nan"), 3.0]})

    @resolver.safe_run(
        reraise=False,
        auto_fix_context=lambda df: {"frame": df, "fill_value": 0},
    )
    def fit(df: pd.DataFrame):
        LinearRegression().fit(df[["x"]], [1, 2, 3])
        return df[["x"]].to_numpy()

    result = fit(df_with_nan)
    assert result is not None


def test_autofix_declines_when_no_rule_applies() -> None:
    """A resolver with auto_fix=True must still fall back to normal
    reraise/None behavior when no rule's preconditions are met -- it must
    never guess."""
    resolver = ErrorResolver(auto_fix=True)

    @resolver.safe_run(reraise=False)  # no auto_fix_context supplied
    def boom() -> int:
        return {"a": 1}["b"]  # type: ignore[index]

    assert boom() is None


def test_autofix_try_fix_returns_unapplied_result_for_unhandled_exception() -> None:
    from aksci.error_handler.autofix import try_fix

    result = try_fix(RuntimeError("something unrelated"), {})
    assert result.applied is False
    assert result.rule_name == "none"


def test_micro_pipeline_in_memory_chunking() -> None:
    pipeline = MicroPipeline(chunk_size=2)
    pipeline.add_stage("double", lambda x: x * 2)
    pipeline.add_stage("add_one", lambda x: x + 1)
    result = list(pipeline.run([1, 2, 3]))
    assert result == [3, 5, 7]


def test_micro_pipeline_run_csv(tmp_path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"value": range(10)}).to_csv(csv_path, index=False)

    pipeline = MicroPipeline(chunk_size=3)
    pipeline.add_stage("double_value", lambda df: df.assign(value=df["value"] * 2))

    chunks = list(pipeline.run_csv(str(csv_path)))
    total_rows = sum(len(c) for c in chunks)
    assert total_rows == 10
    assert all(chunk["value"].iloc[0] % 2 == 0 for chunk in chunks)
