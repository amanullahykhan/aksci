"""
Unified data-science facade.

One consistent, typed API over pandas, Polars, NumPy, SciPy, and
scikit-learn, so common operations (standardizing a dataframe, fitting a
regression, running a t-test) read the same way regardless of which
backend the underlying data happens to be in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
from sklearn.linear_model import LinearRegression as _SKLinearRegression

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:  # pragma: no cover - polars is a hard dependency, but degrade gracefully
    _HAS_POLARS = False

AnyFrame = Union[pd.DataFrame, "pl.DataFrame"]


class UnifiedFrame:
    """A thin, backend-agnostic wrapper around a pandas or Polars DataFrame.

    Wraps whichever backend you hand it and exposes the same handful of
    high-level operations for both, so you can write one line of code that
    works either way -- and convert between backends explicitly when you
    need a library that only accepts one of them (e.g. scikit-learn wants
    pandas/NumPy; some newer, faster ETL work is easier in Polars).
    """

    def __init__(self, data: AnyFrame) -> None:
        self._data: AnyFrame = data
        self._backend: Literal["pandas", "polars"] = (
            "polars" if _HAS_POLARS and isinstance(data, pl.DataFrame) else "pandas"
        )

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def shape(self) -> tuple:
        return self._data.shape

    def to_pandas(self) -> pd.DataFrame:
        """Return the underlying data as a pandas DataFrame (converting if needed)."""
        if self._backend == "pandas":
            return self._data  # type: ignore[return-value]
        return self._data.to_pandas()  # type: ignore[union-attr]

    def to_polars(self) -> "pl.DataFrame":
        """Return the underlying data as a Polars DataFrame (converting if needed)."""
        if not _HAS_POLARS:
            raise ImportError("polars is not installed. Run: pip install polars")
        if self._backend == "polars":
            return self._data  # type: ignore[return-value]
        return pl.from_pandas(self._data)  # type: ignore[arg-type]

    def standardize(self, columns: Optional[List[str]] = None) -> "UnifiedFrame":
        """Z-score standardize numeric columns (mean 0, std 1), on either backend.

        Parameters
        ----------
        columns: which columns to standardize. Defaults to all numeric columns.
        """
        if self._backend == "pandas":
            df = self._data.copy()  # type: ignore[union-attr]
            cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
            df[cols] = (df[cols] - df[cols].mean()) / df[cols].std(ddof=0)
            return UnifiedFrame(df)
        else:
            df = self._data  # type: ignore[assignment]
            numeric_types = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)
            cols = columns or [
                c for c, dt in zip(df.columns, df.dtypes) if dt in numeric_types  # type: ignore[union-attr]
            ]
            exprs = [((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c) for c in cols]
            return UnifiedFrame(df.with_columns(exprs))  # type: ignore[union-attr]

    def dropna(self) -> "UnifiedFrame":
        """Drop rows containing any null values, on either backend."""
        if self._backend == "pandas":
            return UnifiedFrame(self._data.dropna())  # type: ignore[union-attr]
        return UnifiedFrame(self._data.drop_nulls())  # type: ignore[union-attr]

    def describe(self):
        """Summary statistics, delegating to the underlying backend's `.describe()`."""
        return self._data.describe()

    def __repr__(self) -> str:
        return f"UnifiedFrame(backend={self._backend!r}, shape={self.shape})"


@dataclass
class GradientDescentResult:
    """Output of `ml.gradient_descent`: learned parameters and the loss curve."""
    weights: np.ndarray
    bias: float
    loss_history: List[float]


class ml:
    """Simplified, well-documented machine-learning shortcuts.

    Not a replacement for scikit-learn -- a thin, teaching-friendly layer
    over it, plus one hand-rolled algorithm (gradient descent) written so
    every step is visible, for when you want to see the mechanics rather
    than call a black box.
    """

    @staticmethod
    def linear_regression(X: np.ndarray, y: np.ndarray) -> _SKLinearRegression:
        """Fit ordinary least-squares linear regression via scikit-learn.

        Returns the fitted `sklearn.linear_model.LinearRegression` object,
        so `.coef_`, `.intercept_`, and `.predict()` all work as normal.
        """
        model = _SKLinearRegression()
        model.fit(X, y)
        return model

    @staticmethod
    def gradient_descent(
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 500,
    ) -> GradientDescentResult:
        """Batch gradient descent for linear regression, written from scratch.

        This exists for learning: every update step is explicit NumPy code
        (no hidden optimizer), so you can see exactly how weights and bias
        move on each epoch and inspect the loss curve afterward.

        Parameters
        ----------
        X: shape (n_samples, n_features)
        y: shape (n_samples,)
        learning_rate: step size for each gradient update.
        epochs: number of full passes over the data.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0.0
        loss_history: List[float] = []

        for _ in range(epochs):
            y_pred = X @ weights + bias
            error = y_pred - y
            loss = float(np.mean(error**2))
            loss_history.append(loss)

            grad_w = (2.0 / n_samples) * (X.T @ error)
            grad_b = (2.0 / n_samples) * np.sum(error)

            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b

        return GradientDescentResult(weights=weights, bias=bias, loss_history=loss_history)


class stats:
    """Shortcuts over the most commonly used `scipy.stats` routines."""

    @staticmethod
    def ttest(a: Iterable[float], b: Iterable[float]):
        """Independent two-sample t-test. Returns scipy's TtestResult."""
        return _scipy_stats.ttest_ind(a, b)

    @staticmethod
    def correlation(a: Iterable[float], b: Iterable[float]) -> dict:
        """Pearson correlation coefficient and p-value between two samples."""
        r, p_value = _scipy_stats.pearsonr(a, b)
        return {"r": float(r), "p_value": float(p_value)}
