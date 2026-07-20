"""Unified facade over pandas, Polars, NumPy, SciPy, and scikit-learn."""
from .facade import GradientDescentResult, UnifiedFrame, ml, stats

__all__ = ["UnifiedFrame", "ml", "stats", "GradientDescentResult"]
