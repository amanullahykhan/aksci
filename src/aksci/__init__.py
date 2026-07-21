"""
AK-SCI: A Unified, AI-Assisted Data Science Toolkit
=====================================================
Author: Amanullah Khan

AK-SCI gives you:
  - ErrorResolver     : offline ML-based diagnosis of runtime errors, with
                         optional cloud AI-assist for deeper explanations.
  - MicroPipeline      : memory-bounded, chunked streaming for large datasets.
  - UnifiedFrame       : one API over pandas and Polars DataFrames.
  - ml, stats          : simplified, well-documented ML and stats shortcuts
                         over scikit-learn and SciPy.

Quick start
-----------
>>> from aksci import ErrorResolver, UnifiedFrame
>>> resolver = ErrorResolver()
>>>
>>> @resolver.safe_run(reraise=False)
... def risky():
...     return {"a": 1}["b"]
>>> risky()  # prints a diagnostic report to stderr, returns None
"""
from .ai_core.client import AIClient
from .error_handler.resolver import ErrorResolver
from .micro_pipeline.buffer import BoundedBuffer, MicroPipeline
from .unified_api.facade import GradientDescentResult, UnifiedFrame, ml, stats

__version__ = "0.1.1"
__author__ = "Amanullah Khan"

__all__ = [
    "ErrorResolver",
    "MicroPipeline",
    "BoundedBuffer",
    "UnifiedFrame",
    "ml",
    "stats",
    "GradientDescentResult",
    "AIClient",
]