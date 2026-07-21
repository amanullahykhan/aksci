"""Error handling: AI-assisted diagnosis and safe-run decorators."""
from . import autofix
from .autofix import AutoFixResult
from .resolver import ErrorResolver

__all__ = ["ErrorResolver", "autofix", "AutoFixResult"]
