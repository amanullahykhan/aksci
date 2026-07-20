"""
Opt-in auto-fix rules.

Everything here is off by default and must be explicitly enabled via
`ErrorResolver(auto_fix=True)` (or by calling `autofix.try_fix(...)`
directly). No rule here ever changes what your code *means* -- each one
only recovers from a narrow, well-understood failure in a way a careful
developer would type by hand, and every recovery is logged so it's never
silent.

Design constraints for every rule in `AUTO_FIX_RULES`:
  - Whitelisted: the rule only fires for the specific exception type +
    category it was written for, never as a generic catch-all.
  - Conservative: if a rule can't be confident about the fix, it declines
    (returns None) rather than guessing.
  - Logged: `try_fix` always returns an `AutoFixResult` describing exactly
    what happened, even when it changed nothing.
  - Reversible in spirit: rules recover a *value* to let the calling code
    continue (e.g. a fuzzy-matched column, a filled NaN) -- they never
    mutate source files, delete data, or hide the original exception from
    the diagnostic report.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class AutoFixResult:
    """Outcome of attempting an auto-fix."""
    applied: bool
    rule_name: str
    original_error: str
    recovered_value: Any = None
    description: str = ""


def _fix_missing_column_fuzzy_match(
    exc: BaseException, context: dict
) -> Optional[AutoFixResult]:
    """KeyError on a DataFrame-like object with a close-enough column name.

    Only fires if `context` supplies `available_columns` explicitly -- it
    never introspects arbitrary objects to avoid guessing wrong. Requires
    a fuzzy match cutoff of 0.6, same threshold as `suggest_column`, so
    the standalone suggestion path and the auto-fix path never disagree.
    """
    if not isinstance(exc, KeyError):
        return None
    available = context.get("available_columns")
    frame = context.get("frame")
    if not available:
        return None
    missing = str(exc.args[0]) if exc.args else ""
    matches = difflib.get_close_matches(missing, list(available), n=1, cutoff=0.6)
    if not matches:
        return None
    best = matches[0]
    recovered = None
    if frame is not None:
        try:
            recovered = frame[best]
        except Exception:
            return None
    return AutoFixResult(
        applied=True,
        rule_name="missing_column_fuzzy_match",
        original_error=str(exc),
        recovered_value=recovered,
        description=f"Column '{missing}' not found; used closest match '{best}' instead.",
    )


def _fix_null_values_fillna(
    exc: BaseException, context: dict
) -> Optional[AutoFixResult]:
    """ValueError from NaN/inf in a DataFrame/Series/array, when the caller
    opts in by supplying `frame` and a `fill_value` (default 0)."""
    msg = str(exc).lower()
    if not isinstance(exc, ValueError):
        return None
    if "nan" not in msg and "infinity" not in msg and "inf" not in msg:
        return None
    frame = context.get("frame")
    if frame is None:
        return None
    fill_value = context.get("fill_value", 0)
    try:
        if hasattr(frame, "fillna"):
            recovered = frame.fillna(fill_value)
        else:
            import numpy as np

            recovered = np.nan_to_num(frame, nan=fill_value)
    except Exception:
        return None
    return AutoFixResult(
        applied=True,
        rule_name="null_values_fillna",
        original_error=str(exc),
        recovered_value=recovered,
        description=f"Data contained NaN/inf; filled with {fill_value!r} to continue.",
    )


def _fix_division_by_zero_guard(
    exc: BaseException, context: dict
) -> Optional[AutoFixResult]:
    """ZeroDivisionError, when the caller supplies `numerator` and opts
    in with a `default` to use instead of raising."""
    if not isinstance(exc, ZeroDivisionError):
        return None
    if "default" not in context:
        return None
    default = context["default"]
    return AutoFixResult(
        applied=True,
        rule_name="division_by_zero_guard",
        original_error=str(exc),
        recovered_value=default,
        description=f"Division by zero; used fallback value {default!r} instead.",
    )


def _fix_type_coercion(
    exc: BaseException, context: dict
) -> Optional[AutoFixResult]:
    """TypeError/ValueError from a str where a number was expected, when
    the caller supplies `raw_value` and a `target_type` (int/float)."""
    if not isinstance(exc, (TypeError, ValueError)):
        return None
    raw_value = context.get("raw_value")
    target_type = context.get("target_type")
    if raw_value is None or target_type not in (int, float):
        return None
    try:
        recovered = target_type(str(raw_value).strip())
    except (TypeError, ValueError):
        return None
    return AutoFixResult(
        applied=True,
        rule_name="type_coercion",
        original_error=str(exc),
        recovered_value=recovered,
        description=(
            f"Coerced {raw_value!r} to {target_type.__name__}({recovered!r}) "
            "to continue."
        ),
    )


#: Ordered list of (name, rule_function) pairs. Order matters only in that
#: the first rule to return a non-None result wins -- rules are written to
#: be mutually exclusive by exception type/context key, so ordering rarely
#: matters in practice, but is kept deterministic (declaration order).
AUTO_FIX_RULES: list[tuple[str, Callable[[BaseException, dict], Optional[AutoFixResult]]]] = [
    ("missing_column_fuzzy_match", _fix_missing_column_fuzzy_match),
    ("null_values_fillna", _fix_null_values_fillna),
    ("division_by_zero_guard", _fix_division_by_zero_guard),
    ("type_coercion", _fix_type_coercion),
]


def try_fix(exc: BaseException, context: Optional[dict] = None) -> AutoFixResult:
    """Attempt every whitelisted rule against `exc` and return the first
    that applies. `context` supplies whatever a rule needs to act safely
    (e.g. `available_columns`, `frame`, `fill_value`, `default`,
    `raw_value`, `target_type`) -- rules that need context you didn't
    supply simply decline rather than guessing.

    Always returns an `AutoFixResult`; check `.applied` to see whether
    anything actually happened.
    """
    ctx = context or {}
    for name, rule in AUTO_FIX_RULES:
        try:
            result = rule(exc, ctx)
        except Exception:
            # A rule itself misbehaving must never crash the caller or
            # mask the original exception -- treat it as "no fix".
            continue
        if result is not None:
            return result
    return AutoFixResult(applied=False, rule_name="none", original_error=str(exc))
