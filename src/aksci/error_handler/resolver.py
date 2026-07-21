"""
AI-assisted error resolution.

Design principle: this module explains and suggests -- it never silently
rewrites your code. The only "auto" behavior it performs is returning a
safe fallback value from a decorated function (opt-in via `reraise=False`)
or suggesting the closest matching column/key name (which you must accept
explicitly by using it). This keeps debugging fast without hiding bugs.
"""
from __future__ import annotations

import difflib
import functools
import sys
import traceback
from typing import Any, Callable, List, Optional, TypeVar

from ..ai_core.client import AIAssistUnavailable, AIClient
from ..ai_core.diagnostics import Diagnosis, LocalDiagnosticModel
from .autofix import AutoFixResult, try_fix

F = TypeVar("F", bound=Callable[..., Any])


class ErrorResolver:
    """Central AI-assisted error handler.

    Parameters
    ----------
    ai_client:
        Optional `AIClient` for deeper cloud-based explanations. If omitted
        or unconfigured, only the local offline model is used.
    auto_fix:
        If True, `safe_run` will attempt the whitelisted auto-fix rules in
        `error_handler.autofix` before falling back to `reraise`/`None`
        behavior. Off by default -- see `autofix.py` for exactly what each
        rule does and does not do. Auto-fix never runs unless you turn it
        on, and every fix it applies is logged to stderr.
    """

    def __init__(
        self, ai_client: Optional[AIClient] = None, auto_fix: bool = False
    ) -> None:
        self._model = LocalDiagnosticModel()
        self._ai_client = ai_client
        self._auto_fix = auto_fix

    def diagnose(self, exc: BaseException) -> Diagnosis:
        """Classify an exception using the local embedded model."""
        return self._model.diagnose(type(exc).__name__, str(exc))

    def explain(self, exc: BaseException) -> str:
        """Build a human-readable diagnostic report for an exception."""
        diag = self.diagnose(exc)
        lines = [
            f"[AKSCI] Caught {type(exc).__name__}: {exc}",
            f"  Category   : {diag.category} (confidence {diag.confidence:.0%})",
            f"  Likely fix : {diag.suggestion}",
        ]
        if self._ai_client is not None and self._ai_client.available:
            try:
                ai_result = self._ai_client.diagnose_error(
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    traceback_text=traceback.format_exc(),
                )
                if ai_result.get("explanation"):
                    lines.append(f"  AI insight : {ai_result['explanation']}")
                if ai_result.get("fix_code"):
                    lines.append(f"  Suggested code:\n{ai_result['fix_code']}")
            except AIAssistUnavailable:
                pass  # local diagnosis already provided above
        return "\n".join(lines)

    @staticmethod
    def suggest_column(missing: str, available: List[str]) -> Optional[str]:
        """Fuzzy-match a mistyped column/key name against real ones.

        Returns the closest match (or None) -- the caller decides whether
        to use it. This never modifies data on its own.
        """
        matches = difflib.get_close_matches(missing, available, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def try_auto_fix(
        self, exc: BaseException, context: Optional[dict] = None
    ) -> AutoFixResult:
        """Attempt the whitelisted auto-fix rules against `exc`.

        Works regardless of the `auto_fix` constructor flag -- that flag
        only controls whether `safe_run` calls this automatically. Always
        returns an `AutoFixResult`; check `.applied`. See `autofix.py` for
        what each rule requires in `context` and what it will and won't do.
        """
        return try_fix(exc, context)

    def safe_run(
        self,
        func: Optional[F] = None,
        *,
        reraise: bool = True,
        auto_fix_context: Optional[Callable[..., dict]] = None,
    ) -> Callable:
        """Decorator that diagnoses exceptions raised inside `func`.

        By default it prints the diagnosis and re-raises (so you still see
        the real traceback). Pass `reraise=False` to instead swallow the
        exception and return None -- useful for optional/best-effort steps
        in a larger script, not for silently hiding real bugs.

        If this resolver was built with `ErrorResolver(auto_fix=True)`,
        pass `auto_fix_context` -- a callable taking the same `(*args,
        **kwargs)` as `func` and returning a context dict (see
        `autofix.try_fix`) -- to let a whitelisted rule recover a value
        and return it instead of raising/swallowing. If no rule applies,
        behavior falls through to `reraise`/`None` as usual.

        Important: auto-fix replaces the *entire return value* of `func`
        for that call -- it cannot resume execution partway through a
        function body that failed on its second or third risky operation.
        It works best wrapping a function that does exactly one risky
        thing (e.g. `df[col]`, `a / b`, `float(x)`) rather than a
        multi-step function where any one of several lines might fail.
        For a multi-step function, wrap each risky line in its own
        small helper and decorate that instead.
        """

        def decorator(fn: F) -> F:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    print(self.explain(exc), file=sys.stderr)
                    if self._auto_fix:
                        ctx = auto_fix_context(*args, **kwargs) if auto_fix_context else {}
                        fix = self.try_auto_fix(exc, ctx)
                        if fix.applied:
                            print(
                                f"  [AKSCI auto-fix] {fix.rule_name}: {fix.description}",
                                file=sys.stderr,
                            )
                            return fix.recovered_value
                    if reraise:
                        raise
                    return None

            return wrapper  # type: ignore[return-value]

        if func is not None:
            return decorator(func)
        return decorator

    def install_global_hook(self) -> None:
        """Route all uncaught exceptions in the process through AK-SCI's
        diagnostic report, printed before the normal traceback."""

        def hook(exc_type: type, exc_value: BaseException, tb: Any) -> None:
            print(self.explain(exc_value), file=sys.stderr)
            traceback.print_exception(exc_type, exc_value, tb)

        sys.excepthook = hook
