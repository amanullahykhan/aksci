"""
Optional cloud AI-assist client.

AK-SCI's error resolution and diagnostics work fully offline via the local
model in `ai_core.diagnostics`. This client is a strictly opt-in layer: it
only activates if the user supplies an API key, and it never runs
automatically. It exists to give deeper, natural-language explanations and
suggested fixes for errors the local classifier is unsure about.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from .prompts import ERROR_DIAGNOSIS_SYSTEM_PROMPT


class AIAssistUnavailable(RuntimeError):
    """Raised when AI-assist mode is requested but not configured/installed."""


class AIClient:
    """Thin wrapper around the Anthropic API for optional AI-assisted diagnostics.

    Parameters
    ----------
    api_key:
        Anthropic API key. If omitted, falls back to the
        ``AKSCI_ANTHROPIC_API_KEY`` environment variable. If neither is
        set, ``available`` is False and every method raises
        ``AIAssistUnavailable`` instead of failing silently.
    model:
        Model string to use for requests.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6") -> None:
        self.api_key = api_key or os.environ.get("AKSCI_ANTHROPIC_API_KEY")
        self.model = model
        self._client: Any = None

    @property
    def available(self) -> bool:
        """True if an API key is configured. Does not verify the key is valid."""
        return bool(self.api_key)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic  # local import: optional dependency
            except ImportError as exc:
                raise AIAssistUnavailable(
                    "AI-assist mode needs the 'anthropic' package. "
                    "Install it with: pip install aksci[ai]"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def diagnose_error(
        self,
        error_type: str,
        error_message: str,
        traceback_text: str = "",
        code_context: str = "",
    ) -> dict:
        """Ask the model for a root cause, fix, and explanation.

        Returns a dict with keys: root_cause, fix_code, explanation, confidence.
        Raises AIAssistUnavailable if no API key is configured.
        """
        if not self.available:
            raise AIAssistUnavailable(
                "No API key configured. Pass api_key=... or set "
                "AKSCI_ANTHROPIC_API_KEY."
            )
        client = self._get_client()
        user_prompt = (
            f"Error type: {error_type}\n"
            f"Message: {error_message}\n"
            f"Traceback:\n{traceback_text}\n\n"
            f"Code context:\n{code_context}"
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=600,
            system=ERROR_DIAGNOSIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Model didn't return clean JSON -- degrade gracefully rather than crash.
            return {
                "root_cause": "unparsed_response",
                "fix_code": "",
                "explanation": text,
                "confidence": 0.0,
            }
