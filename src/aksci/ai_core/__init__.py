"""AI core: local diagnostic model, optional cloud AI-assist client, and prompts."""
from .client import AIClient, AIAssistUnavailable
from .diagnostics import LocalDiagnosticModel, Diagnosis

__all__ = ["AIClient", "AIAssistUnavailable", "LocalDiagnosticModel", "Diagnosis"]
