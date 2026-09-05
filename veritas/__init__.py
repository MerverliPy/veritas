"""veritas — accurate and reliable agent research team."""

__version__ = "0.1.0"

from .schema import (Claim, Evidence, Plan, Query, Report, Source, Surface,
                     SubQuestion, Verdict)
from .llm import BaseLLM, DeepSeekClient, FakeLLM, LLMError
from .pipeline.runner import Runner

__all__ = [
    "__version__",
    "Claim", "Evidence", "Plan", "Query", "Report", "Source", "Surface",
    "SubQuestion", "Verdict",
    "BaseLLM", "DeepSeekClient", "FakeLLM", "LLMError",
    "Runner",
]
