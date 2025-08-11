"""Shared utilities for Ideological-Political course assistants.

This package centralizes reusable agents, LLM wrappers, prompts, and vector
helpers so multiple subject-specific projects can import them without code
duplication.
"""

__all__ = [
    "base_agent",
    "base_dialogue_agent",
    "base_kg_agent",
    "base_retrieval_agent",
    "llm_wrapper",
    "multimodal_agent",
    "prompts",
    "vector_utils",
]


