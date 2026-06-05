"""LLM local para las tareas T2 y T4 de la PoC."""

from .local_llm import DEFAULT_LLM_MODEL_ID, LocalLLM, load_local_llm

__all__ = ["DEFAULT_LLM_MODEL_ID", "LocalLLM", "load_local_llm"]
