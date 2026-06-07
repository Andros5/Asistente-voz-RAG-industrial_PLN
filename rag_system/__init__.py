"""Sistema RAG de la PoC PLN T2 -> T3 -> T4."""

from .config import Settings, load_settings


def call_rag_system(query_en: str, top_k: int | None = None, mode: str | None = None) -> str:
    """Entrada compatible con el bucle conversacional existente."""
    from .pipeline import call_rag_system as _call_rag_system

    return _call_rag_system(query_en=query_en, top_k=top_k, mode=mode)


def call_rag_system_with_timing(query_en: str, top_k: int | None = None, mode: str | None = None):
    """Entrada T3 con evidencia y diagnostico de latencia."""
    from .pipeline import call_rag_system_with_timing as _call_rag_system_with_timing

    return _call_rag_system_with_timing(query_en=query_en, top_k=top_k, mode=mode)


__all__ = ["Settings", "load_settings", "call_rag_system", "call_rag_system_with_timing"]
