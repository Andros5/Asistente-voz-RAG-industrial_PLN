from __future__ import annotations

from dataclasses import dataclass
import time

from llm_system import DEFAULT_LLM_MODEL_ID, load_local_llm
from rag_system.pipeline import call_rag_system_with_timing
from rag_system.poc_prompts import PROMPT_T2A, PROMPT_T2B, PROMPT_T4A, PROMPT_T4B


@dataclass
class PoCTimings:
    load_s: float
    t2a_s: float
    t2b_s: float
    t3_s: float
    t3_embedding_s: float
    t3_retrieval_s: float
    t3_bm25_s: float
    t3_vector_s: float
    t3_fusion_s: float
    t4a_s: float
    t4b_s: float
    total_s: float

    def as_dict(self) -> dict[str, float]:
        return {
            "load_s": self.load_s,
            "t2a_s": self.t2a_s,
            "t2b_s": self.t2b_s,
            "t3_s": self.t3_s,
            "t3_embedding_s": self.t3_embedding_s,
            "t3_retrieval_s": self.t3_retrieval_s,
            "t3_bm25_s": self.t3_bm25_s,
            "t3_vector_s": self.t3_vector_s,
            "t3_fusion_s": self.t3_fusion_s,
            "t4a_s": self.t4a_s,
            "t4b_s": self.t4b_s,
            "total_s": self.total_s,
        }


@dataclass
class PoCResult:
    original_query_es: str
    normalized_query_es: str
    normalized_query_en: str
    retrieved_evidence: str
    answer_en: str
    final_answer_es: str
    timings: PoCTimings


class PoCRunner:
    """Ejecutor reutilizable de la PoC completa."""

    def __init__(
        self,
        *,
        mode: str = "hybrid",
        top_k: int = 5,
        model_id: str = DEFAULT_LLM_MODEL_ID,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        dtype: str = "auto",
        local_files_only: bool = False,
        warmup: bool = True,
    ) -> None:
        self.mode = mode
        self.top_k = top_k
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.warmup = warmup
        self._llm = None

    def load(self, *, print_steps: bool = True):
        if self._llm is None:
            if print_steps:
                print("[Debug] Cargando tokenizer y modelo...", flush=True)
            self._llm = load_local_llm(
                model_id=self.model_id,
                dtype=self.dtype,
                local_files_only=self.local_files_only,
                warmup=self.warmup,
            )
        return self._llm

    def run(
        self,
        query_es: str,
        *,
        stream_final: bool = False,
        print_steps: bool = True,
        print_timings: bool = False,
    ) -> PoCResult:
        load_start = time.perf_counter()
        llm = self.load(print_steps=print_steps)
        load_s = time.perf_counter() - load_start
        total_start = time.perf_counter()

        if print_steps:
            print("\n[T2a] Normalizando consulta...", flush=True)
        start = time.perf_counter()
        normalized_query_es = llm.query(
            PROMPT_T2A,
            f"Consulta transcrita: {query_es}",
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        t2a_s = time.perf_counter() - start
        if print_steps:
            print(normalized_query_es, flush=True)

        if print_steps:
            print("\n[T2b] Traduciendo al ingles...", flush=True)
        start = time.perf_counter()
        normalized_query_en = llm.query(
            PROMPT_T2B,
            f"Consulta normalizada en espanol: {normalized_query_es}",
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        t2b_s = time.perf_counter() - start
        if print_steps:
            print(normalized_query_en, flush=True)

        if print_steps:
            print("\n[T3] Recuperando evidencia...", flush=True)
        start = time.perf_counter()
        retrieved_evidence, t3_timing = call_rag_system_with_timing(
            normalized_query_en,
            mode=self.mode,
            top_k=self.top_k,
        )
        t3_s = time.perf_counter() - start
        if print_steps:
            print(retrieved_evidence, flush=True)

        if print_steps:
            print("\n[T4a] Generando respuesta tecnica en ingles...", flush=True)
        input_t4a = f"Normalized query: {normalized_query_en}\n\nRetrieved evidence:\n{retrieved_evidence}"
        start = time.perf_counter()
        answer_en = llm.query(
            PROMPT_T4A,
            input_t4a,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        t4a_s = time.perf_counter() - start
        if print_steps:
            print(answer_en, flush=True)

        if print_steps:
            print("\n[T4b] Traduciendo y adaptando al espanol...", flush=True)
        input_t4b = (
            f"Consulta original (contexto): {query_es}\n\n"
            f"Respuesta tecnica en ingles: {answer_en}"
        )
        start = time.perf_counter()
        final_answer_es = llm.query(
            PROMPT_T4B,
            input_t4b,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stream=stream_final,
        )
        t4b_s = time.perf_counter() - start
        if print_steps and not stream_final:
            print(final_answer_es, flush=True)
        total_s = time.perf_counter() - total_start

        timings = PoCTimings(
            load_s=load_s,
            t2a_s=t2a_s,
            t2b_s=t2b_s,
            t3_s=t3_s,
            t3_embedding_s=t3_timing.embedding_s,
            t3_retrieval_s=t3_timing.retrieval_s,
            t3_bm25_s=t3_timing.bm25_s,
            t3_vector_s=t3_timing.vector_s,
            t3_fusion_s=t3_timing.fusion_s,
            t4a_s=t4a_s,
            t4b_s=t4b_s,
            total_s=total_s,
        )
        if print_timings:
            if stream_final:
                print(flush=True)
            print("\n[Timings]", flush=True)
            for name, value in timings.as_dict().items():
                print(f"{name}: {value:.3f}s", flush=True)

        return PoCResult(
            original_query_es=query_es,
            normalized_query_es=normalized_query_es,
            normalized_query_en=normalized_query_en,
            retrieved_evidence=retrieved_evidence,
            answer_en=answer_en,
            final_answer_es=final_answer_es,
            timings=timings,
        )


def run_poc(
    query_es: str,
    *,
    mode: str = "hybrid",
    top_k: int = 5,
    model_id: str = DEFAULT_LLM_MODEL_ID,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    dtype: str = "auto",
    local_files_only: bool = False,
    stream_final: bool = False,
    print_steps: bool = True,
    print_timings: bool = False,
) -> PoCResult:
    runner = PoCRunner(
        mode=mode,
        top_k=top_k,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        dtype=dtype,
        local_files_only=local_files_only,
        warmup=True,
    )
    return runner.run(
        query_es,
        stream_final=stream_final,
        print_steps=print_steps,
        print_timings=print_timings,
    )
