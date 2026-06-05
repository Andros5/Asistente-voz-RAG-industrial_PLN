from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass

DEFAULT_LLM_MODEL_ID = "mistralai/Ministral-3-8B-Instruct-2512"


def _prepare_runtime_environment() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _is_mistral3_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return "mistral-3" in normalized or "ministral-3" in normalized


def _messages(system_prompt: str, user_text: str, structured: bool) -> list[dict]:
    instruction_block = (
        "Sigue estrictamente estas instrucciones.\n\n"
        f"{system_prompt.strip()}\n\n"
        "Entrada:\n"
        f"{user_text.strip()}\n\n"
        "Salida:"
    )
    if structured:
        return [
            {"role": "user", "content": [{"type": "text", "text": instruction_block}]},
        ]
    return [
        {"role": "user", "content": instruction_block},
    ]


def _first_model_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    for parameter in model.parameters():
        if getattr(parameter, "device", None) is not None and parameter.device.type != "meta":
            return parameter.device
    return "cpu"


def _resolve_dtype(dtype_name: str):
    import torch

    if dtype_name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError("dtype debe ser uno de: auto, bfloat16, float16, float32")


@dataclass
class LocalLLM:
    tokenizer: object
    model: object
    device: object
    structured_chat: bool

    def _generation_config(self, *, do_sample: bool):
        generation_config = deepcopy(self.model.generation_config)
        generation_config.do_sample = do_sample
        generation_config.max_length = None
        if not do_sample:
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None
        return generation_config

    def warmup(self) -> None:
        import torch

        dummy = _messages("You are a concise assistant.", "Hi", self.structured_chat)
        tokenized = self.tokenizer.apply_chat_template(
            dummy,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        with torch.no_grad():
            generation_config = self._generation_config(do_sample=False)
            generation_config.max_new_tokens = 1
            generation_config.pad_token_id = self.tokenizer.eos_token_id
            self.model.generate(
                **tokenized,
                generation_config=generation_config,
            )

    def query(
        self,
        system_prompt: str,
        user_text: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> str:
        import torch
        from transformers import TextStreamer

        messages = _messages(system_prompt, user_text, self.structured_chat)
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        do_sample = temperature > 0.0
        generation_config = self._generation_config(do_sample=do_sample)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.pad_token_id = self.tokenizer.eos_token_id
        if do_sample:
            generation_config.temperature = temperature

        gen_kwargs = {
            "generation_config": generation_config,
        }
        if stream:
            gen_kwargs["streamer"] = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.no_grad():
            output_ids = self.model.generate(**tokenized, **gen_kwargs)

        input_length = tokenized["input_ids"].shape[1]
        return self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True).strip()


def load_local_llm(
    model_id: str = DEFAULT_LLM_MODEL_ID,
    *,
    dtype: str = "auto",
    local_files_only: bool = False,
    warmup: bool = True,
) -> LocalLLM:
    _prepare_runtime_environment()
    _load_dotenv_if_available()

    import torch

    resolved_dtype = _resolve_dtype(dtype)
    device_map = "auto" if torch.cuda.is_available() else None
    model_kwargs = {
        "dtype": resolved_dtype,
        "local_files_only": local_files_only,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    if _is_mistral3_model(model_id):
        from transformers import Mistral3ForConditionalGeneration

        try:
            from transformers import MistralCommonBackend as MistralTokenizerBackend
        except ImportError:
            from transformers import MistralCommonTokenizer as MistralTokenizerBackend

        tokenizer = MistralTokenizerBackend.from_pretrained(model_id, local_files_only=local_files_only)
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs,
        )
        structured_chat = True
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        structured_chat = False

    llm = LocalLLM(
        tokenizer=tokenizer,
        model=model,
        device=_first_model_device(model),
        structured_chat=structured_chat,
    )
    if warmup:
        llm.warmup()
    return llm
