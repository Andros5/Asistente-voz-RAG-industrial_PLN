import torch
from transformers import GenerationConfig

from llm_system.local_llm import LocalLLM


class TokenBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors, return_dict):
        return TokenBatch({"input_ids": torch.tensor([[10, 20, 30]])})

    def decode(self, token_ids, skip_special_tokens):
        return "ok"


class FakeModel:
    def __init__(self):
        self.generation_config = GenerationConfig(max_length=262144)
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor([[10, 20, 30, 40]])


def test_local_llm_passes_new_token_limit_without_max_length_clash():
    model = FakeModel()
    llm = LocalLLM(
        tokenizer=FakeTokenizer(),
        model=model,
        device="cpu",
        structured_chat=False,
    )

    assert llm.query("System", "User", max_new_tokens=512) == "ok"

    generate_kwargs = model.generate_calls[-1]
    generation_config = generate_kwargs["generation_config"]

    assert generate_kwargs["max_new_tokens"] == 512
    assert generate_kwargs["use_model_defaults"] is False
    assert generation_config.max_length is None
    assert generation_config.max_new_tokens is None
