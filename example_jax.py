"""Example usage of nano-vllm JAX."""

from __future__ import annotations

import os
from pathlib import Path

from nanovllm_jax import LLM, SamplingParams
from transformers import AutoTokenizer


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    default_model = repo_root / "models/qwen/Qwen-3-0.6B"
    model_path = Path(os.environ.get("NANOVLLM_MODEL_PATH", str(default_model))).expanduser()

    if not model_path.exists():
        raise SystemExit(
            f"Model path does not exist: {model_path}\n"
            "Set NANOVLLM_MODEL_PATH to a local HuggingFace model directory."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(str(model_path), enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n" + "=" * 50)
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
