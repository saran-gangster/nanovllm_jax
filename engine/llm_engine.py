"""LLM Engine for text generation.

Main inference engine that coordinates:
- Request management and tokenization
- Scheduling (prefill vs decode batching)
- Model execution via ModelRunner
- Output decoding
"""

import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm_jax.config import Config
from nanovllm_jax.sampling_params import SamplingParams
from nanovllm_jax.engine.sequence import Sequence
from nanovllm_jax.engine.scheduler import Scheduler
from nanovllm_jax.engine.model_runner import ModelRunner


class LLMEngine:
    """Main LLM inference engine.
    
    Coordinates all components for text generation:
    - Tokenizer for encoding/decoding
    - Scheduler for request batching
    - ModelRunner for GPU execution
    
    Unlike the PyTorch version, JAX handles multi-device coordination
    through its single-controller model, so we don't need explicit
    multiprocessing with shared memory.
    
    Attributes:
        model_runner: Executes model on GPU.
        tokenizer: HuggingFace tokenizer.
        scheduler: Manages request batching.
    """
    
    def __init__(self, model: str, **kwargs):
        """Initialize the LLM engine.
        
        Args:
            model: Path to the HuggingFace model directory.
            **kwargs: Additional Config parameters.
        """
        # Extract config kwargs
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        
        # For JAX, we use a simplified single-process model
        # Multi-device TP is handled via JAX's sharding primitives
        self.model_runner = ModelRunner(config, tp_rank=0)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        
        # Initialize scheduler
        self.scheduler = Scheduler(config)
        
        # Register cleanup
        atexit.register(self.exit)
    
    def exit(self):
        """Cleanup resources."""
        if hasattr(self, 'model_runner'):
            self.model_runner.exit()
            del self.model_runner
    
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Add a generation request.
        
        Args:
            prompt: Text prompt or pre-tokenized token IDs.
            sampling_params: Sampling parameters for this request.
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
    
    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """Execute one generation step.
        
        Schedules a batch (prefill or decode), runs the model,
        and processes the outputs.
        
        Returns:
            Tuple of:
            - List of (seq_id, completion_tokens) for finished sequences
            - Number of tokens processed (positive for prefill, negative for decode)
        """
        # Schedule batch
        seqs, is_prefill = self.scheduler.schedule()
        
        # Run model
        token_ids = self.model_runner.run(seqs, is_prefill)
        
        # Process outputs
        self.scheduler.postprocess(seqs, token_ids)
        
        # Collect finished sequences
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs if seq.is_finished
        ]
        
        # Token count (positive for prefill, negative for decode)
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        
        return outputs, num_tokens
    
    def is_finished(self) -> bool:
        """Check if all requests are complete."""
        return self.scheduler.is_finished()
    
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        """Generate completions for a batch of prompts.
        
        Args:
            prompts: List of text prompts or token ID lists.
            sampling_params: Sampling params (single or per-prompt).
            use_tqdm: Whether to show progress bar.
        
        Returns:
            List of dicts with "text" and "token_ids" for each prompt.
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        # Handle single sampling_params for all prompts
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # Add all requests
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        # Generate
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            
            # Update progress
            if use_tqdm:
                elapsed = perf_counter() - t
                if num_tokens > 0:
                    prefill_throughput = num_tokens / elapsed
                else:
                    decode_throughput = -num_tokens / elapsed
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # Collect outputs
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        # Sort by sequence ID to maintain order
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        
        # Decode tokens to text
        outputs = [
            {
                "text": self.tokenizer.decode(token_ids, skip_special_tokens=True),
                "token_ids": token_ids
            }
            for token_ids in outputs
        ]
        
        if use_tqdm:
            pbar.close()
        
        return outputs
