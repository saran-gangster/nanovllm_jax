from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Sampling parameters for text generation.
    
    Attributes:
        temperature: Sampling temperature. Must be > 0 (no greedy sampling).
        max_tokens: Maximum number of tokens to generate.
        ignore_eos: If True, continue generation past EOS token.
    """
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature <= 1e-10:
            raise ValueError("temperature must be > 1e-10")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
