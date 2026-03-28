"""nano-vLLM (JAX backend).

Exports are loaded lazily so host-side utilities (scheduler/block manager)
can be imported without initializing full model dependencies.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .llm import LLM
    from .sampling_params import SamplingParams

try:
    __version__ = version("nanovllm-jax")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["LLM", "SamplingParams", "__version__"]


def __getattr__(name: str):
    if name == "SamplingParams":
        from .sampling_params import SamplingParams

        return SamplingParams
    if name == "LLM":
        try:
            from .llm import LLM
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                f"Missing optional dependency '{e.name}' required for the JAX backend.\n"
                "Install JAX for your platform first, then install the package:\n"
                "  pip install -e ."
            ) from e
        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
