"""Engine components for LLM inference.

Heavy modules are imported lazily to keep host-side utilities cheap to import.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .sequence import Sequence, SequenceStatus
    from .block_manager import BlockManager, Block
    from .scheduler import Scheduler
    from .model_runner import ModelRunner
    from .llm_engine import LLMEngine

__all__ = [
    "Sequence",
    "SequenceStatus",
    "BlockManager",
    "Block",
    "Scheduler",
    "ModelRunner",
    "LLMEngine",
]


def __getattr__(name: str):
    if name in {"Sequence", "SequenceStatus"}:
        from .sequence import Sequence, SequenceStatus

        return {"Sequence": Sequence, "SequenceStatus": SequenceStatus}[name]
    if name in {"BlockManager", "Block"}:
        from .block_manager import BlockManager, Block

        return {"BlockManager": BlockManager, "Block": Block}[name]
    if name == "Scheduler":
        from .scheduler import Scheduler

        return Scheduler
    if name == "ModelRunner":
        from .model_runner import ModelRunner

        return ModelRunner
    if name == "LLMEngine":
        from .llm_engine import LLMEngine

        return LLMEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
