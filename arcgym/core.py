"""
Minimal base classes and data structures for agents and environments.

Inlined from rllm to remove the submodule dependency — only the parts
actually used by arcgym are kept.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Agent data structures
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """A single step in an agent trajectory."""
    observation: Any = None
    action: Any = None
    model_response: str = ""
    chat_completions: list[dict[str, str]] = field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    """Ordered sequence of Steps for one episode."""
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "agent"
    steps: list[Step] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Minimal abstract agent interface."""

    @property
    def trajectory(self) -> Trajectory:
        return Trajectory()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        raise NotImplementedError

    def update_from_model(self, response: str, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        return


class BaseEnv(ABC):
    """Minimal abstract environment interface."""

    @abstractmethod
    def reset(self) -> tuple[dict, dict]:
        pass

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        pass

    def close(self):
        return
