"""Minimal base classes and data structures for agents and environments."""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    observation: Any = None
    action: Any = None
    model_response: str = ""
    chat_completions: list[dict[str, str]] = field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "agent"
    steps: list[Step] = field(default_factory=list)


class BaseAgent(ABC):
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
    @abstractmethod
    def reset(self) -> tuple[dict, dict]:
        pass

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        pass

    def close(self):
        return
