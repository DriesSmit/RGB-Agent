"""
ARC-AGI-3 environment wrapper using the arc_agi toolkit.

This adapter provides a Gym-compatible interface for ARC-AGI-3 games
using the official arc_agi toolkit for all backend communication.
"""
from __future__ import annotations

from typing import Any, Mapping

import arc_agi
from arc_agi import OperationMode
from arcengine import FrameDataRaw, GameAction, GameState

from arcgym.core import BaseEnv


class ArcAgi3Env(BaseEnv):
    """
    Wraps the ARC-AGI toolkit with Gym-compatible reset/step mechanics.
    """

    def __init__(
        self,
        game_id: str,
        max_actions: int = 80,
        reward_mode: str = "binary",
        reward_scale: float = 1.0,
        # Arcade pass-through args
        arc_api_key: str = "",
        arc_base_url: str = "https://three.arcprize.org",
        operation_mode: OperationMode = OperationMode.NORMAL,
    ) -> None:
        self.game_id = game_id
        self.max_actions = max_actions
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self._arc = arc_agi.Arcade(
            arc_api_key=arc_api_key,
            arc_base_url=arc_base_url,
            operation_mode=operation_mode,
        )
        self._env = None
        self._actions_taken = 0
        self._last_obs: FrameDataRaw | None = None
        self._external_scorecard = False
        self._scorecard_id: str | None = None

    @classmethod
    def from_arcade(
        cls,
        arcade: arc_agi.Arcade,
        game_id: str,
        scorecard_id: str,
        max_actions: int = 80,
        reward_mode: str = "binary",
        reward_scale: float = 1.0,
    ) -> "ArcAgi3Env":
        """Create an env with an externally-managed scorecard (for Swarm mode).

        The Swarm opens one scorecard for all games and owns its lifecycle.
        Envs created via this factory will NOT open/close their own scorecards.
        """
        inst = cls.__new__(cls)
        inst.game_id = game_id
        inst.max_actions = max_actions
        inst.reward_mode = reward_mode
        inst.reward_scale = reward_scale
        inst._arc = arcade
        inst._scorecard_id = scorecard_id
        inst._env = None
        inst._actions_taken = 0
        inst._last_obs = None
        inst._external_scorecard = True
        return inst

    def reset(self, task: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and return the initial observation."""
        game_id = (task or {}).get("game_id", self.game_id)
        if not self._external_scorecard:
            tags = (task or {}).get("tags", [])
            self._scorecard_id = self.open_scorecard(tags=tags)
        self._env = self._arc.make(game_id, scorecard_id=self._scorecard_id)
        obs = self._env.reset()
        self._last_obs = obs
        observation = self._format_observation(obs)
        return observation

    def step(self, action_payload: Any) -> tuple[dict, float, bool]:
        """Take a step in the environment."""
        if self._env is None or self._last_obs is None:
            raise RuntimeError("ArcAgi3Env.step called before reset.")

        action, payload, reasoning = self._coerce_action(action_payload)
        obs = self._env.step(action, data=payload, reasoning=reasoning)
        if obs is None:
            raise ConnectionError("ARC API returned None — connection likely dropped")
        self._actions_taken += 1
        self._last_obs = obs
        reward = self._compute_reward(obs)
        done = obs.state in (GameState.WIN, GameState.GAME_OVER) or self._actions_taken >= self.max_actions
        observation = self._format_observation(obs)
        return observation, reward, done

    def close(self) -> None:
        """Close the environment and scorecard.

        If the scorecard is externally managed (Swarm mode), only resets
        local state — the Swarm is responsible for closing the scorecard.
        """
        if not self._external_scorecard:
            self.close_scorecard(self._scorecard_id)
        self._env = None
        self._last_obs = None
        self._actions_taken = 0

    def open_scorecard(self, tags: list[str] | None = None) -> str:
        """Open a new scorecard."""
        return self._arc.open_scorecard(tags=tags)

    def close_scorecard(self, card_id: str | None = None):
        """Close a scorecard by ID."""
        return self._arc.close_scorecard(card_id)

    def get_scorecard(self) -> str:
        """Get the scorecard ID."""
        return self._arc.get_scorecard(self._scorecard_id)

    def _format_observation(self, obs: FrameDataRaw) -> dict[str, Any]:
        """Format FrameDataRaw into observation dict."""
        return {
            "game_id": obs.game_id,
            "state": obs.state.name,
            "score": obs.levels_completed,
            "frame": [layer.tolist() if hasattr(layer, "tolist") else layer for layer in obs.frame],
            "available_actions": obs.available_actions,
            "guid": obs.guid,
        }

    _REASONING_MAX_BYTES = 16000  # ActionInput hard limit is 16384; keep a small buffer

    def _coerce_action(self, action_payload: Any) -> tuple[GameAction, dict[str, Any], Any | None]:
        """Convert action payload into GameAction, data dict, and reasoning."""
        if isinstance(action_payload, Mapping):
            action = action_payload.get("action")
            reasoning = action_payload.get("reasoning")
            if isinstance(reasoning, str):
                encoded = reasoning.encode("utf-8")
                if len(encoded) > self._REASONING_MAX_BYTES:
                    reasoning = encoded[: self._REASONING_MAX_BYTES].decode("utf-8", errors="ignore")
            payload = {k: v for k, v in action_payload.items() if k not in {"action", "reasoning"}}
            return action, payload, reasoning
        raise TypeError(f"Unsupported action payload type: {type(action_payload)}")

    def _compute_reward(self, obs: FrameDataRaw) -> float:
        """Compute reward from observation."""
        if self.reward_mode == "score":
            base = obs.levels_completed
        elif self.reward_mode == "binary":
            base = 1.0 if obs.state == GameState.WIN else 0.0
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode!r}")
        return float(base) * float(self.reward_scale)

