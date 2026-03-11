"""re_arc environment wrapper using local offline environments."""
from __future__ import annotations

from typing import Any, Mapping

from arcengine import FrameDataRaw, GameAction
from re_arc import EnvSampler

from rgb_agent.environment import BaseEnv


class ReArcEnv(BaseEnv):
    """Gym-compatible interface for re_arc games."""

    _REASONING_MAX_BYTES = 16000

    def __init__(
        self,
        game_id: str,
        max_actions: int = 80,
        reward_mode: str = "transition",
        reward_scale: float = 1.0,
        seed: int | None = None,
        augment: bool = False,
        augmentation_config: dict[str, bool] | None = None,
        environments_dir: str | None = None,
    ) -> None:
        self.game_id = game_id
        self.max_actions = max_actions
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self.seed = seed
        self._sampler = EnvSampler(
            include=[game_id],
            augment=augment,
            augmentation_config=augmentation_config,
            environments_dir=environments_dir,
            seed=seed,
        )
        self._env = None
        self._actions_taken = 0
        self._last_obs: FrameDataRaw | None = None

    def reset(self, task: dict | None = None) -> dict[str, Any]:
        task = task or {}
        game_id = task.get("game_id", self.game_id)
        seed = task.get("seed", self.seed)
        self._env = self._sampler.make(game_id=game_id, seed=seed)
        obs = self._env.reset()
        self._actions_taken = 0
        self._last_obs = obs
        return self._format_observation(obs)

    def step(self, action_payload: Any) -> tuple[dict[str, Any], float, bool]:
        if self._env is None or self._last_obs is None:
            raise RuntimeError("step() called before reset()")

        action, payload, _reasoning = self._coerce_action(action_payload)
        step_result = self._env.step(action, data=payload)
        if not isinstance(step_result, tuple) or len(step_result) != 4:
            raise RuntimeError(f"Unexpected step result type from re_arc env: {type(step_result)}")

        obs, reward, done, _info = step_result
        if obs is None:
            raise RuntimeError("re_arc env returned no observation after step()")

        self._actions_taken += 1
        self._last_obs = obs
        done = bool(done) or self._actions_taken >= self.max_actions
        reward = self._compute_reward(obs, reward)
        return self._format_observation(obs), reward, done

    def close(self) -> None:
        self._env = None
        self._last_obs = None
        self._actions_taken = 0

    def _format_observation(self, obs: FrameDataRaw) -> dict[str, Any]:
        return {
            "game_id": obs.game_id,
            "state": obs.state.name,
            "score": obs.levels_completed,
            "frame": [layer.tolist() if hasattr(layer, "tolist") else layer for layer in obs.frame],
            "available_actions": obs.available_actions,
            "guid": None,
        }

    def _coerce_action(self, action_payload: Any) -> tuple[GameAction, dict[str, Any], Any | None]:
        if isinstance(action_payload, Mapping):
            action = action_payload.get("action")
            if isinstance(action, str):
                action = GameAction.from_name(action)
            elif isinstance(action, int):
                action = GameAction(action)
            if not isinstance(action, GameAction):
                raise TypeError(f"Unsupported action type: {type(action)}")

            reasoning = action_payload.get("reasoning")
            if isinstance(reasoning, str):
                encoded = reasoning.encode("utf-8")
                if len(encoded) > self._REASONING_MAX_BYTES:
                    reasoning = encoded[:self._REASONING_MAX_BYTES].decode("utf-8", errors="ignore")

            payload = {k: v for k, v in action_payload.items() if k not in {"action", "reasoning"}}
            return action, payload, reasoning

        raise TypeError(f"Unsupported action payload type: {type(action_payload)}")

    def _compute_reward(self, obs: FrameDataRaw, transition_reward: float) -> float:
        if self.reward_mode == "transition":
            base = float(transition_reward)
        elif self.reward_mode == "score":
            base = float(obs.levels_completed)
        elif self.reward_mode == "binary":
            base = 1.0 if obs.state.name == "WIN" else 0.0
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode!r}")
        return float(base) * float(self.reward_scale)
