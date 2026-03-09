"""Base ARC agent with rolling context, state-action memory, and hint injection."""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

from arcgym.core import BaseAgent, Step, Trajectory
from arcengine import GameAction
from arcgym.utils.grid_utils import (
    compute_grid_diff,
    format_grid_ascii,
    get_click_info,
    hash_grid_state,
)

log = logging.getLogger(__name__)


class BaseArcAgent(BaseAgent):
    """Two-phase agent (observation then action) with rolling context window.

    Subclasses override _call_observation_model and _call_action_model.
    """

    def __init__(
        self,
        *,
        name: str = "base_arc_agent",
        game_id: str | None = None,
        context_window_size: int = 5,
        show_tried_actions: bool = True,
        include_strategy_in_context: bool = False,
    ) -> None:
        self.name = name
        self.context_window_size = context_window_size
        self.show_tried_actions = show_tried_actions
        self.include_strategy_in_context = include_strategy_in_context
        self.game_id = game_id

        self._step_history: deque = deque(maxlen=self.context_window_size)
        self._state_action_memory: dict[str, dict[str, dict[str, Any]]] = {}
        self.reset()

    # -- lifecycle -------------------------------------------------------------

    def reset(self) -> None:
        self._trajectory = Trajectory(name=self.name)
        self._last_observation: dict[str, Any] | None = None
        self._action_counter: int = 0
        self._pending_action: dict[str, Any] | None = None
        self._step_history = deque(maxlen=self.context_window_size)
        self._last_observation_prompt: str = ""
        self._last_observation_response: str = ""
        self._last_action_prompt: str = ""
        self._last_action_response: str = ""
        self._state_action_memory = {}
        self._last_executed_action: str | None = None
        self._pending_state_action: dict | None = None
        self._external_hint: str | None = None
        self._persistent_hint: str | None = None

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    # -- hints (called by harness / analyzer) ----------------------------------

    def set_external_hint(self, hint: str) -> None:
        """One-shot strategic hint for the next observation prompt."""
        self._external_hint = hint
        self._persistent_hint = None

    def set_persistent_hint(self, plan: str) -> None:
        """Short plan that persists on every prompt until the next analysis."""
        self._persistent_hint = plan

    # -- grid helpers ----------------------------------------------------------

    def _format_grid(self, grid: list[list[int]]) -> str:
        return format_grid_ascii(grid)

    def _process_frame(self, obs: dict) -> tuple[list[list[int]], str]:
        frame_3d = obs.get("frame", [])
        grid_raw = [list(row) for row in frame_3d[-1]] if frame_3d else []
        return grid_raw, self._format_grid(grid_raw) if grid_raw else ""

    # -- state-action memory ---------------------------------------------------

    def _record_state_action(self, state_hash: str, action_key: str, result: dict[str, Any]) -> None:
        self._state_action_memory.setdefault(state_hash, {})[action_key] = result

    def _get_tried_actions(self, state_hash: str) -> dict[str, dict[str, Any]]:
        return self._state_action_memory.get(state_hash, {})

    def _format_state_action_context(self, grid: list[list[int]]) -> str:
        if not self.show_tried_actions:
            return ""
        tried = self._get_tried_actions(hash_grid_state(grid))
        if not tried:
            return ""
        lines = ["**From Current State, Already Tried:**\n"]
        for action_key, result in tried.items():
            changed = result.get("changed", False)
            diff = result.get("diff", "")
            marker = "changed" if changed else "no change"
            lines.append(f"- {action_key}: {marker}")
            if diff and changed:
                lines.append(f"    Diff: {diff}")
        lines.append("")
        return "\n".join(lines)

    # -- step history ----------------------------------------------------------

    def _format_step_history(self, include_strategy: bool = True) -> str:
        if not self._step_history:
            return ""
        lines = ["**Recent History:**\n"]
        for entry in self._step_history:
            dup = " NO STATE CHANGE" if entry.get("no_state_change") else ""
            pre = entry.get("grid_raw", [])
            post = entry.get("post_grid_raw")
            diff = compute_grid_diff(pre, post) if post is not None else "(pending)"
            text = (
                f"Step {entry['step']}: {entry['action']}, Score={entry['score']}{dup}\n"
                f"  Changes: {diff}\n"
            )
            if include_strategy:
                obs_resp = entry.get("obs_response", "")
                if obs_resp:
                    text += f"  [Strategy]: {obs_resp}\n"
            lines.append(text)
        return "\n".join(lines) + "\n"

    # -- core loop -------------------------------------------------------------

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict = None, **_: Any) -> None:
        self._last_observation = observation

        prompts = []
        if self._last_observation_prompt:
            prompts.append({"role": "observation_phase", "content": self._last_observation_prompt})
        if self._last_observation_response:
            prompts.append({"role": "observation_response", "content": self._last_observation_response})
        if self._last_action_prompt:
            prompts.append({"role": "action_phase", "content": self._last_action_prompt})
        if self._last_action_response:
            prompts.append({"role": "action_response", "content": self._last_action_response})

        step = Step(observation=observation, reward=reward, done=done, info=info, chat_completions=prompts)
        self._trajectory.steps.append(step)

        if self._step_history:
            grid_raw, _ = self._process_frame(observation)
            pre_grid = self._step_history[-1].get("grid_raw", [])
            no_change = (pre_grid == grid_raw)
            self._step_history[-1]["no_state_change"] = no_change
            self._step_history[-1]["post_grid_raw"] = grid_raw

            if self._pending_state_action:
                diff = compute_grid_diff(pre_grid, grid_raw)
                record = {"changed": not no_change, "diff": diff if not no_change else ""}
                record.update(self._pending_state_action.get("extra", {}))
                self._record_state_action(
                    self._pending_state_action["state_hash"],
                    self._pending_state_action["action_key"],
                    record,
                )
                self._pending_state_action = None

    def update_from_model(self, action_payload: dict | None = None, **_: Any) -> dict:
        action_dict = action_payload or self._pending_action
        obs_text = action_dict.get("obs_text", "")
        response_text = f"Observation: {obs_text}\nAction: {action_dict['name']}"

        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = response_text
            self._trajectory.steps[-1].action = action_dict

        obs = self._last_observation or {}
        grid_raw, _ = self._process_frame(obs)
        action_name = action_dict["name"]

        if action_name == "ACTION6":
            data = action_dict.get("data", {})
            x, y = data.get("x", 0), data.get("y", 0)
            label, comp_id = get_click_info(grid_raw, x, y)
            action_display = f"ACTION6(x={x}, y={y}, {label})"
            self._pending_state_action = {
                "state_hash": hash_grid_state(grid_raw),
                "action_key": f"click_{comp_id}",
                "extra": {"x": x, "y": y},
            }
        else:
            action_display = action_name
            self._pending_state_action = {
                "state_hash": hash_grid_state(grid_raw),
                "action_key": action_name,
                "extra": {},
            }

        self._step_history.append({
            "step": self._action_counter,
            "action": action_display,
            "score": obs.get("score", 0),
            "state": obs.get("state", "UNKNOWN"),
            "grid_raw": grid_raw,
            "no_state_change": False,
            "obs_response": self._last_observation_response if self.include_strategy_in_context else "",
        })

        self._action_counter += 1
        self._pending_action = None
        self._last_executed_action = action_name

        action = GameAction.from_name(action_name)
        result = {"action": action, "reasoning": response_text}
        if action == GameAction.ACTION6:
            x_pos = max(0, min(63, int(action_dict["data"].get("x", 0))))
            y_pos = max(0, min(63, int(action_dict["data"].get("y", 0))))
            result["x"] = y_pos
            result["y"] = x_pos
        return result

    async def call_llm(self) -> dict:
        obs = self._last_observation or {}
        state = obs.get("state", "NOT_PLAYED")

        # Auto-reset on game over
        if state in ("NOT_PLAYED", "GAME_OVER") and self._last_executed_action != "RESET":
            action_dict = {"name": "RESET", "data": {}, "obs_text": "Game Over, starting new game.", "action_text": ""}
            self._pending_action = action_dict
            return action_dict

        grid_raw, grid_text = self._process_frame(obs)
        score = obs.get("score", 0)

        obs_text = await self._call_observation_model(grid_text, score, grid_raw)
        action_dict = await self._call_action_model(grid_text, obs_text)
        action_dict["obs_text"] = obs_text
        self._pending_action = action_dict
        return action_dict

    async def _call_observation_model(self, grid: str, score: int, grid_raw: list) -> str:
        raise NotImplementedError

    async def _call_action_model(self, grid: str, last_obs: str) -> dict:
        raise NotImplementedError
