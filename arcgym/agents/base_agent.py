"""
Base ARC agent with rolling context.

Provides the core agent loop: grid formatting, rolling step history,
state-action memory, and external hint injection.  Subclasses override
_call_observation_model / _call_action_model to implement their own
observation and action logic.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List

from arcgym.core import BaseAgent, Step, Trajectory

from arcengine import GameAction
from arcgym.utils.grid_utils import (
    compute_grid_diff,
    format_grid_ascii,
    get_click_info,
    hash_grid_state,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseArcAgent(BaseAgent):
    """
    ARC agent with rolling context, state-action memory, and
    two-phase calls (observation then action).  Subclasses must
    override _call_observation_model and _call_action_model.
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

        # Initialised properly in reset(), but must exist before super().__init__
        # which may call reset() during construction.
        self._step_history: deque = deque(maxlen=self.context_window_size)
        self._state_action_memory: Dict[str, Dict[str, Dict[str, Any]]] = {}

        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Hints (called by harness / analyzer)
    # ------------------------------------------------------------------

    def set_external_hint(self, hint: str) -> None:
        """Inject a one-shot strategic hint for the next observation prompt."""
        self._external_hint = hint
        self._persistent_hint = None

    def set_persistent_hint(self, plan: str) -> None:
        """Set a short plan that persists on every prompt until the next analysis."""
        self._persistent_hint = plan

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _format_grid(self, grid: List[List[int]]) -> str:
        return format_grid_ascii(grid)

    def _process_frame(self, obs: dict) -> tuple[List[List[int]], str]:
        """Extract grid_raw and formatted grid string from an observation."""
        frame_3d = obs.get("frame", [])
        grid_raw = [list(row) for row in frame_3d[-1]] if frame_3d else []
        return grid_raw, self._format_grid(grid_raw) if grid_raw else ""

    # ------------------------------------------------------------------
    # State-action memory
    # ------------------------------------------------------------------

    def _record_state_action(self, state_hash: str, action_key: str, result: Dict[str, Any]) -> None:
        if state_hash not in self._state_action_memory:
            self._state_action_memory[state_hash] = {}
        self._state_action_memory[state_hash][action_key] = result

    def _get_tried_actions_from_state(self, state_hash: str) -> Dict[str, Dict[str, Any]]:
        return self._state_action_memory.get(state_hash, {})

    def _format_state_action_context(self, grid: List[List[int]]) -> str:
        if not self.show_tried_actions:
            return ""
        sh = hash_grid_state(grid)
        tried = self._get_tried_actions_from_state(sh)
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

    # ------------------------------------------------------------------
    # Step history
    # ------------------------------------------------------------------

    def _format_step_history(self, include_strategy: bool = True) -> str:
        if not self._step_history:
            return ""
        history_lines = ["**Recent History:**\n"]
        for step_info in self._step_history:
            dup = " NO STATE CHANGE" if step_info.get("no_state_change") else ""
            pre_grid = step_info.get("grid_raw", [])
            post_grid = step_info.get("post_grid_raw")
            diff = compute_grid_diff(pre_grid, post_grid) if post_grid is not None else "(pending)"
            entry = (
                f"Step {step_info['step']}: {step_info['action']}, Score={step_info['score']}{dup}\n"
                f"  Changes: {diff}\n"
            )
            if include_strategy:
                obs_response = step_info.get("obs_response", "")
                if obs_response:
                    entry += f"  [Strategy]: {obs_response}\n"
            history_lines.append(entry)
        return "\n".join(history_lines) + "\n"

    # ------------------------------------------------------------------
    # Core loop: update_from_env → call_llm → update_from_model
    # ------------------------------------------------------------------

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict = None, **_: Any) -> None:
        self._last_observation = observation

        # Build prompt/response log
        full_prompts = []
        if self._last_observation_prompt:
            full_prompts.append({"role": "observation_phase", "content": self._last_observation_prompt})
        if self._last_observation_response:
            full_prompts.append({"role": "observation_response", "content": self._last_observation_response})
        if self._last_action_prompt:
            full_prompts.append({"role": "action_phase", "content": self._last_action_prompt})
        if self._last_action_response:
            full_prompts.append({"role": "action_response", "content": self._last_action_response})

        step = Step(observation=observation, reward=reward, done=done, info=info, chat_completions=full_prompts)
        self._trajectory.steps.append(step)

        # Detect state change by comparing post-action grid with pre-action grid
        if self._step_history:
            grid_raw, _ = self._process_frame(observation)
            pre_grid = self._step_history[-1].get("grid_raw", [])
            no_state_change = (pre_grid == grid_raw)
            self._step_history[-1]["no_state_change"] = no_state_change
            self._step_history[-1]["post_grid_raw"] = grid_raw

            # Finalize state-action memory entry
            if self._pending_state_action:
                diff = compute_grid_diff(pre_grid, grid_raw)
                record = {"changed": not no_state_change, "diff": diff if not no_state_change else ""}
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

        # Record step in history
        obs = self._last_observation or {}
        grid_raw, _ = self._process_frame(obs)

        action_name = action_dict["name"]
        if action_name == "ACTION6":
            data = action_dict.get("data", {})
            x_grid = data.get("x", 0)
            y_grid = data.get("y", 0)
            label, comp_id = get_click_info(grid_raw, x_grid, y_grid)
            action_display = f"ACTION6(x={x_grid}, y={y_grid}, {label})"
            self._pending_state_action = {
                "state_hash": hash_grid_state(grid_raw),
                "action_key": f"click_{comp_id}",
                "extra": {"x": x_grid, "y": y_grid},
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
        self._last_executed_action = action_dict["name"]

        # Build action payload for environment
        action = GameAction.from_name(action_dict["name"])
        action_dict2 = {"action": action, "reasoning": response_text}
        if action == GameAction.ACTION6:
            x_pos = max(0, min(63, int(action_dict["data"].get("x", 0))))
            y_pos = max(0, min(63, int(action_dict["data"].get("y", 0))))
            action_dict2["x"] = y_pos
            action_dict2["y"] = x_pos
        return action_dict2

    async def call_llm(self) -> dict:
        """Two-phase call: observation then action."""
        obs = self._last_observation or {}
        state = obs.get("state", "NOT_PLAYED")

        # Auto-RESET on game over (but not if we just reset)
        if state in ("NOT_PLAYED", "GAME_OVER") and self._last_executed_action != "RESET":
            action_dict = {"name": "RESET", "data": {}, "obs_text": "Game Over, starting new game.", "action_text": ""}
            self._pending_action = action_dict
            return action_dict

        grid_raw, grid_text = self._process_frame(obs)
        score = obs.get("score", 0)

        # Phase 1: Observation
        obs_text = await self._call_observation_model(grid_text, score, grid_raw)

        # Phase 2: Action
        action_dict = await self._call_action_model(grid_text, obs_text)
        action_dict["obs_text"] = obs_text

        self._pending_action = action_dict
        return action_dict

    async def _call_observation_model(self, grid: str, score: int, grid_raw: List) -> str:
        """Subclasses must override to build observation context."""
        raise NotImplementedError

    async def _call_action_model(self, grid: str, last_obs: str) -> dict:
        """Subclasses must override to produce an action dict."""
        raise NotImplementedError
