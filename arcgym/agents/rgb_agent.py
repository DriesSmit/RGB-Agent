"""RGBAgent: queue-based agent for ARC-AGI-3 puzzles."""
from __future__ import annotations

import json
import logging
import os
import re
from collections import deque
from typing import Any

from arcgym.agents.base_agent import BaseArcAgent

log = logging.getLogger(__name__)

_LOG_REASONING_MAX_CHARS = int(os.environ.get("ARCGYM_REASONING_LOG_CHARS", "0"))


class QueueExhausted(RuntimeError):
    pass


_VALID_ACTIONS = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "RESET"}


def _truncate_log_text(text: Any, limit: int) -> str:
    value = " ".join(str(text).split())
    if limit <= 0 or len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


class ActionQueue:
    """Holds and serves a batch of parsed actions."""

    def __init__(self) -> None:
        self._queue: deque[dict] = deque()
        self.plan_total: int = 0
        self.plan_index: int = 0

    def clear(self) -> None:
        self._queue.clear()
        self.plan_total = 0
        self.plan_index = 0

    def __len__(self) -> int:
        return len(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)

    def pop(self) -> dict:
        action = self._queue.popleft()
        self.plan_index += 1
        return action

    def load(self, actions_text: str) -> bool:
        """Parse [ACTIONS] JSON and load the queue. Returns True on success."""
        clean = re.sub(r"```(?:json)?\s*", "", actions_text).strip()

        parsed = None
        decoder = json.JSONDecoder()
        for char in ("{", "["):
            idx = clean.find(char)
            if idx >= 0:
                try:
                    parsed, _ = decoder.raw_decode(clean, idx)
                    break
                except json.JSONDecodeError:
                    continue

        if parsed is None:
            log.warning("ActionQueue.load: could not parse: %s", actions_text[:200])
            return False

        if isinstance(parsed, list):
            parsed = {"plan": parsed, "reasoning": ""}

        plan = parsed.get("plan", parsed.get("actions", []))
        if not isinstance(plan, list) or not plan:
            log.warning("ActionQueue.load: empty or invalid plan")
            return False

        self._queue.clear()
        for step in plan:
            if isinstance(step, str):
                m = re.match(r"ACTION6\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", step)
                if m:
                    name, data = "ACTION6", {"x": int(m.group(1)), "y": int(m.group(2))}
                else:
                    name, data = step, {}
            else:
                name = step.get("action")
                if not name:
                    log.warning("skipping step with no action key: %s", step)
                    continue
                data = (
                    {"x": int(step.get("x", 0)), "y": int(step.get("y", 0))}
                    if name == "ACTION6" else {}
                )
            if name not in _VALID_ACTIONS:
                log.warning("skipping unrecognized action: %s", name)
                continue
            self._queue.append({"name": name, "data": data, "obs_text": "", "action_text": ""})

        self.plan_total = len(self._queue)
        self.plan_index = 0
        reasoning = parsed.get("reasoning", "")
        log.info("loaded %d-step plan: %s — %s",
                 self.plan_total,
                 [s if isinstance(s, str) else s.get("action") for s in plan],
                 _truncate_log_text(reasoning, _LOG_REASONING_MAX_CHARS))
        return True


class RGBAgent(BaseArcAgent):
    """Queue-based agent for ARC-AGI-3 puzzles."""

    def __init__(self, *, plan_size: int = 5, **kwargs: Any) -> None:
        self._queue = ActionQueue()
        self._last_score: int = 0
        self._score_changed: bool = False
        self._use_queued: bool = False
        self._plan_size = plan_size
        super().__init__(**kwargs)

    def reset(self) -> None:
        super().reset()
        self._queue.clear()
        self._last_score = 0
        self._score_changed = False
        self._use_queued = False

    @property
    def is_overhead_action(self) -> bool:
        return False

    @property
    def plan_index(self) -> int:
        return self._queue.plan_index

    @property
    def plan_total(self) -> int:
        return self._queue.plan_total

    def render_board(self) -> str | None:
        _, grid_text = self._process_frame(self._last_observation or {})
        return grid_text or None

    def set_action_plan(self, actions_text: str) -> bool:
        return self._queue.load(actions_text)

    def update_from_env(self, observation, reward, done, info=None):
        super().update_from_env(observation, reward, done, info)
        obs = observation if isinstance(observation, dict) else {}
        score = obs.get("score", 0)
        if score != self._last_score:
            if self._queue:
                log.info("score %d->%d: flushing %d queued actions",
                         self._last_score, score, len(self._queue))
                self._queue.clear()
            self._score_changed = True
            self._last_score = score

    async def call_llm(self):
        self._use_queued = bool(self._queue and not self._score_changed)
        if not self._use_queued:
            self._score_changed = False
        return await super().call_llm()

    async def _call_observation_model(self, grid: str, score: int, grid_raw: list) -> str:
        history = self._format_step_history()
        tried = self._format_state_action_context(grid_raw)

        hint_block = ""
        if self._external_hint:
            hint_block = f"\n[STRATEGIC ANALYSIS FROM LOG REVIEW]\n{self._external_hint}\n"
            self._external_hint = None
        elif self._persistent_hint:
            hint_block = f"\n[CURRENT PLAN]\n{self._persistent_hint}\n"

        context = (
            f"{hint_block}"
            f"{history}"
            f"{tried}"
            f"**Current State:**\n"
            f"Score: {score}\n"
            f"Step: {self._action_counter}\n\n"
            f"**Current Matrix** 64x64 (ASCII characters):\n{grid}\n"
        )

        if self._use_queued:
            label = f"step {self._queue.plan_index + 1}/{self._queue.plan_total}"
            context += f"\n[Executing pre-planned action ({label}) — no model call]\n"
            self._last_observation_prompt = f"[Queued plan {label}]\n\n{context}"
            self._last_observation_response = f"[Pre-planned action {label}]"
        else:
            self._last_observation_prompt = f"[Observation context]\n\n{context}"
            self._last_observation_response = "[Observation model — context assembled]"

        return context

    async def _call_action_model(self, grid: str, last_obs: str) -> dict:
        if self._use_queued and self._queue:
            action = self._queue.pop()
            label = f"plan step {self._queue.plan_index}/{self._queue.plan_total}"

            action["obs_text"] = last_obs
            action["action_text"] = f"[queued {label}]"
            self._pending_action = action

            self._last_action_prompt = f"[Queued {label} — no model call]"
            self._last_action_response = (
                f"Tool Call: {action['name']}({json.dumps(action['data'])})\n"
                f"Content: Executing pre-planned action ({label})"
            )
            log.info("queue drain -> %s (%s, %d remaining)",
                     action.get("name"), label, len(self._queue))
            return action

        log.info("queue empty — ending episode")
        raise QueueExhausted("Queue empty, no actions from analyzer")
