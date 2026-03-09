"""Queue-driven agent that executes batch action plans from an external analyzer.

No LLM calls are made by this agent. Actions are queued via set_action_plan()
and popped one per step. QueueExhausted is raised when the queue empties,
signaling the harness to invoke the analyzer for a new plan.
"""
from __future__ import annotations

import json
import logging
import re
from collections import deque
from typing import Any

from arcgym.agents.base_agent import BaseArcAgent

log = logging.getLogger(__name__)


class QueueExhausted(RuntimeError):
    pass


class ClaudeCodeActionAgent(BaseArcAgent):

    _VALID_ACTIONS = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "RESET"}

    def __init__(self, *, plan_size: int = 5, **kwargs: Any) -> None:
        self._action_queue: deque[dict] = deque()
        self._last_score: int = 0
        self._score_changed: bool = False
        self._use_queued: bool = False
        self._plan_total: int = 0
        self._plan_index: int = 0
        self._plan_size = plan_size
        super().__init__(**kwargs)

    def reset(self) -> None:
        super().reset()
        self._action_queue.clear()
        self._last_score = 0
        self._score_changed = False
        self._use_queued = False
        self._plan_total = 0
        self._plan_index = 0

    @property
    def is_overhead_action(self) -> bool:
        return False

    # -- flush queue on score change (level transition) ------------------------

    def update_from_env(self, observation, reward, done, info=None):
        super().update_from_env(observation, reward, done, info)
        obs = observation if isinstance(observation, dict) else {}
        score = obs.get("score", 0)
        if score != self._last_score:
            if self._action_queue:
                log.info("score %d->%d: flushing %d queued actions",
                         self._last_score, score, len(self._action_queue))
                self._action_queue.clear()
            self._score_changed = True
            self._last_score = score

    # -- receive batch action plan from analyzer -------------------------------

    def set_action_plan(self, actions_text: str) -> bool:
        """Parse [ACTIONS] JSON from the analyzer and load the queue.

        Accepts {"plan": [...]} objects, bare arrays, and markdown-fenced JSON.
        Returns True if the plan was loaded.
        """
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
            log.warning("set_action_plan: could not parse: %s", actions_text[:200])
            return False

        if isinstance(parsed, list):
            parsed = {"plan": parsed, "reasoning": ""}

        plan = parsed.get("plan", parsed.get("actions", []))
        if not isinstance(plan, list) or not plan:
            log.warning("set_action_plan: empty or invalid plan")
            return False

        self._action_queue.clear()
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
            if name not in self._VALID_ACTIONS:
                log.warning("skipping unrecognized action: %s", name)
                continue
            self._action_queue.append({"name": name, "data": data, "obs_text": "", "action_text": ""})

        self._plan_total = len(self._action_queue)
        self._plan_index = 0
        reasoning = parsed.get("reasoning", "")
        log.info("loaded %d-step plan: %s — %s",
                 self._plan_total,
                 [s if isinstance(s, str) else s.get("action") for s in plan],
                 reasoning[:100])
        return True

    # -- main loop -------------------------------------------------------------

    async def call_llm(self):
        self._use_queued = bool(self._action_queue and not self._score_changed)
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
            label = f"step {self._plan_index + 1}/{self._plan_total}"
            context += f"\n[Executing pre-planned action ({label}) — no model call]\n"
            self._last_observation_prompt = f"[Queued plan {label}]\n\n{context}"
            self._last_observation_response = f"[Pre-planned action {label}]"
        else:
            self._last_observation_prompt = f"[Observation context]\n\n{context}"
            self._last_observation_response = "[Observation model — context assembled]"

        return context

    async def _call_action_model(self, grid: str, last_obs: str) -> dict:
        if self._use_queued and self._action_queue:
            action = self._action_queue.popleft()
            self._plan_index += 1
            label = f"plan step {self._plan_index}/{self._plan_total}"

            action["obs_text"] = last_obs
            action["action_text"] = f"[queued {label}]"
            self._pending_action = action

            self._last_action_prompt = f"[Queued {label} — no model call]"
            self._last_action_response = (
                f"Tool Call: {action['name']}({json.dumps(action['data'])})\n"
                f"Content: Executing pre-planned action ({label})"
            )
            log.info("queue drain -> %s (%s, %d remaining)",
                     action.get("name"), label, len(self._action_queue))
            return action

        log.info("queue empty — ending episode")
        raise QueueExhausted("Queue empty, no actions from analyzer")
