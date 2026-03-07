"""
Claude Code Action Agent — analyzer-driven batch planning.

The external analyzer (analyzer.py) reads
the prompt log every N actions and outputs a batch action plan in an
[ACTIONS] section.  The harness parses this and calls set_action_plan()
to load the queue.

Architecture:
  set_action_plan()        -> Receives JSON plan from analyzer, loads queue.
  call_llm()               -> If queue non-empty and no score change, pops
                              from queue (no LLM call).  Otherwise raises
                              QueueExhausted to end the episode.
  update_from_env()        -> Flushes queue on score change (level transition).
  _call_observation_model()-> Builds full context string (no LLM call).
  _call_action_model()     -> Returns queued action or raises QueueExhausted.
"""
from __future__ import annotations

import json
import logging
import re
from collections import deque
from typing import Any, List

from arcgym.agents.base_agent import BaseArcAgent

log = logging.getLogger(__name__)


class QueueExhausted(RuntimeError):
    """Raised when the action queue is empty and no analyzer plan is available."""
    pass


class ClaudeCodeActionAgent(BaseArcAgent):
    """
    Agent driven by an external analyzer's batch action plans.

    Actions are queued via set_action_plan() and executed one per step
    without any LLM calls.  When the queue is empty, QueueExhausted is
    raised so the harness can invoke the analyzer for a new plan.
    """

    def __init__(self, *, plan_size: int = 5, **kwargs: Any) -> None:
        self._action_queue: deque[dict] = deque()
        self._last_score: int = 0
        self._score_changed: bool = False
        self._use_queued_action: bool = False
        self._plan_total: int = 0
        self._plan_index: int = 0
        self._plan_size = plan_size
        super().__init__(**kwargs)

    def reset(self) -> None:
        super().reset()
        self._action_queue.clear()
        self._last_score = 0
        self._score_changed = False
        self._use_queued_action = False
        self._plan_total = 0
        self._plan_index = 0

    @property
    def is_overhead_action(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Flush queue on score change (level transition)
    # ------------------------------------------------------------------

    def update_from_env(self, observation, reward, done, info=None):
        super().update_from_env(observation, reward, done, info)
        obs = observation if isinstance(observation, dict) else {}
        score = obs.get("score", 0)
        if score != self._last_score:
            if self._action_queue:
                log.info(
                    "[ClaudeCodeActionAgent] score %d->%d: flushing %d queued actions",
                    self._last_score, score, len(self._action_queue),
                )
                self._action_queue.clear()
            self._score_changed = True
            self._last_score = score

    # ------------------------------------------------------------------
    # Receive batch action plan from analyzer
    # ------------------------------------------------------------------

    _VALID_ACTIONS = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "RESET"}

    def set_action_plan(self, actions_text: str) -> bool:
        """Parse an [ACTIONS] JSON blob from the analyzer and load the queue.

        Expected format:
          {"plan": [{"action": "ACTION6", "x": 3, "y": 7}, ...], "reasoning": "..."}
        Also accepts bare arrays, markdown-fenced JSON, and trailing text.
        Returns True if plan was loaded successfully, False otherwise.
        """
        clean = re.sub(r"```(?:json)?\s*", "", actions_text).strip()

        parsed = None
        decoder = json.JSONDecoder()
        for start_char in ("{", "["):
            idx = clean.find(start_char)
            if idx >= 0:
                try:
                    parsed, _ = decoder.raw_decode(clean, idx)
                    break
                except json.JSONDecodeError:
                    continue

        if parsed is None:
            log.warning("[ClaudeCodeActionAgent] set_action_plan: could not parse: %s", actions_text[:200])
            return False

        if isinstance(parsed, list):
            parsed = {"plan": parsed, "reasoning": ""}

        plan = parsed.get("plan", parsed.get("actions", []))
        if not isinstance(plan, list) or not plan:
            log.warning("[ClaudeCodeActionAgent] set_action_plan: empty or invalid plan")
            return False

        self._action_queue.clear()
        for step in plan:
            if isinstance(step, str):
                m = re.match(r"ACTION6\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", step)
                if m:
                    name = "ACTION6"
                    data = {"x": int(m.group(1)), "y": int(m.group(2))}
                else:
                    name = step
                    data = {}
            else:
                name = step.get("action")
                if not name:
                    log.warning("[ClaudeCodeActionAgent] skipping step with no action key: %s", step)
                    continue
                data = (
                    {"x": int(step.get("x", 0)), "y": int(step.get("y", 0))}
                    if name == "ACTION6" else {}
                )
            if name not in self._VALID_ACTIONS:
                log.warning("[ClaudeCodeActionAgent] skipping unrecognized action: %s", name)
                continue
            self._action_queue.append({"name": name, "data": data, "obs_text": "", "action_text": ""})

        self._plan_total = len(self._action_queue)
        self._plan_index = 0
        reasoning = parsed.get("reasoning", "")
        log.info(
            "[ClaudeCodeActionAgent] loaded %d-step plan: %s — %s",
            self._plan_total,
            [s if isinstance(s, str) else s.get("action") for s in plan],
            reasoning[:100],
        )
        return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def call_llm(self):
        self._use_queued_action = bool(self._action_queue and not self._score_changed)
        if not self._use_queued_action:
            self._score_changed = False
        return await super().call_llm()

    # ------------------------------------------------------------------
    # Observation: build context string, no LLM call
    # ------------------------------------------------------------------

    async def _call_observation_model(self, grid: str, score: int, grid_raw: List) -> str:
        history_context = self._format_step_history()
        state_action_context = self._format_state_action_context(grid_raw)

        hint_block = ""
        if self._external_hint:
            hint_block = f"\n[STRATEGIC ANALYSIS FROM LOG REVIEW]\n{self._external_hint}\n"
            self._external_hint = None
        elif self._persistent_hint:
            hint_block = f"\n[CURRENT PLAN]\n{self._persistent_hint}\n"

        context = (
            f"{hint_block}"
            f"{history_context}"
            f"{state_action_context}"
            f"**Current State:**\n"
            f"Score: {score}\n"
            f"Step: {self._action_counter}\n\n"
            f"**Current Matrix** 64x64 (ASCII characters):\n{grid}\n"
        )

        if self._use_queued_action:
            step_label = f"step {self._plan_index + 1}/{self._plan_total}"
            context += f"\n[Executing pre-planned action ({step_label}) — no model call]\n"
            self._last_observation_prompt = f"[Queued plan {step_label}]\n\n{context}"
            self._last_observation_response = f"[Pre-planned action {step_label}]"
        else:
            self._last_observation_prompt = f"[Observation context]\n\n{context}"
            self._last_observation_response = "[Observation model — context assembled]"

        return context

    # ------------------------------------------------------------------
    # Action: pop from queue or raise QueueExhausted
    # ------------------------------------------------------------------

    async def _call_action_model(self, grid: str, last_obs: str) -> dict:
        if self._use_queued_action and self._action_queue:
            action = self._action_queue.popleft()
            self._plan_index += 1
            step_label = f"plan step {self._plan_index}/{self._plan_total}"

            action["obs_text"] = last_obs
            action["action_text"] = f"[queued {step_label}]"
            self._pending_action = action

            self._last_action_prompt = f"[Queued {step_label} — no model call]"
            self._last_action_response = (
                f"Tool Call: {action['name']}({json.dumps(action['data'])})\n"
                f"Content: Executing pre-planned action ({step_label})"
            )
            log.info(
                "[ClaudeCodeActionAgent] queue drain -> %s (%s, %d remaining)",
                action.get("name"), step_label, len(self._action_queue),
            )
            return action

        log.info("[ClaudeCodeActionAgent] queue empty — ending episode")
        raise QueueExhausted("Queue empty, no actions from analyzer")
