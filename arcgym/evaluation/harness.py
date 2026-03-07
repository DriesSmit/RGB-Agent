"""
Core game evaluation loop.

Provides ``evaluate_single_game`` which is called by the Swarm runner
(one invocation per game thread).
"""
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

import requests

from arcgym.agents.claude_code_action_agent import QueueExhausted
from arcgym.environments import ArcAgi3Env
from arcengine import GameState
from arcgym.metrics.structures import GameMetrics, LevelMetrics, AttemptMetrics

log = logging.getLogger(__name__)

ROOT_URL = os.environ.get("ROOT_URL", "https://three.arcprize.org")

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
MAX_RETRIES = 5
INITIAL_BACKOFF = 1  # seconds


def _run_with_retries(func_to_run: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run *func_to_run* with exponential backoff on network errors."""
    retries = 0
    backoff = INITIAL_BACKOFF
    while True:
        try:
            return func_to_run(*args, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if retries >= MAX_RETRIES:
                log.error(f"Final attempt failed for {func_to_run.__name__} after {retries} retries.")
                raise
            log.warning(
                f"API error for {func_to_run.__name__}: {type(e).__name__}. "
                f"Retrying in {backoff}s (attempt {retries + 1}/{MAX_RETRIES})"
            )
            time.sleep(backoff)
            retries += 1
            backoff *= 2


def _render_post_action_grid(agent) -> str | None:
    """Render the current board from the agent's last observation."""
    _, grid_text = agent._process_frame(agent._last_observation or {})
    return grid_text or None


# ---------------------------------------------------------------------------
# Single-game evaluation loop
# ---------------------------------------------------------------------------

def evaluate_single_game(
    agent,
    env: ArcAgi3Env,
    game_id: str,
    agent_name: str,
    max_actions_per_game: int,
    run_index: int,
    tags: Optional[List[str]] = None,
    prompts_log_path: Optional[Path] = None,
    analyzer=None,
    log_post_board: bool = False,
    analyzer_retries: int = 5,
) -> GameMetrics:
    """Run a single game loop, returning collected metrics."""

    run_metrics = GameMetrics(
        game_id=game_id,
        agent_name=agent_name,
        run_index=run_index,
        start_time=time.time(),
    )
    run_metrics.status = "IN_PROGRESS"

    current_level_number = 1
    current_level_metrics = LevelMetrics(level_number=current_level_number)
    current_attempt_number = 1
    current_attempt_metrics = AttemptMetrics(attempt_number=current_attempt_number)
    attempt_start_time = run_metrics.start_time

    max_score = 0
    total_actions_this_run = 0
    arc_state: GameState | None = None
    arc_score = 0

    _loop = asyncio.new_event_loop()

    try:
        agent.reset()

        # --- Reset game state ---
        def _reset_game_state():
            observation = _run_with_retries(
                env.reset,
                task={"game_id": game_id, "max_actions": max_actions_per_game, "tags": tags},
            )
            _arc_state = GameState[observation.get("state") or "NOT_PLAYED"]
            _arc_score = observation.get("score", 0) or 0

            obs_guid = observation.get("guid")
            if obs_guid and not run_metrics.guid:
                run_metrics.guid = obs_guid
                run_metrics.replay_url = f"{ROOT_URL}/replay/{game_id}/{obs_guid}"
                log.info(f"[{game_id} Run {run_index}] Replay URL: {run_metrics.replay_url}")
                if prompts_log_path:
                    guid_path = prompts_log_path.parent / "run_info.txt"
                    guid_path.write_text(
                        f"game_id: {game_id}\n"
                        f"guid: {obs_guid}\n"
                        f"replay_url: {run_metrics.replay_url}\n"
                        f"scorecard_id: {getattr(env, '_scorecard_id', 'unknown')}\n"
                        f"command: {Path(sys.argv[0]).name} {' '.join(sys.argv[1:])}\n"
                    )

            agent.update_from_env(observation=observation, reward=0.0, done=False)
            return _arc_state, _arc_score

        arc_state, arc_score = _reset_game_state()

        def _record_attempt(status: str, game_over: bool = False) -> float:
            attempt_end_time = time.time()
            current_attempt_metrics.duration_seconds = attempt_end_time - attempt_start_time
            current_attempt_metrics.status = status
            if game_over:
                current_attempt_metrics.game_overs += 1
            current_level_metrics.attempts.append(current_attempt_metrics)
            return attempt_end_time

        # Log the initial board state
        if prompts_log_path:
            _init_grid = _render_post_action_grid(agent)
            if _init_grid:
                with open(prompts_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Action 0 | Level {current_level_number} | Attempt {current_attempt_number} | INITIAL STATE\n")
                    f.write(f"Score: {arc_score} | State: {arc_state.name}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(f"[INITIAL BOARD STATE]\n{_init_grid}\n\n")

        def _fire_analyzer_and_load_plan(action_num: int, retry_nudge: str = "") -> bool:
            """Fire the analyzer, parse hint, load action plan. Returns True if plan loaded."""
            if not analyzer:
                return False
            # Append current board state so the analyzer sees the latest
            if prompts_log_path and not log_post_board:
                _post_grid = _render_post_action_grid(agent)
                if _post_grid:
                    with open(prompts_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"[POST-ACTION BOARD STATE]\nScore: {arc_score}\n{_post_grid}\n\n")
            hint = analyzer(prompts_log_path, action_num, retry_nudge=retry_nudge)
            if not hint:
                log.warning(f"[harness] analyzer returned None at action {action_num}")
                return False
            # Normalize trailing whitespace
            hint = "\n".join(line.rstrip() for line in hint.split("\n"))
            _actions_text = None
            _actions_sep = "\n[ACTIONS]\n"
            if _actions_sep in hint:
                hint, _actions_text = hint.split(_actions_sep, 1)
                _actions_text = _actions_text.strip()
            _plan_sep = "\n[PLAN]\n"
            if _plan_sep in hint:
                full_hint, plan = hint.split(_plan_sep, 1)
                full_hint = full_hint.strip()
                plan = plan.strip()
            else:
                full_hint = hint
                plan = hint
            agent.set_external_hint(full_hint)
            agent.set_persistent_hint(plan)
            if _actions_text:
                loaded = agent.set_action_plan(_actions_text)
                if loaded:
                    log.info(f"[harness] analyzer at action {action_num}: loaded action plan ({len(_actions_text)} chars)")
                    return True
                log.warning(f"[harness] analyzer at action {action_num}: set_action_plan rejected the plan")
                return False
            log.warning(f"[harness] analyzer at action {action_num}: hint received but NO [ACTIONS] section")
            return False

        # --- Main game loop ---
        while total_actions_this_run < max_actions_per_game:
            try:
                action_dict = _loop.run_until_complete(agent.call_llm())
            except QueueExhausted:
                log.info(f"[harness] queue exhausted at action {total_actions_this_run} — firing analyzer")
                _RETRY_NUDGE = (
                    "CRITICAL: Your previous response was missing the [ACTIONS] section. "
                    "You MUST end your response with an [ACTIONS] section containing a JSON action plan. "
                    "Do NOT write actions to a file — output them directly in your response text."
                )
                _analyzer_loaded = False
                for _attempt in range(analyzer_retries):
                    _nudge = _RETRY_NUDGE if _attempt > 0 else ""
                    # On resumed sessions, always nudge since model tends to skip [ACTIONS]
                    if not _nudge and total_actions_this_run > 0:
                        _nudge = _RETRY_NUDGE
                    log.info(f"[harness] analyzer attempt {_attempt + 1}/{analyzer_retries} "
                             f"action={total_actions_this_run} nudge={bool(_nudge)}")
                    if _fire_analyzer_and_load_plan(total_actions_this_run, retry_nudge=_nudge):
                        _analyzer_loaded = True
                        break
                    log.warning(f"[harness] analyzer attempt {_attempt + 1}/{analyzer_retries} failed — retrying")
                if not _analyzer_loaded:
                    raise
                action_dict = _loop.run_until_complete(agent.call_llm())

            action_obj = agent.update_from_model(response=action_dict)
            observation, reward, done = _run_with_retries(env.step, action_obj)

            total_actions_this_run += 1
            current_attempt_metrics.actions += 1

            previous_arc_score = arc_score
            arc_state = GameState[observation.get("state") or "NOT_PLAYED"]
            arc_score = observation.get("score", 0) or 0
            max_score = max(max_score, arc_score)
            run_metrics.highest_level_reached = max(run_metrics.highest_level_reached, current_level_number)

            agent.update_from_env(observation=observation, reward=reward, done=done)

            # Log prompts/responses
            if prompts_log_path and agent.trajectory.steps:
                last_step = agent.trajectory.steps[-1]
                with open(prompts_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    plan_step_info = ""
                    if agent._plan_total > 0:
                        plan_step_info = f" | Plan Step {agent._plan_index}/{agent._plan_total}"
                    f.write(f"Action {total_actions_this_run} | Level {current_level_number} | Attempt {current_attempt_number}{plan_step_info}\n")
                    f.write(f"Score: {arc_score} | State: {arc_state.name}\n")
                    f.write(f"{'='*80}\n\n")

                    if last_step.chat_completions:
                        for msg in last_step.chat_completions:
                            role = msg.get('role', 'unknown')
                            content = msg.get('content', '')
                            tool_calls = msg.get('tool_calls', [])
                            f.write(f"[{role.upper()}]\n")
                            if content:
                                f.write(f"{content}\n")
                            if tool_calls:
                                for tc in tool_calls:
                                    fn = tc.get('function', {}) if isinstance(tc, dict) else {}
                                    f.write(f"Tool: {fn.get('name', tc)}({fn.get('arguments', '')})\n")
                            f.write("\n")

            # Log post-action board state
            if log_post_board and prompts_log_path:
                _post_grid = _render_post_action_grid(agent)
                if _post_grid:
                    with open(prompts_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"[POST-ACTION BOARD STATE]\nScore: {arc_score}\n{_post_grid}\n\n")

            # --- Handle Level Completion ---
            level_completed = (arc_score > previous_arc_score and
                               arc_state not in (GameState.WIN, GameState.GAME_OVER))

            if level_completed:
                attempt_end_time = _record_attempt("COMPLETED")
                current_level_metrics.status = "COMPLETED"
                run_metrics.level_metrics[current_level_number] = current_level_metrics

                log.info(
                    f"[{game_id} Run {run_index}] Level {current_level_number} COMPLETED. "
                    f"Attempt {current_attempt_number} actions: {current_attempt_metrics.actions}. Score: {arc_score}."
                )

                current_level_number += 1
                run_metrics.highest_level_reached = max(run_metrics.highest_level_reached, current_level_number)
                current_level_metrics = LevelMetrics(level_number=current_level_number)
                current_attempt_number = 1
                current_attempt_metrics = AttemptMetrics(attempt_number=current_attempt_number)
                attempt_start_time = attempt_end_time
                continue

            if arc_state == GameState.GAME_OVER:
                _record_attempt("GAME_OVER", game_over=True)
                current_level_metrics.status = "GAME_OVER"
                run_metrics.level_metrics[current_level_number] = current_level_metrics
                run_metrics.status = "TIMEOUT"
                log.warning(
                    f"[{game_id} Run {run_index}] Game Over on Level {current_level_number}, "
                    f"Attempt {current_attempt_number}. Actions: {current_attempt_metrics.actions}."
                )
                current_attempt_number += 1
                current_attempt_metrics = AttemptMetrics(attempt_number=current_attempt_number)
                attempt_start_time = time.time()

            if arc_state == GameState.WIN:
                _record_attempt("COMPLETED")
                current_level_metrics.status = "COMPLETED"
                run_metrics.level_metrics[current_level_number] = current_level_metrics
                run_metrics.status = "COMPLETED_RUN"
                log.info(
                    f"[{game_id} Run {run_index}] Game COMPLETED! "
                    f"Level {current_level_number} actions: {current_attempt_metrics.actions}. Score: {arc_score}"
                )
                break

    except QueueExhausted as e:
        log.info(f"[{game_id} Run {run_index}] Episode ended (queue exhausted): {e}")
        run_metrics.status = "QUEUE_EXHAUSTED"

    except Exception as e:
        run_metrics.status = "ERROR"
        run_metrics.error_message = str(e)
        current_attempt_metrics.status = "ERROR"
        current_level_metrics.status = "ERROR"
        log.error(f"[{game_id} Run {run_index}] Exception: {e}", exc_info=True)

    finally:
        run_metrics.end_time = time.time()
        run_metrics.run_duration_seconds = run_metrics.end_time - run_metrics.start_time

        # Finalize attempt status if still in progress
        if current_attempt_metrics.status == "IN_PROGRESS":
            current_attempt_metrics.duration_seconds = run_metrics.end_time - attempt_start_time
            if run_metrics.status == "ERROR":
                current_attempt_metrics.status = "ERROR"
            elif arc_state == GameState.WIN:
                current_attempt_metrics.status = "COMPLETED"
                run_metrics.status = "COMPLETED_RUN"
            else:
                current_attempt_metrics.status = "TIMEOUT"
                if run_metrics.status == "IN_PROGRESS":
                    run_metrics.status = "TIMEOUT"

        if not current_level_metrics.attempts or current_level_metrics.attempts[-1].attempt_number != current_attempt_metrics.attempt_number:
            current_level_metrics.attempts.append(current_attempt_metrics)
        if current_level_metrics.status == "IN_PROGRESS":
            current_level_metrics.status = current_attempt_metrics.status

        run_metrics.level_metrics[current_level_number] = current_level_metrics
        run_metrics.run_total_actions = sum(lm.total_actions for lm in run_metrics.level_metrics.values())
        run_metrics.total_game_overs_across_run = sum(lm.total_game_overs for lm in run_metrics.level_metrics.values())
        run_metrics.total_state_changes_across_run = sum(lm.total_state_changes for lm in run_metrics.level_metrics.values())
        run_metrics.final_score = max_score

        if run_metrics.guid and not run_metrics.replay_url:
            run_metrics.replay_url = f"{ROOT_URL}/replay/{game_id}/{run_metrics.guid}"

        _loop.close()

    return run_metrics
