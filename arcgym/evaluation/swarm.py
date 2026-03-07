"""
ARC-AGI-3 Swarm: run one scorecard across multiple games in parallel threads.

Mirrors the official arcprize/ARC-AGI-3-Agents Swarm pattern while
reusing the harness's ``evaluate_single_game`` for the per-game loop
(analyzer, prompt logging, QueueExhausted retries — all included).

Usage:
    python -m arcgym.evaluation.swarm --suite standard_suite
    python -m arcgym.evaluation.swarm --game ls20-cb3b57cc,ft09-9ab2447a
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Type

import requests
from dotenv import load_dotenv

import arc_agi
from arc_agi import OperationMode

from arcgym.agents import AVAILABLE_AGENTS
from arcgym.environments import ArcAgi3Env
from arcgym.evaluation.config import EVALUATION_GAMES
from arcgym.evaluation.harness import evaluate_single_game
from arcgym.metrics.structures import GameMetrics
from arcgym.metrics.reporting import generate_console_report, save_summary_report, calculate_stats

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root / .env loading
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
load_dotenv(dotenv_path=_project_root / ".env.example")
load_dotenv(dotenv_path=_project_root / ".env", override=True)

ROOT_URL = os.environ.get("ROOT_URL", "https://three.arcprize.org")


# ---------------------------------------------------------------------------
# Swarm
# ---------------------------------------------------------------------------
class Swarm:
    """Manages a single scorecard and runs one agent per game in daemon threads.

    Mirrors the official ARC-AGI-3 Swarm pattern:
    1. Opens ONE scorecard for all games
    2. Creates one agent + env per game
    3. Spawns daemon threads, joins all
    4. Closes scorecard when done

    The per-game loop delegates entirely to the harness's
    ``evaluate_single_game``, which handles the analyzer, prompt logging,
    QueueExhausted retries, and detailed metrics.
    """

    def __init__(
        self,
        agent_class: Type,
        agent_kwargs: dict[str, Any],
        arcade: arc_agi.Arcade,
        games: list[str],
        tags: list[str],
        max_actions: int = 500,
        analyzer_hook: Any = None,
        prompts_log_dir: Path | None = None,
        log_post_board: bool = True,
        analyzer_retries: int = 5,
    ) -> None:
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs
        self._arcade = arcade
        self.games = games
        self.tags = tags
        self.max_actions = max_actions
        self.analyzer_hook = analyzer_hook
        self.prompts_log_dir = prompts_log_dir
        self.log_post_board = log_post_board
        self.analyzer_retries = analyzer_retries

        self.card_id: str | None = None
        self.scorecard: Any = None
        self.results: dict[str, GameMetrics] = {}
        self._lock = threading.Lock()

    # -- public API ----------------------------------------------------------

    def run(self) -> dict[str, GameMetrics]:
        """Open scorecard -> run all games -> close scorecard -> return results."""
        self.card_id = self._arcade.open_scorecard(tags=self.tags)
        log.info(
            "[Swarm] Opened scorecard %s for %d game(s)", self.card_id, len(self.games)
        )

        threads: list[threading.Thread] = []
        for game_id in self.games:
            t = threading.Thread(
                target=self._run_game, args=(self.card_id, game_id), daemon=True
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.scorecard = self._arcade.close_scorecard(self.card_id)
        log.info("[Swarm] Closed scorecard %s", self.card_id)

        return self.results

    # -- per-game thread -----------------------------------------------------

    def _run_game(self, card_id: str, game_id: str) -> None:
        """Thread target: create env + agent, delegate to evaluate_single_game."""
        try:
            env = ArcAgi3Env.from_arcade(
                arcade=self._arcade,
                game_id=game_id,
                scorecard_id=card_id,
                max_actions=self.max_actions,
            )

            agent = self.agent_class(**self.agent_kwargs, game_id=game_id)

            prompts_log_path = None
            if self.prompts_log_dir:
                game_dir = self.prompts_log_dir / game_id.split("-")[0]
                game_dir.mkdir(parents=True, exist_ok=True)
                prompts_log_path = game_dir / "logs.txt"
                prompts_log_path.write_text("")

            metrics = evaluate_single_game(
                agent=agent,
                env=env,
                game_id=game_id,
                agent_name=self.agent_kwargs.get("name", "swarm_agent"),
                max_actions_per_game=self.max_actions,
                run_index=1,
                tags=self.tags,
                prompts_log_path=prompts_log_path,
                analyzer=self.analyzer_hook,
                log_post_board=self.log_post_board,
                analyzer_retries=self.analyzer_retries,
            )

            with self._lock:
                self.results[game_id] = metrics

        except Exception as exc:
            log.error("[Swarm] Game %s failed: %s", game_id, exc, exc_info=True)
            with self._lock:
                self.results[game_id] = GameMetrics(
                    game_id=game_id,
                    agent_name=self.agent_kwargs.get("name", "swarm_agent"),
                    start_time=time.time(),
                    status="ERROR",
                    error_message=str(exc),
                )
        finally:
            try:
                env.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    # Suppress duplicate logs from arc_agi's own StreamHandler
    logging.getLogger("arc_agi").propagate = False

    parser = argparse.ArgumentParser(
        description="Run ARC-AGI-3 Swarm evaluation (one scorecard, parallel games)."
    )
    parser.add_argument(
        "--agent", "-a",
        default="claude_code_action_agent",
        choices=list(AVAILABLE_AGENTS.keys()),
        help="Agent to run.",
    )
    parser.add_argument(
        "--game", "-g",
        help="Comma-separated game IDs (e.g. ls20-cb3b57cc,ft09-9ab2447a). "
             "Omit to use --suite or fetch all games from API.",
    )
    parser.add_argument(
        "--suite", "-s",
        choices=list(EVALUATION_GAMES.keys()),
        help="Predefined evaluation suite.",
    )
    parser.add_argument(
        "--tags", "-t",
        help="Comma-separated tags for the scorecard.",
    )
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument(
        "--operation-mode", default="online",
        choices=["normal", "online", "offline"],
    )

    # -- Analyzer flags -------------------------------------------------------
    parser.add_argument("--analyzer-interval", dest="analyzer_interval",
                        type=int, default=10)
    parser.add_argument("--analyzer-model", dest="analyzer_model",
                        default="claude-opus-4-6")
    parser.add_argument("--analyzer-retries", dest="analyzer_retries",
                        type=int, default=5)

    args = parser.parse_args()

    # -- Resolve game list ---------------------------------------------------
    # Build prefix lookup from all known game IDs
    _all_known = {gid for ids in EVALUATION_GAMES.values() for gid in ids}
    _prefix_map: dict[str, str] = {}
    for gid in _all_known:
        prefix = gid.split("-")[0]
        _prefix_map[prefix] = gid

    games: list[str] = []
    if args.game:
        raw = [g.strip() for g in args.game.split(",") if g.strip()]
        games = [_prefix_map.get(g, g) for g in raw]
    elif args.suite:
        games = EVALUATION_GAMES[args.suite]
    else:
        api_key = os.getenv("ARC_API_KEY", "")
        try:
            resp = requests.get(
                f"{ROOT_URL}/api/games",
                headers={"X-API-Key": api_key, "Accept": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            games = [g["game_id"] for g in resp.json()]
            log.info("[Swarm] Fetched %d games from API", len(games))
        except Exception as exc:
            log.error("[Swarm] Failed to fetch games from API: %s", exc)
            sys.exit(1)

    if not games:
        log.error("[Swarm] No games to run. Provide --game, --suite, or set ARC_API_KEY.")
        sys.exit(1)

    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    tags.append(f"swarm-{args.agent}")

    # -- Shared Arcade -------------------------------------------------------
    arcade = arc_agi.Arcade(
        arc_api_key=os.getenv("ARC_API_KEY", ""),
        arc_base_url=ROOT_URL,
        operation_mode=OperationMode(args.operation_mode),
    )

    # -- Analyzer hook (always required — agent is queue-driven) -------------
    agent_name_cli = args.agent

    from arcgym.utils.analyzer import make_opencode_analyzer

    analyzer_hook = make_opencode_analyzer(
        interval=0,
        use_subscription=False,
        allow_bash=True,
        action_mode="all",
        plan_size=args.analyzer_interval,
        allow_self_read=False,
        model=args.analyzer_model,
        fast=False,
        resume_session=True,
    )
    log.info("[Swarm] OpenCode analyzer enabled (interval=%d, model=%s)",
             args.analyzer_interval, args.analyzer_model)

    # -- Run directory & prompts log -----------------------------------------
    timestamp = datetime.now().strftime("%m%dT%H%M%S")
    run_dir = Path("evaluation_results") / f"{timestamp}_swarm_{agent_name_cli}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prompts_log_dir = run_dir

    # -- Agent kwargs --------------------------------------------------------
    agent_kwargs: dict[str, Any] = {
        "name": agent_name_cli,
        "plan_size": args.analyzer_interval,
    }

    # -- Run swarm -----------------------------------------------------------
    swarm = Swarm(
        agent_class=AVAILABLE_AGENTS[args.agent],
        agent_kwargs=agent_kwargs,
        arcade=arcade,
        games=games,
        tags=tags,
        max_actions=args.max_actions,
        analyzer_hook=analyzer_hook,
        prompts_log_dir=prompts_log_dir,
        log_post_board=True,
        analyzer_retries=args.analyzer_retries,
    )

    # Run in daemon thread with SIGINT handling
    runner = threading.Thread(target=swarm.run, daemon=True)
    runner.start()

    def _sigint_handler(sig: int, frame: Any) -> None:
        print("[Swarm] SIGINT received — cleaning up containers...", flush=True)
        # Normal exit triggers atexit handlers (container cleanup)
        sys.exit(1)

    signal.signal(signal.SIGINT, _sigint_handler)

    # Join with timeout so the main thread can process SIGINT
    while runner.is_alive():
        runner.join(timeout=1)

    # -- Save results ---------------------------------------------------------
    results_list = list(swarm.results.values())

    # Console report
    print(f"\nScorecard ID: {swarm.card_id}")
    print(f"Results:      {run_dir}")
    for m in sorted(results_list, key=lambda r: r.game_id):
        if m.replay_url:
            print(f"  Replay:     {m.replay_url}")

    # Print official scorecard
    if swarm.scorecard:
        sc = swarm.scorecard
        print(f"\n{'='*60}")
        print(f"ARC Scorecard  —  overall score: {sc.score:.1f}")
        print(f"  Environments: {sc.total_environments_completed}/{sc.total_environments}")
        print(f"  Levels:       {sc.total_levels_completed}/{sc.total_levels}")
        print(f"  Actions:      {sc.total_actions}")
        for env in sc.environments:
            run = env.runs[0] if env.runs else None
            if not run:
                continue
            game_label = env.id or "unknown"
            state = run.state.name if run.state else "?"
            print(f"\n  {game_label}  score={run.score:.1f}  state={state}  actions={run.actions}")
            if run.level_scores:
                for i, (ls, la, lb) in enumerate(zip(
                    run.level_scores,
                    run.level_actions or [],
                    run.level_baseline_actions or [],
                )):
                    baseline_str = str(lb) if lb >= 0 else "n/a"
                    print(f"    Level {i+1}: efficiency={ls:.1f}  actions={la}  baseline={baseline_str}")
            if run.message:
                print(f"    Note: {run.message}")
        print(f"{'='*60}")

        # Save scorecard JSON
        scorecard_path = run_dir / "scorecard.json"
        scorecard_path.write_text(sc.model_dump_json(indent=2))
        log.info("Scorecard saved to %s", scorecard_path)

    if results_list:
        generate_console_report(results_list, "swarm", agent_name_cli, 1, scorecard=swarm.scorecard)
        game_stats, overall_summary = calculate_stats(results_list)
        summary_path = run_dir / "summary.txt"
        save_summary_report(
            str(summary_path),
            game_stats, overall_summary, results_list,
            agent_name_cli, "swarm", 1,
            scorecard=swarm.scorecard,
        )
        log.info("Summary saved to %s", summary_path)
    else:
        log.error("No results collected.")


if __name__ == "__main__":
    main()
