"""Run one scorecard across multiple games in parallel threads.

Usage:
    rgb-swarm --suite all --max-actions 500
    rgb-swarm --game ls20,ft09
    rgb-swarm --env-source re_arc --game memory-0001
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

import arc_agi
from arc_agi import OperationMode
from re_arc import EnvSampler

from rgb_agent.agent import OpenCodeAgent
from rgb_agent.environment.runner import GameRunner
from rgb_agent.environment import ArcAgi3Env, ReArcEnv
from rgb_agent.environment.config import EVALUATION_GAMES
from rgb_agent.metrics.structures import GameMetrics, Status
from rgb_agent.metrics.reporting import generate_console_report, save_summary_report, calculate_stats

log = logging.getLogger(__name__)

_CONSOLE_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
_FILE_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
load_dotenv(dotenv_path=_project_root / ".env.example")
load_dotenv(dotenv_path=_project_root / ".env", override=True)

ROOT_URL = os.environ.get("ROOT_URL", "https://three.arcprize.org")
_DEFAULT_ANALYZER_MODEL = os.environ.get("RGB_ANALYZER_MODEL", os.environ.get("ARCGYM_ANALYZER_MODEL", "local-qwen"))

_RE_ARC_GAME_ALIASES = {
    "memory": "memory-0001",
}


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=_CONSOLE_LOG_FORMAT)
    logging.getLogger("arc_agi").propagate = False


def _attach_run_log(run_dir: Path) -> logging.FileHandler:
    run_log_path = run_dir / "run.log"
    handler = logging.FileHandler(run_log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_FILE_LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    return handler


class Swarm:
    """Runs one agent per game in daemon threads."""

    def __init__(
        self,
        inner_agent_kwargs: dict[str, Any],
        env_source: str,
        arcade: arc_agi.Arcade | None,
        games: list[str],
        tags: list[str],
        max_actions: int = 500,
        analyzer_hook: Any = None,
        prompts_log_dir: Path | None = None,
        log_post_board: bool = True,
        analyzer_retries: int = 5,
        re_arc_seed: int | None = None,
        re_arc_augment: bool = False,
        re_arc_environments_dir: str | None = None,
    ) -> None:
        self.inner_agent_kwargs = inner_agent_kwargs
        self.env_source = env_source
        self._arcade = arcade
        self.games = games
        self.tags = tags
        self.max_actions = max_actions
        self.analyzer_hook = analyzer_hook
        self.prompts_log_dir = prompts_log_dir
        self.log_post_board = log_post_board
        self.analyzer_retries = analyzer_retries
        self.re_arc_seed = re_arc_seed
        self.re_arc_augment = re_arc_augment
        self.re_arc_environments_dir = re_arc_environments_dir

        self.card_id: str | None = None
        self.scorecard: Any = None
        self.results: dict[str, GameMetrics] = {}
        self._lock = threading.Lock()

    def run(self) -> dict[str, GameMetrics]:
        card_id: str | None = None
        if self.env_source == "arc_agi":
            if self._arcade is None:
                raise RuntimeError("arcade instance is required for arc_agi runs")
            card_id = self._arcade.open_scorecard(tags=self.tags)
            self.card_id = card_id
            log.info("Opened scorecard %s for %d game(s)", self.card_id, len(self.games))
        else:
            log.info("Running re_arc for %d game(s) (no scorecard)", len(self.games))

        threads = [
            threading.Thread(target=self._run_game, args=(card_id, gid), daemon=True)
            for gid in self.games
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if self.env_source == "arc_agi":
            if self._arcade is None or self.card_id is None:
                raise RuntimeError("scorecard close requested without an active arcade/card")
            self.scorecard = self._arcade.close_scorecard(self.card_id)
            log.info("Closed scorecard %s", self.card_id)
        return self.results

    def _run_game(self, card_id: str | None, game_id: str) -> None:
        env = None
        try:
            if self.env_source == "arc_agi":
                if self._arcade is None or card_id is None:
                    raise RuntimeError("arc_agi run requires arcade + scorecard_id")
                env = ArcAgi3Env.from_arcade(
                    arcade=self._arcade,
                    game_id=game_id,
                    scorecard_id=card_id,
                    max_actions=self.max_actions,
                )
            elif self.env_source == "re_arc":
                env = ReArcEnv(
                    game_id=game_id,
                    max_actions=self.max_actions,
                    seed=self.re_arc_seed,
                    augment=self.re_arc_augment,
                    environments_dir=self.re_arc_environments_dir,
                )
            else:
                raise ValueError(f"Unknown env source: {self.env_source!r}")

            prompts_log_path = None
            if self.prompts_log_dir:
                game_dir = self.prompts_log_dir / game_id.split("-")[0]
                game_dir.mkdir(parents=True, exist_ok=True)
                prompts_log_path = game_dir / "logs.txt"
                prompts_log_path.write_text("")

            runner = GameRunner(
                env=env,
                game_id=game_id,
                agent_name=self.inner_agent_kwargs.get("name", "swarm_agent"),
                max_actions_per_game=self.max_actions,
                tags=self.tags,
                prompts_log_path=prompts_log_path,
                analyzer=self.analyzer_hook,
                log_post_board=self.log_post_board,
                analyzer_retries=self.analyzer_retries,
                agent_kwargs=self.inner_agent_kwargs,
            )
            metrics = runner.run()

            with self._lock:
                self.results[game_id] = metrics

        except Exception as exc:
            log.error("Game %s failed: %s", game_id, exc, exc_info=True)
            with self._lock:
                self.results[game_id] = GameMetrics(
                    game_id=game_id,
                    agent_name=self.inner_agent_kwargs.get("name", "swarm_agent"),
                    start_time=time.time(),
                    status=Status.ERROR,
                    error_message=str(exc),
                )
        finally:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass


def _resolve_arc_agi_games(args: argparse.Namespace) -> list[str]:
    all_known = {gid for ids in EVALUATION_GAMES.values() for gid in ids}
    prefix_map = {gid.split("-")[0]: gid for gid in all_known}

    if args.game:
        raw = [g.strip() for g in args.game.split(",") if g.strip()]
        return [prefix_map.get(g, g) for g in raw]
    if args.suite:
        return EVALUATION_GAMES[args.suite]

    api_key = os.getenv("ARC_API_KEY", "")
    resp = requests.get(
        f"{ROOT_URL}/api/games",
        headers={"X-API-Key": api_key, "Accept": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    games = [g["game_id"] for g in resp.json()]
    log.info("Fetched %d games from ARC API", len(games))
    return games


def _resolve_re_arc_games(args: argparse.Namespace) -> list[str]:
    available = EnvSampler.list_game_ids(environments_dir=args.re_arc_environments_dir)
    prefix_map = {gid.split("-")[0]: gid for gid in available}
    alias_map = {key: value for key, value in _RE_ARC_GAME_ALIASES.items() if value in available}

    if args.game:
        raw = [g.strip().lower() for g in args.game.split(",") if g.strip()]
    elif args.suite:
        raw = [g.strip().lower() for g in EVALUATION_GAMES[args.suite] if g.strip()]
    else:
        raise ValueError("No games to run. Provide --game (e.g. memory-0001) or --suite.")

    resolved: list[str] = []
    unknown: list[str] = []
    for value in raw:
        game_id = alias_map.get(value, prefix_map.get(value, value))
        if game_id in available:
            resolved.append(game_id)
        else:
            unknown.append(value)

    if unknown:
        raise ValueError(
            f"Unknown re_arc game(s): {unknown}. "
            "Use --game memory-0001, --game taps, or another id from re_arc.EnvSampler.list_game_ids()."
        )

    return resolved


def main() -> None:
    _configure_logging()

    parser = argparse.ArgumentParser(description="Run RGB swarm evaluation.")
    parser.add_argument("--agent", "-a", default="rgb_agent")
    parser.add_argument("--env-source", default="arc_agi", choices=["arc_agi", "re_arc"])
    parser.add_argument(
        "--game", "-g",
        help="Comma-separated game IDs (e.g. ls20-cb3b57cc,ft09-9ab2447a,memory-0001,taps).",
    )
    parser.add_argument("--suite", "-s", choices=list(EVALUATION_GAMES.keys()))
    parser.add_argument("--tags", "-t", help="Comma-separated tags.")
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--operation-mode", default="online", choices=["normal", "online", "offline"])
    parser.add_argument("--re-arc-seed", dest="re_arc_seed", type=int, default=None)
    parser.add_argument("--re-arc-augment", dest="re_arc_augment", action="store_true")
    parser.add_argument("--re-arc-environments-dir", dest="re_arc_environments_dir", default=None)
    parser.add_argument(
        "--interval", "-n", "--analyzer-interval",
        dest="analyzer_interval",
        type=int,
        default=10,
        help="Actions per analyzer batch plan.",
    )
    parser.add_argument(
        "--model", "-m", "--analyzer-model",
        dest="analyzer_model",
        default=_DEFAULT_ANALYZER_MODEL,
        help="Analyzer model or alias (default: %(default)s). Useful aliases: local-qwen, opus, sonnet.",
    )
    parser.add_argument(
        "--retries", "--analyzer-retries",
        dest="analyzer_retries",
        type=int,
        default=5,
        help="Max analyzer retry attempts.",
    )

    args = parser.parse_args()

    try:
        if args.env_source == "arc_agi":
            games = _resolve_arc_agi_games(args)
        else:
            games = _resolve_re_arc_games(args)
    except Exception as exc:
        log.error("Failed to resolve games: %s", exc)
        sys.exit(1)

    if not games:
        log.error("No games to run.")
        sys.exit(1)

    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    tags.append(f"swarm-{args.agent}")

    arcade = None
    if args.env_source == "arc_agi":
        arcade = arc_agi.Arcade(
            arc_api_key=os.getenv("ARC_API_KEY", ""),
            arc_base_url=ROOT_URL,
            operation_mode=OperationMode(args.operation_mode),
        )

    timestamp = datetime.now().strftime("%m%dT%H%M%S")
    run_dir = Path("evaluation_results") / f"{timestamp}_{args.env_source}_swarm_{args.agent}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log_handler = _attach_run_log(run_dir)

    try:
        log.info("Run log: %s", run_dir / "run.log")
        log.info(
            "Run config env_source=%s games=%s model=%s max_actions=%d interval=%d retries=%d",
            args.env_source,
            ",".join(games),
            args.analyzer_model,
            args.max_actions,
            args.analyzer_interval,
            args.analyzer_retries,
        )

        agent = OpenCodeAgent(
            model=args.analyzer_model,
            plan_size=args.analyzer_interval,
        )
        log.info("Analyzer enabled (interval=%d, model=%s)", args.analyzer_interval, args.analyzer_model)

        inner_agent_kwargs: dict[str, Any] = {
            "name": args.agent,
            "plan_size": args.analyzer_interval,
        }

        swarm = Swarm(
            inner_agent_kwargs=inner_agent_kwargs,
            env_source=args.env_source,
            arcade=arcade,
            games=games,
            tags=tags,
            max_actions=args.max_actions,
            analyzer_hook=agent.analyze,
            prompts_log_dir=run_dir,
            log_post_board=True,
            analyzer_retries=args.analyzer_retries,
            re_arc_seed=args.re_arc_seed,
            re_arc_augment=args.re_arc_augment,
            re_arc_environments_dir=args.re_arc_environments_dir,
        )

        runner = threading.Thread(target=swarm.run, daemon=True)
        runner.start()

        def sigint_handler(sig: int, frame: Any) -> None:
            print("[Swarm] SIGINT received — cleaning up...", flush=True)
            sys.exit(1)

        signal.signal(signal.SIGINT, sigint_handler)

        while runner.is_alive():
            runner.join(timeout=1)

        results_list = list(swarm.results.values())

        print(f"\nEnvironment:  {args.env_source}")
        if swarm.card_id:
            print(f"Scorecard ID: {swarm.card_id}")
        print(f"Results:      {run_dir}")
        print(f"Run Log:      {run_dir / 'run.log'}")
        for m in sorted(results_list, key=lambda r: r.game_id):
            if m.replay_url:
                print(f"  Replay:     {m.replay_url}")

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
                label = env.id or "unknown"
                state = run.state.name if run.state else "?"
                print(f"\n  {label}  score={run.score:.1f}  state={state}  actions={run.actions}")
                if run.level_scores:
                    for i, (ls, la, lb) in enumerate(zip(
                        run.level_scores,
                        run.level_actions or [],
                        run.level_baseline_actions or [],
                    )):
                        baseline = str(lb) if lb >= 0 else "n/a"
                        print(f"    Level {i+1}: efficiency={ls:.1f}  actions={la}  baseline={baseline}")
                if run.message:
                    print(f"    Note: {run.message}")
            print(f"{'='*60}")

            scorecard_path = run_dir / "scorecard.json"
            scorecard_path.write_text(sc.model_dump_json(indent=2))
            log.info("Scorecard saved to %s", scorecard_path)
    finally:
        logging.getLogger().removeHandler(run_log_handler)
        run_log_handler.close()

    if results_list:
        generate_console_report(results_list, "swarm", args.agent, 1, scorecard=swarm.scorecard)
        game_stats, overall = calculate_stats(results_list)
        summary_path = run_dir / "summary.txt"
        save_summary_report(
            str(summary_path), game_stats, overall, results_list,
            args.agent, "swarm", 1, scorecard=swarm.scorecard,
        )
        log.info("Summary saved to %s", summary_path)
    else:
        log.error("No results collected.")


if __name__ == "__main__":
    main()
