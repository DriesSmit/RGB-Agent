"""
Reporting utilities for generating evaluation summaries and statistics.
"""
import textwrap
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from datetime import datetime
import statistics

from .structures import GameMetrics


def calculate_stats(results: List[GameMetrics]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Calculates per-game, per-level, and overall summary statistics."""
    game_level_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    game_summary_stats = defaultdict(lambda: defaultdict(list))

    total_runs_all = 0
    total_completed_all = 0
    total_duration_all = 0.0

    for res in results:
        game_id = res.game_id
        total_runs_all += 1
        total_duration_all += res.run_duration_seconds

        game_summary_stats[game_id]['status'].append(res.status)
        game_summary_stats[game_id]['final_score'].append(res.final_score)
        game_summary_stats[game_id]['run_total_actions'].append(res.run_total_actions)
        game_summary_stats[game_id]['run_duration'].append(res.run_duration_seconds)
        game_summary_stats[game_id]['total_game_overs'].append(res.total_game_overs_across_run)
        game_summary_stats[game_id]['highest_level'].append(res.highest_level_reached)

        if res.status == "COMPLETED_RUN":
            total_completed_all += 1

        for level_num, level_data in res.level_metrics.items():
            game_level_stats[game_id][level_num]['total_actions'].append(level_data.total_actions)
            game_level_stats[game_id][level_num]['total_game_overs'].append(level_data.total_game_overs)
            game_level_stats[game_id][level_num]['total_state_changes'].append(level_data.total_state_changes)
            game_level_stats[game_id][level_num]['status'].append(level_data.status)

            success_actions = level_data.actions_in_successful_attempt
            if success_actions is not None:
                game_level_stats[game_id][level_num]['success_actions'].append(success_actions)

            game_level_stats[game_id][level_num]['attempt_durations'].extend(
                [a.duration_seconds for a in level_data.attempts]
            )

    overall_summary = {
        "total_runs": total_runs_all,
        "total_completed": total_completed_all,
        "overall_completion_rate": (float(total_completed_all) / total_runs_all * 100.0) if total_runs_all else 0.0,
        "average_duration_all": total_duration_all / total_runs_all if total_runs_all else 0.0,
    }

    processed_game_stats = {}
    all_game_ids = set(game_summary_stats.keys()) | set(game_level_stats.keys())

    for game_id in all_game_ids:
        summary_data = game_summary_stats.get(game_id, defaultdict(list))
        level_data_raw = game_level_stats.get(game_id, defaultdict(lambda: defaultdict(list)))

        runs = len(summary_data['status'])
        completed_runs = summary_data['status'].count("COMPLETED_RUN")

        avg_final_score = statistics.mean(summary_data['final_score']) if summary_data['final_score'] else 0.0
        avg_run_total_actions = statistics.mean(summary_data['run_total_actions']) if summary_data['run_total_actions'] else 0.0
        avg_run_duration = statistics.mean(summary_data['run_duration']) if summary_data['run_duration'] else 0.0
        avg_total_game_overs = statistics.mean(summary_data['total_game_overs']) if summary_data['total_game_overs'] else 0.0
        avg_highest_level = statistics.mean(summary_data['highest_level']) if summary_data['highest_level'] else 1.0
        completion_rate = (float(completed_runs) / runs * 100.0) if runs else 0.0

        aggregated_levels = {}
        if level_data_raw:
            max_level_num = max(level_data_raw.keys()) if level_data_raw else 0
            for level_num in range(1, max_level_num + 1):
                level_stats = level_data_raw.get(level_num, defaultdict(list))

                attempt_count = len(level_stats['status'])
                completed_count = level_stats['status'].count("COMPLETED")

                avg_success_actions = statistics.mean(level_stats['success_actions']) if level_stats['success_actions'] else 0.0
                avg_total_actions = statistics.mean(level_stats['total_actions']) if level_stats['total_actions'] else 0.0
                avg_duration = statistics.mean(level_stats['attempt_durations']) if level_stats['attempt_durations'] else 0.0
                avg_state_changes = statistics.mean(level_stats['total_state_changes']) if level_stats['total_state_changes'] else 0.0
                avg_game_overs = statistics.mean(level_stats['total_game_overs']) if level_stats['total_game_overs'] else 0.0
                level_completion_rate = (float(completed_count) / attempt_count * 100.0) if attempt_count else 0.0

                aggregated_levels[level_num] = {
                    "attempts": attempt_count,
                    "completed": completed_count,
                    "avg_total_actions": avg_total_actions,
                    "avg_success_actions": avg_success_actions,
                    "avg_duration_per_attempt": avg_duration,
                    "avg_total_state_changes": avg_state_changes,
                    "avg_total_game_overs": avg_game_overs,
                    "completion_rate": level_completion_rate,
                }

        processed_game_stats[game_id] = {
            "num_runs": runs,
            "completed_runs": completed_runs,
            "avg_final_score": avg_final_score,
            "avg_run_total_actions": avg_run_total_actions,
            "avg_run_duration": avg_run_duration,
            "run_completion_rate": completion_rate,
            "avg_total_game_overs_per_run": avg_total_game_overs,
            "avg_highest_level": avg_highest_level,
            "level_stats": aggregated_levels,
        }

    return processed_game_stats, overall_summary


def _build_report_lines(
    game_stats: Dict[str, Dict[str, Any]],
    overall_summary: Dict[str, Any],
    results_data: List[GameMetrics],
    agent_name: str,
    suite_name: str,
    num_runs_per_game: int,
    scorecard: Any = None,
) -> List[str]:
    """Build the full report as a list of lines."""
    lines = []

    lines.append(" Evaluation Report ")
    lines.append(f"Agent: {agent_name}")
    lines.append(f"Suite: {suite_name}")
    lines.append(f"Requested Runs per Game: {num_runs_per_game}")
    lines.append(f"Generated At: {datetime.now().isoformat()}")
    lines.append("-" * 50)

    # Overall Summary
    lines.append("\n## Overall Summary")
    lines.append("-" * 50)
    lines.append(f"Total Runs Attempted: {overall_summary['total_runs']}")
    lines.append(f"Total Runs Completed (Full Game Win): {overall_summary['total_completed']}")
    lines.append(f"Overall Game Completion Rate: {overall_summary['overall_completion_rate']:.1f}%")
    lines.append(f"Average Run Duration (all runs): {overall_summary['average_duration_all']:.2f}s")
    lines.append("-" * 50)

    # Per-Game Summary
    lines.append("\n## Per-Game Summary (Averaged Across Runs)")

    if not game_stats:
        lines.append("No game results to display.")
    else:
        for game_id, stats in sorted(game_stats.items()):
            lines.append("\n" + "=" * 80)
            lines.append(f"Game ID: {game_id}")
            lines.append("=" * 80)

            lines.append("  Run Summary:")
            lines.append(f"    Total Runs: {stats['num_runs']}")
            lines.append(f"    Completed Runs: {stats['completed_runs']} ({stats['run_completion_rate']:.1f}%)")
            lines.append(f"    Avg Final Score: {stats['avg_final_score']:.1f}")
            lines.append(f"    Avg Highest Level: {stats['avg_highest_level']:.1f}")
            lines.append(f"    Avg Total Actions: {stats['avg_run_total_actions']:.1f}")
            lines.append(f"    Avg Run Duration: {stats['avg_run_duration']:.2f}s")
            lines.append(f"    Avg Game Overs: {stats['avg_total_game_overs_per_run']:.1f}")

            if stats['level_stats']:
                lines.append("\n  Level Statistics:")
                lvl_header = f"    {'Lvl':>3} | {'Avg Total Actions':>18} | {'Avg Success Actions':>20} | {'Avg Total GOs':>14} | {'Avg State D':>12} | {'Cmpl Rate':>11} | {'Attempts':>10}"
                lines.append("    " + "-" * (len(lvl_header) - 4))
                lines.append(lvl_header)
                lines.append("    " + "-" * (len(lvl_header) - 4))
                for lvl_num, lvl_stat in sorted(stats['level_stats'].items()):
                    avg_total_act_str = f"{lvl_stat['avg_total_actions']:.1f}"
                    avg_success_act_str = f"{lvl_stat['avg_success_actions']:.1f}" if lvl_stat['avg_success_actions'] > 0 else "N/A"
                    lines.append(
                        f"    {lvl_num:>3} | {avg_total_act_str:>18} | {avg_success_act_str:>20} | "
                        f"{lvl_stat['avg_total_game_overs']:>14.1f} | {lvl_stat['avg_total_state_changes']:>12.1f} | "
                        f"{lvl_stat['completion_rate']:>10.1f}% | {lvl_stat['attempts']:>10}"
                    )
            else:
                lines.append("\n  No level statistics collected.")

    # Detailed Run List
    lines.append("\n" + "=" * 80)
    lines.append("\n## Detailed Run List")
    lines.append("-" * 80)
    if not results_data:
        lines.append("No runs recorded.")
    else:
        sorted_results = sorted(results_data, key=lambda r: (r.game_id, r.run_index))
        current_game_id = None
        for res in sorted_results:
            if res.game_id != current_game_id:
                lines.append(f"\nGame: {res.game_id}")
                current_game_id = res.game_id

            details = f"-> {res.replay_url or 'N/A'}"
            if res.status == "ERROR" and res.error_message:
                details = f"-> ERROR: {textwrap.shorten(res.error_message.replace(chr(10), ' '), width=70, placeholder='...')}"

            lines.append(
                f"  Run {res.run_index:>2}: {res.status:<15} Score={res.final_score:>4}, "
                f"HighestLvl={res.highest_level_reached:>2}, Actions={res.run_total_actions:>4}, "
                f"Dur={res.run_duration_seconds:.2f}s, GOs={res.total_game_overs_across_run:>3} {details}"
            )
    lines.append("-" * 80)

    # ARC Scorecard
    if scorecard:
        lines.append("\n" + "=" * 80)
        lines.append("## ARC Scorecard (Efficiency)")
        lines.append("=" * 80)
        lines.append(f"  Overall Score: {scorecard.score:.1f}")
        lines.append(f"  Environments:  {scorecard.total_environments_completed}/{scorecard.total_environments}")
        lines.append(f"  Levels:        {scorecard.total_levels_completed}/{scorecard.total_levels}")
        lines.append(f"  Total Actions: {scorecard.total_actions}")
        for env in scorecard.environments:
            run = env.runs[0] if env.runs else None
            if not run:
                continue
            game_label = env.id or "unknown"
            state = run.state.name if run.state else "?"
            lines.append(f"\n  {game_label}  score={run.score:.1f}  state={state}  actions={run.actions}")
            if run.level_scores:
                for i, (ls, la, lb) in enumerate(zip(
                    run.level_scores,
                    run.level_actions or [],
                    run.level_baseline_actions or [],
                )):
                    baseline_str = str(lb) if lb >= 0 else "n/a"
                    lines.append(f"    Level {i+1}: efficiency={ls:.1f}  actions={la}  baseline={baseline_str}")
            if run.message:
                lines.append(f"    Note: {run.message}")
        lines.append("-" * 80)

    lines.append("\n End of Report ")
    return lines


def generate_console_report(
    results_data: List[GameMetrics],
    suite_name: str,
    agent_name: str,
    num_runs_per_game: int,
    scorecard: Any = None,
):
    """Print evaluation report to console."""
    if not results_data:
        print("No results to report.")
        return
    game_stats, overall_summary = calculate_stats(results_data)
    for line in _build_report_lines(game_stats, overall_summary, results_data, agent_name, suite_name, num_runs_per_game, scorecard):
        print(line)


def save_summary_report(
    filepath: str,
    game_stats: Dict[str, Dict[str, Any]],
    overall_summary: Dict[str, Any],
    results_data: List[GameMetrics],
    agent_name: str,
    suite_name: str,
    num_runs_per_game: int,
    scorecard: Any = None,
):
    """Save evaluation report to a text file."""
    lines = _build_report_lines(game_stats, overall_summary, results_data, agent_name, suite_name, num_runs_per_game, scorecard)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
