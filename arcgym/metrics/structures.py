"""
Data structures for tracking evaluation metrics.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

@dataclass
class AttemptMetrics:
    """Metrics for a single attempt (try) at a specific level."""
    attempt_number: int # 1, 2, 3...
    actions: int = 0
    duration_seconds: float = 0.0
    state_changes: int = 0
    game_overs: int = 0 # Will be 1 if this attempt ended in a GAME_OVER, 0 otherwise
    status: str = "IN_PROGRESS" # IN_PROGRESS, COMPLETED, TIMEOUT, ERROR, GAME_OVER

@dataclass
class LevelMetrics:
    """Metrics for a single level, containing all attempts."""
    level_number: int
    attempts: List[AttemptMetrics] = field(default_factory=list)
    status: str = "IN_PROGRESS" # IN_PROGRESS, COMPLETED, TIMEOUT, ERROR

    @property
    def total_actions(self) -> int:
        """Total actions taken across all attempts for this level."""
        return sum(a.actions for a in self.attempts)

    @property
    def total_game_overs(self) -> int:
        """Total game overs encountered on this level."""
        return sum(a.game_overs for a in self.attempts)

    @property
    def total_state_changes(self) -> int:
        """Total state changes across all attempts for this level."""
        return sum(a.state_changes for a in self.attempts)

    @property
    def actions_in_successful_attempt(self) -> Optional[int]:
        """Actions taken in the final, successful attempt. None if not completed."""
        if self.status == "COMPLETED" and self.attempts:
            # The last attempt is the successful one
            return self.attempts[-1].actions
        return None

    @property
    def state_change_percentage(self) -> float:
        """Percentage of actions that resulted in a state change across all attempts."""
        total_acts = self.total_actions
        if total_acts == 0:
            return 0.0
        return (self.total_state_changes / total_acts) * 100.0

@dataclass
class GameMetrics:
    """Metrics collected for a single game evaluation run."""
    #   Identification  
    game_id: str
    agent_name: str # This will be the "dynamic" name from the CLI
    run_index: int = 1
    guid: Optional[str] = None 

    #   Overall Run Performance  
    run_total_actions: int = 0 # Sum of total_actions from all LevelMetrics
    final_score: int = 0
    highest_level_reached: int = 1
    status: str = "PENDING" # PENDING, IN_PROGRESS, COMPLETED_RUN, TIMEOUT, ERROR
    error_message: Optional[str] = None # <-- ADDED THIS FIELD

    #   Detailed Level Data  
    # Stores metrics for each level, including all attempts
    level_metrics: Dict[int, LevelMetrics] = field(default_factory=dict) 

    #   Timing  
    start_time: float = field(default_factory=time.time)
    end_time: float = field(default_factory=time.time)
    run_duration_seconds: float = 0.0

    #   Replay  
    replay_url: Optional[str] = None 

    #   Aggregated Totals (for convenience, calculated at the end)
    total_state_changes_across_run: int = 0
    total_game_overs_across_run: int = 0
    duplicate_actions: int = 0  # Count of duplicate state-action pairs


