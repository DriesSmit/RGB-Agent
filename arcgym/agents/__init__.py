"""ArcGym agent package."""

from .claude_code_action_agent import ClaudeCodeActionAgent

AVAILABLE_AGENTS = {
    "claude_code_action_agent": ClaudeCodeActionAgent,
}

__all__ = [
    "ClaudeCodeActionAgent",
    "AVAILABLE_AGENTS",
]
