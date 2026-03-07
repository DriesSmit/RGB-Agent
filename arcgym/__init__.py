"""
This package hosts extension modules that sit on top of the upstream
`rllm` agent/env/workflow stack. Nothing here is tightly coupled to
training infrastructure; instead we expose registry helpers so new
components can be imported and registered from CLI/config entry points.
"""

# Make registry import optional - it requires verl which may not be available
try:
    from .registry import register_arcgym_components
    __all__ = ["register_arcgym_components"]
except ImportError as e:
    # Registry unavailable
    # Agents can still be imported directly from arcgym.agents
    import logging
    logging.getLogger(__name__).debug(f"Registry not available: {e}")
    __all__ = []
