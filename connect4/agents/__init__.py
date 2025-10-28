"""Reusable baseline ConnectX agents exposed under stable names."""

from __future__ import annotations

from typing import Callable, Dict

from .adaptive_midrange import my_agent as adaptive_midrange
from .alpha_beta_v9 import my_agent as alpha_beta_v9
from .time_boxed_pruner import my_agent as time_boxed_pruner

Agent = Callable[[object, object], int]

BASELINE_AGENTS: Dict[str, Agent] = {
    "alpha_beta_v9": alpha_beta_v9,
    "adaptive_midrange": adaptive_midrange,
    "time_boxed_pruner": time_boxed_pruner,
}

__all__ = [
    "Agent",
    "BASELINE_AGENTS",
    "adaptive_midrange",
    "alpha_beta_v9",
    "time_boxed_pruner",
]
