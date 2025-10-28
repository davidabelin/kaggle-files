"""Documentation helpers for logging Rock Paper Scissors training data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol


@dataclass
class MatchStep:
    """Single turn of a Rock Paper Scissors match."""

    step: int
    agent_action: int
    opponent_action: int
    reward: int


@dataclass
class MatchRecord:
    """Full match history for supervised learning."""

    agent_name: str
    opponent_name: str
    steps: List[MatchStep]


class MatchLogger(Protocol):
    """Protocol describing the expected dataset logging interface."""

    def log_match(self, match: MatchRecord) -> None:
        ...


def iter_training_examples(match: MatchRecord) -> Iterable[dict]:
    """Yield dictionary-based samples suitable for model training."""

    for step in match.steps:
        yield {
            "agent": match.agent_name,
            "opponent": match.opponent_name,
            "step": step.step,
            "agent_action": step.agent_action,
            "opponent_action": step.opponent_action,
            "reward": step.reward,
        }
