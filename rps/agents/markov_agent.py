"""Markov-chain-based heuristic agent for Rock Paper Scissors."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List

import numpy as np


@dataclass
class MarkovAgentConfig:
    """Configuration parameters for the Markov agent."""

    order: int = 2
    refresh_interval: int = 250
    deterministic_horizon: int = 500
    mirror_horizon: int = 900


def make_markov_agent(config: MarkovAgentConfig | None = None) -> Callable[["Observation", "Configuration"], int]:
    """Create a stateful Markov agent compatible with the Kaggle API."""

    if config is None:
        config = MarkovAgentConfig()

    table = defaultdict(lambda: [1, 1, 1])
    action_seq: List[int] = []

    def agent(observation: "Observation", configuration: "Configuration") -> int:
        nonlocal table, action_seq

        if observation.step % config.refresh_interval == 0:
            table = defaultdict(lambda: [1, 1, 1])
            action_seq = []

        if len(action_seq) <= 2 * config.order + 1:
            action = int(np.random.randint(3))
            if observation.step > 0:
                action_seq.extend([observation.lastOpponentAction, action])
            else:
                action_seq.append(action)
            return action

        key = "".join(str(a) for a in action_seq[:-1])
        table[key][observation.lastOpponentAction] += 1

        action_seq[:-2] = action_seq[2:]
        action_seq[-2] = observation.lastOpponentAction

        key = "".join(str(a) for a in action_seq[:-1])
        if observation.step < config.deterministic_horizon:
            next_opp = int(np.argmax(table[key]))
        else:
            scores = np.asarray(table[key], dtype=float)
            next_opp = int(np.random.choice(3, p=scores / scores.sum()))

        action = (next_opp + 1) % 3
        if observation.step > config.mirror_horizon:
            action = next_opp

        action_seq[-1] = action
        return int(action)

    return agent


__all__ = ["MarkovAgentConfig", "make_markov_agent"]
