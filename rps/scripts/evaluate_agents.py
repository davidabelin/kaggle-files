"""Utility script to compare bundled Rock Paper Scissors heuristic agents."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from rps.agents import make_decision_tree_agent, make_markov_agent


@dataclass
class Observation:
    step: int
    lastOpponentAction: int | None


@dataclass
class Configuration:
    episodeSteps: int = 1000


AgentFactory = Callable[[], Callable[[Observation, Configuration], int]]


def _play_episode(
    agent_a_factory: AgentFactory, agent_b_factory: AgentFactory, steps: int
) -> Tuple[int, int]:
    agent_a = agent_a_factory()
    agent_b = agent_b_factory()

    observation_a = Observation(step=0, lastOpponentAction=None)
    observation_b = Observation(step=0, lastOpponentAction=None)
    configuration = Configuration(episodeSteps=steps)

    score_a = 0
    score_b = 0

    for step in range(steps):
        action_a = agent_a(observation_a, configuration)
        action_b = agent_b(observation_b, configuration)

        result = (action_a - action_b) % 3
        if result == 1:
            score_a += 1
        elif result == 2:
            score_b += 1

        observation_a = Observation(step=step + 1, lastOpponentAction=action_b)
        observation_b = Observation(step=step + 1, lastOpponentAction=action_a)

    return score_a, score_b


def evaluate_pair(
    agent_a_factory: AgentFactory,
    agent_b_factory: AgentFactory,
    episodes: int,
    steps: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    scores_a: List[int] = []
    scores_b: List[int] = []

    for _ in range(episodes):
        np.random.seed(rng.integers(0, 2**32 - 1, dtype=np.uint32).item())
        score_a, score_b = _play_episode(agent_a_factory, agent_b_factory, steps)
        scores_a.append(score_a)
        scores_b.append(score_b)

    total_rounds = episodes * steps
    return {
        "agent_a_win_rate": sum(scores_a) / total_rounds,
        "agent_b_win_rate": sum(scores_b) / total_rounds,
        "draw_rate": 1.0 - (sum(scores_a) + sum(scores_b)) / total_rounds,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10, help="number of simulated episodes")
    parser.add_argument("--steps", type=int, default=1000, help="rounds per episode")
    parser.add_argument("--seed", type=int, default=7, help="seed controlling randomness across episodes")
    args = parser.parse_args(list(argv) if argv is not None else None)

    factories = {
        "markov": make_markov_agent,
        "decision_tree": make_decision_tree_agent,
    }

    agent_names = list(factories.keys())

    results = []
    for i, agent_a_name in enumerate(agent_names):
        for agent_b_name in agent_names[i + 1 :]:
            stats = evaluate_pair(
                factories[agent_a_name], factories[agent_b_name], args.episodes, args.steps, args.seed
            )
            results.append((agent_a_name, agent_b_name, stats))

    for agent_a_name, agent_b_name, stats in results:
        print(f"{agent_a_name} vs {agent_b_name}")
        print(f"  {agent_a_name} win rate: {stats['agent_a_win_rate']:.3f}")
        print(f"  {agent_b_name} win rate: {stats['agent_b_win_rate']:.3f}")
        print(f"  draws: {stats['draw_rate']:.3f}\n")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
