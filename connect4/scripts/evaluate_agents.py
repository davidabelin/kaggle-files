"""Evaluate the bundled ConnectX baseline agents head-to-head."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

from kaggle_environments import make

from connect4.agents import BASELINE_AGENTS, Agent

DEFAULT_MATCHES = 20


def normalise_reward(reward: float) -> float:
    """Convert the [-1, 1] ConnectX reward into a [0, 1] score."""
    return (reward + 1.0) / 2.0


def head_to_head(agent_a: Agent, agent_b: Agent, matches: int) -> float:
    """Play ``matches`` games and return the average score for ``agent_a``."""
    if matches <= 0:
        raise ValueError("matches must be a positive integer")

    env = make("connectx", debug=False)
    scores: List[float] = []

    for match in range(matches):
        env.reset()
        if match % 2 == 0:
            env.run([agent_a, agent_b])
            reward = env.state[0].reward
        else:
            env.run([agent_b, agent_a])
            reward = env.state[1].reward

        # Invalid moves return None; treat as a hard loss.
        if reward is None:
            reward = -1.0
        scores.append(normalise_reward(reward))

    return sum(scores) / len(scores)


def evaluate_agents(agents: Dict[str, Agent], matches: int) -> Dict[str, Dict[str, float]]:
    """Return a symmetric lookup table of average scores."""
    names = list(agents)
    results: Dict[str, Dict[str, float]] = {name: {} for name in names}

    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if i == j:
                results[left][right] = 0.0
            elif right in results and left in results[right]:
                results[left][right] = 1.0 - results[right][left]
            else:
                results[left][right] = round(head_to_head(agents[left], agents[right], matches), 3)

    return results


def write_csv(results: Dict[str, Dict[str, float]], path: Path) -> None:
    """Serialise the matchup table to ``path``."""
    names = list(results)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([""] + names)
        for name in names:
            writer.writerow([name] + [results[name][opponent] for opponent in names])


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matches",
        type=int,
        default=DEFAULT_MATCHES,
        help="number of head-to-head games to run per pairing (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "agent_vs_agent.csv",
        help="target CSV file to update (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> Path:
    args = parse_args(argv)
    results = evaluate_agents(BASELINE_AGENTS, args.matches)
    write_csv(results, args.output)
    return args.output


if __name__ == "__main__":
    main()
