"""Tiny smoke test to verify baseline agents import and run a single short matchup.

This is intentionally small and fast: it runs one head-to-head pairing between two
baseline agents using the existing evaluation helpers.
"""

from __future__ import annotations

from connect4.agents import BASELINE_AGENTS
from connect4.scripts.evaluate_agents import head_to_head


def main() -> None:
    names = list(BASELINE_AGENTS)
    if len(names) < 2:
        print("Not enough agents to run smoke test.")
        return

    a, b = names[0], names[1]
    print(f"Running smoke test: {a} vs {b} (1 match)")
    score = head_to_head(BASELINE_AGENTS[a], BASELINE_AGENTS[b], matches=1)
    print(f"Average normalized score for {a} vs {b}: {score}")


if __name__ == "__main__":
    main()
