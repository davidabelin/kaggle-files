# Connect4 Baseline Agents

This repository snapshots a collection of heuristic agents that were developed for the Kaggle
[ConnectX](https://www.kaggle.com/competitions/connectx) environment.  The retained baselines live in
`connect4/agents` and are exposed via stable import names so that they can be plugged directly into
Kaggle submissions or local experiments.

## Available baseline heuristics

* **`connect4.agents.alpha_beta_v9`** – Depth-adaptive alpha-beta search tuned for the
  `test_agent_v9` baseline.

  ```python
  from connect4.agents.alpha_beta_v9 import agent

  def my_agent(obs, config):
      return agent(obs, config)
  ```

* **`connect4.agents.adaptive_midrange`** – Mid-game weighted heuristic that adjusts search depth
  dynamically.

  ```python
  from connect4.agents.adaptive_midrange import agent as adaptive_agent

  def agent(obs, config):
      return adaptive_agent(obs, config)
  ```

* **`connect4.agents.time_boxed_pruner`** – Alpha-beta search with light time boxing for Kaggle's
  execution window.

  ```python
  from connect4.agents.time_boxed_pruner import agent as pruner

  def my_agent(obs, config):
      return pruner(obs, config)
  ```

Each module exposes both `my_agent` (the historical entry point) and an alias named `agent` that
matches Kaggle's expected callable signature.  You can also import the aggregated helpers:

```python
from connect4.agents import BASELINE_AGENTS
move = BASELINE_AGENTS["alpha_beta_v9"](obs, config)
```

## Evaluating the baselines

The script `connect4/scripts/evaluate_agents.py` recreates the agent-vs-agent score matrix used for
comparison.  By default it writes the results back to `connect4/agent_vs_agent.csv`.

```bash
python -m connect4.scripts.evaluate_agents --matches 20
```

> **Note:** The script relies on the `kaggle_environments` package, which is pre-installed inside the
> official Kaggle notebooks.  In offline environments you may need to install it manually before
> running the evaluator.

## Legacy agents

Earlier heuristic iterations are preserved in `connect4/agents/legacy`.  They are no longer imported
by default but remain available for historical reference.
