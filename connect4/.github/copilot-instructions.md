## How to be productive in this Connect4 (ConnectX) repo

This file gives concise, actionable guidance so an AI coding agent (Copilot-style) can contribute safely and productively.

Key facts
- Package root: `connect4/` — code expects imports like `from connect4.agents import ...`.
- Agents live in `connect4/agents/`. Each agent module exposes `my_agent` and sets `agent = my_agent` so it matches Kaggle's expected signature `(obs, config) -> int`.
- Aggregated registry: `connect4/agents/__init__.py` exports `BASELINE_AGENTS` (map of name -> callable). Use this for evaluation and experiments.

Architecture / data flow (short)
- Agents are pure functions that accept the Kaggle `obs` and `config` objects and return a column index.
- The `scripts/evaluate_agents.py` uses `kaggle_environments.make('connectx')` to run head-to-head matches using the `BASELINE_AGENTS` registry and writes `agent_vs_agent.csv` by default.
- Message passing: the project relies on the external `kaggle_environments` package for simulated environment objects (obs, config, env.run()).

Project-specific conventions
- Entrypoints: prefer the `agent` name (alias to `my_agent`) for Kaggle compatibility. When adding agents follow the pattern in existing modules (`my_agent` implementation, then `agent = my_agent` and `__all__ = [...]`).
- Heuristic files place tunable constants at top (A,B,C...) and provide helper functions `drop_piece`, `get_score`/`get_heuristic`, and `alphabeta`/`score_move`. Reuse these helpers when adding new heuristic agents.
- Tie-breaking: many agents prefer center-first moves via preference lists (e.g. `[3,4,2]` in existing code). Keep similar tie-break behavior when extending agents for comparable results.
- Time handling: `time_boxed_pruner.py` reads `config.get('actTimeout', ...)` and exposes optional `START_TIME` and `cutoff_time` parameters — tests or local calls can override these to simulate Kaggle timings.

Developer workflows and commands
- Run the evaluator locally (requires `kaggle_environments` and `numpy`):
  - Windows (cmd): `python -m connect4.scripts.evaluate_agents --matches 20`
  - Output file: `connect4/agent_vs_agent.csv` by default (can pass `--output PATH`).
- Dependencies: install at least `numpy` and `kaggle_environments` to run evaluation and play agents locally. If adding CI, ensure these are installed before tests.

Patterns & examples (copyable)
- Importing a baseline in a new script:

  ```python
  from connect4.agents.alpha_beta_v9 import agent

  def my_agent(obs, config):
      return agent(obs, config)
  ```

- Evaluating matchups programmatically (use `BASELINE_AGENTS`):

  ```python
  from connect4.agents import BASELINE_AGENTS
  from connect4.scripts.evaluate_agents import evaluate_agents

  results = evaluate_agents(BASELINE_AGENTS, matches=10)
  ```

What to avoid / assumptions
- Do not assume Kaggle-only runtime: some `config` keys such as `actTimeout` may be missing in offline runs. Use safe fallbacks (see `time_boxed_pruner.py`).
- Avoid changing the public `BASELINE_AGENTS` mapping shape or agent signature — other scripts depend on the callable shape (obs, config) -> int.

Where to look for implementation examples
- `connect4/agents/alpha_beta_v9.py` — simple alphabeta + heuristic; good example for adding a small search-based agent.
- `connect4/agents/time_boxed_pruner.py` — shows time-boxed search and how to read `config` actTimeout.
- `connect4/agents/__init__.py` — registry and exported names.
- `connect4/scripts/evaluate_agents.py` — evaluation harness and CSV output.

If anything is unclear
- Ask for which part of the agent lifecycle you need (writing new agent, running evaluation, benchmarking). Provide the file you intend to edit and a brief goal (e.g., "add Monte Carlo rollout agent that uses BASELINE_AGENTS for comparison").

Please review this guidance and tell me if you'd like more examples, CI-oriented instructions (requirements), or a short bench harness for new agents.
If you want quick verification (recommended), the repository includes a tiny smoke-test and a CI example that run one short matchup between two baseline agents.

Quick local smoke test
- Create a virtual environment, install requirements, then run the smoke test module. Example (Windows cmd):

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m connect4.scripts.smoke_test
```

CI example (GitHub Actions)
- A small workflow is provided at `.github/workflows/ci.yml`. It installs Python, installs `-r requirements.txt`, and runs the smoke test to catch import/runtime issues early.

Files added by this guidance
- `requirements.txt` — pins minimal runtime deps (`numpy`, `kaggle_environments`).
- `connect4/scripts/smoke_test.py` — small runner that imports `BASELINE_AGENTS` and plays a single head-to-head game to ensure imports and env calls work.
- `.github/workflows/ci.yml` — example CI job that runs the smoke test on push/PR.

If you'd like, I can shorten the steps further or add explicit pinned versions in `requirements.txt`.
