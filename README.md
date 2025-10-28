# Rock Paper Scissors Heuristics

This repository collects a pair of battle-tested heuristic agents for the Kaggle
Rock Paper Scissors environment and tooling for evaluating and training future
learning-based opponents.

## Project layout

```
rps/
├── agents/
│   ├── decision_tree_classifier.py
│   └── markov_agent.py
└── scripts/
    └── evaluate_agents.py
training/
└── rps_dataset_stub.py
```

## Running the bundled agents head-to-head

The evaluation script simulates multiple Kaggle-style matches between the
heuristics and summarises their win rates. The command below runs ten episodes of
1,000 rounds each and prints the aggregated results:

```bash
python -m rps.scripts.evaluate_agents --episodes 10 --steps 1000
```

Use the `--episodes`, `--steps`, and `--seed` flags to customise the experiment.
The script exposes the `Observation` and `Configuration` dataclasses expected by
Kaggle so custom agents can be benchmarked by adding them to the `factories`
dictionary in `evaluate_agents.py`.

## Integrating a learning agent

Both heuristic factories, `make_markov_agent` and `make_decision_tree_agent`,
return callables compatible with Kaggle's Rock Paper Scissors API. A learning
agent can be evaluated against them by implementing the same callable signature
and extending the evaluation script:

1. Implement a factory function that returns your agent callable.
2. Import the factory into `rps/scripts/evaluate_agents.py` and register it in
   the `factories` dictionary.
3. Rerun the evaluation command to compare the agent with the heuristics.

## Logging training data

The `training/rps_dataset_stub.py` module documents a simple schema for logging
match histories. The `MatchRecord` and `MatchStep` dataclasses capture the
sequence of actions and rewards for a match, while `iter_training_examples`
converts them into dictionaries that can be ingested by supervised learning
pipelines. Implementations of `MatchLogger` are expected to persist the records
for downstream training jobs.
