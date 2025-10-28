"""Decision-tree heuristic for Rock Paper Scissors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
from sklearn.tree import DecisionTreeClassifier


@dataclass
class DecisionTreeAgentConfig:
    """Hyper-parameters controlling the behaviour of the decision tree agent."""

    window: int = 5
    min_samples: int = 25
    random_state: int = 42


def _construct_local_features(rollouts: Dict[str, List[int]]) -> np.ndarray:
    features = np.array([[step % k for step in rollouts["steps"]] for k in (2, 3, 5)], dtype=float)
    features = np.append(features, rollouts["steps"])
    features = np.append(features, rollouts["actions"])
    features = np.append(features, rollouts["opp-actions"])
    return features


def _construct_global_features(rollouts: Dict[str, List[int]]) -> np.ndarray:
    features: List[float] = []
    for key in ("actions", "opp-actions"):
        for choice in range(3):
            actions_count = float(np.mean([r == choice for r in rollouts[key]]))
            features.append(actions_count)
    return np.array(features, dtype=float)


def _construct_features(short_stats: Dict[str, List[int]], long_stats: Dict[str, List[int]]) -> np.ndarray:
    return np.concatenate([_construct_local_features(short_stats), _construct_global_features(long_stats)])


def make_decision_tree_agent(
    config: DecisionTreeAgentConfig | None = None,
) -> Callable[["Observation", "Configuration"], int]:
    """Create a stateful decision tree agent compatible with the Kaggle API."""

    if config is None:
        config = DecisionTreeAgentConfig()

    rollouts_hist: Dict[str, List[int]] = {"steps": [], "actions": [], "opp-actions": []}
    last_move: Dict[str, int] | None = None
    data: Dict[str, List] = {"x": [], "y": []}
    test_sample: np.ndarray | None = None

    classifier = DecisionTreeClassifier(random_state=config.random_state)

    def _update_rollouts(last_action: Dict[str, int], opponent_action: int) -> None:
        rollouts_hist["steps"].append(last_action["step"])
        rollouts_hist["actions"].append(last_action["action"])
        rollouts_hist["opp-actions"].append(opponent_action)

    def agent(observation: "Observation", configuration: "Configuration") -> int:
        nonlocal last_move, test_sample

        if observation.step == 0:
            last_move = None
            test_sample = None
            rollouts_hist["steps"].clear()
            rollouts_hist["actions"].clear()
            rollouts_hist["opp-actions"].clear()
            data["x"].clear()
            data["y"].clear()

        if last_move is not None and observation.step > 0:
            _update_rollouts(last_move, observation.lastOpponentAction)

        if observation.step <= config.min_samples + config.window:
            action = int(np.random.randint(3))
            last_move = {"step": observation.step, "action": action}
            return action

        if not data["x"]:
            for i in range(len(rollouts_hist["steps"]) - config.window + 1):
                short_stats = {key: rollouts_hist[key][i : i + config.window] for key in rollouts_hist}
                long_stats = {key: rollouts_hist[key][: i + config.window] for key in rollouts_hist}
                features = _construct_features(short_stats, long_stats)
                data["x"].append(features)
            test_sample = data["x"][-1].reshape(1, -1)
            data["x"] = data["x"][:-1]
            data["y"] = rollouts_hist["opp-actions"][config.window :]
        else:
            short_stats = {key: rollouts_hist[key][-config.window :] for key in rollouts_hist}
            features = _construct_features(short_stats, rollouts_hist)
            data["x"].append(test_sample[0])
            data["y"] = rollouts_hist["opp-actions"][config.window :]
            test_sample = features.reshape(1, -1)

        classifier.fit(np.asarray(data["x"]), np.asarray(data["y"]))
        next_opp_action_pred = classifier.predict(test_sample)[0]

        action = int((next_opp_action_pred + 1) % 3)
        last_move = {"step": observation.step, "action": action}
        return action

    return agent


__all__ = ["DecisionTreeAgentConfig", "make_decision_tree_agent"]
