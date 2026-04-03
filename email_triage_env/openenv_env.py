"""
OpenEnv-compatible Python interface for the Email Triage environment.

This class provides canonical reset()/step()/state() methods and deterministic
benchmark task evaluation helpers used by inference and validation scripts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .server import app as _unused_app  # Ensures server module is loaded.
from .server.app import (
    ActionModel,
    BENCHMARK_TASKS,
    ClassifyEmailRequest,
    ObservationModel,
    ResetRequest,
    RewardModel,
    StepInfoModel,
    StepResultModel,
    get_state,
    reset_episode,
    tool_classify_email,
)


class EmailTriageOpenEnv:
    """In-process OpenEnv-style interface.

    Methods:
        - reset(...) -> ObservationModel
        - step(action) -> StepResultModel
        - state() -> Dict[str, Any]
    """

    def __init__(self) -> None:
        self._last_state: Dict[str, Any] = {}

    def reset(
        self,
        episode_length: int = 5,
        difficulty: str = "easy",
        partial_info: bool = False,
        seed: Optional[int] = None,
    ) -> ObservationModel:
        result = reset_episode(
            ResetRequest(
                episode_length=episode_length,
                difficulty=difficulty,
                partial_info=partial_info,
                seed=seed,
            )
        )
        self._last_state = result.get("state", {})
        return ObservationModel(**result["observation"])

    def step(self, action: ActionModel) -> StepResultModel:
        raw = tool_classify_email(ClassifyEmailRequest(**action.model_dump()))
        next_obs = raw.get("next_email")
        reward = RewardModel(
            reward=raw["reward"],
            done=raw["done"],
            step=raw["step"],
            total_reward=raw["total_reward"],
            field_breakdown=raw["field_breakdown"],
        )
        self._last_state = get_state()
        return StepResultModel(
            observation=ObservationModel(**next_obs) if next_obs else None,
            reward=reward,
            done=reward.done,
            info=StepInfoModel(
                difficulty=self._last_state.get("difficulty", "easy"),
                reveals_used=0,
            ),
        )

    def state(self) -> Dict[str, Any]:
        self._last_state = get_state()
        return self._last_state


class DeterministicTaskGrader:
    """Deterministic grader helper for benchmark tasks."""

    def __init__(self, env: Optional[EmailTriageOpenEnv] = None) -> None:
        self.env = env or EmailTriageOpenEnv()

    def list_tasks(self) -> List[Dict[str, Any]]:
        return BENCHMARK_TASKS

    def grade_task(self, task_id: str, actions: List[ActionModel]) -> float:
        task = next((t for t in BENCHMARK_TASKS if t["task_id"] == task_id), None)
        if task is None:
            raise ValueError(f"Unknown task_id: {task_id}")

        self.env.reset(
            episode_length=task["episode_length"],
            difficulty=task["difficulty"],
            partial_info=task["partial_info"],
            seed=task["seed"],
        )

        rewards: List[float] = []
        for action in actions[: task["episode_length"]]:
            result = self.env.step(action)
            rewards.append(result.reward.reward)
            if result.done:
                break

        if not rewards:
            return 0.0

        score = sum(rewards) / float(task["episode_length"])
        return max(0.0, min(1.0, round(score, 4)))
