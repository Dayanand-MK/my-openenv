"""
Email Triage & Response Environment
OpenEnv-compliant environment for training AI agents on real-world email handling.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Optional

from pydantic import BaseModel, Field

from email_data import EMAILS, TASK_CONFIGS
from tasks import Task1Grader, Task2Grader, Task3Grader


# ─────────────────────────────────────────────
#  Typed Models  (OpenEnv spec)
# ─────────────────────────────────────────────

class EmailItem(BaseModel):
    id: str
    subject: str
    sender: str
    body: str
    timestamp: str


class EmailTriageObservation(BaseModel):
    task_id: str = Field(description="Which task is active: task1 | task2 | task3")
    emails: list[EmailItem] = Field(description="Emails the agent must process")
    instruction: str = Field(description="Natural-language instruction for the agent")
    step_number: int = Field(description="Current step in the episode")
    max_steps: int = Field(description="Maximum steps allowed")
    previous_feedback: Optional[str] = Field(
        default=None, description="Feedback from the previous step"
    )
    done: bool = Field(default=False)


class EmailTriageAction(BaseModel):
    """
    task1 → classifications: {"email_id": "label", ...}
    task2 → priority_order: ["email_id1", "email_id2", ...]
    task3 → reply: {"email_id": "...", "subject": "...", "body": "..."}
    """
    classifications: Optional[dict[str, str]] = Field(
        default=None,
        description="Task 1: map each email_id to one of spam|urgent|billing|support|general",
    )
    priority_order: Optional[list[str]] = Field(
        default=None,
        description="Task 2: email_ids ranked most→least urgent",
    )
    reply: Optional[dict[str, str]] = Field(
        default=None,
        description="Task 3: keys email_id, subject, body",
    )


class EmailTriageReward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    breakdown: dict[str, float] = Field(default_factory=dict)
    feedback: str = ""


class StepResult(BaseModel):
    observation: EmailTriageObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
#  Environment
# ─────────────────────────────────────────────

GRADERS = {
    "task1": Task1Grader,
    "task2": Task2Grader,
    "task3": Task3Grader,
}


class EmailTriageEnv:
    """
    OpenEnv-compliant Email Triage environment.

    Three tasks:
      task1 (easy)   – classify emails into categories
      task2 (medium) – rank emails by urgency
      task3 (hard)   – draft a professional reply
    """

    def __init__(self, task_id: str = "task1", seed: int = 42):
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from {list(TASK_CONFIGS)}")
        self.task_id = task_id
        self.seed = seed
        self._rng = random.Random(seed)
        self._config = TASK_CONFIGS[task_id]
        self._grader = GRADERS[task_id]()

        # episode state
        self._step = 0
        self._max_steps: int = self._config["max_steps"]
        self._emails: list[EmailItem] = []
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._last_feedback: Optional[str] = None

    # ── public API ──────────────────────────

    def reset(self) -> StepResult:
        """Reset episode and return initial observation."""
        self._rng = random.Random(self.seed)
        self._step = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._last_feedback = None

        pool = copy.deepcopy(EMAILS)
        self._rng.shuffle(pool)
        n = self._config["num_emails"]
        self._emails = [EmailItem(**e) for e in pool[:n]]

        obs = self._make_obs()
        return StepResult(observation=obs, reward=0.0, done=False, info={})

    def step(self, action: EmailTriageAction) -> StepResult:
        """Execute one agent action and return (obs, reward, done, info)."""
        if self._done:
            obs = self._make_obs()
            return StepResult(observation=obs, reward=0.0, done=True,
                              info={"warning": "Episode already finished"})

        self._step += 1

        reward_obj: EmailTriageReward = self._grader.score(
            action=action,
            emails=self._emails,
            step=self._step,
            max_steps=self._max_steps,
        )

        self._cumulative_reward += reward_obj.value
        self._last_feedback = reward_obj.feedback

        # episode ends when max_steps reached or perfect score
        self._done = (self._step >= self._max_steps) or (reward_obj.value >= 1.0)

        obs = self._make_obs()
        return StepResult(
            observation=obs,
            reward=reward_obj.value,
            done=self._done,
            info={
                "step": self._step,
                "cumulative_reward": self._cumulative_reward,
                "breakdown": reward_obj.breakdown,
            },
        )

    def state(self) -> dict[str, Any]:
        """Return the full current state (for debugging / checkpointing)."""
        return {
            "task_id": self.task_id,
            "step": self._step,
            "max_steps": self._max_steps,
            "done": self._done,
            "cumulative_reward": self._cumulative_reward,
            "emails": [e.model_dump() for e in self._emails],
            "last_feedback": self._last_feedback,
        }

    async def close(self):
        """Async teardown hook (no-op for this env)."""
        pass

    # ── helpers ─────────────────────────────

    def _make_obs(self) -> EmailTriageObservation:
        return EmailTriageObservation(
            task_id=self.task_id,
            emails=self._emails,
            instruction=self._config["instruction"],
            step_number=self._step,
            max_steps=self._max_steps,
            previous_feedback=self._last_feedback,
            done=self._done,
        )
