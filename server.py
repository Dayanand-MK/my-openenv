"""
FastAPI server — exposes the Email Triage environment via HTTP.
Endpoints:
  POST /reset          – start a new episode
  POST /step           – execute one action
  GET  /state          – inspect current state
  GET  /health         – liveness probe
  GET  /tasks          – list available tasks
"""

from __future__ import annotations

import os
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import EmailTriageAction, EmailTriageEnv, StepResult

app = FastAPI(
    title="Email Triage OpenEnv",
    description="An OpenEnv environment for training AI agents on email triage tasks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One global env per task_id (simple; suitable for single-agent eval)
_envs: dict[str, EmailTriageEnv] = {}


def _get_env(task_id: str) -> EmailTriageEnv:
    if task_id not in _envs:
        _envs[task_id] = EmailTriageEnv(task_id=task_id)
    return _envs[task_id]


# ── Request / Response schemas ───────────────

class ResetRequest(BaseModel):
    task_id: str = "task1"
    seed: int = 42


class StepRequest(BaseModel):
    task_id: str = "task1"
    action: EmailTriageAction


# ── Endpoints ────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "env": "email-triage-env", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    from email_data import TASK_CONFIGS
    return {
        tid: {
            "name": cfg["name"],
            "difficulty": cfg["difficulty"],
            "num_emails": cfg["num_emails"],
            "max_steps": cfg["max_steps"],
        }
        for tid, cfg in TASK_CONFIGS.items()
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None) -> StepResult:
    try:
        if req is None:
            req = ResetRequest()
        env = EmailTriageEnv(task_id=req.task_id, seed=req.seed)
        _envs[req.task_id] = env
        return env.reset()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest) -> StepResult:
    env = _get_env(req.task_id)
    return env.step(req.action)


@app.get("/state")
def state(task_id: str = Query(default="task1")) -> dict[str, Any]:
    env = _get_env(task_id)
    return env.state()


def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)

# This allows openenv to call main() directly
if __name__ == "__main__":
    main()
