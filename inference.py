"""
inference.py — Baseline agent for the Email Triage OpenEnv environment.

Reads credentials from environment variables:
  API_BASE_URL   The LLM API endpoint  (default: https://api.openai.com/v1)
  MODEL_NAME     The model identifier   (default: gpt-4o-mini)
  HF_TOKEN       Your API key

Logs strictly follow the [START] / [STEP] / [END] format required by the
OpenEnv evaluation harness.

Usage:
  export API_BASE_URL=https://api.openai.com/v1
  export MODEL_NAME=gpt-4o-mini
  export HF_TOKEN=sk-...
  python inference.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx
from openai import OpenAI

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "---TOKEN---")

ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TASKS = ["task1", "task2", "task3"]
MAX_STEPS_PER_TASK = {"task1": 3, "task2": 4, "task3": 5}
MAX_TOTAL_REWARD = {"task1": 3.0, "task2": 4.0, "task3": 5.0}
SUCCESS_SCORE_THRESHOLD = 0.6

# ─────────────────────────────────────────────
#  Logging helpers  (required format)
# ─────────────────────────────────────────────

def log_start(*, task: str, env: str, model: str) -> None:
    print(
        json.dumps({"event": "START", "task": task, "env": env, "model": model}),
        flush=True,
    )

def log_step(*, step: int, action: Any, reward: float, done: bool, error: Any) -> None:
    print(
        json.dumps({
            "event": "STEP",
            "step": step,
            "action": str(action)[:300],
            "reward": round(reward, 4),
            "done": done,
            "error": str(error) if error else None,
        }),
        flush=True,
    )

def log_end(*, success: bool, steps: int, score: float, rewards: list[float]) -> None:
    print(
        json.dumps({
            "event": "END",
            "success": success,
            "steps": steps,
            "score": round(score, 4),
            "rewards": [round(r, 4) for r in rewards],
        }),
        flush=True,
    )

# ─────────────────────────────────────────────
#  OpenAI client
# ─────────────────────────────────────────────

def make_client() -> OpenAI:
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ─────────────────────────────────────────────
#  Agent prompt builders
# ─────────────────────────────────────────────

def build_prompt(task_id: str, obs: dict, step: int, last_reward: float, history: list[str]) -> str:
    emails_text = ""
    for e in obs.get("emails", []):
        emails_text += (
            f"\n---\nID: {e['id']}\n"
            f"Subject: {e['subject']}\n"
            f"From: {e['sender']}\n"
            f"Body: {e['body']}\n"
        )

    history_text = "\n".join(history[-3:]) if history else "None"
    feedback = obs.get("previous_feedback") or "None"

    return f"""You are an expert email triage assistant.

TASK: {task_id.upper()}
INSTRUCTION: {obs.get('instruction', '')}

EMAILS:
{emails_text}

STEP: {step}  |  LAST REWARD: {last_reward:.2f}
PREVIOUS FEEDBACK: {feedback}
RECENT HISTORY: {history_text}

Respond with ONLY valid JSON matching the instruction format. No explanation, no markdown."""


def parse_action(task_id: str, raw: str) -> dict:
    """Parse the model's raw text into the correct action dict."""
    # Strip markdown fences if present
    raw = raw.strip()
    for fence in ["```json", "```"]:
        raw = raw.replace(fence, "")
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from somewhere in the text
        import re
        match = re.search(r'[\[{].*[\]}]', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            return {}

    if task_id == "task1":
        return {"classifications": data if isinstance(data, dict) else {}}
    elif task_id == "task2":
        return {"priority_order": data if isinstance(data, list) else []}
    elif task_id == "task3":
        return {"reply": data if isinstance(data, dict) else {}}
    return {}


# ─────────────────────────────────────────────
#  HTTP helpers for the environment server
# ─────────────────────────────────────────────

async def env_reset(client: httpx.AsyncClient, task_id: str) -> dict:
    r = await client.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": 42})
    r.raise_for_status()
    return r.json()


async def env_step(client: httpx.AsyncClient, task_id: str, action: dict) -> dict:
    r = await client.post(f"{ENV_BASE_URL}/step", json={"task_id": task_id, "action": action})
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────
#  Single-task runner
# ─────────────────────────────────────────────

async def run_task(task_id: str, llm: OpenAI, http: httpx.AsyncClient) -> float:
    max_steps = MAX_STEPS_PER_TASK[task_id]
    max_total = MAX_TOTAL_REWARD[task_id]
    rewards: list[float] = []
    history: list[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env="email-triage-env", model=MODEL_NAME)

    try:
        result = await env_reset(http, task_id)
        obs = result["observation"]
        last_reward = 0.0

        for step in range(1, max_steps + 1):
            if result.get("done"):
                break

            prompt = build_prompt(task_id, obs, step, last_reward, history)

            # Call LLM
            try:
                response = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1024,
                )
                raw_action = response.choices[0].message.content or ""
            except Exception as e:
                log_step(step=step, action="", reward=0.0, done=False, error=e)
                break

            action_dict = parse_action(task_id, raw_action)

            # Step environment
            try:
                result = await env_step(http, task_id, action_dict)
            except Exception as e:
                log_step(step=step, action=raw_action, reward=0.0, done=False, error=e)
                break

            obs = result["observation"]
            reward = result.get("reward") or 0.0
            done = result.get("done", False)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            history.append(f"Step {step}: reward {reward:+.2f}")

            log_step(step=step, action=raw_action, reward=reward, done=done, error=None)

            if done:
                break

        score = sum(rewards) / max_total if max_total > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

async def main() -> None:
    llm = make_client()
    all_scores: dict[str, float] = {}

    async with httpx.AsyncClient(timeout=60.0) as http:
        # Wait for env server to be ready
        for attempt in range(30):
            try:
                r = await http.get(f"{ENV_BASE_URL}/health")
                if r.status_code == 200:
                    break
            except Exception:
                pass
            print(f"[DEBUG] Waiting for env server... attempt {attempt+1}", flush=True)
            await asyncio.sleep(2)

        for task_id in TASKS:
            print(f"\n{'='*50}", flush=True)
            print(f"[DEBUG] Running {task_id}", flush=True)
            score = await run_task(task_id, llm, http)
            all_scores[task_id] = score

    print("\n" + "="*50, flush=True)
    print(json.dumps({"event": "SUMMARY", "scores": all_scores}), flush=True)
    overall = sum(all_scores.values()) / len(all_scores)
    print(f"[DEBUG] Overall average score: {overall:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
