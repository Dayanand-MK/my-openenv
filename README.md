---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - email
  - nlp
license: mit
---

# 📧 Email Triage OpenEnv

An [OpenEnv](https://github.com/openenv/openenv)-compliant reinforcement-learning environment
where AI agents learn to handle real-world email tasks: classification, prioritisation, and
professional reply drafting.

---

## Why this environment?

Email overload is one of the most universal productivity challenges.  
This environment lets researchers train and evaluate AI agents on the three core skills
any inbox-management system needs:

| Skill | Real-world value |
|---|---|
| Classify emails | Route to the right team / filter spam |
| Prioritise inbox | Ensure critical issues get attention first |
| Draft replies | Automate first-response for common queries |

---

## Environment description

The environment simulates an inbox with 20 realistic emails spanning billing issues,
customer complaints, spam, security alerts, and general correspondence.
Each episode presents the agent with a subset of these emails and a specific task.

---

## Action space

Actions are typed Pydantic models (`EmailTriageAction`).  
Only one field is used per task:

| Task | Field | Type | Description |
|---|---|---|---|
| task1 | `classifications` | `dict[str, str]` | Map email_id → category label |
| task2 | `priority_order` | `list[str]` | Email ids ranked most → least urgent |
| task3 | `reply` | `dict` | Keys: `email_id`, `subject`, `body` |

Valid category labels for task1: `spam` · `urgent` · `billing` · `support` · `general`

---

## Observation space

Each step returns an `EmailTriageObservation` with:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Active task: `task1` / `task2` / `task3` |
| `emails` | `list[EmailItem]` | Emails to process (id, subject, sender, body, timestamp) |
| `instruction` | `str` | Natural-language task description |
| `step_number` | `int` | Current step in the episode |
| `max_steps` | `int` | Maximum allowed steps |
| `previous_feedback` | `str` | Scoring breakdown from the previous step |
| `done` | `bool` | Whether the episode has ended |

---

## Tasks

### Task 1 — Email Classification (Easy)
- **Goal:** Label each of 8 emails as `spam`, `urgent`, `billing`, `support`, or `general`
- **Max steps:** 3
- **Reward:** Per-email accuracy with partial credit for semantically close labels
- **Baseline score:** ~0.72

### Task 2 — Inbox Prioritisation (Medium)
- **Goal:** Rank 10 emails from most → least urgent
- **Max steps:** 4
- **Reward:** Normalised Kendall tau + bonus for placing critical emails in top 3
- **Baseline score:** ~0.61

### Task 3 — Professional Reply Drafting (Hard)
- **Goal:** Identify the complaint email and write a professional reply
- **Max steps:** 5
- **Reward:** Rubric covering target selection, acknowledgement, apology, next steps, timeline, and tone
- **Baseline score:** ~0.55

---

## Reward function

Rewards are always in `[0.0, 1.0]` and are returned at **every step** (dense rewards).

- **Task 1:** Accuracy score with –0.05 step penalty per step beyond step 1
- **Task 2:** `0.7 × kendall_tau + 0.3 × top3_overlap` minus step penalty
- **Task 3:** Weighted rubric (correct target 20%, acknowledgement 20%, apology 15%, next steps 20%, timeline 10%, tone 15%) minus step penalty

---

## Setup & usage

### Prerequisites

- Python 3.10+
- Docker
- A Hugging Face account

### Local development

```bash
git clone https://huggingface.co/spaces/<your-username>/email-triage-env
cd email-triage-env

pip install -r requirements.txt

# Start the environment server
python server.py

# In another terminal, run the baseline agent
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
python inference.py
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

### OpenEnv validation

```bash
pip install openenv-core
openenv validate
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/tasks` | List all tasks with metadata |
| POST | `/reset` | Start a new episode |
| POST | `/step` | Execute one agent action |
| GET | `/state` | Inspect current episode state |

### Example: reset + step

```python
import httpx, json

BASE = "http://localhost:7860"

# Reset
r = httpx.post(f"{BASE}/reset", json={"task_id": "task1", "seed": 42})
obs = r.json()["observation"]

# Step with a classification action
action = {
    "task_id": "task1",
    "action": {
        "classifications": {
            e["id"]: "spam" for e in obs["emails"]
        }
    }
}
r = httpx.post(f"{BASE}/step", json=action)
print(r.json())
```

---

## Baseline scores

| Task | Difficulty | GPT-4o-mini score |
|---|---|---|
| task1 (classification) | Easy | 0.72 |
| task2 (prioritisation) | Medium | 0.61 |
| task3 (reply drafting) | Hard | 0.55 |

---

## Project structure

```
email-triage-env/
├── environment.py   # OpenEnv core (step/reset/state, typed models)
├── email_data.py    # Email dataset + task configuration
├── tasks.py         # Graders for all 3 tasks
├── server.py        # FastAPI HTTP server
├── inference.py     # Baseline agent script
├── openenv.yaml     # OpenEnv metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## License

MIT
