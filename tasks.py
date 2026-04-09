"""
Graders for the three Email Triage tasks.

Each grader implements:
    score(action, emails, step, max_steps) -> EmailTriageReward

Scores are always in [0.0, 1.0].
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from environment import EmailTriageAction, EmailTriageReward
    from email_data import EmailItem


# ─────────────────────────────────────────────
#  Shared reward model (local import to avoid
#  circular imports — graders are imported by
#  environment.py which defines EmailTriageReward)
# ─────────────────────────────────────────────

class _Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    breakdown: dict[str, float] = Field(default_factory=dict)
    feedback: str = ""


# ─────────────────────────────────────────────
#  TASK 1 — Email Classification  (Easy)
# ─────────────────────────────────────────────

class Task1Grader:
    """
    Scores how accurately the agent classifies each email.

    Partial credit:
      • 1.0 per correct label  (full credit)
      • 0.5 for semantically close mis-labels (e.g. urgent vs support)
      • 0.0 for wrong label
    Penalty:
      • –0.1 for every step wasted beyond step 1 (incentivises getting it right first time)
    """

    CLOSE_PAIRS: set[frozenset] = {
        frozenset({"urgent", "support"}),
        frozenset({"billing", "support"}),
        frozenset({"general", "support"}),
    }

    def score(self, action, emails, step: int, max_steps: int) -> _Reward:
        from email_data import EMAILS

        # Build ground-truth map
        gt: dict[str, str] = {e["id"]: e["category"] for e in EMAILS}

        if action.classifications is None:
            return _Reward(
                value=0.0,
                feedback="No classifications provided. Return a dict of {email_id: category}.",
            )

        total = len(emails)
        correct = 0.0
        details: list[str] = []

        for email in emails:
            eid = email.id
            predicted = (action.classifications.get(eid) or "").strip().lower()
            truth = gt.get(eid, "")

            if predicted == truth:
                correct += 1.0
                details.append(f"{eid}: ✓ {predicted}")
            elif frozenset({predicted, truth}) in self.CLOSE_PAIRS:
                correct += 0.5
                details.append(f"{eid}: ~ {predicted} (expected {truth})")
            else:
                details.append(f"{eid}: ✗ {predicted} (expected {truth})")

        raw = correct / total if total > 0 else 0.0

        # Step penalty: –0.05 per extra step
        step_penalty = max(0.0, (step - 1) * 0.05)
        value = max(0.0, round(raw - step_penalty, 4))

        return _Reward(
            value=value,
            breakdown={"accuracy": raw, "step_penalty": -step_penalty},
            feedback="\n".join(details),
        )


# ─────────────────────────────────────────────
#  TASK 2 — Inbox Prioritisation  (Medium)
# ─────────────────────────────────────────────

class Task2Grader:
    """
    Scores the quality of an urgency ranking using normalised Kendall tau distance.

    A perfect ranking gets 1.0.  A reversed ranking gets ~0.0.
    Partial credit is awarded for every correctly-ordered adjacent pair.

    Additionally rewards placing all urgency-5 emails in the top 3.
    """

    def score(self, action, emails, step: int, max_steps: int) -> _Reward:
        from email_data import EMAILS

        gt_urgency: dict[str, int] = {e["id"]: e["urgency"] for e in EMAILS}
        email_ids = [e.id for e in emails]

        if action.priority_order is None:
            return _Reward(
                value=0.0,
                feedback="No priority_order provided. Return a list of email_ids ranked by urgency.",
            )

        predicted = [eid for eid in action.priority_order if eid in set(email_ids)]
        # If some ids are missing, append them at the end in arbitrary order
        missing = [eid for eid in email_ids if eid not in predicted]
        predicted += missing

        # Ground-truth ranking (stable sort by urgency desc)
        ideal = sorted(email_ids, key=lambda x: gt_urgency.get(x, 0), reverse=True)

        # Normalised Kendall tau
        n = len(predicted)
        concordant = 0
        total_pairs = n * (n - 1) / 2

        pos_pred = {eid: i for i, eid in enumerate(predicted)}
        pos_ideal = {eid: i for i, eid in enumerate(ideal)}

        for i in range(n):
            for j in range(i + 1, n):
                a, b = ideal[i], ideal[j]
                if pos_pred[a] < pos_pred[b]:
                    concordant += 1

        tau = concordant / total_pairs if total_pairs > 0 else 1.0

        # Bonus: top-3 contains the highest-urgency emails
        top3_predicted = set(predicted[:3])
        top3_ideal = set(ideal[:3])
        top3_overlap = len(top3_predicted & top3_ideal) / 3

        # Step penalty
        step_penalty = max(0.0, (step - 1) * 0.04)
        raw = 0.7 * tau + 0.3 * top3_overlap
        value = max(0.0, round(raw - step_penalty, 4))

        return _Reward(
            value=value,
            breakdown={"kendall_tau": tau, "top3_overlap": top3_overlap, "step_penalty": -step_penalty},
            feedback=(
                f"Kendall tau: {tau:.2f}  Top-3 overlap: {top3_overlap:.2f}\n"
                f"Your order:  {predicted}\n"
                f"Ideal order: {ideal}"
            ),
        )


# ─────────────────────────────────────────────
#  TASK 3 — Professional Reply Drafting  (Hard)
# ─────────────────────────────────────────────

class Task3Grader:
    """
    Holistic rubric for a customer-support reply.

    Dimensions (each 0–1, then weighted):
      1. Correct target (0.2) – replied to an actual complaint email
      2. Acknowledgement (0.2) – mentions the customer's problem
      3. Apology (0.15)        – contains a genuine apology
      4. Next steps (0.2)      – explains what will happen
      5. Timeline (0.1)        – gives a concrete timeframe
      6. Tone (0.15)           – professional, empathetic language
    """

    APOLOGY_PHRASES = [
        "sorry", "apologise", "apologize", "apologies", "regret",
        "sincerely sorry", "deeply sorry",
    ]
    NEXT_STEP_PHRASES = [
        "will", "shall", "team", "investigate", "escalate", "refund",
        "resolve", "fix", "contact", "reach out", "follow up", "looking into",
    ]
    TIMELINE_PHRASES = [
        "24 hours", "48 hours", "business day", "within", "by", "shortly",
        "immediately", "asap", "as soon as", "today",
    ]
    NEGATIVE_PHRASES = [
        "unfortunately nothing", "cannot help", "not our problem",
        "your fault", "deal with it",
    ]

    def score(self, action, emails, step: int, max_steps: int) -> _Reward:
        from email_data import TASK_CONFIGS

        complaint_ids: set[str] = TASK_CONFIGS["task3"]["complaint_ids"]

        if action.reply is None:
            return _Reward(
                value=0.0,
                feedback="No reply provided. Return {email_id, subject, body}.",
            )

        email_id = (action.reply.get("email_id") or "").strip()
        subject = (action.reply.get("subject") or "").strip()
        body = (action.reply.get("body") or "").strip().lower()

        # 1. Correct target
        target_score = 1.0 if email_id in complaint_ids else 0.0

        # 2. Acknowledgement: body mentions key complaint words
        ack_keywords = ["order", "charge", "payment", "crash", "issue", "problem",
                        "delay", "refund", "error", "incident"]
        ack_score = min(1.0, sum(kw in body for kw in ack_keywords) / 2)

        # 3. Apology
        apology_score = 1.0 if any(p in body for p in self.APOLOGY_PHRASES) else 0.0

        # 4. Next steps
        next_step_score = min(1.0, sum(p in body for p in self.NEXT_STEP_PHRASES) / 3)

        # 5. Timeline
        timeline_score = 1.0 if any(p in body for p in self.TIMELINE_PHRASES) else 0.0

        # 6. Tone (penalise negative phrases, reward length as proxy for effort)
        negative_penalty = 0.3 * sum(p in body for p in self.NEGATIVE_PHRASES)
        length_bonus = min(0.5, len(body.split()) / 100)  # up to 0.5 for 100+ word reply
        has_subject = 1.0 if subject.lower().startswith("re:") else 0.5
        tone_score = max(0.0, min(1.0, 0.5 * has_subject + length_bonus - negative_penalty))

        # Weighted combination
        weights = {
            "correct_target": 0.20,
            "acknowledgement": 0.20,
            "apology": 0.15,
            "next_steps": 0.20,
            "timeline": 0.10,
            "tone": 0.15,
        }
        scores = {
            "correct_target": target_score,
            "acknowledgement": ack_score,
            "apology": apology_score,
            "next_steps": next_step_score,
            "timeline": timeline_score,
            "tone": tone_score,
        }

        raw = sum(weights[k] * scores[k] for k in weights)

        # Step penalty
        step_penalty = max(0.0, (step - 1) * 0.03)
        value = max(0.0, round(raw - step_penalty, 4))

        feedback_lines = [f"  {k}: {v:.2f} (weight {weights[k]:.2f})" for k, v in scores.items()]
        return _Reward(
            value=value,
            breakdown={**scores, "step_penalty": -step_penalty},
            feedback="Dimension scores:\n" + "\n".join(feedback_lines),
        )
