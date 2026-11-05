"""Small, evidence-aware presentation choices for the research interface."""

from __future__ import annotations

import re

from ui_data import DEMO_QUESTIONS


MODEL_NAME = re.compile(
    r"\b(?:m30t|matrice\s*30t?|m4t|matrice\s*4t?|evo\s*max\s*4n|matrice\s*210|m210)\b",
    re.IGNORECASE,
)
MANUAL_TOPIC = re.compile(
    r"\b(?:failsafe|return.to.home|rth|obstacle|signal.lost.action|set\s+to|setting|manual|flight.mode)\b",
    re.IGNORECASE,
)
CROSS_SOURCE = re.compile(
    r"\b(?:faa|regulat\w*|incident\w*|report\w*|account\w*|compare\w*|across|conflict\w*|evidence)\b",
    re.IGNORECASE,
)


def source_scope(question: str) -> tuple[str, ...] | None:
    """Keep a clearly manual-focused question narrow; compare all tiers otherwise."""
    if MODEL_NAME.search(question) and MANUAL_TOPIC.search(question) and not CROSS_SOURCE.search(question):
        return ("manual",)
    return None


def contextualize_question(question: str, previous_turn: dict | None) -> str:
    """Carry a single known model into a short follow-up without rewriting the user's words."""
    if not previous_turn or MODEL_NAME.search(question):
        return question
    models = previous_turn.get("result", {}).get("intake", {}).get("models", [])
    followup = re.match(r"(?i)^\s*(?:what about|and |why |when |does it|can it|how about)", question)
    if len(models) == 1 and followup and len(question.split()) <= 28:
        return f"For the {models[0]}, {question[0].lower() + question[1:]}"
    return question


RECORDED_ANSWERS = {
    DEMO_QUESTIONS[0]: (
        "The Autel manual says Return to Home can start after a lost link when that action is configured. "
        "An unverified incident account says RTH failed. The available documents do not show which settings "
        "and conditions applied in that event, so they do not establish that the drone will always return home."
    ),
    DEMO_QUESTIONS[1]: (
        "The available documents do not establish a lost-link failure rate for the DJI Matrice 4T. "
        "A rate needs both a verified failure count and total flight hours; this collection does not provide them."
    ),
    DEMO_QUESTIONS[2]: (
        "The DJI Matrice 30T can be set to return home, land, or hover when the remote controller signal is lost. "
        "If Return to Home is selected, the manual describes Failsafe RTH starting after more than six seconds "
        "of lost signal when the Home Point was recorded and the compass is working normally."
    ),
}


def answer_text(run: dict, *, recorded: bool = False) -> str:
    """Put the question's answer first without inventing claims or making another model call."""
    if recorded and run.get("question") in RECORDED_ANSWERS:
        return RECORDED_ANSWERS[run["question"]]
    question = run.get("question", "").lower()
    summary = (run.get("brief", {}).get("short_answer") or "").strip()
    if not summary:
        return "The available sources did not support an answer to this question."
    gaps = " ".join(g.get("text", "") for g in run.get("brief", {}).get("gaps", [])).lower()
    if ("failure rate" in question or "per 1,000 flight hours" in question) and (
        "no" in gaps and ("rate" in gaps or "flight hour" in gaps)
    ):
        return ("The available documents do not establish a lost-link failure rate for this model. "
                "A rate needs both a verified failure count and total flight hours; this collection does not provide them.")
    sentences = re.split(r"(?<=[.!?])\s+", summary)
    if source_scope(run.get("question", "")) == ("manual",):
        for index, sentence in enumerate(sentences):
            if "manual" in sentence.lower() or MODEL_NAME.search(sentence):
                sentences = sentences[index:]
                break
    relevant = [s for s in sentences if not s.lower().startswith(("faa guidance", "from faa guidance", "also per ac"))
                and not s.lower().startswith(("gaps", "the main gap"))]
    if not any(word in question for word in ("incident", "report", "evidence", "compare", "conflict")):
        relevant = [s for s in relevant if not s.lower().startswith(("incident reports", "unverified incident"))]
    answer = " ".join(relevant[:3]).strip()
    return answer or summary


def review_note(run: dict) -> str | None:
    """Explain the workflow status using the decisions saved in the trace."""
    if run.get("status") != "human_review_required":
        return None
    drops = [d for d in run.get("decisions", []) if d.get("action") == "drop"]
    if drops:
        question = run.get("question", "").lower()
        claims = [c.get("text", "").lower() for section in run.get("brief", {}).get("sections", [])
                  for c in section.get("claims", [])]
        unresolved = " The obstacle-avoidance part of your question remains unresolved." if (
            "obstacle" in question and not any("obstacle" in c for c in claims)
        ) else ""
        conflicts = [g.get("text", "") for g in run.get("brief", {}).get("gaps", [])
                     if g.get("kind") == "conflict"] if source_scope(run.get("question", "")) is None else []
        conflict_note = f" Open issue: {conflicts[0]}" if conflicts else ""
        return (f"{len(drops)} drafted statement(s) did not pass the source checks after the allowed revisions, "
                f"so they were left out of the answer.{unresolved}{conflict_note} Check the relevant cited "
                "passages before treating the answer as settled.")
    return ("The sources leave a material conflict or unsupported claim. Check the cited passages "
            "before treating the answer as settled.")


def followup_question(run: dict) -> str | None:
    """Offer a narrower, standalone question only where it clarifies a real omission."""
    question = run.get("question", "").lower()
    if run.get("status") == "human_review_required" and "obstacle" in question:
        models = run.get("intake", {}).get("models", [])
        if len(models) == 1:
            return (f"What does the {models[0]} manual say about obstacle avoidance "
                    "during Failsafe Return to Home in low light?")
    return None
