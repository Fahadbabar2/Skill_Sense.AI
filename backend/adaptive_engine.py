"""
Adaptive Question Engine
-------------------------
Implements a simple version of Computerized Adaptive Testing (CAT):
- Start at medium difficulty.
- Correct answer -> next question is harder.
- Wrong answer -> next question is easier.
- Never repeats a question already asked.
- Session ends after a fixed number of questions or when the question bank
  for a difficulty is exhausted.

This is deliberately rule-based (no ML) for the question *selection* step.
The ML model is only used afterward, once, to predict overall skill level
from the collected feature vector. Keeping these separate is important:
CAT logic must be fast and deterministic, while skill prediction benefits
from a trained model on richer features.
"""

import json
import random
import re
import time
import uuid
from pathlib import Path

QUESTIONS_PATH = Path(__file__).parent.parent / "data" / "questions.json"
DIFFICULTY_ORDER = ["easy", "medium", "hard"]

with open(QUESTIONS_PATH) as f:
    ALL_QUESTIONS = json.load(f)

QUESTIONS_BY_DIFFICULTY = {
    d: [q for q in ALL_QUESTIONS if q["difficulty"] == d] for d in DIFFICULTY_ORDER
}

# In-memory session store. Fine for a portfolio project / single-user demo.
# Swap for Redis or a DB table if this ever needs to survive a server restart
# or serve concurrent users reliably.
SESSIONS = {}

MAX_QUESTIONS = 8


def _pick_question(difficulty, asked_ids):
    pool = [q for q in QUESTIONS_BY_DIFFICULTY[difficulty] if q["id"] not in asked_ids]
    if not pool:
        # fall back to an adjacent difficulty if this one is exhausted
        for alt in DIFFICULTY_ORDER:
            pool = [q for q in QUESTIONS_BY_DIFFICULTY[alt] if q["id"] not in asked_ids]
            if pool:
                break
    if not pool:
        return None
    return random.choice(pool)


def start_session():
    session_id = str(uuid.uuid4())
    first_q = _pick_question("medium", set())
    SESSIONS[session_id] = {
        "asked_ids": set(),
        "current_difficulty": "medium",
        "history": [],
        "question_start_time": time.time(),
        "current_question": first_q,
    }
    SESSIONS[session_id]["asked_ids"].add(first_q["id"])
    return session_id, _public_question(first_q)


def _public_question(q):
    """Strip answer keys before sending to the client."""
    public = {"id": q["id"], "difficulty": q["difficulty"], "type": q["type"], "question": q["question"]}
    if q["type"] == "mcq":
        public["options"] = q["options"]
    return public


def _score_answer(question, answer_text):
    """
    Very lightweight scoring:
    - MCQ: exact match.
    - short_answer / code: keyword overlap as a stand-in for NLP semantic
      similarity (swap this function for a Sentence-Transformers cosine
      similarity call later without touching anything else).
    """
    if question["type"] == "mcq":
        correct = answer_text.strip().lower() == question["answer"].strip().lower()
        return correct, 1.0 if correct else 0.0

    text = answer_text.lower()
    keywords = question.get("keywords", [])
    if not keywords:
        return True, 1.0
    hits = sum(1 for k in keywords if re.search(re.escape(k.lower()), text))
    ratio = hits / len(keywords)
    correct = ratio >= 0.35  # threshold: needs roughly a third of expected concepts present
    return correct, round(ratio, 2)


def submit_answer(session_id, answer_text, hints_used=0):
    session = SESSIONS[session_id]
    q = session["current_question"]
    time_taken = round(time.time() - session["question_start_time"], 1)

    correct, quality_score = _score_answer(q, answer_text)

    session["history"].append({
        "question_id": q["id"],
        "difficulty": q["difficulty"],
        "category": q["category"],
        "correct": correct,
        "quality_score": quality_score,
        "time_taken": time_taken,
        "hints_used": hints_used,
    })

    # adapt difficulty
    idx = DIFFICULTY_ORDER.index(session["current_difficulty"])
    if correct and idx < len(DIFFICULTY_ORDER) - 1:
        idx += 1
    elif not correct and idx > 0:
        idx -= 1
    session["current_difficulty"] = DIFFICULTY_ORDER[idx]

    finished = len(session["history"]) >= MAX_QUESTIONS
    next_q_public = None
    if not finished:
        next_q = _pick_question(session["current_difficulty"], session["asked_ids"])
        if next_q is None:
            finished = True
        else:
            session["asked_ids"].add(next_q["id"])
            session["current_question"] = next_q
            session["question_start_time"] = time.time()
            next_q_public = _public_question(next_q)

    return {
        "correct": correct,
        "quality_score": quality_score,
        "finished": finished,
        "next_question": next_q_public,
    }


def get_feature_vector(session_id):
    """
    Aggregate raw per-question history into the feature vector the ML
    model expects. This is the bridge between the adaptive engine and
    ml/train_model.py -- keep the feature names/order in sync with that file.
    """
    history = SESSIONS[session_id]["history"]
    if not history:
        raise ValueError("No answers submitted yet.")

    n = len(history)
    correct_flags = [h["correct"] for h in history]
    times = [h["time_taken"] for h in history]

    easy = [h for h in history if h["difficulty"] == "easy"]
    medium = [h for h in history if h["difficulty"] == "medium"]
    hard = [h for h in history if h["difficulty"] == "hard"]

    def acc(subset):
        return sum(1 for h in subset if h["correct"]) / len(subset) if subset else 0.0

    features = {
        "overall_accuracy": sum(correct_flags) / n,
        "easy_accuracy": acc(easy),
        "medium_accuracy": acc(medium),
        "hard_accuracy": acc(hard),
        "avg_time_taken": sum(times) / n,
        "avg_quality_score": sum(h["quality_score"] for h in history) / n,
        "total_hints_used": sum(h["hints_used"] for h in history),
        "hard_questions_attempted": len(hard),
    }
    return features, history


def get_session(session_id):
    return SESSIONS.get(session_id)
