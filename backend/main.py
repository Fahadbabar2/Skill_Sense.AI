"""
FastAPI backend for SkillSense AI.

Endpoints:
    POST /start                 -> begin a session, get first question
    POST /submit_answer         -> submit an answer, get next question (or finish)
    GET  /report/{session_id}   -> full report with ML skill prediction

Run with:
    uvicorn backend.main:app --reload --port 8000
(run from the project root so the `backend` package resolves correctly)
"""

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend import adaptive_engine as engine

app = FastAPI(title="SkillSense AI")

# Streamlit runs on a different port locally, so CORS needs to be open for the demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent / "models" / "skill_model.pkl"
LABEL_ENCODER_PATH = Path(__file__).parent / "models" / "label_encoder.pkl"

FEATURE_ORDER = [
    "overall_accuracy",
    "easy_accuracy",
    "medium_accuracy",
    "hard_accuracy",
    "avg_time_taken",
    "avg_quality_score",
    "total_hints_used",
    "hard_questions_attempted",
    "claimed_experience_years",
    "claimed_skill_count",
]

_model = None
_label_encoder = None


def _load_model():
    global _model, _label_encoder
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail="Model not trained yet. Run ml/generate_synthetic_data.py then ml/train_model.py first.",
            )
        _model = joblib.load(MODEL_PATH)
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return _model, _label_encoder


class AnswerSubmission(BaseModel):
    session_id: str
    answer_text: str
    hints_used: int = 0


@app.post("/start")
def start():
    session_id, question = engine.start_session()
    return {"session_id": session_id, "question": question}


@app.post("/submit_answer")
def submit_answer(payload: AnswerSubmission):
    if engine.get_session(payload.session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /start first.")
    result = engine.submit_answer(payload.session_id, payload.answer_text, payload.hints_used)
    return result


@app.get("/report/{session_id}")
def report(
    session_id: str,
    resume_claim: str = "Not provided",
    claimed_experience_years: float = 0,
    claimed_skill_count: int = 0,
):
    session = engine.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    features, history = engine.get_feature_vector(session_id)
    features["claimed_experience_years"] = claimed_experience_years
    features["claimed_skill_count"] = claimed_skill_count
    model, label_encoder = _load_model()

    x = np.array([[features[f] for f in FEATURE_ORDER]])
    pred_idx = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    predicted_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = round(float(max(proba)) * 100, 1)

    # simple per-category strengths/weaknesses from raw history
    by_category = {}
    for h in history:
        cat = h["category"]
        by_category.setdefault(cat, []).append(h["correct"])
    category_scores = {
        cat: round(sum(flags) / len(flags) * 100, 1) for cat, flags in by_category.items()
    }
    strengths = sorted(category_scores, key=category_scores.get, reverse=True)[:3]
    weaknesses = sorted(category_scores, key=category_scores.get)[:3]

    return {
        "resume_claim": resume_claim,
        "ai_estimated_level": predicted_label,
        "confidence_pct": confidence,
        "overall_accuracy_pct": round(features["overall_accuracy"] * 100, 1),
        "category_scores": category_scores,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "questions_answered": len(history),
        "raw_features": features,
    }
