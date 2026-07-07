# SkillSense AI (MVP)

Adaptive Python skill assessment: the system asks harder or easier questions
based on how you're doing, then uses a trained ML model to predict your
actual skill level from your answer *behavior* — not your resume.

This is a scoped-down, weekend-buildable version of a larger concept
(see `PROJECT_BRIEF.md` if you kept the original doc). It's built to be
honest about what's real vs simplified, which matters more for a portfolio
than fake completeness.

## What's real here
- **Adaptive question engine** — genuine Computerized Adaptive Testing (CAT)
  logic: correct → harder, wrong → easier, no repeats.
- **Feature extraction** — turns a raw answer history into a numeric feature
  vector (accuracy by difficulty tier, timing, hint usage, etc.).
- **Trained ML model** — a real XGBoost classifier, trained/validated/tested
  with a proper 70/15/15 split, with accuracy + confusion matrix reported.
- **End-to-end product** — FastAPI backend + Streamlit UI, actually runs.

## What's simplified (and why)
- **No live code execution sandbox.** Code questions are scored by keyword/
  concept matching instead of running submitted code in Docker. Real code
  execution needs sandboxing for security — worth doing, not worth doing
  first.
- **No BERT/CodeBERT semantic scoring.** Same keyword-matching stand-in.
  Swappable later — see `_score_answer()` in `adaptive_engine.py`, it's
  isolated in one function specifically so this is a drop-in upgrade.
- **Synthetic training data**, not real candidates. There's no existing
  public dataset linking "answer behavior" to "verified skill level" — you'd
  have to collect it by running this on real people and getting independent
  verification (grades, certifications). Synthetic data lets you build and
  validate the *pipeline* now; swapping in real data later doesn't require
  touching the model code, just `ml/synthetic_candidates.csv`.
- **In-memory sessions**, not a database. Fine for a single-user demo;
  restarting the backend clears active sessions.

## Project structure
```
skillsense-ai/
├── data/questions.json          question bank (difficulty + category tagged)
├── backend/
│   ├── main.py                  FastAPI app (start / submit_answer / report)
│   ├── adaptive_engine.py       CAT logic + feature extraction
│   └── models/                  trained model artifacts (generated)
├── ml/
│   ├── generate_synthetic_data.py
│   └── train_model.py
└── frontend/app.py               Streamlit UI
```

## Resume features (added on top of the base MVP)

The model now also trains on two features derived from a Kaggle resume
dataset (`AI-Powered Resume Screening Dataset`, `ml/data/AI_Resume_Screening.csv`):
- `claimed_experience_years` — from the resume's `Experience (Years)` column
- `claimed_skill_count` — count of comma-separated skills listed

**Important caveat, stated plainly for your writeup:** the resume rows and
the behavioral rows are from two unrelated sources (no dataset exists
pairing a real resume with a real behavioral assessment for the same
person). `ml/merge_resume_data.py` creates the pairing *randomly*, so the
resulting model can genuinely use resume-derived features technically, but
that doesn't mean it reflects a real-world relationship between resumes
and actual skill. This also means the model can, in principle, be swayed
by a confidently-lied-about resume, which cuts against the project's
original "catch the exaggeration" premise — worth discussing as a known
limitation and trade-off rather than hiding it.

To regenerate everything with resume features included:
```bash
python ml/generate_synthetic_data.py
python ml/merge_resume_data.py
python ml/train_model.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run it (3 steps)

**1. Train the model (one-time, ~10 seconds):**
```bash
python ml/generate_synthetic_data.py
python ml/train_model.py
```
This prints test accuracy, a classification report, and a confusion matrix —
keep that output, it's your evidence the model works.

**2. Start the backend** (from the project root, so imports resolve):
```bash
uvicorn backend.main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` to poke the API directly if you want.

**3. Start the frontend** (in a second terminal):
```bash
streamlit run frontend/app.py
```
Opens at `http://localhost:8501`.

## How to extend it next (in priority order)
1. **Real code execution** — run candidate code in a Docker sandbox against
   pytest test cases instead of keyword matching. This is the single biggest
   credibility upgrade.
2. **Semantic answer scoring** — swap keyword matching in `_score_answer()`
   for a Sentence-Transformers cosine similarity against a reference answer.
3. **Real training data** — run sessions with real people, get their level
   independently verified, retrain on that instead of synthetic profiles.
4. **Explainability** — add SHAP values to the report so "why Intermediate"
   has a real answer, not just a label.
5. **Persistent storage** — swap the in-memory `SESSIONS` dict for
   PostgreSQL so sessions survive restarts and support multiple users.

## Notes for your writeup
- The 96% test accuracy on synthetic data measures the ML pipeline working
  correctly, not real-world predictive accuracy — be upfront about this
  distinction in a portfolio or course submission.
- Ethical considerations from the original brief still apply here in
  miniature: this scores based on limited signals (8 questions, keyword
  matching) and shouldn't be presented as more authoritative than a real
  technical interview.
