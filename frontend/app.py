"""
Streamlit UI for SkillSense AI.

Run (with the backend already running in another terminal):
    streamlit run frontend/app.py
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="SkillSense AI", page_icon="🧠", layout="centered")
st.title("🧠 SkillSense AI")
st.caption("Adaptive Python skill assessment — answers, not resumes, decide your level.")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
    st.session_state.question = None
    st.session_state.finished = False
    st.session_state.resume_claim = ""
    st.session_state.claimed_experience_years = 0
    st.session_state.claimed_skill_count = 0

# --- Landing / start screen ---
if st.session_state.session_id is None:
    st.session_state.resume_claim = st.text_input(
        "What skill level does your resume claim? (label shown in the final comparison)",
        placeholder="e.g. Expert",
    )
    st.session_state.claimed_experience_years = st.number_input(
        "Claimed years of experience", min_value=0, max_value=40, value=0
    )
    skills_text = st.text_input(
        "Skills listed on resume (comma-separated)",
        placeholder="e.g. Python, SQL, Machine Learning, Docker",
    )
    st.session_state.claimed_skill_count = len([s for s in skills_text.split(",") if s.strip()])

    if st.button("Start Assessment", type="primary"):
        resp = requests.post(f"{API_URL}/start").json()
        st.session_state.session_id = resp["session_id"]
        st.session_state.question = resp["question"]
        st.rerun()

# --- Question screen ---
elif not st.session_state.finished:
    q = st.session_state.question
    st.markdown(f"**Difficulty:** {q['difficulty'].capitalize()}")
    st.subheader(q["question"])

    if q["type"] == "mcq":
        answer = st.radio("Choose one:", q["options"], key=q["id"])
    else:
        answer = st.text_area("Your answer:", key=q["id"], height=150)

    hints_used = st.number_input("Hints used (0 if none)", min_value=0, max_value=5, value=0)

    if st.button("Submit Answer", type="primary"):
        payload = {
            "session_id": st.session_state.session_id,
            "answer_text": answer,
            "hints_used": hints_used,
        }
        result = requests.post(f"{API_URL}/submit_answer", json=payload).json()

        if result["correct"]:
            st.success("Correct — next question will be harder.")
        else:
            st.warning("Not quite — next question will be easier.")

        if result["finished"]:
            st.session_state.finished = True
        else:
            st.session_state.question = result["next_question"]
        st.rerun()

# --- Report screen ---
else:
    st.success("Assessment complete!")
    resume_claim = st.session_state.resume_claim or "Not provided"
    report = requests.get(
        f"{API_URL}/report/{st.session_state.session_id}",
        params={
            "resume_claim": resume_claim,
            "claimed_experience_years": st.session_state.claimed_experience_years,
            "claimed_skill_count": st.session_state.claimed_skill_count,
        },
    ).json()

    col1, col2, col3 = st.columns(3)
    col1.metric("Resume Claim", report["resume_claim"])
    col2.metric("AI Estimated Level", report["ai_estimated_level"])
    col3.metric("Confidence", f"{report['confidence_pct']}%")

    st.metric("Overall Accuracy", f"{report['overall_accuracy_pct']}%")

    st.subheader("Category Breakdown")
    st.bar_chart(report["category_scores"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strengths**")
        for s in report["strengths"]:
            st.markdown(f"- {s}")
    with c2:
        st.markdown("**Weaknesses**")
        for w in report["weaknesses"]:
            st.markdown(f"- {w}")

    with st.expander("Raw feature vector (debug)"):
        st.json(report["raw_features"])

    if st.button("Start Over"):
        st.session_state.session_id = None
        st.session_state.finished = False
        st.rerun()
