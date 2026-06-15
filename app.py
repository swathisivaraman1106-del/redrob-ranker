import streamlit as st
import json
from datetime import date
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🤖")
st.title("🤖 Redrob Intelligent Candidate Ranker")
st.markdown("Ranks candidates for **Senior AI Engineer** role using multi-signal scoring.")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_candidates():
    candidates = []
    with open("sample_candidates.json") as f:
        candidates = json.load(f)
    return candidates

model = load_model()
candidates = load_candidates()

JD_REQUIREMENTS = """
Production experience with embeddings and vector search systems.
Python programming with strong code quality.
NLP and information retrieval experience.
Ranking and recommendation systems at scale.
LLM integration, fine-tuning, prompt engineering.
"""
jd_embedding = model.encode(JD_REQUIREMENTS, convert_to_tensor=True)

GOOD_TITLES = ["machine learning", "ml engineer", "ai engineer", "data scientist",
               "nlp engineer", "research engineer", "applied scientist",
               "software engineer", "backend engineer", "data engineer"]
BAD_TITLES = ["hr", "accountant", "graphic designer", "content writer",
              "marketing", "sales", "civil engineer", "mechanical engineer",
              "customer support", "operations manager"]
SERVICES = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]
CORE_SKILLS = ["python", "nlp", "embeddings", "vector", "retrieval", "llm",
               "transformer", "pytorch", "tensorflow", "faiss", "pinecone",
               "machine learning", "deep learning", "rag", "huggingface"]

def score_candidate(c):
    title = c["profile"]["current_title"].lower()
    ts = 1.0 if any(g in title for g in GOOD_TITLES) else (0.0 if any(b in title for b in BAD_TITLES) else 0.3)

    skills = c["skills"]
    total = sum(
        {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.85, "expert": 1.0}.get(s["proficiency"], 0.5)
        for s in skills
        if any(core in s["name"].lower() for core in CORE_SKILLS)
        and not (s["proficiency"] == "expert" and s.get("duration_months", 0) == 0)
    )
    ss = min(total / 10.0, 1.0)

    yoe = c["profile"]["years_of_experience"]
    exp_score = 1.0 if 5 <= yoe <= 9 else (0.7 if 4 <= yoe <= 11 else 0.4)
    product_months = sum(j["duration_months"] for j in c["career_history"] if not any(s in j["company"].lower() for s in SERVICES))
    total_months = sum(j["duration_months"] for j in c["career_history"])
    cs = (exp_score * 0.5) + ((product_months / total_months if total_months else 0.5) * 0.5)

    sig = c["redrob_signals"]
    today = date(2026, 6, 12)
    days = (today - date.fromisoformat(sig["last_active_date"])).days
    bs = (0.30 if days <= 30 else 0.15 if days <= 90 else 0.05) + \
         (0.20 if sig["open_to_work_flag"] else 0) + \
         sig["recruiter_response_rate"] * 0.20 + \
         (0.15 if sig["notice_period_days"] <= 30 else 0.10 if sig["notice_period_days"] <= 60 else 0.05) + \
         sig["interview_completion_rate"] * 0.15

    profile_text = " ".join([s["name"] for s in skills] + [j["title"] for j in c["career_history"]])
    sem = float(util.cos_sim(model.encode(profile_text, convert_to_tensor=True), jd_embedding).item())

    fs = (ts * 0.35) + (ss * 0.20) + (cs * 0.20) + (min(bs, 1.0) * 0.15) + (max(0, sem) * 0.10)
    return fs, ts, ss, cs, min(bs, 1.0), max(0, sem)

st.sidebar.header("Filters")
min_score = st.sidebar.slider("Minimum Score", 0.0, 1.0, 0.5)
top_n = st.sidebar.slider("Show Top N", 5, 50, 10)

if st.button("🚀 Rank Candidates"):
    with st.spinner("Scoring candidates..."):
        results = []
        for c in candidates:
            fs, ts, ss, cs, bs, sem = score_candidate(c)
            if fs >= min_score:
                results.append((fs, ts, ss, cs, bs, sem, c))
        results.sort(key=lambda x: (-x[0], x[6]["candidate_id"]))

    st.success(f"Found {len(results)} candidates above {min_score} score")

    for rank, (fs, ts, ss, cs, bs, sem, c) in enumerate(results[:top_n], 1):
        with st.expander(f"#{rank} — {c['profile']['current_title']} | Score: {fs:.3f}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Experience:** {c['profile']['years_of_experience']} years")
                st.write(f"**Industry:** {c['profile']['current_industry']}")
                top_skills = [s['name'] for s in c['skills'] if s.get('duration_months', 0) > 0][:5]
                st.write(f"**Skills:** {', '.join(top_skills)}")
            with col2:
                st.metric("Title", f"{ts:.2f}")
                st.metric("Skills", f"{ss:.2f}")
                st.metric("Career", f"{cs:.2f}")
                st.metric("Behavior", f"{bs:.2f}")
                st.metric("Semantic", f"{sem:.2f}")