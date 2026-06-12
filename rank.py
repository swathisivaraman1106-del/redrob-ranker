import json
import csv
from datetime import date
from sentence_transformers import SentenceTransformer, util

# --- LOAD DATA ---
candidates = []
with open("candidates.jsonl", "r") as f:
    for line in f:
        candidates.append(json.loads(line))

print(f"Total candidates: {len(candidates)}")

# --- LOAD SEMANTIC MODEL ---
print("Loading semantic model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

JD_REQUIREMENTS = """
Production experience with embeddings and vector search systems.
Python programming with strong code quality.
NLP and information retrieval experience.
Ranking and recommendation systems at scale.
LLM integration, fine-tuning, prompt engineering.
Evaluation frameworks for ranking systems like NDCG and MRR.
Experience with FAISS, Pinecone, Weaviate, Elasticsearch or similar.
Product company experience shipping ML systems to real users.
"""

jd_embedding = model.encode(JD_REQUIREMENTS, convert_to_tensor=True)
print("Model loaded!")

# --- STEP 1: Title Score ---

GOOD_TITLES = [
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "nlp engineer", "research engineer", "applied scientist",
    "software engineer", "backend engineer", "data engineer",
    "deep learning", "computer vision"
]

BAD_TITLES = [
    "hr", "accountant", "graphic designer", "content writer",
    "marketing", "sales", "civil engineer", "mechanical engineer",
    "customer support", "operations manager", "project manager"
]

def title_score(candidate):
    title = candidate["profile"]["current_title"].lower()
    for good in GOOD_TITLES:
        if good in title:
            return 1.0
    for bad in BAD_TITLES:
        if bad in title:
            return 0.0
    return 0.3


# --- STEP 2: Skills Score ---

CORE_SKILLS = [
    "python", "nlp", "embeddings", "vector", "retrieval", "ranking",
    "llm", "fine-tuning", "transformer", "bert", "machine learning",
    "deep learning", "pytorch", "tensorflow", "scikit-learn",
    "elasticsearch", "faiss", "pinecone", "weaviate", "qdrant",
    "milvus", "sentence-transformers", "huggingface", "rag",
    "recommendation", "search", "neural", "xgboost", "spark"
]

PROFICIENCY_WEIGHT = {
    "beginner": 0.25,
    "intermediate": 0.5,
    "advanced": 0.85,
    "expert": 1.0
}

def skills_score(candidate):
    skills = candidate["skills"]
    if not skills:
        return 0.0

    total = 0.0
    for skill in skills:
        skill_name = skill["name"].lower()
        for core in CORE_SKILLS:
            if core in skill_name or skill_name in core:
                if skill["proficiency"] == "expert" and skill.get("duration_months", 0) == 0:
                    continue
                weight = PROFICIENCY_WEIGHT.get(skill["proficiency"], 0.5)
                total += weight
                break

    return min(total / 10.0, 1.0)


# --- STEP 3: Career Score ---

SERVICES_COMPANIES = [
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "hexaware",
    "l&t infotech", "ltimindtree"
]

def career_score(candidate):
    yoe = candidate["profile"]["years_of_experience"]
    career = candidate["career_history"]

    if 5 <= yoe <= 9:
        exp_score = 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        exp_score = 0.7
    elif 3 <= yoe < 4 or 11 < yoe <= 13:
        exp_score = 0.4
    else:
        exp_score = 0.1

    product_months = 0
    services_months = 0
    for job in career:
        company = job["company"].lower()
        duration = job["duration_months"]
        is_services = any(s in company for s in SERVICES_COMPANIES)
        if is_services:
            services_months += duration
        else:
            product_months += duration

    total_months = product_months + services_months
    if total_months == 0:
        product_score = 0.5
    else:
        product_score = product_months / total_months

    return (exp_score * 0.5) + (product_score * 0.5)


# --- STEP 4: Behavioral Score ---

def behavioral_score(candidate):
    signals = candidate["redrob_signals"]
    today = date(2026, 6, 12)

    score = 0.0

    last_active = date.fromisoformat(signals["last_active_date"])
    days_inactive = (today - last_active).days
    if days_inactive <= 30:
        score += 0.30
    elif days_inactive <= 90:
        score += 0.15
    elif days_inactive <= 180:
        score += 0.05

    if signals["open_to_work_flag"]:
        score += 0.20

    score += signals["recruiter_response_rate"] * 0.20

    notice = signals["notice_period_days"]
    if notice <= 30:
        score += 0.15
    elif notice <= 60:
        score += 0.10
    elif notice <= 90:
        score += 0.05

    score += signals["interview_completion_rate"] * 0.15

    return min(score, 1.0)


# --- STEP 5: Semantic Scoring (batch mode) ---

print("Building candidate profiles for semantic scoring...")
profile_texts = []
for c in candidates:
    skills = [s["name"] for s in c["skills"] if s.get("duration_months", 0) > 0]
    titles = [j["title"] for j in c["career_history"]]
    profile_texts.append(" ".join(skills + titles) or "unknown")

print("Running semantic encoding on all candidates (batch mode)...")
candidate_embeddings = model.encode(
    profile_texts,
    batch_size=512,
    show_progress_bar=True,
    convert_to_tensor=True
)
similarities = util.cos_sim(candidate_embeddings, jd_embedding).squeeze().tolist()
print("Semantic scoring done!")


# --- STEP 6: Final Score ---

def final_score(ts, ss, cs, bs, sem):
    return (ts * 0.35) + (ss * 0.20) + (cs * 0.20) + (bs * 0.15) + (sem * 0.10)


def reasoning(candidate, fs):
    title = candidate["profile"]["current_title"]
    yoe = candidate["profile"]["years_of_experience"]
    signals = candidate["redrob_signals"]
    skills = [
        s["name"] for s in candidate["skills"]
        if s.get("duration_months", 0) > 0
    ][:3]

    return (
        f"{title} with {yoe}y exp; "
        f"skills: {', '.join(skills) if skills else 'none relevant'}; "
        f"response rate {signals['recruiter_response_rate']:.2f}; "
        f"notice {signals['notice_period_days']}d."
    )


# --- STEP 7: Score all + generate CSV ---

print("Calculating final scores...")
scored = []
for i, c in enumerate(candidates):
    ts = title_score(c)
    ss = skills_score(c)
    cs = career_score(c)
    bs = behavioral_score(c)
    sem = max(0.0, min(float(similarities[i]), 1.0))
    fs = final_score(ts, ss, cs, bs, sem)
    scored.append((fs, c))

# sort by score descending, tie-break by candidate_id ascending
scored.sort(key=lambda x: (-round(x[0], 6), x[1]["candidate_id"]))

top100 = scored[:100]

with open("submission.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])

    for rank, (fs, c) in enumerate(top100, start=1):
        cid = c["candidate_id"]
        r = reasoning(c, fs)
        writer.writerow([cid, rank, round(fs, 6), r])

print("Done! submission.csv created.")
print("\nTop 10 candidates:")
for rank, (fs, c) in enumerate(top100[:10], start=1):
    print(f"  {rank}. {c['profile']['current_title']:40s} score={fs:.4f}")