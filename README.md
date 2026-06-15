# Redrob Intelligent Candidate Ranking System

## Approach

This system ranks candidates for a Senior AI Engineer role using a multi-signal scoring pipeline that goes beyond keyword matching.

## Scoring Components

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Title Score | 35% | Is the candidate actually an AI/ML/Data engineer? |
| Skills Score | 20% | Do they have relevant technical skills with real experience? |
| Career Score | 20% | Product company experience? Right years of experience (5-9)? |
| Behavioral Score | 15% | Active recently? Open to work? Good response rate? |
| Semantic Score | 10% | How well does their profile semantically match the JD? |

## Key Design Decisions

- **Anti-keyword-stuffing**: Title score heavily penalizes irrelevant roles (HR, Marketing, Accountant) even if they list AI skills
- **Honeypot detection**: Skills marked "expert" with 0 months experience are ignored
- **Services company penalty**: Candidates with 100% TCS/Infosys/Wipro experience are downweighted per JD requirements
- **Behavioral signals**: Candidates inactive 6+ months are heavily downweighted regardless of profile quality
- **Semantic matching**: Uses sentence-transformers (all-MiniLM-L6-v2) to understand profile-JD fit beyond keywords

## Tech Stack

- Python 3.11
- sentence-transformers
- scikit-learn

## How to Run

```
pip install sentence-transformers
python rank.py
python validate_submission.py submission.csv
```