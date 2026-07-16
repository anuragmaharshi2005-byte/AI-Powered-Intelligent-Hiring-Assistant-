"""
utils.py
--------
Small shared helpers and constants used across the project:
- file paths
- the curated skill vocabulary (built from the resumes_dataset.jsonl skill tags)
- a couple of generic helper functions that don't belong in any single module

Keeping these in one place avoids repeating the same paths / skill list
in preprocessing.py, similarity.py, feedback.py and chatbot.py.
"""

import json
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DATASET_PATH = os.path.join(DATA_DIR, "resumes_dataset.jsonl")
SKILLS_PATH = os.path.join(DATA_DIR, "skills_list.json")

CLASSIFIER_PATH = os.path.join(MODELS_DIR, "resume_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")


# ---------------------------------------------------------------------------
# Skill vocabulary
# ---------------------------------------------------------------------------
def load_skill_vocabulary():
    """
    Loads the curated skill vocabulary from data/skills_list.json.

    This list was built by scanning the "Skills" field of every resume in
    resumes_dataset.jsonl and collecting the unique skill tags. It's a small,
    clean vocabulary (~95 skills) which keeps skill-matching fast and easy
    to reason about, instead of trying to do open-ended keyword extraction
    over free text.
    """
    with open(SKILLS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def score_bucket(match_score):
    """
    Buckets a 0-100 match score into a qualitative label.
    Used by feedback.py and the Streamlit dashboard so both stay consistent.
    """
    if match_score >= 75:
        return "Strong Match"
    elif match_score >= 50:
        return "Moderate Match"
    else:
        return "Weak Match"


def truncate(text, max_chars=400):
    """Utility used when printing previews of long resume/JD text in the UI."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."
