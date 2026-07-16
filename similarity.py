"""
similarity.py
--------------
Everything related to comparing ONE resume against ONE job description:

1. compute_match_score()  -> semantic similarity score (0-100)
2. compare_skills()       -> matching / missing / extra skills
3. classify_resume()      -> predicted job category using the trained
                              Logistic Regression model (train_model.py)

Semantic similarity uses Sentence-Transformers (all-MiniLM-L6-v2) to embed
the resume and the job description, then cosine similarity between the two
embeddings gives the match score. This captures meaning ("built REST APIs"
vs "developed backend services") rather than exact keyword overlap, which
is why it's used instead of plain TF-IDF for the headline score.

If sentence-transformers / its model weights aren't available (e.g. no
internet on first run), we fall back to a TF-IDF + cosine similarity score
instead, so the app still works end-to-end -- just with slightly less
nuanced matching. A warning is surfaced to the caller so this is visible
rather than silent.
"""

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text
from utils import CLASSIFIER_PATH, LABEL_ENCODER_PATH, VECTORIZER_PATH, load_skill_vocabulary

_EMBEDDER = None
_EMBEDDER_AVAILABLE = None  # None = not checked yet, True/False after first attempt


def _get_embedder():
    """
    Lazily loads the sentence-transformer model once and reuses it.
    Returns None if the library/model can't be loaded, so callers can
    fall back to TF-IDF similarity instead of crashing the app.
    """
    global _EMBEDDER, _EMBEDDER_AVAILABLE
    if _EMBEDDER_AVAILABLE is False:
        return None
    if _EMBEDDER is not None:
        return _EMBEDDER

    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        _EMBEDDER_AVAILABLE = True
    except Exception:
        _EMBEDDER_AVAILABLE = False
        _EMBEDDER = None

    return _EMBEDDER


def compute_match_score(resume_text, jd_text):
    """
    Returns (score, method) where score is 0-100 and method is either
    "semantic" (sentence-transformers) or "tfidf" (fallback).
    """
    embedder = _get_embedder()

    if embedder is not None:
        embeddings = embedder.encode([resume_text, jd_text])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        method = "semantic"
    else:
        cleaned = [clean_text(resume_text), clean_text(jd_text)]
        vectorizer = TfidfVectorizer()
        try:
            tfidf = vectorizer.fit_transform(cleaned)
            sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        except ValueError:
            # Happens if one of the documents is empty after cleaning
            sim = 0.0
        method = "tfidf"

    score = round(float(np.clip(sim, 0, 1)) * 100, 1)
    return score, method


def extract_skills(text, vocabulary=None):
    """
    Finds which skills from the curated vocabulary appear in a piece of
    text. Simple case-insensitive substring matching -- deliberately not
    fuzzy, since the vocabulary is small and clean enough that exact
    matching on word boundaries is reliable and easy to explain to a user
    (no "why did it think I know X" surprises).
    """
    if vocabulary is None:
        vocabulary = load_skill_vocabulary()

    text_lower = f" {text.lower()} "
    found = []
    for skill in vocabulary:
        skill_lower = skill.lower()
        # handle skills with punctuation (e.g. "Node.js", "HTML/CSS") by
        # matching the raw substring; for plain alphabetic skills, match
        # on word boundaries to avoid partial-word false positives
        # (e.g. "Go" inside "Google").
        if skill_lower.replace(".", "").replace("/", "").isalpha() and " " not in skill_lower:
            import re
            pattern = r"\b" + re.escape(skill_lower) + r"\b"
            if re.search(pattern, text_lower):
                found.append(skill)
        else:
            if skill_lower in text_lower:
                found.append(skill)

    return sorted(set(found))


def compare_skills(resume_text, jd_text, vocabulary=None):
    """
    Returns a dict with:
        matching_skills : skills present in BOTH resume and JD
        missing_skills  : skills the JD asks for but the resume doesn't have
        extra_skills    : skills the resume has that the JD didn't ask for
    """
    if vocabulary is None:
        vocabulary = load_skill_vocabulary()

    resume_skills = set(extract_skills(resume_text, vocabulary))
    jd_skills = set(extract_skills(jd_text, vocabulary))

    return {
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "matching_skills": sorted(resume_skills & jd_skills),
        "missing_skills": sorted(jd_skills - resume_skills),
        "extra_skills": sorted(resume_skills - jd_skills),
    }


def classify_resume(resume_text, top_k=3):
    """
    Uses the trained Logistic Regression model to predict which job
    category a resume most likely belongs to, with class probabilities.

    Returns a list of (category, probability) tuples, sorted descending,
    truncated to top_k. Returns None if model artifacts haven't been
    trained yet (i.e. train_model.py hasn't been run).
    """
    try:
        model = joblib.load(CLASSIFIER_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    except FileNotFoundError:
        return None

    cleaned = clean_text(resume_text)
    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]

    ranked = sorted(zip(label_encoder.classes_, probs), key=lambda x: x[1], reverse=True)
    return [(cat, round(float(p) * 100, 1)) for cat, p in ranked[:top_k]]
