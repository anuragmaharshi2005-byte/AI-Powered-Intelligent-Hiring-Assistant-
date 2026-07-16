"""
preprocessing.py
-----------------
Basic text cleaning used before we run TF-IDF / embeddings on resume and
job description text. Nothing fancy: lowercase, strip punctuation/numbers,
remove stopwords. This is the same style of cleaning the training dataset
(resumes_dataset.jsonl) already had applied to it, so keeping it simple here
keeps the training data and the live upload text on the same footing.
"""

import re

# A small built-in fallback list, used only if NLTK's full stopword corpus
# isn't available locally AND can't be downloaded (e.g. running fully
# offline before `python -c "import nltk; nltk.download('stopwords')"`
# has ever been run). It's short but covers the highest-frequency words,
# so cleaning still works reasonably well -- it just won't be as thorough
# as the full ~180-word NLTK list.
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "to", "from", "in", "on", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "it", "its",
    "as", "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "i", "you", "he", "she", "we", "they",
    "them", "his", "her", "their", "our", "your", "not", "no", "so",
}

try:
    import nltk
    from nltk.corpus import stopwords

    # Downloaded once on first import; NLTK caches this locally afterwards
    # so it won't try to re-download every time the app runs.
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
        except Exception:
            pass  # no internet -- fall through to the fallback list below

    try:
        STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        STOPWORDS = set(_FALLBACK_STOPWORDS)
except ImportError:
    STOPWORDS = set(_FALLBACK_STOPWORDS)

# A few resume-specific filler words that survive normal stopword lists but
# don't carry any signal for matching (contact-detail boilerplate etc.)
EXTRA_STOPWORDS = {
    "resume", "cv", "curriculum", "vitae", "email", "phone", "address",
    "city", "state", "references", "available", "request",
}


def clean_text(text):
    """
    Lowercases, strips emails/URLs/numbers/punctuation, and removes
    stopwords. Returns a single cleaned string (not a token list) so it can
    be fed straight into TfidfVectorizer or a sentence-transformer.
    """
    text = text.lower()

    # Drop emails and URLs before anything else — they add noise and
    # occasionally leak personal info into downstream text fields.
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and t not in EXTRA_STOPWORDS and len(t) > 1]

    return " ".join(tokens)


def clean_for_display(text, max_len=2000):
    """
    A much lighter cleanup used only when showing raw extracted text back
    to the user in the UI (we want it readable, not stripped of stopwords).
    Just collapses excess whitespace.
    """
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]
