"""
chatbot.py
----------
A small conversational interface that answers candidate questions about
their resume-vs-JD analysis, e.g.:
    "Why is my score low?"
    "Which skills are missing?"
    "How can I improve my resume?"
    "Am I suitable for this role?"
    "Explain my strengths."

Design choice -- intent matching + retrieval over the analysis, not a
free-form LLM call:
    The brief explicitly requires the chatbot to "only use the uploaded
    documents as context and should not hallucinate." The safest way to
    guarantee that is to not generate free text with a language model at
    all for the factual parts -- instead, every answer is built directly
    from the SAME analysis dict produced by similarity.py + feedback.py
    that's already shown on the dashboard. The chatbot's job is just to
    figure out *which* piece of that analysis the user is asking about
    (via keyword/intent matching over the question) and phrase it back
    conversationally. This means the chatbot can never say something the
    dashboard doesn't already support with numbers.

    For a question that doesn't match any known intent, the bot says so
    honestly instead of guessing -- again, to avoid hallucinating an
    answer it has no grounding for.
"""

import re

# Each intent: a name, a list of keyword patterns to match in the user's
# question, and a function that builds the answer from the analysis dict.
# Order matters -- more specific intents are checked before generic ones.
_INTENTS = []


def _register(name, keywords):
    def decorator(func):
        _INTENTS.append({"name": name, "keywords": keywords, "handler": func})
        return func
    return decorator


@_register("score_explanation", ["why", "low", "score", "match score", "how did you calculate"])
def _answer_score(analysis):
    score = analysis["match_score"]
    bucket = analysis["feedback"]["bucket"]
    matching = analysis["skill_comparison"]["matching_skills"]
    missing = analysis["skill_comparison"]["missing_skills"]
    text = (
        f"Your match score is {score}/100 ({bucket}). It's calculated from the semantic "
        f"similarity between your resume text and the job description -- not just keyword "
        f"overlap, but overall meaning."
    )
    if missing:
        text += (
            f" A likely reason it isn't higher: {len(missing)} skill(s) from the job "
            f"description weren't found in your resume -- specifically {', '.join(missing)}."
        )
    elif matching:
        text += " All the job description's key skills were found in your resume, so the remaining gap is likely in wording/context rather than missing skills."
    return text


@_register("missing_skills", ["missing", "skill gap", "what skills", "don't have", "lack"])
def _answer_missing(analysis):
    missing = analysis["skill_comparison"]["missing_skills"]
    if not missing:
        return "No missing required skills were detected -- your resume covers every skill listed in the job description."
    return (
        f"The job description mentions {len(missing)} skill(s) that weren't found in your resume: "
        f"{', '.join(missing)}. If you have experience with any of these, make sure they're stated "
        f"explicitly rather than implied."
    )


@_register("improve", ["improve", "how can i", "what should i", "better", "tips", "suggest"])
def _answer_improve(analysis):
    suggestions = analysis["feedback"]["suggestions"]
    return " ".join(suggestions)


@_register("suitability", ["suitable", "am i fit", "should i apply", "good fit", "qualify"])
def _answer_suitability(analysis):
    score = analysis["match_score"]
    bucket = analysis["feedback"]["bucket"]
    if bucket == "Strong Match":
        return (
            f"Based on a {score}/100 match score, you look like a strong fit for this role on paper. "
            f"That said, a resume score is only one signal -- it doesn't capture soft skills or interview performance."
        )
    elif bucket == "Moderate Match":
        return (
            f"You're a moderate fit ({score}/100). It's worth applying, especially if you can address "
            f"some of the missing skills before submitting, or address them directly in a cover letter."
        )
    else:
        return (
            f"At {score}/100, your resume as written doesn't show strong overlap with this role's requirements. "
            f"It's not necessarily disqualifying, but you'd likely want to tailor your resume more closely to "
            f"this specific job description first."
        )


@_register("strengths", ["strength", "good at", "what am i good", "positives"])
def _answer_strengths(analysis):
    strengths = analysis["feedback"]["strengths"]
    return " ".join(strengths)


@_register("predicted_category", ["category", "role am i", "what kind of", "classify", "classified"])
def _answer_category(analysis):
    note = analysis["feedback"]["classifier_note"]
    if not note:
        return "The resume category model hasn't been trained yet, so I can't classify your resume right now."
    return note


def _match_intent(question):
    q = question.lower()
    for intent in _INTENTS:
        for kw in intent["keywords"]:
            if kw in q:
                return intent
    return None


def answer_question(question, analysis):
    """
    Parameters
    ----------
    question : str -- the candidate's typed question
    analysis : dict -- must contain keys: match_score, skill_comparison,
        feedback (i.e. everything similarity.py + feedback.py already
        computed for the current resume/JD pair; app.py assembles this
        once per upload and reuses it for every chatbot turn).

    Returns
    -------
    str : the chatbot's answer.
    """
    if not question or not question.strip():
        return "Ask me something about your match score, missing skills, or how to improve your resume."

    intent = _match_intent(question)
    if intent is None:
        return (
            "I can only answer questions grounded in your uploaded resume and job description analysis -- "
            "things like your match score, missing skills, strengths, or suitability for the role. "
            "Try asking one of those directly, e.g. \"Why is my score low?\" or \"What skills am I missing?\""
        )

    return intent["handler"](analysis)
