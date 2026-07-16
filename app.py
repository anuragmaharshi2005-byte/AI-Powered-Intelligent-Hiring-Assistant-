"""
app.py
------
Streamlit front-end for the Resume Screening & Candidate Feedback System.

Pages (via the sidebar):
    Home                  - project intro
    Upload & Analyze       - upload resume + JD, run the pipeline
    Match Score             - semantic match score visualization
    Skill Comparison       - matching / missing / extra skills
    Personalized Feedback  - explainable feedback report
    Chatbot                 - ask questions about your own analysis
    About                    - project write-up

Run with:  streamlit run app.py
(Run `python train_model.py` once beforehand to generate the model files
 the classifier feature needs.)
"""

import streamlit as st

st.set_page_config(page_title="Resume Screening & Feedback System", page_icon="🧾", layout="wide")

# ---------------------------------------------------------------------------
# Startup dependency check
# ---------------------------------------------------------------------------
# Streamlit itself has to be installed just to get this far (it's the command
# being run), but everything else this app depends on -- plotly, scikit-learn
# (via similarity.py/train_model.py), sentence-transformers, pdfplumber -- is
# only imported once the app starts executing. If one of those is missing,
# Python would normally throw a raw ModuleNotFoundError traceback in the
# browser, which looks broken rather than "please install a package". This
# catches that case and shows a clear, actionable message instead.
try:
    import plotly.graph_objects as go
    from parser import parse_document
    from preprocessing import clean_for_display
    from similarity import compute_match_score, compare_skills, classify_resume
    from feedback import generate_feedback
    from chatbot import answer_question
except ImportError as e:
    st.error(
        f"Missing dependency: **{e.name}**\n\n"
        "It looks like not all required packages are installed. "
        "From the project folder, run:\n\n"
        "```\npip install -r requirements.txt\n```\n\n"
        "then restart the app with `streamlit run app.py`."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Session state -- holds the current resume/JD text and computed analysis
# so results survive navigating between sidebar pages.
# ---------------------------------------------------------------------------
for key in ["resume_text", "jd_text", "jd_title", "analysis"]:
    if key not in st.session_state:
        st.session_state[key] = None


def run_analysis(resume_text, jd_text, jd_title=None):
    """Runs the full pipeline once and caches the result in session_state."""
    score, method = compute_match_score(resume_text, jd_text)
    skill_comparison = compare_skills(resume_text, jd_text)
    predicted_category = classify_resume(resume_text)
    fb = generate_feedback(score, skill_comparison, predicted_category, jd_title)

    return {
        "match_score": score,
        "score_method": method,
        "skill_comparison": skill_comparison,
        "predicted_category": predicted_category,
        "feedback": fb,
    }


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🧾 Resume Screening")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Upload & Analyze", "Match Score", "Skill Comparison", "Personalized Feedback", "Chatbot", "About"],
)

if st.session_state["analysis"] is None and page not in ("Home", "Upload & Analyze", "About"):
    st.sidebar.warning("Upload a resume + job description first.")


# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if page == "Home":
    st.title("AI-Powered Resume Screening & Candidate Feedback System")
    st.write(
        "Upload your resume and a job description to get a semantic match score, "
        "a breakdown of matching/missing skills, personalized improvement feedback, "
        "and a chatbot you can ask follow-up questions to."
    )
    st.markdown("#### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload**\n\nUpload your resume and a job description (PDF or text).")
    with col2:
        st.markdown("**2. Analyze**\n\nSentence embeddings + a trained classifier score and explain the match.")
    with col3:
        st.markdown("**3. Improve**\n\nGet specific, explainable feedback and ask the chatbot follow-up questions.")

    st.info("Start with **Upload & Analyze** in the sidebar.")


# ---------------------------------------------------------------------------
# UPLOAD & ANALYZE
# ---------------------------------------------------------------------------
elif page == "Upload & Analyze":
    st.title("Upload & Analyze")

    input_mode = st.radio("Job description input", ["Upload file", "Paste text"], horizontal=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Resume")
        resume_file = st.file_uploader("Upload resume (PDF or TXT)", type=["pdf", "txt"], key="resume_upload")

    with col2:
        st.subheader("Job Description")
        jd_title = st.text_input("Role title (optional, used in feedback)", placeholder="e.g. Backend Developer")
        if input_mode == "Upload file":
            jd_file = st.file_uploader("Upload job description (PDF or TXT)", type=["pdf", "txt"], key="jd_upload")
            jd_pasted = None
        else:
            jd_file = None
            jd_pasted = st.text_area("Paste job description text", height=200)

    if st.button("Run Analysis", type="primary"):
        if resume_file is None:
            st.error("Please upload a resume.")
        elif jd_file is None and not jd_pasted:
            st.error("Please upload or paste a job description.")
        else:
            with st.spinner("Extracting text and running the pipeline..."):
                try:
                    resume_text = parse_document(resume_file)
                    jd_text = jd_pasted if jd_pasted else parse_document(jd_file)

                    st.session_state["resume_text"] = resume_text
                    st.session_state["jd_text"] = jd_text
                    st.session_state["jd_title"] = jd_title
                    st.session_state["analysis"] = run_analysis(resume_text, jd_text, jd_title)

                    st.success("Analysis complete. See the other pages in the sidebar for results.")
                except Exception as e:
                    st.error(f"Something went wrong while processing your files: {e}")

    if st.session_state["resume_text"]:
        with st.expander("Preview extracted resume text"):
            st.write(clean_for_display(st.session_state["resume_text"]))
    if st.session_state["jd_text"]:
        with st.expander("Preview extracted job description text"):
            st.write(clean_for_display(st.session_state["jd_text"]))


# ---------------------------------------------------------------------------
# MATCH SCORE
# ---------------------------------------------------------------------------
elif page == "Match Score":
    st.title("Match Score")
    analysis = st.session_state["analysis"]
    if analysis is None:
        st.warning("Run an analysis first from the Upload & Analyze page.")
    else:
        score = analysis["match_score"]
        bucket = analysis["feedback"]["bucket"]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"Match Score — {bucket}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2E7D32" if score >= 75 else "#F9A825" if score >= 50 else "#C62828"},
                "steps": [
                    {"range": [0, 50], "color": "#FFEBEE"},
                    {"range": [50, 75], "color": "#FFF8E1"},
                    {"range": [75, 100], "color": "#E8F5E9"},
                ],
            },
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Calculated using {'semantic sentence embeddings' if analysis['score_method'] == 'semantic' else 'TF-IDF cosine similarity (fallback — sentence-transformer model unavailable)'}."
        )

        if analysis["predicted_category"]:
            st.subheader("Predicted resume category")
            cats = analysis["predicted_category"]
            fig2 = go.Figure(go.Bar(
                x=[c[1] for c in cats],
                y=[c[0] for c in cats],
                orientation="h",
                marker_color="#1565C0",
            ))
            fig2.update_layout(xaxis_title="Confidence (%)", yaxis_title="", height=300)
            st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# SKILL COMPARISON
# ---------------------------------------------------------------------------
elif page == "Skill Comparison":
    st.title("Skill Comparison")
    analysis = st.session_state["analysis"]
    if analysis is None:
        st.warning("Run an analysis first from the Upload & Analyze page.")
    else:
        sc = analysis["skill_comparison"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### ✅ Matching")
            st.write(", ".join(sc["matching_skills"]) or "None detected")
        with col2:
            st.markdown("#### ❌ Missing")
            st.write(", ".join(sc["missing_skills"]) or "None — great coverage!")
        with col3:
            st.markdown("#### ➕ Extra (not in JD)")
            st.write(", ".join(sc["extra_skills"]) or "None detected")

        st.markdown("---")
        total_jd = len(sc["jd_skills"])
        if total_jd:
            coverage = round(len(sc["matching_skills"]) / total_jd * 100, 1)
            st.metric("Required-skill coverage", f"{coverage}%", f"{len(sc['matching_skills'])}/{total_jd} skills")
        else:
            st.info("No recognizable skills were found in the job description text (from the curated skill list).")


# ---------------------------------------------------------------------------
# PERSONALIZED FEEDBACK
# ---------------------------------------------------------------------------
elif page == "Personalized Feedback":
    st.title("Personalized Feedback")
    analysis = st.session_state["analysis"]
    if analysis is None:
        st.warning("Run an analysis first from the Upload & Analyze page.")
    else:
        fb = analysis["feedback"]
        st.subheader(fb["bucket"])
        st.write(fb["summary"])

        st.markdown("#### Strengths")
        for s in fb["strengths"]:
            st.markdown(f"- {s}")

        st.markdown("#### Gaps")
        for g in fb["gaps"]:
            st.markdown(f"- {g}")

        st.markdown("#### Suggestions")
        for s in fb["suggestions"]:
            st.markdown(f"- {s}")

        if fb["classifier_note"]:
            st.markdown("#### Category check")
            st.info(fb["classifier_note"])


# ---------------------------------------------------------------------------
# CHATBOT
# ---------------------------------------------------------------------------
elif page == "Chatbot":
    st.title("Ask About Your Resume")
    analysis = st.session_state["analysis"]
    if analysis is None:
        st.warning("Run an analysis first from the Upload & Analyze page.")
    else:
        st.caption(
            "This chatbot only answers using your uploaded resume/JD analysis above — "
            "it won't make up information beyond that."
        )

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for role, msg in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.write(msg)

        suggestions = ["Why is my score low?", "Which skills are missing?", "How can I improve my resume?", "Am I suitable for this role?"]
        cols = st.columns(len(suggestions))
        clicked = None
        for c, s in zip(cols, suggestions):
            if c.button(s):
                clicked = s

        question = st.chat_input("Ask a question about your resume analysis...") or clicked

        if question:
            st.session_state["chat_history"].append(("user", question))
            answer = answer_question(question, analysis)
            st.session_state["chat_history"].append(("assistant", answer))
            st.rerun()


# ---------------------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------------------
elif page == "About":
    st.title("About This Project")
    st.markdown(
        """
This project is an AI-powered Resume Screening and Candidate Feedback System, built as a
data science / applied ML project.

**Pipeline**
1. **Parsing** — extracts text from uploaded PDF/TXT resumes and job descriptions.
2. **Preprocessing** — cleans text (lowercasing, stopword removal) before vectorization.
3. **Semantic Matching** — Sentence-Transformers (`all-MiniLM-L6-v2`) embeds both documents;
   cosine similarity gives the match score (falls back to TF-IDF similarity if the model
   can't be downloaded).
4. **Skill Comparison** — matches a curated ~95-skill vocabulary (built from a labelled resume
   dataset) against both documents to find matching/missing/extra skills.
5. **Classification** — a Logistic Regression model (TF-IDF features), trained on 3,500 labelled
   resumes across 36 job categories, predicts which role a resume most resembles, as an
   independent sanity check on top of the similarity score.
6. **Feedback Generation** — a template-based, retrieval-style engine turns the analysis into
   explainable, personalized feedback (every sentence traces back to a specific number or skill).
7. **Chatbot** — an intent-matching chatbot answers candidate questions using only the computed
   analysis, so it can't hallucinate information not already grounded in the documents.

**Dataset**: `resumes_dataset.jsonl` — 3,500 resumes across 36 job categories (used to train the
classifier and build the skill vocabulary).

See the README for the full write-up, folder structure, and future improvements.
        """
    )
