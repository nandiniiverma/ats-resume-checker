import streamlit as st

from utils import extract_text_from_pdf, preprocess_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(
    page_title="ATS Resume Checker",
    page_icon="📄",
    layout="wide"
)

st.title("📄 ATS Resume Score Checker")

st.write("Upload Resume and Paste Job Description")

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=250
)

if st.button("Check ATS Score"):

    if uploaded_file and job_description:
        TECH_SKILLS = [
    "python",
    "java",
    "c++",
    "sql",
    "aws",
    "linux",
    "git",
    "streamlit",
    "machine",
    "learning",
    "tensorflow",
    "pandas",
    "numpy",
    "rest",
    "api",
    "microservices",
    "oop",
    "algorithms",
    "data",
    "structures"
    ]

        resume_text = extract_text_from_pdf(uploaded_file)

        resume_text = preprocess_text(resume_text)
        job_description = preprocess_text(job_description)

        documents = [resume_text, job_description]

        vectorizer = TfidfVectorizer()

        tfidf_matrix = vectorizer.fit_transform(documents)

        score = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )[0][0]

        ats_score = round(score * 100, 2)

        st.success(f"ATS Match Score: {ats_score}%")
        st.progress(int(ats_score))
        if ats_score >= 80:
            st.success("Excellent Match")
        elif ats_score >= 60:
            st.warning("Good Match")
        else:
            st.error("Needs Improvement")

        resume_words = {
        word for word in resume_text.split()
        if word not in ENGLISH_STOP_WORDS
        }

        jd_words = {
            word for word in job_description.split()
            if word not in ENGLISH_STOP_WORDS
        }

        missing_keywords = []
        for skill in TECH_SKILLS:
            if skill in job_description and skill not in resume_text:
                missing_keywords.append(skill)

        top_missing = list(missing_keywords)[:20]

        st.subheader("Missing Keywords")

        for word in top_missing:
            st.write("❌", word)


    else:
        st.warning("Please upload resume and paste job description.")