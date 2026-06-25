import streamlit as st
import plotly.graph_objects as go

from utils import extract_text_from_pdf, preprocess_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(
    page_title="ATS Resume Checker",
    page_icon="📎",
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
    "machine learning",
    "tensorflow",
    "pandas",
    "numpy",
    "rest",
    "api",
    "microservices",
    "oop",
    "algorithms",
    "data structures"
    
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

        

        

        resume_words = {
        word for word in resume_text.split()
        if word not in ENGLISH_STOP_WORDS
        }

        jd_words = {
            word for word in job_description.split()
            if word not in ENGLISH_STOP_WORDS
        }

        missing_keywords = []
        matched_keywords=[]
        for skill in TECH_SKILLS:
            if skill in job_description and skill in resume_text:
                matched_keywords.append(skill)
            elif skill in job_description and skill not in resume_text:
                missing_keywords.append(skill)
        total_keywords=len(matched_keywords)+len(missing_keywords)

        if total_keywords>0:
            keyword_score=(len(matched_keywords)/total_keywords)*100
        else:
            keyword_score=0

        similarity_score = round(score * 100, 2)

        final_score = round(
        0.3 * similarity_score +
        0.7 * keyword_score,
        2
        )
        st.success(f"ATS Match Score: {final_score}%")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_score,

            title={"text": "ATS Score"},

            gauge={
                "axis": {"range": [0, 100]},

                "steps": [
                    {"range": [0, 40],"color": "#ff6b6b"},
                    {"range": [40, 70],"color": "#f1c30e"},
                    {"range": [70, 100],"color":"#52ca49"}
                ]
            }
        ))
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Matched Skills",
                len(matched_keywords)
            )

        with col2:
            st.metric(
                "Missing Skills",
                len(missing_keywords)
            )

        with col3:
            st.metric(
                "ATS Score",
                f"{final_score}%"
            )
        st.plotly_chart(
            fig,
            use_container_width=True
        )
        if final_score >= 80:
            st.success("Excellent Match")
        elif final_score >= 60:
            st.warning("Good Match")
        else:
            st.error("Needs Improvement")

        top_missing = list(missing_keywords)[:20]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Matched Keywords")

            for word in matched_keywords:
                st.write(word)

        with col2:
            st.subheader("❌ Missing Keywords")

            for word in top_missing:
                 st.write(word)

       

        st.subheader("Resume Suggestions")
        if missing_keywords:
            st.write(
                "consider adding these skills to your resume if you have worked with them."
            )    
        else:
            st.success("great! your resume covers most of the requires skills")

        
        



    else:
        st.warning("Please upload resume and paste job description.")
