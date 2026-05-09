import pandas as pd
import numpy as np
import streamlit as st
import joblib

st.set_page_config(
    page_title="Lab 5.0 | Clearance Time Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load Models
@st.cache_resource
def load_models():
    return (
        joblib.load("classification_model.pkl"),
        joblib.load("regression_model.pkl")
    )

classification_model, regression_model = load_models()

# UI
st.title("🎓 AI 5.0 Lab 5.0 & 6.0")
st.subheader("Student Clearance Performance Predictor")
st.markdown("---")

tab1, tab2 = st.tabs([
    "💳 Payment Status (Classification)",
    "📅 Processing Days (Regression)"
])

# =========================
# CLASSIFICATION
# =========================
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        course = st.selectbox("Course", ["BSIT"])
        year = st.selectbox("Year Level", ["1st","2nd","3rd","4th"])
        sem = st.selectbox("Semester", ["1st","2nd"])

    with c2:
        dept = st.number_input("Department Count", 1, 10, 3)
        report = st.number_input("Report Score", 0, 10, 2)
        prev = st.number_input("Previous Approval Time (min)", 0, 5000, 120)

    course_val = 1
    year_val = {"1st":1,"2nd":2,"3rd":3,"4th":4}[year]
    sem_val = {"1st":1,"2nd":2}[sem]

    if st.button("🔍 PREDICT PAYMENT STATUS"):
        inp = np.array([[course_val, year_val, sem_val, dept, report, prev]])
        res = classification_model.predict(inp)[0]

        if res == 1:
            st.success("✅ RESULT: PAID")
        else:
            st.error("❌ RESULT: NOT PAID")

# =========================
# REGRESSION
# =========================
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        course_r = st.selectbox("Course", ["BSIT"], key="c_r")
        year_r = st.selectbox("Year Level", ["1st","2nd","3rd","4th"], key="y_r")
        sem_r = st.selectbox("Semester", ["1st","2nd"], key="s_r")

    with c2:
        payment = st.selectbox("Payment Status", [0,1])
        dept_r = st.number_input("Department Count ", 1, 10, 3)
        report_r = st.number_input("Report Score ", 0, 10, 2)
        prev_r = st.number_input("Previous Approval Time ", 0, 5000, 120)

    course_val_r = 1
    year_val_r = {"1st":1,"2nd":2,"3rd":3,"4th":4}[year_r]
    sem_val_r = {"1st":1,"2nd":2}[sem_r]

    if st.button("📊 PREDICT DAYS"):
        inp = np.array([[course_val_r, year_val_r, sem_val_r, payment, dept_r, report_r, prev_r]])
        res = regression_model.predict(inp)[0]
        st.info(f"⏳ Estimated Processing Days: {res:.2f}")