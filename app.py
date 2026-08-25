"""
AI Career Intelligence Platform
=================================
Streamlit application implementing the "Upgraded Version" pipeline:

Student Profile -> Dashboard -> Explainable AI -> Skill Gap Radar ->
Job Role Matcher -> Resume Analyzer -> Personalized Roadmap ->
Industry Benchmarking -> Placement Simulator (What-If) -> Progress
Tracker -> AI Career Chatbot -> Smart Recommendations

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import json
import datetime as dt

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

from job_roles import JOB_ROLES, RESUME_SKILL_KEYWORDS
import mentor

# --------------------------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="CareerTwin AI | Employability & Placement Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "models")
DATA_PATH = os.path.join(BASE, "data", "student_employability_dataset.csv")
LOG_PATH = os.path.join(BASE, "data", "progress_log.csv")

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    :root {
        --ink: #12142b;
        --muted: #6b7280;
        --violet: #7c3aed;
        --violet-dark: #5b21b6;
        --teal: #14b8a6;
        --amber: #f59e0b;
        --rose: #f43f5e;
        --card-bg: rgba(255,255,255,0.72);
        --card-border: rgba(124,58,237,0.14);
    }

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Poppins', sans-serif; color: var(--ink); }
    p, span, label, div { color: var(--ink); }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(124,58,237,0.10) 0%, transparent 45%),
            radial-gradient(circle at 95% 10%, rgba(20,184,166,0.10) 0%, transparent 40%),
            linear-gradient(180deg, #fbfbff 0%, #f3f2fb 100%);
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14102b 0%, #1d1740 55%, #241a52 100%);
        border-right: none;
    }
    section[data-testid="stSidebar"] * { color: #eae7fb !important; }
    section[data-testid="stSidebar"] h3 { color: #ffffff !important; letter-spacing: .3px; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.14); }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stTextInput label { color: #c9c3f0 !important; font-weight: 600; font-size: 0.82rem; }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] input {
        background-color: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        color: #fff !important; border-radius: 10px;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px; padding: 8px 10px; margin-bottom: 6px;
        transition: all .15s ease;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(124,58,237,0.35); border-color: rgba(124,58,237,0.6);
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] { background-color: var(--teal) !important; }
    .stSlider [data-baseweb="slider"] > div > div { background: linear-gradient(90deg, var(--violet), var(--teal)) !important; }

    /* ---------- GENERIC ---------- */
    .metric-card {
        background: var(--card-bg); backdrop-filter: blur(6px);
        border-radius: 18px; padding: 20px 22px; border: 1px solid var(--card-border);
        box-shadow: 0 8px 24px rgba(76,29,149,0.08);
        border-top: 4px solid var(--violet);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .metric-card:hover { transform: translateY(-4px); box-shadow: 0 14px 30px rgba(76,29,149,0.16); }
    .metric-card b { color: var(--muted); font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
    .metric-card h2 { margin: 4px 0 8px 0; font-size: 1.9rem; }
    .metric-card h3 { margin: 4px 0 8px 0; }

    .badge {
        display:inline-block; padding: 4px 14px; border-radius: 999px;
        font-weight: 700; font-size: 0.78rem; margin-right: 6px; letter-spacing: .02em;
    }
    .badge-green { background:#e6faf3; color:#0f9d68; border:1px solid #9be8cc;}
    .badge-yellow{ background:#fff6df; color:#b7791f; border:1px solid #fbd899;}
    .badge-red   { background:#ffe9ec; color:#c81e3a; border:1px solid #f7aab8;}

    .section-title {
        font-size: 1.5rem; font-weight: 800; margin-top: 0.8rem; margin-bottom: 0.6rem;
        padding-left: 14px; color: var(--ink);
        border-left: 6px solid var(--violet);
        background: linear-gradient(90deg, rgba(124,58,237,0.08), transparent 70%);
        border-radius: 6px; padding-top: 4px; padding-bottom: 4px;
    }

    .pill {
        display:inline-block; background:#f1edfd; border:1px solid #ded4fb;
        border-radius: 999px; padding: 6px 14px; margin: 3px; font-size:0.82rem;
        color: var(--violet-dark); font-weight: 600;
    }

    div[data-testid="stMetricValue"] { font-size: 1.7rem; color: var(--ink); font-family: 'Poppins', sans-serif; }
    div[data-testid="stMetricLabel"] { color: var(--muted); }

    div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, var(--violet) 0%, #4f46e5 100%);
        color: #ffffff !important; border: none; border-radius: 12px;
        padding: 0.6rem 1.4rem; font-weight: 700; letter-spacing: .01em;
        transition: transform 0.15s ease, box-shadow 0.15s ease, background-color 0.15s ease;
        box-shadow: 0 4px 14px rgba(99,102,241,0.35);
    }
    div[data-testid="stButton"] button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px); box-shadow: 0 10px 22px rgba(79,70,229,0.4);
        color: #ffffff !important; border: none;
    }

    /* ---------- HERO ---------- */
    .hero {
        position: relative; overflow: hidden;
        margin: 4px 0 28px 0; padding: 34px 34px; border-radius: 22px;
        background: linear-gradient(120deg, #3b0764 0%, #6d28d9 45%, #14b8a6 130%);
        color: #ffffff; box-shadow: 0 16px 40px rgba(76,29,149,0.35);
    }
    .hero::after {
        content: ""; position: absolute; top: -60px; right: -60px;
        width: 220px; height: 220px; border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 70%);
    }
    .hero-eyebrow {
        display:inline-block; background: rgba(255,255,255,0.16); color:#fff;
        padding: 4px 14px; border-radius: 999px; font-size: 0.75rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase; margin-bottom: 10px;
    }
    .hero h1 { color: #ffffff; margin: 4px 0 6px 0; font-size: 2.1rem; font-weight: 800; }
    .hero p { color: #ece9ff; margin: 0; font-size: 1.02rem; max-width: 640px; }

    /* ---------- HOW IT WORKS ---------- */
    .how-wrap { display:flex; flex-wrap:wrap; gap: 14px; margin-top: 10px; }
    .how-step {
        flex: 1 1 210px; position: relative; background:#ffffff; border:1px solid var(--card-border);
        border-radius: 18px; padding: 20px 18px 18px 18px; box-shadow: 0 6px 18px rgba(30,20,60,0.06);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .how-step:hover { transform: translateY(-4px); box-shadow: 0 14px 28px rgba(124,58,237,0.18); }
    .how-num {
        position: absolute; top: -14px; left: 18px; width: 30px; height: 30px; border-radius: 50%;
        background: linear-gradient(135deg, var(--violet), var(--teal)); color: #fff; font-weight: 800;
        display:flex; align-items:center; justify-content:center; font-size: 0.9rem;
        box-shadow: 0 4px 10px rgba(76,29,149,0.35);
    }
    .how-icon { font-size: 1.6rem; margin: 8px 0 8px 0; }
    .how-title { font-weight: 700; font-size: 0.98rem; color: var(--ink); margin-bottom: 4px; }
    .how-desc { font-size: 0.83rem; color: var(--muted); line-height: 1.4; }
    .how-arrow { align-self: center; font-size: 1.4rem; color: var(--violet); opacity: 0.5; padding: 0 2px; }

    /* ---------- FEATURE GRID ---------- */
    .feat-grid { display:flex; flex-wrap:wrap; gap: 14px; margin-top: 16px; }
    .feat-card {
        flex: 1 1 220px; background: linear-gradient(160deg, #ffffff 0%, #faf9ff 100%);
        border: 1px solid var(--card-border); border-radius: 18px;
        padding: 18px 18px; box-shadow: 0 6px 16px rgba(30,20,60,0.06);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color .18s ease;
    }
    .feat-card:hover { transform: translateY(-5px); box-shadow: 0 16px 30px rgba(124,58,237,0.18); border-color: rgba(124,58,237,0.4); }
    .feat-icon {
        width: 42px; height: 42px; border-radius: 12px; display:flex; align-items:center; justify-content:center;
        font-size: 1.25rem; background: linear-gradient(135deg, rgba(124,58,237,0.14), rgba(20,184,166,0.14));
        margin-bottom: 10px;
    }
    .feat-title { font-weight:700; font-size:0.95rem; color:var(--ink); margin-bottom:4px; }
    .feat-desc { font-size:0.82rem; color:#6b7280; line-height:1.35; }

    /* ---------- ROADMAP ---------- */
    .roadmap-item {
        border-radius: 16px; padding: 16px 20px; margin-bottom: 12px; border-left: 6px solid var(--violet);
        background: #ffffff; box-shadow: 0 4px 14px rgba(30,20,60,0.07);
        transition: transform .15s ease;
    }
    .roadmap-item:hover { transform: translateX(3px); }

    /* ---------- CHAT ---------- */
    .chat-bubble-user {
        background: linear-gradient(135deg, var(--violet), #4f46e5); color:#fff; padding:10px 16px;
        border-radius: 16px 16px 3px 16px; margin: 6px 0; display:inline-block; max-width: 80%;
        box-shadow: 0 4px 10px rgba(79,70,229,0.3);
    }
    .chat-bubble-bot {
        background:#f1edfd; color: var(--ink); padding:10px 16px; border-radius: 16px 16px 16px 3px;
        margin: 6px 0; display:inline-block; max-width: 85%; border: 1px solid #e2d9fb;
    }

    /* ---------- SIDEBAR BRAND ---------- */
    .sidebar-brand-title { font-size: 1.15rem; font-weight: 800; color: #ffffff !important; margin-bottom: 2px; }
    .sidebar-brand-tag { font-size: 0.78rem; color: #b7aef0 !important; margin-bottom: 4px; }
    .sidebar-workflow { font-size: 0.72rem; line-height: 1.6; color: #b7aef0 !important; }

    /* ---------- QUICK TAKE ---------- */
    .quicktake-card {
        display:flex; gap:16px; align-items:flex-start;
        background: linear-gradient(135deg, #ede9fe 0%, #e6fbf6 100%);
        border: 1px solid rgba(124,58,237,0.22); border-radius: 18px;
        padding: 20px 24px; box-shadow: 0 10px 26px rgba(76,29,149,0.12);
    }
    .quicktake-icon {
        width: 46px; height: 46px; min-width: 46px; border-radius: 14px;
        background: linear-gradient(135deg, var(--violet), var(--teal));
        display:flex; align-items:center; justify-content:center; font-size: 1.4rem;
        box-shadow: 0 6px 14px rgba(76,29,149,0.3);
    }
    .quicktake-label {
        font-weight: 800; color: var(--violet-dark); font-size: 0.78rem;
        text-transform: uppercase; letter-spacing: .08em; margin-bottom: 4px;
    }
    .quicktake-text { color: var(--ink); font-size: 0.97rem; line-height: 1.55; }

    /* ---------- STATUS BANNER ---------- */
    .status-banner {
        border-radius: 14px; padding: 14px 20px; margin: 6px 0 18px 0;
        font-weight: 600; font-size: 0.92rem; display:flex; align-items:center; gap:10px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CHART_FONT_COLOR = "#12142b"
CHART_PAPER_BG = "rgba(0,0,0,0)"
CHART_PLOT_BG = "rgba(0,0,0,0)"

# --------------------------------------------------------------------------------------
# LOAD MODEL ARTIFACTS
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    clf = joblib.load(os.path.join(MODEL_DIR, "placement_classifier.pkl"))
    reg = joblib.load(os.path.join(MODEL_DIR, "package_regressor.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return clf, reg, scaler, encoders, feature_columns, metrics


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


clf, reg, scaler, encoders, FEATURE_COLUMNS, METRICS = load_artifacts()
df_reference = load_dataset()

NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in ["Gender", "Branch", "Placement_Training"]]
CATEGORICAL_FEATURES = ["Gender", "Branch", "Placement_Training"]

try:
    import shap
    SHAP_AVAILABLE = True
    explainer = shap.TreeExplainer(clf)
except Exception:
    SHAP_AVAILABLE = False
    explainer = None

# --------------------------------------------------------------------------------------
# CORE HELPER FUNCTIONS
# --------------------------------------------------------------------------------------
def build_feature_row(profile: dict) -> pd.DataFrame:
    row = {}
    for col in NUMERIC_FEATURES:
        row[col] = profile[col]
    for col in CATEGORICAL_FEATURES:
        le = encoders[col]
        val = profile[col]
        if val not in le.classes_:
            val = le.classes_[0]
        row[col] = le.transform([val])[0]
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def compute_employability_score(profile: dict) -> float:
    score = (
        profile["CGPA"] / 10 * 20 + min(profile["Internships"], 3) / 3 * 12 +
        min(profile["Projects"], 5) / 5 * 10 + min(profile["Certifications"], 4) / 4 * 6 +
        profile["Aptitude_Score"] / 100 * 14 + profile["Technical_Skill"] / 10 * 12 +
        profile["Coding_Skill"] / 10 * 8 + profile["Communication_Skill"] / 10 * 8 +
        profile["Soft_Skill"] / 10 * 4 + (3 if profile["Placement_Training"] == "Yes" else 0) +
        profile["Extracurricular_Score"] / 10 * 2 + profile["Leadership_Score"] / 10 * 1 -
        profile["Backlogs"] * 4.2
    )
    return float(np.clip(score, 0, 100))


def predict_profile(profile: dict):
    X = build_feature_row(profile)
    X_scaled = scaler.transform(X)
    proba = clf.predict_proba(X_scaled)[0, 1]
    label = int(proba >= 0.5)
    package = float(reg.predict(X_scaled)[0]) if proba >= 0.35 else 0.0
    return label, float(proba), max(package, 0.0)


def get_explanation(profile: dict, top_n=8):
    X = build_feature_row(profile)
    X_scaled = scaler.transform(X)

    if SHAP_AVAILABLE:
        try:
            shap_values = explainer.shap_values(X_scaled)
            vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
            contrib = dict(zip(FEATURE_COLUMNS, vals))
            expl_df = pd.DataFrame({"Feature": list(contrib.keys()), "Impact": list(contrib.values())})
            expl_df["AbsImpact"] = expl_df["Impact"].abs()
            expl_df = expl_df.sort_values("AbsImpact", ascending=False).head(top_n)
            expl_df["Direction"] = np.where(expl_df["Impact"] > 0, "Increases", "Decreases")
            return expl_df[["Feature", "Impact", "Direction"]], "SHAP (exact model attribution)"
        except Exception:
            pass

    importances = METRICS["feature_importances"]
    means = df_reference[NUMERIC_FEATURES].mean().to_dict()
    stds = df_reference[NUMERIC_FEATURES].std().to_dict()
    rows = []
    for feat in FEATURE_COLUMNS:
        imp = importances.get(feat, 0.01)
        if feat in NUMERIC_FEATURES:
            z = (profile[feat] - means[feat]) / (stds[feat] + 1e-6)
            if feat == "Backlogs":
                z = -z
            impact = imp * z
        else:
            impact = imp * (0.5 if profile[feat] in ["Yes"] else 0.0)
        rows.append({"Feature": feat, "Impact": impact})
    expl_df = pd.DataFrame(rows)
    expl_df["AbsImpact"] = expl_df["Impact"].abs()
    expl_df = expl_df.sort_values("AbsImpact", ascending=False).head(top_n)
    expl_df["Direction"] = np.where(expl_df["Impact"] > 0, "Increases", "Decreases")
    return expl_df[["Feature", "Impact", "Direction"]], "Feature-importance heuristic (install `shap` for exact attribution)"


def compute_role_match(profile: dict):
    results = []
    for role, spec in JOB_ROLES.items():
        weights = spec["weights"]
        total, achieved = 0.0, 0.0
        gaps = []
        for feat, target in weights.items():
            if feat == "Backlogs_Inverse":
                actual = max(0, 10 - profile["Backlogs"] * 2.5)
                target_val = target
            else:
                actual = profile.get(feat, 0)
                target_val = target
            total += target_val
            ratio = min(actual / target_val, 1.0) if target_val > 0 else 1.0
            achieved += ratio * target_val
            if actual < target_val:
                gaps.append({
                    "skill": feat.replace("_", " ").replace("Inverse", "(fewer backlogs)"),
                    "current": round(actual, 1), "required": target_val, "gap": round(target_val - actual, 1),
                })
        match_pct = round((achieved / total) * 100, 1) if total > 0 else 0
        gaps = sorted(gaps, key=lambda g: -g["gap"])
        results.append({
            "role": role, "icon": spec["icon"], "match": match_pct, "gaps": gaps,
            "core_skills": spec["core_skills"], "description": spec["description"], "weights": weights,
        })
    return sorted(results, key=lambda r: -r["match"])


def extract_resume_text(uploaded_file):
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        import pdfplumber
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    return uploaded_file.read().decode("utf-8", errors="ignore")


def analyze_resume(text: str):
    text_lower = text.lower()
    found = {}
    for category, keywords in RESUME_SKILL_KEYWORDS.items():
        matched = [kw for kw in keywords if kw in text_lower]
        found[category] = matched
    total_possible = sum(len(v) for v in RESUME_SKILL_KEYWORDS.values())
    total_found = sum(len(v) for v in found.values())
    coverage = round((total_found / total_possible) * 100, 1) if total_possible else 0
    word_count = len(text.split())
    return found, coverage, word_count


def generate_improvement_plan(profile: dict, probability: float, gaps: list):
    plan = []
    if profile["CGPA"] < 7.0:
        plan.append(("Academics", "Raise CGPA above 7.0", "Aim for consistent 8+ in upcoming semesters; target weak subjects with weekly revision.", "High"))
    if profile["Backlogs"] > 0:
        plan.append(("Academics", "Clear pending backlogs", "Backlogs strongly reduce placement odds — prioritize clearing them next attempt.", "Critical"))
    if profile["Internships"] < 2:
        plan.append(("Experience", "Complete at least 2 internships", "Apply to 2-3 month internships (virtual internships on Internshala/AICTE count too).", "High"))
    if profile["Projects"] < 3:
        plan.append(("Experience", "Build 3+ solid projects", "Pick projects showing end-to-end skills and host them on GitHub.", "High"))
    if profile["Certifications"] < 2:
        plan.append(("Skills", "Earn 2+ relevant certifications", "NPTEL / Coursera / Udemy certifications in your target domain build credibility fast.", "Medium"))
    if profile["Aptitude_Score"] < 65:
        plan.append(("Aptitude", "Improve quantitative & logical aptitude", "Practice 30 mins/day on IndiaBix/PrepInsta; most core companies screen with aptitude tests.", "High"))
    if profile["Technical_Skill"] < 6.5 or profile["Coding_Skill"] < 6.5:
        plan.append(("Skills", "Strengthen coding & technical fundamentals", "Solve 3 DSA problems/day on LeetCode/HackerRank; revise core CS subjects.", "Critical"))
    if profile["Communication_Skill"] < 6.0:
        plan.append(("Soft Skills", "Improve communication skills", "Join a speaking club, do 2 mock interviews a week, practice structured self-intro.", "Medium"))
    if profile["Placement_Training"] == "No":
        plan.append(("Preparation", "Enroll in placement training", "Structured training covering resume, GD, and interviews meaningfully lifts placement odds.", "Medium"))
    if profile["LinkedIn_GitHub_Activity"] < 5:
        plan.append(("Visibility", "Build your LinkedIn & GitHub presence", "Recruiters screen profiles — post projects, keep GitHub active, network on LinkedIn.", "Low"))
    if gaps:
        top_gap = gaps[0]
        plan.append(("Role Fit", f"Close gap in {top_gap['skill']}", f"You're {top_gap['gap']} points below target for your best-matching role — focused practice here has outsized impact.", "High"))

    priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    plan = sorted(plan, key=lambda x: priority_order.get(x[3], 4))
    if not plan:
        plan.append(("Overall", "Maintain your strong profile", "You're in great shape — keep practicing mock interviews and stay updated with industry trends.", "Low"))
    return plan


def probability_badge(prob):
    if prob >= 0.70:
        return '<span class="badge badge-green">High Chance</span>'
    elif prob >= 0.40:
        return '<span class="badge badge-yellow">Moderate Chance</span>'
    return '<span class="badge badge-red">Needs Improvement</span>'


def priority_color(p):
    return {"Critical": "#dc2626", "High": "#ea580c", "Medium": "#ca8a04", "Low": "#16a34a"}.get(p, "#6366f1")


def load_progress_log():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=["timestamp", "student_name", "probability", "employability_score", "package"])


def save_progress_entry(name, probability, emp_score, package):
    log = load_progress_log()
    new_row = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "student_name": name, "probability": round(probability * 100, 2),
        "employability_score": round(emp_score, 2), "package": round(package, 2),
    }
    log = pd.concat([log, pd.DataFrame([new_row])], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)
    return log


def gauge_chart(value, title, max_val=100, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": suffix, "font": {"color": CHART_FONT_COLOR}},
        title={"text": title, "font": {"color": CHART_FONT_COLOR, "size": 16}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": CHART_FONT_COLOR},
            "bar": {"color": "#7c3aed"},
            "steps": [
                {"range": [0, max_val * 0.4], "color": "#ffe9ec"},
                {"range": [max_val * 0.4, max_val * 0.7], "color": "#fff6df"},
                {"range": [max_val * 0.7, max_val], "color": "#e6faf3"},
            ],
        },
    ))
    fig.update_layout(paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG, height=260,
                       margin=dict(l=20, r=20, t=50, b=10), font_color=CHART_FONT_COLOR)
    return fig


# --------------------------------------------------------------------------------------
# SESSION STATE DEFAULTS
# --------------------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "section" not in st.session_state:
    st.session_state.section = "🏠 Dashboard"

_PROFILE_DEFAULTS = {
    "Name": "Student",
    "Gender": encoders["Gender"].classes_.tolist()[0],
    "Branch": encoders["Branch"].classes_.tolist()[0],
    "CGPA": 7.5,
    "Backlogs": 0,
    "Internships": 1,
    "Projects": 2,
    "Certifications": 1,
    "Aptitude_Score": 60,
    "Technical_Skill": 6.0,
    "Coding_Skill": 6.0,
    "Communication_Skill": 6.5,
    "Soft_Skill": 6.0,
    "Extracurricular_Score": 5.0,
    "Leadership_Score": 5.0,
    "LinkedIn_GitHub_Activity": 5.0,
    "Placement_Training": "Yes",
}
if "profile_data" not in st.session_state:
    st.session_state.profile_data = dict(_PROFILE_DEFAULTS)
PD = st.session_state.profile_data  # single persistent source of truth — only changes when the user edits it

# --------------------------------------------------------------------------------------
# SIDEBAR — BRANDING + NAVIGATION ONLY (profile now lives on the main screen)
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand-title">🎯 CareerTwin AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-brand-tag">Student Employability &amp; Placement Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🧭 Navigate")
    st.session_state.section = st.radio(
        "Go to", [
            "🏠 Dashboard", "ℹ️ About", "🧠 Explainable AI", "🎯 Skill Gap Radar", "💼 Job Role Matcher",
            "📄 Resume Analyzer", "🗺️ Personalized Roadmap", "🏛️ Industry Benchmarking",
            "🎚️ Placement Simulator", "📈 Progress Tracker", "⭐ Smart Recommendations",
            "💬 AI Career Chatbot",
        ], label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<div class="sidebar-workflow"><b>Workflow</b><br>'
        'Understand → Predict → Diagnose →<br>Optimize → Simulate → Act → Track</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Model trained on {len(df_reference):,} student records")

section = st.session_state.section

# ========================================================================================
# 1a. DASHBOARD — HERO + STUDENT PROFILE EDITOR
# (rendered *before* the shared prediction block below so any edit this run is reflected
#  immediately, and PD is otherwise left untouched on every other page/navigation)
# ========================================================================================
if section == "🏠 Dashboard":
    st.markdown(
        """
        <div class="hero">
            <span class="hero-eyebrow">✨ CareerTwin AI</span>
            <h1>AI-Based Student Employability &amp; Placement Prediction</h1>
            <p>From a basic prediction tool to a complete AI Career Intelligence ecosystem —
            grounded in your own live profile data, updated instantly as you change it.</p>
        </div>
        """, unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">👤 Student Profile</div>', unsafe_allow_html=True)
    with st.expander("✏️ Edit your Career Twin profile", expanded=False):
        gender_opts = encoders["Gender"].classes_.tolist()
        branch_opts = encoders["Branch"].classes_.tolist()
        p1, p2, p3 = st.columns(3)
        with p1:
            PD["Name"] = st.text_input("Name", value=PD["Name"], key="w_name")
            PD["Gender"] = st.selectbox("Gender", gender_opts, index=gender_opts.index(PD["Gender"]), key="w_gender")
            PD["Branch"] = st.selectbox("Branch", branch_opts, index=branch_opts.index(PD["Branch"]), key="w_branch")
            PD["CGPA"] = st.slider("CGPA", 4.0, 10.0, value=PD["CGPA"], step=0.1, key="w_cgpa")
            PD["Backlogs"] = st.number_input("Active Backlogs", 0, 10, value=PD["Backlogs"], key="w_backlogs")
        with p2:
            PD["Internships"] = st.number_input("Internships completed", 0, 6, value=PD["Internships"], key="w_internships")
            PD["Projects"] = st.number_input("Projects completed", 0, 10, value=PD["Projects"], key="w_projects")
            PD["Certifications"] = st.number_input("Certifications", 0, 10, value=PD["Certifications"], key="w_certifications")
            PD["Aptitude_Score"] = st.slider("Aptitude Score (/100)", 0, 100, value=PD["Aptitude_Score"], key="w_aptitude")
            PD["Placement_Training"] = st.selectbox("Placement Training Done?", ["Yes", "No"],
                                                      index=0 if PD["Placement_Training"] == "Yes" else 1, key="w_training")
        with p3:
            PD["Technical_Skill"] = st.slider("Technical Skill (/10)", 0.0, 10.0, value=PD["Technical_Skill"], step=0.1, key="w_technical")
            PD["Coding_Skill"] = st.slider("Coding Skill (/10)", 0.0, 10.0, value=PD["Coding_Skill"], step=0.1, key="w_coding")
            PD["Communication_Skill"] = st.slider("Communication Skill (/10)", 0.0, 10.0, value=PD["Communication_Skill"], step=0.1, key="w_communication")
            PD["Soft_Skill"] = st.slider("Soft Skill (/10)", 0.0, 10.0, value=PD["Soft_Skill"], step=0.1, key="w_soft")
            PD["Extracurricular_Score"] = st.slider("Extracurricular Score (/10)", 0.0, 10.0, value=PD["Extracurricular_Score"], step=0.1, key="w_extracurricular")
            PD["Leadership_Score"] = st.slider("Leadership Score (/10)", 0.0, 10.0, value=PD["Leadership_Score"], step=0.1, key="w_leadership")
            PD["LinkedIn_GitHub_Activity"] = st.slider("LinkedIn/GitHub Activity (/10)", 0.0, 10.0, value=PD["LinkedIn_GitHub_Activity"], step=0.1, key="w_linkedin")
        st.caption("Changes apply instantly across every section of the app and stay saved until you edit them again.")

# --------------------------------------------------------------------------------------
# SHARED COMPUTATIONS — used across every section, always built from the persisted PD
# --------------------------------------------------------------------------------------
name = PD["Name"]
profile = {
    "CGPA": PD["CGPA"], "Internships": PD["Internships"], "Projects": PD["Projects"],
    "Certifications": PD["Certifications"], "Aptitude_Score": PD["Aptitude_Score"],
    "Technical_Skill": PD["Technical_Skill"], "Coding_Skill": PD["Coding_Skill"],
    "Communication_Skill": PD["Communication_Skill"], "Soft_Skill": PD["Soft_Skill"],
    "Backlogs": PD["Backlogs"], "Extracurricular_Score": PD["Extracurricular_Score"],
    "Leadership_Score": PD["Leadership_Score"], "LinkedIn_GitHub_Activity": PD["LinkedIn_GitHub_Activity"],
    "Placement_Training": PD["Placement_Training"], "Gender": PD["Gender"], "Branch": PD["Branch"],
}

label, probability, package = predict_profile(profile)
emp_score = compute_employability_score(profile)
role_matches = compute_role_match(profile)
top_role = role_matches[0]
plan = generate_improvement_plan(profile, probability, top_role["gaps"])

# ========================================================================================
# 1b. DASHBOARD — METRICS, STATUS & QUICK TAKE
# ========================================================================================
if section == "🏠 Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><b>Placement Probability</b><br><h2>{probability*100:.1f}%</h2>{probability_badge(probability)}</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><b>Employability Score</b><br><h2>{emp_score:.1f}/100</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><b>Predicted Package</b><br><h2>₹{package:.2f} LPA</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><b>Best-Fit Role</b><br><h3>{top_role["icon"]} {top_role["role"]}</h3>{top_role["match"]}% match</div>', unsafe_allow_html=True)

    if probability >= 0.70:
        _bstyle, _bicon, _btext = "background:#e6faf3;color:#0f9d68;border:1px solid #9be8cc;", "🟢", "Excellent placement readiness. Your Career Twin shows a strong profile — focus on closing the remaining gaps."
    elif probability >= 0.40:
        _bstyle, _bicon, _btext = "background:#fff6df;color:#b7791f;border:1px solid #fbd899;", "🟡", "Moderate placement readiness. A few focused improvements can meaningfully raise your odds."
    else:
        _bstyle, _bicon, _btext = "background:#ffe9ec;color:#c81e3a;border:1px solid #f7aab8;", "🔴", "Placement readiness needs work. Check your Personalized Roadmap for the highest-impact next steps."
    st.markdown(f'<div class="status-banner" style="{_bstyle}"><span>{_bicon}</span><span>{_btext}</span></div>', unsafe_allow_html=True)

    # ---------------- QUICK TAKE ----------------
    st.markdown('<div class="section-title">🤖 Quick Take</div>', unsafe_allow_html=True)
    _quick_take = mentor.answer_question("summary", profile, probability, emp_score, package, role_matches, top_role["gaps"], plan)
    st.markdown(
        f"""
        <div class="quicktake-card">
            <div class="quicktake-icon">🤖</div>
            <div>
                <div class="quicktake-label">AI Career Twin says</div>
                <div class="quicktake-text">{_quick_take}</div>
            </div>
        </div>
        """, unsafe_allow_html=True,
    )

# ========================================================================================
# ABOUT — HOW IT WORKS + KEY FEATURES
# ========================================================================================
elif section == "ℹ️ About":
    st.markdown('<div class="section-title">🧭 How It Works</div>', unsafe_allow_html=True)
    how_steps = [
        ("👤", "Build Your Profile", "Set your academics, skills and activity on the Dashboard — it drives everything below."),
        ("🤖", "Get an AI Prediction", "A trained model instantly estimates placement probability and expected package."),
        ("🔍", "Diagnose the Gaps", "Explainable AI and Skill Gap Radar show exactly what's helping or hurting you."),
        ("🚀", "Act & Track Progress", "Follow your personalized roadmap, simulate what-ifs, and log progress over time."),
    ]
    how_html = '<div class="how-wrap">'
    for i, (icon, title, desc) in enumerate(how_steps, 1):
        how_html += (
            f'<div class="how-step"><div class="how-num">{i}</div>'
            f'<div class="how-icon">{icon}</div>'
            f'<div class="how-title">{title}</div>'
            f'<div class="how-desc">{desc}</div></div>'
        )
    how_html += "</div>"
    st.markdown(how_html, unsafe_allow_html=True)

    st.markdown('<div class="section-title">⚡ Key Features</div>', unsafe_allow_html=True)
    features = [
        ("💬", "AI Career Chatbot", "Personalized advice"),
        ("🎯", "Skill Gap Radar", "Skills vs demand"),
        ("🗺️", "Personalized Roadmap", "Step-by-step plan"),
        ("🏛️", "Industry Benchmarking", "Compare with peers"),
        ("🎚️", "Placement Simulator", "Live what-ifs"),
        ("🧠", "Explainable AI", "See prediction drivers"),
        ("💼", "Job Role Matcher", "Best-fit roles"),
        ("📄", "Resume Analyzer", "Instant skill coverage"),
        ("📈", "Progress Tracker", "Track growth"),
        ("⭐", "Smart Recommendations", "Next-best actions"),
    ]
    feat_html = '<div class="feat-grid">'
    for icon, title, desc in features:
        feat_html += (
            f'<div class="feat-card" style="padding:14px 16px;">'
            f'<div class="feat-icon" style="width:34px;height:34px;font-size:1.05rem;margin-bottom:6px;">{icon}</div>'
            f'<div class="feat-title" style="font-size:0.88rem;margin-bottom:2px;">{title}</div>'
            f'<div class="feat-desc" style="font-size:0.76rem;">{desc}</div></div>'
        )
    feat_html += "</div>"
    st.markdown(feat_html, unsafe_allow_html=True)

# ========================================================================================
# 2. EXPLAINABLE AI
# ========================================================================================
elif section == "🧠 Explainable AI":
    st.markdown('<div class="section-title">🧠 Explainable AI — Why this prediction?</div>', unsafe_allow_html=True)
    expl_df, method = get_explanation(profile)
    st.caption(f"Attribution method: {method}")

    colors = ["#16a34a" if d == "Increases" else "#dc2626" for d in expl_df["Direction"]]
    fig = go.Figure(go.Bar(
        x=expl_df["Impact"], y=expl_df["Feature"], orientation="h",
        marker_color=colors, text=expl_df["Direction"], textposition="outside",
    ))
    fig.update_layout(
        title="Top factors influencing your placement prediction", paper_bgcolor=CHART_PAPER_BG,
        plot_bgcolor=CHART_PLOT_BG, font_color=CHART_FONT_COLOR, height=440,
        xaxis_title="Impact on prediction", yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(expl_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{METRICS['accuracy']*100:.1f}%")
    m2.metric("F1 Score", f"{METRICS['f1_score']:.2f}")
    m3.metric("ROC-AUC", f"{METRICS['roc_auc']:.2f}")
    m4.metric("Package MAE", f"±{METRICS['package_mae']} LPA")

# ========================================================================================
# 3. SKILL GAP RADAR
# ========================================================================================
elif section == "🎯 Skill Gap Radar":
    st.markdown('<div class="section-title">🎯 Skill Gap Radar</div>', unsafe_allow_html=True)
    role_names = [r["role"] for r in role_matches]
    chosen = st.selectbox("Compare against role", role_names, index=0)
    chosen_role = next(r for r in role_matches if r["role"] == chosen)

    axes, current_vals, target_vals = [], [], []
    for feat, target in chosen_role["weights"].items():
        if feat == "Backlogs_Inverse":
            actual = max(0, 10 - profile["Backlogs"] * 2.5)
            norm_target = target
            norm_actual = actual
        else:
            actual = profile.get(feat, 0)
            # normalize aptitude (0-100) onto a comparable 0-10-ish scale for the radar
            if feat == "Aptitude_Score":
                actual, target = actual / 10, target / 10
            elif feat == "CGPA":
                actual, target = actual, target
            norm_target, norm_actual = target, actual
        axes.append(feat.replace("_", " ").replace(" Inverse", " (low backlogs)"))
        current_vals.append(round(norm_actual, 1))
        target_vals.append(round(norm_target, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=target_vals + [target_vals[0]], theta=axes + [axes[0]],
                                   fill="toself", name="Required", line_color="#c7cdf0"))
    fig.add_trace(go.Scatterpolar(r=current_vals + [current_vals[0]], theta=axes + [axes[0]],
                                   fill="toself", name="Your Profile", line_color="#7c3aed"))
    fig.update_layout(
        polar=dict(bgcolor=CHART_PLOT_BG, radialaxis=dict(visible=True, color=CHART_FONT_COLOR)),
        showlegend=True, paper_bgcolor=CHART_PAPER_BG, font_color=CHART_FONT_COLOR, height=500,
        title=f"Your profile vs. {chosen}",
    )
    st.plotly_chart(fig, use_container_width=True)

    if chosen_role["gaps"]:
        st.markdown("**Biggest gaps for this role:**")
        for g in chosen_role["gaps"][:4]:
            st.markdown(f"- **{g['skill']}**: {g['current']} vs {g['required']} required (gap of {g['gap']})")
    else:
        st.success("No gaps — your profile already meets or exceeds this role's target skill levels!")

# ========================================================================================
# 4. JOB ROLE MATCHER
# ========================================================================================
elif section == "💼 Job Role Matcher":
    st.markdown('<div class="section-title">💼 Job Role Matcher — Best Fit Roles</div>', unsafe_allow_html=True)
    for r in role_matches:
        with st.container():
            st.markdown(f"#### {r['icon']} {r['role']} — {r['match']}% match")
            st.progress(min(int(r["match"]), 100))
            st.caption(r["description"])
            st.markdown(" ".join(f'<span class="pill">{s}</span>' for s in r["core_skills"]), unsafe_allow_html=True)
            if r["gaps"]:
                with st.expander(f"See skill gaps for {r['role']}"):
                    for g in r["gaps"]:
                        st.markdown(f"- **{g['skill']}**: {g['current']} vs {g['required']} required")
            st.markdown("---")

# ========================================================================================
# 5. RESUME ANALYZER
# ========================================================================================
elif section == "📄 Resume Analyzer":
    st.markdown('<div class="section-title">📄 Resume Analyzer (AI Powered)</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
    if uploaded:
        text = extract_resume_text(uploaded)
        found, coverage, word_count = analyze_resume(text)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(gauge_chart(coverage, "Keyword Coverage Score", suffix="%"), use_container_width=True)
            st.metric("Word Count", word_count)
        with c2:
            cat_names = list(found.keys())
            counts = [len(v) for v in found.values()]
            fig = px.bar(x=cat_names, y=counts, labels={"x": "Category", "y": "Keywords Found"},
                         title="Skill Keyword Coverage by Category")
            fig.update_layout(paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG, font_color=CHART_FONT_COLOR)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Detected keywords by category:**")
        for cat, kws in found.items():
            if kws:
                st.markdown(f"- **{cat}**: " + ", ".join(kws))
            else:
                st.markdown(f"- **{cat}**: ⚠️ none detected")
    else:
        st.info("Upload a PDF or TXT resume to get an instant AI skill-coverage analysis.")

# ========================================================================================
# 6. PERSONALIZED ROADMAP
# ========================================================================================
elif section == "🗺️ Personalized Roadmap":
    st.markdown('<div class="section-title">🗺️ Personalized Roadmap (Action Plan)</div>', unsafe_allow_html=True)
    for i, (category, title, desc, priority) in enumerate(plan, 1):
        color = priority_color(priority)
        st.markdown(
            f"""<div class="roadmap-item" style="border-left-color:{color};">
            <b>Step {i} · {category}</b>
            <span class="badge" style="background:{color}22;color:{color};border:1px solid {color}55;float:right;">{priority}</span>
            <h4 style="margin:6px 0 4px 0;">{title}</h4>
            <p style="margin:0;color:#4b5568;">{desc}</p>
            </div>""", unsafe_allow_html=True,
        )

# ========================================================================================
# 7. INDUSTRY BENCHMARKING
# ========================================================================================
elif section == "🏛️ Industry Benchmarking":
    st.markdown('<div class="section-title">🏛️ Industry Benchmarking & Percentile Ranking</div>', unsafe_allow_html=True)

    ref_scores = df_reference.apply(lambda r: compute_employability_score(r.to_dict()), axis=1)
    percentile = float(stats.percentileofscore(ref_scores, emp_score))

    c1, c2, c3 = st.columns(3)
    c1.metric("Your Percentile", f"{percentile:.0f}th")
    c2.metric("Cohort Average Score", f"{ref_scores.mean():.1f}/100")
    c3.metric("Your Score", f"{emp_score:.1f}/100")

    fig = px.histogram(ref_scores, nbins=30, title="Where you stand vs. the reference cohort")
    fig.add_vline(x=emp_score, line_color="#dc2626", line_width=3, annotation_text="You")
    fig.update_layout(paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG, font_color=CHART_FONT_COLOR,
                       showlegend=False, xaxis_title="Employability Score", yaxis_title="Number of students")
    st.plotly_chart(fig, use_container_width=True)

    branch_avg = df_reference.groupby("Branch").apply(
        lambda g: g.apply(lambda r: compute_employability_score(r.to_dict()), axis=1).mean()
    ).reset_index(name="avg_score")
    fig2 = px.bar(branch_avg, x="Branch", y="avg_score", title="Average Employability Score by Branch",
                  color="avg_score", color_continuous_scale="Purples")
    fig2.add_hline(y=emp_score, line_color="#dc2626", line_dash="dash", annotation_text="You")
    fig2.update_layout(paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG, font_color=CHART_FONT_COLOR)
    st.plotly_chart(fig2, use_container_width=True)

# ========================================================================================
# 8. PLACEMENT SIMULATOR (WHAT-IF)
# ========================================================================================
elif section == "🎚️ Placement Simulator":
    st.markdown('<div class="section-title">🎚️ Placement Simulator — What-If Analysis</div>', unsafe_allow_html=True)
    st.caption("Drag the sliders to simulate improving your profile and see the impact instantly.")

    sim = dict(profile)
    c1, c2 = st.columns(2)
    with c1:
        sim["CGPA"] = st.slider("Simulated CGPA", 4.0, 10.0, float(profile["CGPA"]), 0.1, key="sim_cgpa")
        sim["Internships"] = st.slider("Simulated Internships", 0, 6, int(profile["Internships"]), key="sim_intern")
        sim["Projects"] = st.slider("Simulated Projects", 0, 10, int(profile["Projects"]), key="sim_proj")
        sim["Backlogs"] = st.slider("Simulated Backlogs", 0, 10, int(profile["Backlogs"]), key="sim_back")
    with c2:
        sim["Coding_Skill"] = st.slider("Simulated Coding Skill", 0.0, 10.0, float(profile["Coding_Skill"]), 0.1, key="sim_code")
        sim["Aptitude_Score"] = st.slider("Simulated Aptitude Score", 0, 100, int(profile["Aptitude_Score"]), key="sim_apt")
        sim["Certifications"] = st.slider("Simulated Certifications", 0, 10, int(profile["Certifications"]), key="sim_cert")
        sim["Placement_Training"] = st.selectbox("Placement Training", ["Yes", "No"],
                                                  index=0 if profile["Placement_Training"] == "Yes" else 1, key="sim_train")

    _, sim_prob, sim_pkg = predict_profile(sim)
    sim_score = compute_employability_score(sim)

    c1, c2, c3 = st.columns(3)
    c1.metric("Placement Probability", f"{sim_prob*100:.1f}%", delta=f"{(sim_prob-probability)*100:+.1f} pts")
    c2.metric("Employability Score", f"{sim_score:.1f}/100", delta=f"{(sim_score-emp_score):+.1f}")
    c3.metric("Predicted Package", f"₹{sim_pkg:.2f} LPA", delta=f"{(sim_pkg-package):+.2f} LPA")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Current", "Simulated"], y=[probability * 100, sim_prob * 100],
                          marker_color=["#a5adc7", "#7c3aed"], text=[f"{probability*100:.1f}%", f"{sim_prob*100:.1f}%"]))
    fig.update_layout(title="Current vs. Simulated Placement Probability", paper_bgcolor=CHART_PAPER_BG,
                       plot_bgcolor=CHART_PLOT_BG, font_color=CHART_FONT_COLOR, yaxis_title="Probability (%)")
    st.plotly_chart(fig, use_container_width=True)

# ========================================================================================
# 9. PROGRESS TRACKER
# ========================================================================================
elif section == "📈 Progress Tracker":
    st.markdown('<div class="section-title">📈 Progress Tracker</div>', unsafe_allow_html=True)
    if st.button("📌 Log current snapshot"):
        save_progress_entry(name, probability, emp_score, package)
        st.success("Snapshot saved!")

    log = load_progress_log()
    my_log = log[log["student_name"] == name] if not log.empty else log
    if my_log.empty:
        st.info("No snapshots logged yet — click the button above to start tracking your progress over time.")
    else:
        my_log["timestamp"] = pd.to_datetime(my_log["timestamp"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=my_log["timestamp"], y=my_log["probability"], mode="lines+markers", name="Probability (%)"))
        fig.add_trace(go.Scatter(x=my_log["timestamp"], y=my_log["employability_score"], mode="lines+markers", name="Employability Score"))
        fig.update_layout(title=f"Progress over time — {name}", paper_bgcolor=CHART_PAPER_BG,
                           plot_bgcolor=CHART_PLOT_BG, font_color=CHART_FONT_COLOR)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(my_log, use_container_width=True, hide_index=True)

# ========================================================================================
# 10. SMART RECOMMENDATIONS
# ========================================================================================
elif section == "⭐ Smart Recommendations":
    st.markdown('<div class="section-title">⭐ Smart Recommendations</div>', unsafe_allow_html=True)
    top3 = plan[:3]
    cols = st.columns(len(top3)) if top3 else []
    for col, (category, title, desc, priority) in zip(cols, top3):
        color = priority_color(priority)
        with col:
            st.markdown(
                f"""<div class="metric-card" style="border-top:4px solid {color};">
                <span class="badge" style="background:{color}22;color:{color};border:1px solid {color}55;">{priority}</span>
                <h4>{title}</h4><p style="color:#4b5568;">{desc}</p></div>""", unsafe_allow_html=True,
            )
    st.markdown("###### Full roadmap available in the **Personalized Roadmap** tab.")

# ========================================================================================
# 11. AI CAREER CHATBOT
# ========================================================================================
elif section == "💬 AI Career Chatbot":
    st.markdown('<div class="section-title">💬 AI Career Chatbot — Ask Your Mentor</div>', unsafe_allow_html=True)
    st.caption("Grounded in your live profile data. Try: \"Which company should I target?\", \"What's my package estimate?\", \"What are my skill gaps?\"")

    for role_, msg in st.session_state.chat_history:
        css = "chat-bubble-user" if role_ == "user" else "chat-bubble-bot"
        align = "right" if role_ == "user" else "left"
        st.markdown(f'<div style="text-align:{align};"><div class="{css}">{msg}</div></div>', unsafe_allow_html=True)

    user_q = st.chat_input("Ask your career mentor anything...")
    if user_q:
        st.session_state.chat_history.append(("user", user_q))
        answer = mentor.answer_question(user_q, profile, probability, emp_score, package, role_matches, top_role["gaps"], plan)
        st.session_state.chat_history.append(("bot", answer))
        st.rerun()
