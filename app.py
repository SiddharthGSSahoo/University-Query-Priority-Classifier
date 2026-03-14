import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
import os

import nltk
nltk.download("stopwords", quiet=True)

from transformers import Preprocess, ArrayFlattener

# ─────────────────────────────────────────────
#  Page Config  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="UniQuery · Priority Classifier",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg:        #0e1117;
    --surface:   #161b27;
    --border:    #252d3d;
    --accent:    #4f8ef7;
    --low:       #22c55e;
    --medium:    #f59e0b;
    --high:      #ef4444;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --radius:    14px;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 780px; }

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 2.8rem 1rem 2rem;
    background: linear-gradient(135deg, #0f1923 0%, #162032 60%, #1a1040 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 20%, rgba(79,142,247,0.12) 0%, transparent 60%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    background: rgba(79,142,247,0.1);
    border: 1px solid rgba(79,142,247,0.3);
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    font-weight: 400;
    margin: 0.3rem 0 0.6rem;
    background: linear-gradient(135deg, #e2e8f0 30%, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: var(--muted);
    font-size: 0.95rem;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Card ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}
.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

/* ── Result Banner ── */
.result-banner {
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    text-align: center;
    margin-top: 1.4rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.result-banner::after {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% -20%, rgba(255,255,255,0.06) 0%, transparent 70%);
}
.result-banner.low    { background: rgba(34,197,94,0.08);  border-color: rgba(34,197,94,0.35); }
.result-banner.medium { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.35); }
.result-banner.high   { background: rgba(239,68,68,0.08);  border-color: rgba(239,68,68,0.35); }

.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-label.low    { color: var(--low); }
.result-label.medium { color: var(--medium); }
.result-label.high   { color: var(--high); }

.result-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    font-weight: 400;
    line-height: 1;
    margin: 0.2rem 0 0.6rem;
}
.result-value.low    { color: var(--low); }
.result-value.medium { color: var(--medium); }
.result-value.high   { color: var(--high); }

.result-desc {
    color: var(--muted);
    font-size: 0.85rem;
    font-weight: 300;
}

/* ── Priority Pill ── */
.pill {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.22rem 0.7rem;
    border-radius: 999px;
}
.pill.low    { background: rgba(34,197,94,0.15);  color: var(--low);    border: 1px solid rgba(34,197,94,0.3); }
.pill.medium { background: rgba(245,158,11,0.15); color: var(--medium); border: 1px solid rgba(245,158,11,0.3); }
.pill.high   { background: rgba(239,68,68,0.15);  color: var(--high);   border: 1px solid rgba(239,68,68,0.3); }

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] > div,
div[data-testid="stTextArea"]  > div > div {
    background: #1a2030 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
div.stButton > button {
    width: 100%;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 1.2rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: opacity 0.2s, transform 0.15s;
    cursor: pointer;
}
div.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
.sidebar-section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.2rem 0 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.4rem 0; }

/* ── Batch Table ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Info boxes ── */
.info-row {
    display: flex;
    gap: 0.8rem;
    margin-top: 1rem;
}
.info-box {
    flex: 1;
    background: #1a2030;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.info-box .ib-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--accent);
}
.info-box .ib-label {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.15rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Load Model (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    pipeline  = joblib.load("models/ModelPipeline.pkl")
    label_map = joblib.load("models/Label_Map.pkl")
    return pipeline, label_map


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
DEPT_OPTIONS = [
    "Computer Science", "Mathematics", "Physics", "Chemistry",
    "Biology", "Engineering", "Economics", "Business",
    "Psychology", "Sociology", "History", "Literature",
    "Fine Arts", "Law", "Medicine", "Nursing", "Education",
    "Environmental Science", "Political Science", "Philosophy",
]

PRIORITY_META = {
    "Low":    ("low",    "🟢", "Routine Query — Can be Handled within Standard SLA."),
    "Medium": ("medium", "🟡", "Moderate Urgency — Should be Reviewed within 24 hours."),
    "High":   ("high",   "🔴", "Critical Query — Requires Immediate Staff Attention."),
}


def predict_single(pipeline, label_map, query: str, department: str) -> str:
    df = pd.DataFrame({"Student_Query": [query], "Department": [department]})
    pred_num = pipeline.predict(df)[0]
    return label_map[pred_num]


def predict_batch(pipeline, label_map, df: pd.DataFrame) -> pd.DataFrame:
    preds_num = pipeline.predict(df[["Student_Query", "Department"]])
    df = df.copy()
    df["Predicted_Priority"] = [label_map[p] for p in preds_num]
    return df


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 UniQuery")
    st.markdown("<div class='sidebar-section-title'>Navigation</div>", unsafe_allow_html=True)
    mode = st.radio("", ["Single Query", "Batch Upload"], label_visibility="collapsed")

    st.markdown("<div class='sidebar-section-title'>About</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.82rem; color:#64748b; line-height:1.6;'>"
        "Classifies student support queries into <b>Low</b>, <b>Medium</b>, or <b>High</b> "
        "priority using an NLP pipeline trained on university query data."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-section-title'>Priority Guide</div>", unsafe_allow_html=True)
    for label, (css, icon, desc) in PRIORITY_META.items():
        st.markdown(
            f"<span class='pill {css}'>{icon} {label}</span>"
            f"<p style='font-size:0.78rem;color:#64748b;margin:0.3rem 0 0.8rem;'>{desc}</p>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
#  Load artifacts
# ─────────────────────────────────────────────
try:
    pipeline, label_map = load_artifacts()
    model_ready = True
except Exception as e:
    model_ready = False
    load_error = str(e)


# ─────────────────────────────────────────────
#  Hero Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">NLP · Classification · v1.0</div>
    <h1>Query Priority Classifier</h1>
    <p>Instantly triage student support queries with machine learning — so your team knows exactly what needs attention first.</p>
</div>
""", unsafe_allow_html=True)

if not model_ready:
    st.error(f"⚠️ Could not load model artifacts.\n\n```{load_error}```")
    st.info("Make sure `models/ModelPipeline.pkl` and `models/Label_Map.pkl` exist.")
    st.stop()


# ═════════════════════════════════════════════
#  MODE A — Single Query
# ═════════════════════════════════════════════
if mode == "Single Query":

    st.markdown("<div class='card-label'>Student Query</div>", unsafe_allow_html=True)
    query_text = st.text_area(
        label="query_input",
        placeholder="e.g.  I have not received my scholarship disbursement and the deadline is tomorrow…",
        height=130,
        label_visibility="collapsed",
    )

    st.markdown("<div class='card-label' style='margin-top:1rem;'>Department</div>", unsafe_allow_html=True)
    department = st.selectbox(
        label="dept_select",
        options=DEPT_OPTIONS,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Classify Priority", use_container_width=True)

    if predict_btn:
        if not query_text.strip():
            st.warning("Please enter a student query before classifying.")
        else:
            with st.spinner("Analysing query…"):
                time.sleep(0.4)          # slight delay so spinner is visible
                result = predict_single(pipeline, label_map, query_text.strip(), department)

            css_cls, icon, desc = PRIORITY_META[result]

            st.markdown(f"""
            <div class="result-banner {css_cls}">
                <div class="result-label {css_cls}">Predicted Priority</div>
                <div class="result-value {css_cls}">{icon} {result}</div>
                <div class="result-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Details expander ──────────────────────────
            with st.expander("🔍 Submission details", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Department**")
                    st.code(department, language=None)
                with col2:
                    st.markdown("**Query length**")
                    st.code(f"{len(query_text.split())} words", language=None)
                st.markdown("**Submitted query**")
                st.code(query_text, language=None)


# ═════════════════════════════════════════════
#  MODE B — Batch Upload
# ═════════════════════════════════════════════
else:
    st.markdown("""
    <div class="card">
        <div class="card-label">Expected CSV Format</div>
        <p style="font-size:0.85rem; color:#64748b; margin:0.4rem 0 0.8rem;">
            Upload a CSV with at least these two columns:
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.code("Student_Query,Department\n"
            "I need help with my exam schedule,Computer Science\n"
            "My scholarship has not arrived,Economics", language="csv")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        try:
            df_input = pd.read_csv(uploaded)

            # ── Validation ──────────────────────────
            required = {"Student_Query", "Department"}
            missing  = required - set(df_input.columns)
            if missing:
                st.error(f"Missing required columns: `{'`, `'.join(missing)}`")
                st.stop()

            st.markdown(f"**{len(df_input):,} queries loaded** — preview:")
            st.dataframe(df_input.head(5), use_container_width=True)

            run_batch = st.button("⚡ Run Batch Classification", use_container_width=True)

            if run_batch:
                with st.spinner(f"Classifying {len(df_input):,} queries…"):
                    df_result = predict_batch(pipeline, label_map, df_input)

                st.success("✅ Classification complete!")

                # ── Summary stats ──────────────────────
                counts = df_result["Predicted_Priority"].value_counts()
                low_c  = counts.get("Low",    0)
                med_c  = counts.get("Medium", 0)
                high_c = counts.get("High",   0)

                st.markdown(f"""
                <div class="info-row">
                    <div class="info-box">
                        <div class="ib-value" style="color:var(--low);">{low_c}</div>
                        <div class="ib-label">🟢 Low Priority</div>
                    </div>
                    <div class="info-box">
                        <div class="ib-value" style="color:var(--medium);">{med_c}</div>
                        <div class="ib-label">🟡 Medium Priority</div>
                    </div>
                    <div class="info-box">
                        <div class="ib-value" style="color:var(--high);">{high_c}</div>
                        <div class="ib-label">🔴 High Priority</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(df_result, use_container_width=True)

                # ── Download ──────────────────────────
                csv_out = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️  Download Results CSV",
                    data=csv_out,
                    file_name="classified_queries.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error processing file: `{e}`")
