# ...existing code...
import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="centered"
)

# ---------- STYLES (strong background + modern buttons) ----------
st.markdown(
    """
    <style>
    :root{
        --card-radius:16px;
        --accent-pink:#ff4b8a;
        --accent-orange:#ffb347;
        --accent-green:#1abc9c;
        --accent-blue:#4f46e5;
    }
    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(255,96,130,0.35) 0, transparent 55%),
            radial-gradient(circle at 100% 0%, rgba(79,70,229,0.32) 0, transparent 55%),
            linear-gradient(180deg,#ffffff 0%, #f4fbff 48%, #ecf3ff 100%);
        padding-top:18px;
    }
    .card {
        background: linear-gradient(135deg,#ffffff 0%, #f7fbff 35%, #f0f4ff 100%);
        border-radius:var(--card-radius);
        padding:20px 22px;
        box-shadow:0 16px 40px rgba(15,23,42,0.12);
        margin-bottom:14px;
        border:1px solid rgba(148,163,184,0.25);
    }

    /* ==== TITLE: bigger, centered, animated gradient ==== */
    .title-main {
        font-weight:900;
        font-size:40px;                 /* bigger title */
        letter-spacing:0.04em;
        background:linear-gradient(90deg,#ff416c,#ffb347,#4f46e5,#22c55e);
        background-size:220% auto;
        -webkit-background-clip:text;
        color:transparent;
        margin-bottom:12px;
        text-align:center;
        animation:titleGradient 6s linear infinite;
    }
    @keyframes titleGradient {
        0% { background-position:0% 50%; }
        50% { background-position:100% 50%; }
        100% { background-position:0% 50%; }
    }

    .subtitle {
        color:#4b5563;
        font-size:14px;
        margin-bottom:10px;
        text-align:center;
    }
    .title-tags {
        margin-top:4px;
        text-align:center;
    }
    .title-tag-chip {
        display:inline-block;
        padding:3px 10px;
        border-radius:999px;
        font-size:10.5px;
        font-weight:600;
        text-transform:uppercase;
        letter-spacing:0.06em;
        background:rgba(15,23,42,0.04);
        color:#111827;
        margin-right:6px;
    }

    /* Sample text label under title */
    .sample-label {
        font-weight:700;
        font-size:13px;
        text-align:center;
        margin-top:4px;
        margin-bottom:6px;
        color:#111827;
        letter-spacing:0.08em;
        text-transform:uppercase;
    }

    /* Global button styling (dynamic gradient like title) */
    .stButton>button {
        border-radius:999px !important;
        border:0 !important;
        padding:0.45rem 1.1rem !important;
        font-weight:700 !important;
        font-size:13px !important;
        box-shadow:0 10px 28px rgba(15,23,42,0.22) !important;
        cursor:pointer;
        transition:all 0.16s ease-out !important;
        background-image:linear-gradient(135deg,#ff6a88,#ffcc70,#4f46e5) !important;
        background-size:220% auto !important;
        color:#111827 !important;
        animation:btnGradient 6s linear infinite;
    }
    @keyframes btnGradient {
        0% { background-position:0% 50%; }
        50% { background-position:100% 50%; }
        100% { background-position:0% 50%; }
    }
    .stButton>button:hover {
        transform:translateY(-1px) scale(1.01);
        box-shadow:0 14px 34px rgba(15,23,42,0.30) !important;
        filter:brightness(1.02);
    }
    .stButton>button:active {
        transform:translateY(0px) scale(0.99);
        box-shadow:0 6px 18px rgba(15,23,42,0.25) !important;
    }

    /* ==== High / Medium / Low sample buttons with distinct colors ==== */
    /* These are the first three st.button on the page (High, Medium, Low) */
    .stButton:nth-of-type(1) > button {
        background-image:linear-gradient(135deg,#ef4444,#f97316,#fb7185) !important;  /* high – red/orange */
        color:#ffffff !important;
    }
    .stButton:nth-of-type(2) > button {
        background-image:linear-gradient(135deg,#facc15,#f97316,#fb923c) !important;  /* medium – amber */
        color:#111827 !important;
    }
    .stButton:nth-of-type(3) > button {
        background-image:linear-gradient(135deg,#22c55e,#16a34a,#4ade80) !important;  /* low – green */
        color:#ffffff !important;
    }

    /* Make Analyse/Reset row feel like a card */
    .actions-card {
        border-radius:var(--card-radius);
        padding:10px 14px 6px 14px;
        margin-top:6px;
        margin-bottom:4px;
        background:linear-gradient(120deg,rgba(255,255,255,0.98),rgba(219,234,254,0.95));
        border:1px solid rgba(129,140,248,0.35);
        box-shadow:0 14px 36px rgba(15,23,42,0.12);
    }

    .metric-pill {
        background: rgba(255,255,255,0.96);
        color:#111;
        padding:10px 14px;
        border-radius:10px;
        display:inline-block;
        margin-right:8px;
        min-width:130px;
        text-align:center;
        box-shadow:0 8px 20px rgba(15,23,42,0.16);
    }
    .metric-value {
        font-weight:900;
        font-size:18px;
        color:#111;
    }
    .info-chip {
        display:inline-block;
        padding:6px 10px;
        border-radius:999px;
        background:#f3f4f6;
        margin-right:6px;
        font-size:12px;
        color:#111827;
        box-shadow:0 4px 10px rgba(15,23,42,0.06);
    }

    /* Overlay fallback styling */
    .overlay {
        position: fixed;
        inset: 0;
        background: radial-gradient(circle at 10% 0%,rgba(15,23,42,0.92),rgba(15,23,42,0.98));
        display:flex;
        align-items:center;
        justify-content:center;
        z-index:9999;
    }
    .overlay-card {
        max-width:640px;
        width:92%;
        background:linear-gradient(145deg,#ffffff,#eef2ff);
        border-radius:18px;
        padding:20px 18px;
        box-shadow:0 40px 90px rgba(0,0,0,0.6);
        overflow:auto;
        max-height:86vh;
        border:1px solid rgba(129,140,248,0.65);
    }

    /* Input labels a bit bolder */
    label[data-testid="stWidgetLabel"] > div {
        font-size:13px;
        font-weight:600;
        color:#111827;
    }

    @media (max-width:640px){
        .title-main{font-size:28px;}
        .card{padding:14px 14px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- MODEL LOADING ----------
def find_file(names):
    for n in names:
        p = Path(n)
        if p.exists():
            return p
    return None

MODEL_PATH = find_file([
    "heart_model.pkl",
    "models/heart_model.pkl",
    r"C:\Users\Tejas Vaghela\OneDrive\Desktop\heart_disease_app\heart_model.pkl",
])
SCALER_PATH = find_file([
    "scaler.pkl",
    "models/scaler.pkl",
    r"C:\Users\Tejas Vaghela\OneDrive\Desktop\heart_disease_app\scaler.pkl",
])

if MODEL_PATH is None:
    st.error("Model file not found. Put heart_model.pkl in project root or models/ folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
uses_pipeline = hasattr(model, "named_steps") or hasattr(model, "steps")
scaler = None
if not uses_pipeline and SCALER_PATH is not None:
    scaler = joblib.load(SCALER_PATH)

# ---------- DEFAULTS & STATE ----------
DEFAULTS = {
    "age": 45, "sex": 1, "cp": 1, "trestbps": 130, "chol": 230, "fbs": 0, "restecg": 1,
    "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

for key in ["show_result", "prediction", "probability", "age_group", "bp_status", "chol_status", "risk_band"]:
    st.session_state.setdefault(key, None)

# ---------- SAMPLE PROFILES (high->high risk, low->low risk) ----------
HIGH_PROFILE = {"age": 30, "sex": 0, "cp": 0, "trestbps": 110, "chol": 170, "fbs": 0, "restecg": 0,
               "thalach": 178, "exang": 0, "oldpeak": 0.1, "slope": 0, "ca": 0, "thal": 1}

MEDIUM_PROFILE = {"age": 58, "sex": 1, "cp": 1, "trestbps": 150, "chol": 240, "fbs": 0, "restecg": 1,
                  "thalach": 135, "exang": 0, "oldpeak": 1.4, "slope": 1, "ca": 1, "thal": 2}
LOW_PROFILE =  {"age": 72, "sex": 1, "cp": 3, "trestbps": 170, "chol": 300, "fbs": 1, "restecg": 2,
                "thalach": 95, "exang": 1, "oldpeak": 3.0, "slope": 2, "ca": 2, "thal": 3}

def apply_profile(profile):
    for k, v in profile.items():
        st.session_state[k] = v
    st.session_state["show_result"] = False
    st.session_state["prediction"] = None
    st.session_state["probability"] = None

# ---------- HELPERS ----------
def interpret_extra():
    s = st.session_state
    age = s.age
    bp = s.trestbps
    chol = s.chol

    s.age_group = "Age: Senior (60+)" if age >= 60 else ("Age: Middle (40-59)" if age >= 40 else "Age: Young (<40)")
    s.bp_status = "BP: High" if bp > 139 else ("BP: Elevated" if bp >= 120 else "BP: Normal")
    s.chol_status = "Cholesterol: High" if chol > 239 else ("Borderline" if chol >= 200 else "Desirable")

    prob = s.probability
    if prob is None:
        s.risk_band = "Risk: N/A"
    else:
        s.risk_band = "Risk: High (≥70%)" if prob >= 0.7 else ("Risk: Medium (40-69%)" if prob >= 0.4 else "Risk: Low (<40%)")

def predict_from_state():
    s = st.session_state
    data = np.array([[s.age, s.sex, s.cp, s.trestbps, s.chol, s.fbs, s.restecg,
                      s.thalach, s.exang, s.oldpeak, s.slope, s.ca, s.thal]], dtype=float)
    pred = None
    prob = None
    try:
        if not uses_pipeline and scaler is not None:
            data = scaler.transform(data)
        pred_raw = model.predict(data)
        pred = int(np.asarray(pred_raw)[0])
        try:
            prob_raw = model.predict_proba(data)
            prob = float(np.asarray(prob_raw)[0][1])
        except Exception:
            prob = None
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    return pred, prob

def run_prediction():
    pred, prob = predict_from_state()
    if pred is not None:
        st.session_state["prediction"] = pred
        st.session_state["probability"] = prob
        interpret_extra()
        st.session_state["show_result"] = True
    else:
        st.session_state["show_result"] = False

# ---------- UI HEADER & SAMPLE BUTTONS ----------
st.markdown(
    """
    <div class="card">
      <div class="title-main">❤️ AI-Powered Heart Disease Risk Assessment</div>
      <div class="subtitle">
        Enter key cardiac vitals and risk factors to estimate the likelihood of heart disease using a trained
        machine-learning model. Click <b>Analyse Risk</b> to view an interactive popup summary.
      </div>
      <div class="title-tags">
        <span class="title-tag-chip">Binary Classification</span>
        <span class="title-tag-chip">Clinical Vitals–Based</span>
        <span class="title-tag-chip">Demo · Not for Diagnosis</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="sample-label">Quick sample risk profiles</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("High", key="sample_high"):
        apply_profile(HIGH_PROFILE)
with c2:
    if st.button("Medium", key="sample_med"):
        apply_profile(MEDIUM_PROFILE)
with c3:
    if st.button("Low", key="sample_low"):
        apply_profile(LOW_PROFILE)

# ---------- INPUTS ----------
with st.container():
    left, right = st.columns(2)
    with left:
        st.number_input("Age (years)", 1, 120, key="age")
        st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1], key="sex")
        st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3], key="cp")
        st.number_input("Resting Blood Pressure (mm Hg)", 70, 220, key="trestbps")
    with right:
        st.number_input("Cholesterol (mg/dl)", 100, 700, key="chol")
        st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [0, 1], key="fbs")
        st.selectbox("Resting ECG (0–2)", [0, 1, 2], key="restecg")
        st.number_input("Max Heart Rate Achieved", 60, 220, key="thalach")

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1], key="exang")
    with col2:
        st.number_input("ST Depression (oldpeak)", 0.0, 10.0, step=0.1, key="oldpeak")
    with col3:
        st.selectbox("Slope (0–2)", [0, 1, 2], key="slope")

    st.selectbox("No. of Major Vessels Colored (0–3)", [0, 1, 2, 3], key="ca")
    st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3], key="thal")

st.markdown(
    '<div class="card" style="font-size:12px;display:flex;align-items:flex-start;gap:8px;">'
    '💡 <div>Use realistic clinical values where possible. This tool is <b>for education and prototyping only</b> '
    'and should not be used for treatment decisions.</div></div>',
    unsafe_allow_html=True,
)

# ---------- ACTIONS ----------
with st.container():
    st.markdown('<div class="actions-card">', unsafe_allow_html=True)
    cols = st.columns([1,1])
    with cols[0]:
        if st.button("🔍 Analyse Risk"):
            run_prediction()
    with cols[1]:
        if st.button("🔄 Reset Inputs"):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.session_state["show_result"] = False
            st.session_state["prediction"] = None
            st.session_state["probability"] = None
            st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RESULT POPUP (st.modal preferred, fallback to components.html) ----------
def modal_content(pred, prob):
    if pred == 1 or (prob is not None and prob >= 0.7):
        risk_label = "High risk — urgent review recommended"
        emoji = "💔"
        bg = "linear-gradient(120deg,#ff416c,#ff4b2b)"
        note = "Higher probability — recommend cardiology review, ECG and further testing; manage risk factors."
    elif prob is not None and 0.4 <= prob < 0.7:
        risk_label = "Medium risk — consider follow-up"
        emoji = "🧡"
        bg = "linear-gradient(120deg,#ffb347,#ffcc33)"
        note = "Intermediate probability — consider further evaluation and lifestyle changes; correlate clinically."
    else:
        risk_label = "Low risk — routine care"
        emoji = "💚"
        bg = "linear-gradient(120deg,#11998e,#38ef7d)"
        note = "Low probability by model. Continue routine care; evaluate symptoms and history."

    prob_text = "N/A" if prob is None else f"{prob*100:.1f}%"
    return dict(risk_label=risk_label, emoji=emoji, bg=bg, note=note, prob_text=prob_text)

if st.session_state.show_result and st.session_state.prediction is not None:
    s = st.session_state
    meta = modal_content(s.prediction, s.probability)

    try:
        with st.modal("Analysis result", clear_on_close=False):
            st.markdown(
                f"""
                <div style="border-radius:16px;padding:14px;background:{meta['bg']};color:#fff;">
                  <div style="display:flex;align-items:center;gap:12px">
                    <div style="font-size:44px">{meta['emoji']}</div>
                    <div>
                      <div style="font-size:20px;font-weight:900">{meta['risk_label']}</div>
                      <div style="font-size:13px;opacity:0.95">
                        Model-based assessment — demo only, not a diagnosis.
                      </div>
                    </div>
                  </div>
                  <div style="margin-top:14px;display:flex;gap:12px;flex-wrap:wrap;">
                    <div class="metric-pill">
                      <div style="font-size:12px;opacity:.9">Estimated Probability</div>
                      <div class="metric-value">{meta['prob_text']}</div>
                    </div>
                    <div class="metric-pill">
                      <div style="font-size:12px;opacity:.9">Predicted Class</div>
                      <div class="metric-value">{s.prediction}</div>
                    </div>
                  </div>
                </div>
                <div style="margin-top:12px;color:#111827">
                  <strong>Clinical-style guidance (for learning only):</strong>
                  <div style="margin-top:6px">{meta['note']}</div>
                </div>
                <div style="margin-top:10px">
                  <span class="info-chip">{s.age_group}</span>
                  <span class="info-chip">{s.bp_status}</span>
                  <span class="info-chip">{s.chol_status}</span>
                  <span class="info-chip">{s.risk_band}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns([1,1])
            with c1:
                if st.button("🔄 Reset & check another"):
                    for k, v in DEFAULTS.items():
                        st.session_state[k] = v
                    st.session_state["show_result"] = False
                    st.session_state["prediction"] = None
                    st.session_state["probability"] = None
                    st.experimental_rerun()
            with c2:
                if st.button("❌ Close"):
                    st.session_state["show_result"] = False
    except Exception:
        html = f"""
        <div class="overlay">
          <div class="overlay-card">
            <div style="display:flex;gap:12px;align-items:center;">
              <div style="font-size:44px;">{meta['emoji']}</div>
              <div>
                <div style="font-size:20px;font-weight:900">{meta['risk_label']}</div>
                <div style="font-size:13px;color:#111827;margin-top:6px">
                  Model-based assessment
                </div>
              </div>
            </div>
            <div style="margin-top:12px;">
              <div style="display:flex;gap:12px;flex-wrap:wrap;">
                <div class="metric-pill">
                  <div style="font-size:12px;opacity:.9">Estimated Probability</div>
                  <div class="metric-value">{meta['prob_text']}</div>
                </div>
                <div class="metric-pill">
                  <div style="font-size:12px;opacity:.9">Predicted Class</div>
                  <div class="metric-value">{s.prediction}</div>
                </div>
              </div>
              <div style="margin-top:12px;color:#111827">{meta['note']}</div>
              <div style="margin-top:10px">
                <span class="info-chip">{s.age_group}</span>
                <span class="info-chip">{s.bp_status}</span>
                <span class="info-chip">{s.chol_status}</span>
                <span class="info-chip">{s.risk_band}</span>
              </div>
            </div>
          </div>
        </div>
        """
        components.html(html, height=520, scrolling=False)
        b1, b2, b3 = st.columns([1,1,1])
        with b1:
            if st.button("🔄 Reset & check another", key="fb_reset"):
                for k, v in DEFAULTS.items():
                    st.session_state[k] = v
                st.session_state["show_result"] = False
                st.session_state["prediction"] = None
                st.session_state["probability"] = None
                st.experimental_rerun()
        with b2:
            if st.button("✳ Scroll to inputs", key="fb_scroll"):
                st.session_state["show_result"] = False
                st.experimental_rerun()
        with b3:
            if st.button("❌ Close", key="fb_close"):
                st.session_state["show_result"] = False
                st.experimental_rerun()

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    '<div style="font-size:11px;color:#6b7280">'
    'Disclaimer: This is a demonstration of a machine-learning model UI and must not be used as a substitute for '
    'professional medical judgment or emergency care.</div>',
    unsafe_allow_html=True,
)
# ...existing code...
