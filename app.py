import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
from utils.preprocess import preprocess_canvas, preprocess_upload

st.set_page_config(
    page_title="LetterLens · Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: #F8FAFC;
    color: #0F172A;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 760px;
}

.hero {
    text-align: center;
    padding: 2.5rem 1.5rem 1.5rem;
    margin-bottom: 0.5rem;
}
.hero-badge {
    display: inline-block;
    background: #EFF6FF;
    color: #3B82F6;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 99px;
    border: 1px solid #BFDBFE;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0 0 0.6rem;
}
.hero h1 span {
    color: #3B82F6;
}
.hero p {
    font-size: 1rem;
    color: #64748B;
    font-weight: 300;
    margin: 0;
    line-height: 1.6;
}

.stats-row {
    display: flex;
    justify-content: center;
    gap: 1.2rem;
    margin: 1.5rem 0 2rem;
    flex-wrap: wrap;
}
.stat-pill {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 99px;
    padding: 0.45rem 1.1rem;
    font-size: 0.82rem;
    color: #475569;
    font-weight: 500;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.stat-pill strong {
    color: #0F172A;
    font-weight: 700;
}

.card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 1.2rem;
}
.card-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.9rem;
}

.result-card {
    background: linear-gradient(135deg, #EFF6FF 0%, #F0FDF4 100%);
    border: 1px solid #BFDBFE;
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    margin-bottom: 1.2rem;
}
.result-letter {
    font-family: 'JetBrains Mono', monospace;
    font-size: 5rem;
    font-weight: 700;
    color: #1D4ED8;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.result-conf {
    font-size: 1rem;
    color: #3B82F6;
    font-weight: 600;
    margin-bottom: 0.2rem;
}
.result-sub {
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 300;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: #F1F5F9;
    padding: 0.3rem;
    border-radius: 10px;
    border: none;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 0.45rem 1.2rem;
    color: #64748B;
    border: none;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #0F172A !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

hr {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 1.5rem 0;
}

.info-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
}
.info-box p {
    margin: 0;
    font-size: 0.85rem;
    color: #64748B;
    line-height: 1.7;
}
.info-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.83rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid #F1F5F9;
    color: #475569;
}
.info-row:last-child { border-bottom: none; }
.info-row span:last-child {
    font-weight: 600;
    color: #0F172A;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}

.canvas-wrap {
    display: flex;
    justify-content: center;
    margin: 0.5rem 0;
}

.upload-hint {
    font-size: 0.8rem;
    color: #94A3B8;
    text-align: center;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

LABELS = [str(i) for i in range(10)]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/letterlens_cnn.keras")

model = load_model()

st.markdown("""
<div class="hero">
    <div class="hero-badge">CodeAlpha Internship · Task 3</div>
    <h1>Letter<span>Lens</span></h1>
    <p>Draw or upload a handwritten digit.<br>Our CNN identifies it instantly.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-row">
    <div class="stat-pill"><strong>99.52%</strong> Test Accuracy</div>
    <div class="stat-pill"><strong>60K</strong> Training Samples</div>
    <div class="stat-pill"><strong>10</strong> Digit Classes</div>
    <div class="stat-pill"><strong>CNN</strong> Deep Learning</div>
</div>
""", unsafe_allow_html=True)

input_arr = None

tab1, tab2 = st.tabs(["  Draw  ", "  Upload  "])

with tab1:
    st.markdown('<div class="card-label">Draw a single digit (0–9) inside the canvas</div>', unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1, 4, 1])
    with col_c:
        canvas = st_canvas(
            fill_color="white",
            stroke_width=20,
            stroke_color="#1e293b",
            background_color="#ffffff",
            height=260,
            width=260,
            drawing_mode="freedraw",
            key="canvas"
        )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Canvas", use_container_width=True):
            st.rerun()
    st.markdown('<p class="upload-hint">Use thick strokes · Write large · Centre the digit</p>', unsafe_allow_html=True)

    if canvas.image_data is not None:
        grayscale = np.mean(canvas.image_data[:, :, :3], axis=2)
        dark_pixels = (grayscale < 200).sum()
        if dark_pixels > 20:
            input_arr = preprocess_canvas(canvas.image_data)

with tab2:
    st.markdown('<div class="card-label">Upload a handwritten digit image (PNG / JPG)</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if uploaded:
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            st.image(uploaded, width=220)
        input_arr = preprocess_upload(uploaded)

st.markdown("<hr>", unsafe_allow_html=True)

if input_arr is not None:
    preds = model.predict(input_arr, verbose=0)[0]
    top3_idx = preds.argsort()[-3:][::-1]
    top3_labels = [LABELS[i] for i in top3_idx]
    top3_scores = [float(preds[i]) * 100 for i in top3_idx]

    st.markdown(f"""
    <div class="result-card">
        <div class="result-letter">{top3_labels[0]}</div>
        <div class="result-conf">{top3_scores[0]:.1f}% confidence</div>
        <div class="result-sub">Predicted digit · MNIST</div>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=top3_labels,
        y=top3_scores,
        marker=dict(
            color=["#3B82F6", "#93C5FD", "#BFDBFE"],
            line=dict(width=0)
        ),
        text=[f"{s:.1f}%" for s in top3_scores],
        textposition="outside",
        textfont=dict(family="Sora, sans-serif", size=13, color="#0F172A"),
        width=0.45
    ))
    fig.update_layout(
        title=dict(
            text="Top 3 Predictions",
            font=dict(family="Sora, sans-serif", size=15, color="#0F172A"),
            x=0
        ),
        yaxis=dict(
            range=[0, 115],
            ticksuffix="%",
            gridcolor="#F1F5F9",
            tickfont=dict(family="Sora, sans-serif", size=11, color="#94A3B8"),
            showline=False,
            zeroline=False
        ),
        xaxis=dict(
            tickfont=dict(family="JetBrains Mono, monospace", size=18, color="#0F172A"),
            showline=False
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        bargap=0.5
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

else:
    st.markdown("""
    <div style="text-align:center; padding: 2.5rem 1rem; color: #94A3B8;">
        <div style="font-size:2.5rem; margin-bottom:0.8rem;">✏️</div>
        <div style="font-size:0.95rem; font-weight:500; color:#64748B;">Draw or upload a digit to see the prediction</div>
        <div style="font-size:0.82rem; margin-top:0.3rem;">Supports digits 0 – 9</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
with st.expander("Model & Architecture Details"):
    st.markdown("""
    <div class="info-box">
        <div class="info-row"><span>Dataset</span><span>MNIST</span></div>
        <div class="info-row"><span>Training samples</span><span>60,000</span></div>
        <div class="info-row"><span>Test accuracy</span><span>99.52%</span></div>
        <div class="info-row"><span>Architecture</span><span>Conv2D × 2 + Dense</span></div>
        <div class="info-row"><span>Input shape</span><span>28 × 28 × 1</span></div>
        <div class="info-row"><span>Output classes</span><span>10 (0 – 9)</span></div>
        <div class="info-row"><span>Optimizer</span><span>Adam</span></div>
        <div class="info-row"><span>Regularization</span><span>Dropout 0.3</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:2.5rem; font-size:0.78rem; color:#CBD5E1;">
    Built by Vishvraj · CodeAlpha ML Internship · 2026
</div>
""", unsafe_allow_html=True)