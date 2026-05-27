import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
from datetime import datetime

from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor,
    plot_tree, export_text
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    r2_score, mean_squared_error
)

st.set_page_config(page_title="Decision Tree", page_icon="🌿", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: #0e1117;
    color: #dde3ec;
}

.main .block-container {
    padding: 2rem 3.5rem;
    max-width: 1150px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #131920 !important;
    border-right: 1px solid #1e2a35;
}
[data-testid="stSidebar"] * { color: #b8c5d1 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #0e1117 !important;
    border: 1px solid #1e2a35 !important;
    border-radius: 10px !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.8rem !important;
    color: #dde3ec !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1e2a35 !important;
}

/* ── Hero ── */
.hero-wrap {
    padding: 2.5rem 0 2rem 0;
    border-bottom: 1px solid #1e2a35;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 24px;
}
.hero-icon {
    width: 56px; height: 56px;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 14px;
    display: flex; align-items: center;
    justify-content: center;
    font-size: 26px;
    flex-shrink: 0;
}
.hero-tag {
    font-family: 'Fira Code', monospace;
    font-size: 0.68rem;
    color: #10b981;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 8px;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ecf0f5;
    margin: 0;
    line-height: 1.15;
}
.hero-sub {
    color: #6b7f8f;
    font-size: 0.92rem;
    margin-top: 6px;
}

/* ── Step header ── */
.step-wrap {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 2.5rem 0 1.2rem 0;
}
.step-pill {
    font-family: 'Fira Code', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    background: rgba(16,185,129,0.12);
    color: #10b981;
    border: 1px solid rgba(16,185,129,0.2);
    padding: 5px 13px;
    border-radius: 20px;
    letter-spacing: 0.1em;
    white-space: nowrap;
}
.step-name {
    font-size: 1.05rem;
    font-weight: 600;
    color: #dde3ec;
}
.step-line {
    flex: 1;
    height: 1px;
    background: #1e2a35;
}

/* ── Stat grid ── */
.stat-grid { display: flex; gap: 12px; margin: 1rem 0 1.4rem 0; }
.stat-box {
    flex: 1;
    background: #131920;
    border: 1px solid #1e2a35;
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
}
.stat-label {
    font-family: 'Fira Code', monospace;
    font-size: 0.65rem;
    color: #4a6070;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
}
.stat-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #10b981;
    line-height: 1;
}

/* ── Result grid ── */
.result-grid { display: flex; gap: 12px; margin: 1.2rem 0; }
.result-box {
    flex: 1;
    background: #131920;
    border: 1px solid #1e2a35;
    border-top: 2px solid #10b981;
    border-radius: 0 0 12px 12px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.result-label {
    font-family: 'Fira Code', monospace;
    font-size: 0.65rem;
    color: #4a6070;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
}
.result-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ecf0f5;
    line-height: 1;
}

/* ── Section label ── */
.sec-label {
    font-family: 'Fira Code', monospace;
    font-size: 0.7rem;
    color: #4a6070;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin: 1.6rem 0 0.6rem 0;
}

/* ── Buttons ── */
.stButton > button {
    background: #10b981 !important;
    color: #051a10 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 26px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: background 0.2s, transform 0.1s !important;
}
.stButton > button:hover {
    background: #34d399 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }

/* ── Selects & radios ── */
.stSelectbox > div > div {
    background: #131920 !important;
    border: 1px solid #1e2a35 !important;
    border-radius: 10px !important;
    color: #dde3ec !important;
    font-size: 0.88rem !important;
}
.stRadio label { color: #b8c5d1 !important; font-size: 0.9rem !important; }
[data-testid="stFileUploader"] {
    background: #131920;
    border: 1px dashed #1e2a35;
    border-radius: 12px;
}

/* ── Alerts ── */
.stSuccess {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-radius: 10px !important;
    color: #34d399 !important;
    font-size: 0.88rem !important;
}
.stError {
    background: rgba(239,68,68,0.1) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 10px !important;
    color: #f87171 !important;
    font-size: 0.88rem !important;
}
.stInfo {
    background: rgba(16,185,129,0.07) !important;
    border: 1px solid rgba(16,185,129,0.18) !important;
    border-radius: 10px !important;
    color: #6ee7b7 !important;
    font-size: 0.88rem !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2a35 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Slider ── */
.stSlider .st-bw { color: #10b981 !important; }

/* ── Labels ── */
label, p { color: #b8c5d1 !important; }
h1,h2,h3,h4 { color: #ecf0f5 !important; }

/* ── Footer ── */
.footer {
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid #1e2a35;
    display: flex;
    justify-content: space-between;
    font-family: 'Fira Code', monospace;
    font-size: 0.65rem;
    color: #253340;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib theme ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#131920",
    "axes.facecolor":   "#131920",
    "axes.edgecolor":   "#1e2a35",
    "axes.labelcolor":  "#b8c5d1",
    "xtick.color":      "#4a6070",
    "ytick.color":      "#4a6070",
    "text.color":       "#b8c5d1",
    "grid.color":       "#1e2a35",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})

# ── Helpers ────────────────────────────────────────────────────────────────────
def step(num, label):
    st.markdown(f"""
    <div class="step-wrap">
        <span class="step-pill">step {num:02d}</span>
        <span class="step-name">{label}</span>
        <div class="step-line"></div>
    </div>""", unsafe_allow_html=True)

def stat_grid(items):
    boxes = "".join(f"""
    <div class="stat-box">
        <div class="stat-label">{l}</div>
        <div class="stat-value">{v}</div>
    </div>""" for l, v in items)
    st.markdown(f'<div class="stat-grid">{boxes}</div>', unsafe_allow_html=True)

def result_grid(items):
    boxes = "".join(f"""
    <div class="result-box">
        <div class="result-label">{l}</div>
        <div class="result-value">{v}</div>
    </div>""" for l, v in items)
    st.markdown(f'<div class="result-grid">{boxes}</div>', unsafe_allow_html=True)

def sec(label):
    st.markdown(f'<div class="sec-label">{label}</div>', unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k in ["df", "df_clean"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ── Dirs ───────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, "data")
RAW_DIR       = os.path.join(BASE, "data", "raw")
PROCESSED_DIR = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-icon">🌿</div>
    <div>
        <div class="hero-tag">◆ Machine Learning Platform</div>
        <div class="hero-title">Decision Tree Dashboard</div>
        <div class="hero-sub">End-to-end Classification &amp; Regression · Visualize · Evaluate</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Fira Code',monospace; font-size:0.68rem; color:#10b981;
                text-transform:uppercase; letter-spacing:0.16em;
                padding: 1rem 0 1rem 0; border-bottom:1px solid #1e2a35; margin-bottom:1rem;">
        ◆ Model Settings
    </div>""", unsafe_allow_html=True)

    task_type    = st.selectbox("Task Type",         ["Classification", "Regression"])
    criterion    = st.selectbox(
        "Criterion",
        ["gini", "entropy", "log_loss"] if task_type == "Classification"
        else ["squared_error", "friedman_mse", "absolute_error", "poisson"]
    )
    max_depth    = st.slider("Max Depth",            1, 20, 5)
    min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
    min_samples_leaf  = st.slider("Min Samples Leaf",  1, 20, 1)
    max_features = st.selectbox("Max Features",      ["None", "sqrt", "log2"])
    splitter     = st.selectbox("Splitter",          ["best", "random"])
    test_size    = st.slider("Test Size",            0.1, 0.5, 0.25, step=0.05)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Fira Code',monospace; font-size:0.65rem; color:#253340;
                line-height:2; text-transform:uppercase; letter-spacing:0.08em;">
        Classification · Regression<br>Gini · Entropy · MSE<br>Tree Visualization
    </div>""", unsafe_allow_html=True)

df = st.session_state.df

# ── Step 1 ─────────────────────────────────────────────────────────────────────
step(1, "Data Ingestion")
option = st.radio("Source", ["Download Dataset", "Upload CSV"], horizontal=True)

if option == "Download Dataset":
    label = "Download Iris Dataset" if task_type == "Classification" else "Download Tips Dataset"
    url   = ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
             if task_type == "Classification" else
             "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
    fname = "iris.csv" if task_type == "Classification" else "tips.csv"
    if st.button(label):
        path = os.path.join(RAW_DIR, fname)
        with open(path, "wb") as f:
            f.write(requests.get(url).content)
        df = pd.read_csv(path)
        st.session_state.df = df
        st.success(f"✓ {fname} downloaded")

if option == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        path = os.path.join(RAW_DIR, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        df = pd.read_csv(path)
        st.session_state.df = df
        st.success(f"✓ {uploaded.name} loaded")

# ── Step 2 ─────────────────────────────────────────────────────────────────────
if df is not None:
    step(2, "Exploratory Data Analysis")

    missing = int(df.isnull().sum().sum())
    stat_grid([
        ("Rows",     str(df.shape[0])),
        ("Columns",  str(df.shape[1])),
        ("Missing",  str(missing)),
        ("Numeric",  str(len(df.select_dtypes(include=np.number).columns))),
    ])

    st.dataframe(df.head(6), use_container_width=True)

    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) > 1:
        sec("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f",
                    cmap=sns.light_palette("#10b981", as_cmap=True),
                    linewidths=0.5, linecolor="#0e1117",
                    ax=ax, cbar_kws={"shrink": 0.75})
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── Step 3 ─────────────────────────────────────────────────────────────────────
if df is not None:
    step(3, "Data Cleaning")
    strategy = st.selectbox("Missing Value Strategy", ["Mean", "Median", "Drop Rows"])
    df_clean = df.copy()
    if strategy == "Drop Rows":
        df_clean.dropna(inplace=True)
    else:
        for col in df_clean.select_dtypes(include=np.number):
            fill = df_clean[col].mean() if strategy == "Mean" else df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(fill)
    st.session_state.df_clean = df_clean
    st.success(f"✓ Cleaned — {len(df_clean)} rows retained")

# ── Step 4 ─────────────────────────────────────────────────────────────────────
step(4, "Save Cleaned Dataset")
if st.button("Save Dataset"):
    if st.session_state.df_clean is None:
        st.error("No cleaned data found. Complete Step 3 first.")
    else:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(PROCESSED_DIR, f"cleaned_{ts}.csv")
        st.session_state.df_clean.to_csv(path, index=False)
        st.success(f"✓ Saved → data/processed/cleaned_{ts}.csv")

# ── Step 5 ─────────────────────────────────────────────────────────────────────
step(5, "Load Dataset for Modelling")
files    = sorted([f for f in os.listdir(PROCESSED_DIR) if f.startswith("cleaned_")])
df_model = None
if files:
    selected = st.selectbox("Choose file", files)
    df_model = pd.read_csv(os.path.join(PROCESSED_DIR, selected))
    st.dataframe(df_model.head(5), use_container_width=True)
else:
    st.info("No cleaned files yet — complete Steps 3 & 4 first.")

# ── Step 6 ─────────────────────────────────────────────────────────────────────
if df_model is not None:
    step(6, "Train Decision Tree")
    target = st.selectbox("Target Column", df_model.columns)

    X = df_model.drop(columns=[target]).copy()
    y = df_model[target].copy()

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if task_type == "Classification" and y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    X = X.select_dtypes(include=np.number)
    feat_names = list(X.columns)
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    mf = None if max_features == "None" else max_features
    params = dict(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=mf,
        splitter=splitter,
        random_state=42,
    )

    if st.button("▶  Train Model"):

        # ── Classification ──────────────────────────────────────────────────
        if task_type == "Classification":
            model  = DecisionTreeClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)

            result_grid([
                ("Accuracy",     f"{acc:.2%}"),
                ("Train Samples",str(len(y_train))),
                ("Test Samples", str(len(y_test))),
                ("Tree Depth",   str(model.get_depth())),
            ])

            sec("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.heatmap(confusion_matrix(y_test, y_pred),
                        annot=True, fmt="d",
                        cmap=sns.light_palette("#10b981", as_cmap=True),
                        linewidths=0.5, linecolor="#0e1117", ax=ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            sec("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

            sec("Feature Importance")
            imp = pd.Series(model.feature_importances_, index=feat_names).sort_values()
            fig, ax = plt.subplots(figsize=(7, max(3, len(feat_names) * 0.45)))
            bars = ax.barh(imp.index, imp.values, color="#10b981", alpha=0.85, height=0.6)
            ax.bar_label(bars, fmt="%.3f", padding=4, color="#b8c5d1", fontsize=9)
            ax.set_xlabel("Importance")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            sec("Decision Tree Plot")
            depth_vis = min(model.get_depth(), 3)
            fig, ax = plt.subplots(figsize=(14, 6))
            plot_tree(model, max_depth=depth_vis, feature_names=feat_names,
                      filled=True, ax=ax, fontsize=8,
                      impurity=True, rounded=True)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ── Regression ─────────────────────────────────────────────────────
        else:
            model  = DecisionTreeRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2     = r2_score(y_test, y_pred)
            mse    = mean_squared_error(y_test, y_pred)
            rmse   = np.sqrt(mse)

            result_grid([
                ("R² Score", f"{r2:.3f}"),
                ("RMSE",     f"{rmse:.3f}"),
                ("MSE",      f"{mse:.3f}"),
                ("Depth",    str(model.get_depth())),
            ])

            sec("Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_test, y_pred, alpha=0.75, color="#10b981",
                       edgecolors="#0e1117", linewidths=0.4, s=55)
            mn = min(float(np.min(y_test)), float(np.min(y_pred)))
            mx = max(float(np.max(y_test)), float(np.max(y_pred)))
            ax.plot([mn, mx], [mn, mx], "--", color="#4a6070", lw=1.2)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"R² = {r2:.3f}", fontsize=10, color="#b8c5d1")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            sec("Feature Importance")
            imp = pd.Series(model.feature_importances_, index=feat_names).sort_values()
            fig, ax = plt.subplots(figsize=(7, max(3, len(feat_names) * 0.45)))
            bars = ax.barh(imp.index, imp.values, color="#10b981", alpha=0.85, height=0.6)
            ax.bar_label(bars, fmt="%.3f", padding=4, color="#b8c5d1", fontsize=9)
            ax.set_xlabel("Importance")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            sec("Decision Tree Plot")
            depth_vis = min(model.get_depth(), 3)
            fig, ax = plt.subplots(figsize=(14, 6))
            plot_tree(model, max_depth=depth_vis, feature_names=feat_names,
                      filled=True, ax=ax, fontsize=8,
                      impurity=True, rounded=True)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            sec("Sample Predictions")
            results = pd.DataFrame({
                "Actual":    np.array(y_test)[:12],
                "Predicted": np.round(y_pred[:12], 4),
                "Error":     np.round(np.abs(np.array(y_test)[:12] - y_pred[:12]), 4),
            })
            st.dataframe(results, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>Decision Tree Dashboard</span>
    <span>Classification · Regression · Visualization</span>
</div>
""", unsafe_allow_html=True)