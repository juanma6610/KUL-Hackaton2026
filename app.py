import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from src.mcm import MultiscaleContextModel  
import plotly.graph_objects as go
import shap
import matplotlib
matplotlib.use('Agg')

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Duolingo Memory Engine",
    layout="wide",
    page_icon="🦉"
)

# --- 2. CUSTOM CSS (Duolingo-inspired theme) ---
st.markdown("""
<style>
    /* Background and font */
    .stApp { background-color: #f7f7f7; }
    h1 { color: #58cc02; font-size: 2.4rem !important; }
    h2, h3 { color: #1cb0f6; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid #e5e5e5;
    }
    [data-testid="stSidebar"] h2 { color: #58cc02; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #fff;
        border: 2px solid #e5e5e5;
        border-radius: 12px;
        padding: 12px 20px;
    }
    [data-testid="stMetricLabel"] p { color: #777; font-size: 0.85rem; }
    [data-testid="stMetricValue"] { color: #1cb0f6; font-size: 2rem; }

    /* Expander */
    .streamlit-expanderHeader { font-weight: 600; color: #58cc02; }

    /* General button */
    .stButton > button {
        background-color: #58cc02;
        color: white;
        border-radius: 12px;
        border: none;
        font-weight: 700;
        padding: 0.5rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD THE SAVED MODEL ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("models/xgboost_baseline0.001.json") 
    return model

model = load_model()

# --- 4. HEADER ---
st.title("🦉 Duolingo Memory Engine")
st.markdown("### Hybrid XGBoost + Cognitive Science (MCM) Spaced Repetition Model")
st.markdown("---")

# --- 5. SIDEBAR ---
st.sidebar.markdown("## 👤 Simulate a Learner")

# Language & grammar
lang = st.sidebar.selectbox(
    "Language Track",
    ["en->es", "en->fr", "en->de", "en->it", "en->pt", "es->en", "it->en", "pt->en"],
    index=0
)
# pos_label uses descriptive names matching enrich_chunk output ("noun", "verb_lexical", ...)
pos_label_options = {
    "Noun": "noun",
    "Verb (lexical)": "verb_lexical",
    "Verb (ser/estar)": "verb_ser",
    "Verb (auxiliary)": "verb_auxiliary",
    "Verb (modal)": "verb_modal",
    "Adjective": "adjective",
    "Adverb": "adverb",
    "Determiner": "determiner",
    "Pronoun": "pronoun",
    "Preposition": "preposition",
    "Conjunction": "conjunction",
    "Proper Noun": "proper_noun",
    "Unknown": "unknown",
}
pos_label_display = st.sidebar.selectbox(
    "Part of Speech",
    list(pos_label_options.keys()),
    index=0
)
pos_label = pos_label_options[pos_label_display]

# NEW: Grammatical feature selectors (all match enrich_chunk output values)
st.sidebar.markdown("#### 🔬 Grammatical Context")
tense = st.sidebar.selectbox(
    "Tense / Mood",
    ["unknown", "present_indicative", "past_participle", "infinitive", "gerund",
     "preterite", "conditional", "future_indicative", "imperative",
     "past_imperfect_indicative", "present_subjunctive"],
    index=0
)
person = st.sidebar.selectbox("Person", ["unknown", "1st_person", "2nd_person", "3rd_person"], index=0)
gram_number = st.sidebar.selectbox("Number", ["unknown", "singular", "plural", "singular_or_plural"], index=0)
gender = st.sidebar.selectbox("Gender", ["unknown", "masculine", "feminine", "neuter", "masculine_or_feminine"], index=0)
case = st.sidebar.selectbox("Case", ["unknown", "nominative", "accusative", "dative", "genitive", "vocative"], index=0)
definiteness = st.sidebar.selectbox("Definiteness", ["unknown", "definite", "indefinite", "demonstrative", "possessive"], index=0)
degree = st.sidebar.selectbox("Adj. Degree", ["unknown", "comparative", "superlative"], index=0)

st.sidebar.markdown("---")

# Practice History
if 'history_seen' not in st.session_state:
    st.session_state.history_seen = 23
if 'history_correct' not in st.session_state:
    st.session_state.history_correct = 13

def enforce_math_logic():
    if st.session_state.history_correct > st.session_state.history_seen:
        st.session_state.history_correct = st.session_state.history_seen

history_seen = st.sidebar.number_input(
    "Total Times Practiced", 
    min_value=1, max_value=200, 
    key='history_seen', 
    on_change=enforce_math_logic
)
history_correct = st.sidebar.number_input(
    "Total Times Correct", 
    min_value=0, max_value=st.session_state.history_seen, 
    key='history_correct'
)

st.sidebar.markdown("#### ⏱️ Timing")
time_lag_days = st.sidebar.slider("Days Since Last Practice", min_value=0.1, max_value=365.0, value=2.0, step=0.5)
hour_of_day = st.sidebar.slider("Hour of Day Practicing", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# NEW: Spacing history mode
st.sidebar.markdown("---")
st.sidebar.markdown("#### 📅 Study History Simulation")
history_mode = st.sidebar.radio(
    "How to simulate past reviews?",
    ["Fixed intervals (every 0.5 days)", "Spaced (1, 3, 7, 14 days)", "Custom sequence"],
    index=0
)
if history_mode == "Custom sequence":
    custom_intervals_str = st.sidebar.text_input(
        "Enter intervals in days (comma-separated)",
        value="0.5, 1, 3, 7"
    )
    try:
        custom_intervals = [float(x.strip()) for x in custom_intervals_str.split(",")]
    except:
        custom_intervals = [0.5]
else:
    custom_intervals = None

SPACING_PRESETS = {
    "Fixed intervals (every 0.5 days)": None,
    "Spaced (1, 3, 7, 14 days)": [1.0, 2.0, 4.0, 7.0],
}

# --- 6. MCM FEATURE ENGINEERING ---
# Use day-scale parameters: tau spans ~1 hour (0.042 days) to ~1 year (365 days)
# Default model params (mu=0.01, nu=1.05) have tau_max ≈ 1.3 days — too short for the slider range.
mcm = MultiscaleContextModel(mu=0.042, nu=1.096, N=100, xi=0.9, eps_r=9.0)

if history_mode == "Fixed intervals (every 0.5 days)":
    for i in range(history_seen):
        is_correct = i < history_correct
        mcm.study(0.5, is_correct)
elif history_mode == "Spaced (1, 3, 7, 14 days)":
    intervals = SPACING_PRESETS["Spaced (1, 3, 7, 14 days)"]
    for idx, interval in enumerate(intervals[:history_seen]):
        is_correct = idx < history_correct
        mcm.study(interval, is_correct)
else:  # custom
    for idx, interval in enumerate(custom_intervals[:history_seen]):
        is_correct = idx < history_correct
        mcm.study(interval, is_correct)

mcm_p = mcm.predict(time_lag_days)

historical_accuracy = history_correct / max(history_seen, 1)
user_global_accuracy = 0.80

input_data = pd.DataFrame({
    'hour_of_day': [hour_of_day],
    'day_of_week': [3],
    'time_lag_days': [time_lag_days],
    'log_delta': [np.log1p(time_lag_days)],
    'lang': [lang],
    'mcm_predicted_p': [mcm_p],
    'historical_accuracy': [historical_accuracy],
    'user_global_accuracy': [user_global_accuracy],
    'right': [np.sqrt(1 + history_correct)],
    'wrong': [np.sqrt(1 + (history_seen - history_correct))],
    # Grammatical features — names match enrich_chunk / model1.py feature matrix
    'pos_label': [pos_label],
    'tense': [tense],
    'person': [person],
    'grammatical_number': [gram_number],
    'gender': [gender],
    'case': [case],
    'definiteness': [definiteness],
    'degree': [degree],
})

for col in ['lang', 'pos_label', 'tense', 'person', 'grammatical_number',
            'gender', 'case', 'definiteness', 'degree']:
    input_data[col] = input_data[col].astype('category')

# --- 7. PREDICTION ---
# Align columns to model's expected features (fallback gracefully)
try:
    expected_features = model.get_booster().feature_names
    # Fill any missing columns with a default
    for feat in expected_features:
        if feat not in input_data.columns:
            input_data[feat] = 0
    input_data_aligned = input_data[expected_features]
    h_pred = model.predict(input_data_aligned)[0]
except Exception:
    h_pred = model.predict(input_data)[0]

h_pred = np.clip(h_pred, 15.0 / (24 * 60), 274.0)
p_pred = 2.0 ** (-time_lag_days / h_pred)

# --- 8. MAIN LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📊 Memory Snapshot")
    
    # Recall probability color indicator
    if p_pred >= 0.8:
        recall_color = "#58cc02"
        recall_label = "Strong"
    elif p_pred >= 0.5:
        recall_color = "#ffc800"
        recall_label = "Fading"
    else:
        recall_color = "#ff4b4b"
        recall_label = "Weak"
    
    st.markdown(f"""
    <div style="
        background: {recall_color}22;
        border: 2px solid {recall_color};
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin-bottom: 16px;
    ">
        <div style="font-size: 3rem; font-weight: 800; color: {recall_color};">
            {p_pred * 100:.1f}%
        </div>
        <div style="color: #555; font-size: 1rem;">Recall Probability — <b>{recall_label}</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.metric(label="Memory Half-Life", value=f"{h_pred:.2f} days")
    st.metric(label="MCM Cognitive Baseline", value=f"{mcm_p * 100:.1f}%")
    st.metric(label="Historical Accuracy", value=f"{historical_accuracy * 100:.1f}%")
    
    # Recommendation
    st.markdown("---")
    threshold = 0.5
    if p_pred >= threshold:
        days_to_threshold = h_pred * np.log2(1.0 / threshold)
        remaining = max(0, days_to_threshold - time_lag_days)
        st.info(f"⏰ **Review recommended in** ~{remaining:.1f} days")
    else:
        st.warning("⚠️ **Review now!** Memory is below 50% recall threshold.")

with col2:
    st.markdown("### 📉 Forgetting Curve")
    
    max_days = max(30.0, float(h_pred * 2.5))
    days = np.linspace(0, max_days, 200)
    p_primary = 2.0 ** (-days / h_pred)
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=days, y=p_primary,
        mode='lines',
        name='Your Learner',
        line=dict(color='#1cb0f6', width=4),
        fill='tozeroy',
        fillcolor='rgba(28,176,246,0.08)',
        hovertemplate='Day %{x:.1f}<br>Prob: %{y:.1%}<extra></extra>'
    ))

    # Threshold line at 50%
    fig.add_hline(
        y=0.5,
        line=dict(color='#ff4b4b', width=2, dash='dot'),
        annotation_text="Review Threshold (50%)",
        annotation_font_color='#ff4b4b'
    )

    # Compare with fast forgetter
    st.sidebar.markdown("---")
    compare_mode = st.sidebar.checkbox("🔁 Compare with 'Fast Forgetter'", value=False)
    if compare_mode:
        h_bench = h_pred * 0.5
        p_bench = 2.0 ** (-days / h_bench)
        fig.add_trace(go.Scatter(
            x=days, y=p_bench,
            mode='lines',
            name='Fast Forgetter',
            line=dict(color='#ff9600', width=2, dash='dash'),
            hovertemplate='Day %{x:.1f}<br>Prob: %{y:.1%}<extra></extra>'
        ))

    # Today marker
    fig.add_trace(go.Scatter(
        x=[time_lag_days], y=[p_pred],
        mode='markers',
        name='Today',
        marker=dict(color='#58cc02', size=14, symbol='diamond', line=dict(width=2, color='white')),
        hovertemplate='Lag: %{x} days<br>Recall: %{y:.1%}<extra></extra>'
    ))

    fig.update_layout(
        xaxis_title="Days Since Last Practice",
        yaxis_title="Probability of Recall",
        yaxis=dict(range=[0, 1.05], tickformat='.0%', gridcolor='#eee'),
        xaxis=dict(gridcolor='#eee'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Arial", size=13),
    )

    st.plotly_chart(fig, use_container_width=True)

# --- 9. SHAP EXPLAINABILITY SECTION ---
st.markdown("---")
with st.expander("🔍 Why did the model predict this? (SHAP Feature Impact)", expanded=False):
    try:
        @st.cache_resource
        def get_shap_explainer(_model):
            return shap.TreeExplainer(_model)

        explainer = get_shap_explainer(model)

        # Align columns again for SHAP
        try:
            shap_input = input_data_aligned
        except NameError:
            shap_input = input_data

        shap_vals = explainer.shap_values(shap_input)
        
        feature_names = shap_input.columns.tolist()
        shap_series = pd.Series(shap_vals[0], index=feature_names).sort_values(key=abs, ascending=False)
        top_features = shap_series.head(8)

        colors = ['#58cc02' if v > 0 else '#ff4b4b' for v in top_features.values]
        
        fig_shap = go.Figure(go.Bar(
            x=top_features.values,
            y=top_features.index,
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.3f}" for v in top_features.values],
            textposition='outside'
        ))
        fig_shap.update_layout(
            title="Top Feature Contributions (SHAP values → Half-Life in days)",
            xaxis_title="SHAP Value (impact on predicted memory half-life)",
            yaxis=dict(autorange='reversed'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=350,
            font=dict(family="Arial", size=12),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption(
            "🟢 Green bars increase the predicted half-life (better retention). "
            "🔴 Red bars decrease it (faster forgetting)."
        )
    except Exception as e:
        st.warning(f"SHAP explanation unavailable with this model version: {e}")

# --- 10. FOOTER ---
st.markdown("---")
st.markdown(
    "💡 *Try changing the **Part of Speech** (Noun vs. Verb vs. Adjective) or the **Tense** field "
    "to see how grammatical complexity affects the model's predicted memory half-life!*"
)