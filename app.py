import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from hlr.model1 import MultiscaleContextModel  
import plotly.graph_objects as go

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Duolingo Memory Engine", layout="wide")
st.title("🧠 Duolingo Spaced Repetition AI")
st.markdown("### Hybrid XGBoost + Cognitive Science (MCM) Model")

# --- 2. LOAD THE SAVED MODEL ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("xgboost_baseline.json") 
    return model

model = load_model()


# --- 4. THE SIDEBAR USER INTERFACE ---
st.sidebar.header("👤 Simulate a Learner")

# Dropdowns for categorical data
lang = st.sidebar.selectbox("Language Track", ["en->es", "en->fr", "en->de", "en->it", "en->pt", "es->en", "it->en", "pt->en"], index=0)
pos_tag = st.sidebar.selectbox("Part of Speech (Grammar)", ["<n>", "<v>", "<adj>", "<adv>", "unknown"], index=1)

# Practice History
if 'history_seen' not in st.session_state:
    st.session_state.history_seen = 23
if 'history_correct' not in st.session_state:
    st.session_state.history_correct = 13

# 2. Create a safety rule: If they lower 'Practiced', forcefully lower 'Correct' so it doesn't crash
def enforce_math_logic():
    if st.session_state.history_correct > st.session_state.history_seen:
        st.session_state.history_correct = st.session_state.history_seen

# 3. Build the UI using 'keys' instead of hardcoded values
history_seen = st.sidebar.number_input(
    "Total Times Practiced", 
    min_value=1, 
    max_value=100, 
    key='history_seen', 
    on_change=enforce_math_logic
)

history_correct = st.sidebar.number_input(
    "Total Times Correct", 
    min_value=0, 
    max_value=st.session_state.history_seen, 
    key='history_correct'
)
# Current Context
time_lag_days = st.sidebar.slider("Days Since Last Practice", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
hour_of_day = st.sidebar.slider("Hour of Day Practicing", min_value=0.0, max_value=4.0, value=0.5, step=0.1)

# --- 5. DYNAMIC FEATURE ENGINEERING ---
# Simulate the MCM memory state based on the user's history
mcm = MultiscaleContextModel()
# We simulate their history by assuming they practiced 1 day apart for their past sessions
for i in range(history_seen):
    is_correct = i < history_correct
    mcm.study(0.5, is_correct) # Practice every 0.5 days

mcm_p = mcm.predict(time_lag_days)

# Build the exact dataframe the model expects
historical_accuracy = history_correct / history_seen
user_global_accuracy = 0.80 # Assumed average user baseline for the demo

input_data = pd.DataFrame({
    'hour_of_day': [hour_of_day],
    'day_of_week': [3], # Assuming Wednesday
    'time_lag_days': [time_lag_days],
    'log_delta': [np.log1p(time_lag_days)],
    'lang': [lang],
    'mcm_predicted_p': [mcm_p],
    'historical_accuracy': [historical_accuracy],
    'user_global_accuracy': [user_global_accuracy],
    'right': [np.sqrt(1 + history_correct)],
    'wrong': [np.sqrt(1 + (history_seen - history_correct))],
    'pos_tag': [pos_tag]
})

# Categorical casting for XGBoost
input_data['lang'] = input_data['lang'].astype('category')
input_data['pos_tag'] = input_data['pos_tag'].astype('category')

# --- 6. MAKE THE PREDICTION ---
# The model predicts Half-Life (h)
h_pred = model.predict(input_data)[0]
h_pred = np.clip(h_pred, 15.0 / (24 * 60), 274.0)

# Convert Half-Life back to Probability of Recall (p)
p_pred = 2.0 ** (-time_lag_days / h_pred)

# --- 7. DISPLAY THE RESULTS ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Model Predictions")
    st.metric(label="Predicted Probability of Recall", value=f"{p_pred * 100:.1f}%")
    st.metric(label="Calculated Memory Half-Life", value=f"{h_pred:.2f} Days")
    st.metric(label="MCM Cognitive Baseline", value=f"{mcm_p * 100:.1f}%")

with col2:
    st.markdown("### 📉 Interactive Forgetting Curve")
    
    # 1. Calculate Primary Curve (The Current User)
    max_days = max(30.0, float(h_pred * 2.5))
    days = np.linspace(0, max_days, 150)
    p_primary = 2.0 ** (-days / h_pred)
    
    fig = go.Figure()

    # --- PRIMARY USER LINE ---
    fig.add_trace(go.Scatter(
        x=days, y=p_primary,
        mode='lines',
        name='Current Learner',
        line=dict(color='#1cb0f6', width=4),
        hovertemplate='Day %{x:.1f}<br>Prob: %{y:.1%}<extra></extra>'
    ))

    # --- OPTIONAL: WHAT-IF SCENARIO ---
    st.sidebar.markdown("---")
    compare_mode = st.sidebar.checkbox("Compare with 'Fast Forgetter'", value=False)
    
    if compare_mode:
        # Simulate a user with 50% lower half-life
        h_bench = h_pred * 0.5
        p_bench = 2.0 ** (-days / h_bench)
        
        fig.add_trace(go.Scatter(
            x=days, y=p_bench,
            mode='lines',
            name='Fast Forgetter',
            line=dict(color='#ff9600', width=2, dash='dash'),
            hovertemplate='Day %{x:.1f}<br>Prob: %{y:.1%}<extra></extra>'
        ))

    # Today Marker
    fig.add_trace(go.Scatter(
        x=[time_lag_days], y=[p_pred],
        mode='markers',
        name='Today',
        marker=dict(color='#ff4b4b', size=14, symbol='diamond', line=dict(width=2, color='white')),
        hovertemplate='Lag: %{x} days<br>Recall: %{y:.1%}<extra></extra>'
    ))

    # Layout Polish
    fig.update_layout(
        xaxis_title="Days Since Last Practice",
        yaxis_title="Probability of Recall",
        yaxis=dict(range=[0, 1.05], tickformat='.0%', gridcolor='lightgray'),
        xaxis=dict(gridcolor='lightgray'),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("💡 *Try changing the Part of Speech from a Noun `<n>` to a Verb `<v>` and watch how the model instantly recalculates the cognitive difficulty!*")