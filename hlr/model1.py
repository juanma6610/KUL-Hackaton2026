"""
Optimization of a spaced repetition model (half-life regression) on a large-scale language learning dataset. See README for details.
"""

import argparse
import math
import os
import random
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import shap
import matplotlib.pyplot as plt
from sys import intern
import optuna



# various constraints on parameters and outputs
MIN_HALF_LIFE = 15.0 / (24 * 60)    # 15 minutes
MAX_HALF_LIFE = 274.                # 9 months
LN2 = math.log(2.)


class MultiscaleContextModel:
    def __init__(self, mu=0.01, nu=1.05, xi=0.9, N=100, eps_r=9.0):
        """
        Initializes the MCM memory state for a single item.
        
        Parameters:
        - mu, nu: Control the distribution of time scales (decay rates).
        - xi: Controls the weighting of different time scales.
        - N: Number of context pools (integrators).
        - eps_r: The boost given to successful retrieval (usually > 1).
        """
        self.N = N
        self.eps_r = eps_r
        
        # 1. Initialize Time Scales (tau) and Weights (gamma)
        indices = np.arange(1, N + 1)
        self.tau = mu * (nu ** indices)
        self.gamma = xi ** indices
        self.gamma = self.gamma / np.sum(self.gamma) # Normalize to sum to 1
        
        # Precompute cumulative sums of gamma (Gamma_i) for the strength calculation
        self.Gamma = np.cumsum(self.gamma)
        
        # 2. Initialize the state of the integrators (x_i)
        # All pools start empty (0.0) before the user has ever seen the word
        self.x = np.zeros(N)

    def decay(self, t):
        """Decays the memory state over time t (in days)."""
        if t > 0:
            self.x = self.x * np.exp(-t / self.tau)

    def get_strengths(self):
        """Calculates the net strength (s_i) at each scale."""
        weighted_x = self.gamma * self.x
        cum_weighted_x = np.cumsum(weighted_x)
        return cum_weighted_x / self.Gamma

    def predict(self, t):
        """Predicts the probability of recall after time t."""
        # Calculate what the state *will be* after time t
        decayed_x = self.x * np.exp(-t / self.tau)
        weighted_x = self.gamma * decayed_x
        
        # Global strength is the sum across all N pools
        s_N = np.sum(weighted_x) / self.Gamma[-1] 
        return np.clip(s_N, 0.0001, 0.9999)

    def study(self, t, recalled):
        """
        Updates the memory state after a study attempt.
        - t: Time elapsed since the LAST study session.
        - recalled: Boolean or 1/0 indicating if the user got it right.
        """
        # 1. Decay the memory by the time elapsed since last review
        self.decay(t)
        
        # 2. Calculate current strength before learning
        s = self.get_strengths()
        
        # 3. Determine learning rate based on retrieval success
        # If they recalled it, the boost is eps_r (e.g., 9). If they forgot, it's 1.
        eps = self.eps_r if recalled else 1.0
        
        # 4. Error-correction update: Pools only fill up if earlier pools failed to represent the item
        delta_x = eps * (1.0 - s)
        self.x = np.clip(self.x + delta_x, 0.0, 1.0)


def generate_mcm_features(df):
    """
    Highly optimized generator that bypasses Pandas iterrows overhead 
    by using NumPy arrays and tuple dictionary keys.
    """
    print("Generating MCM predictions...")
    
    # 1. Sort chronologically (Critical for time-series memory models)
    df = df.sort_values(by=['user_id', 'lexeme_string', 'timestamp'])
    
    # 2. Pre-compute values so we don't do math inside the loop
    t_days_array = (df['delta'] / (60 * 60 * 24)).to_numpy()
    recalled_array = (df['session_correct'] > 0).to_numpy()
    
    # Extract to pure numpy arrays for ultra-fast iteration
    users = df['user_id'].to_numpy()
    words = df['lexeme_string'].to_numpy()
    
    # Pre-allocate output array for speed
    mcm_predictions = np.zeros(len(df))
    
    # Dictionary to hold the state objects
    user_item_states = {}
    
    for i in range(len(df)):
        key = (users[i], words[i]) 
        
        # Initialize MCM for this user-word pair if unseen
        if key not in user_item_states:
            user_item_states[key] = MultiscaleContextModel()
            
        mcm = user_item_states[key]
        
        # Predict probability right now
        mcm_predictions[i] = mcm.predict(t_days_array[i])
        
        # Update the state based on the actual outcome
        mcm.study(t_days_array[i], recalled_array[i])
        
    # 4. Re-attach to the dataframe
    df['mcm_predicted_p'] = mcm_predictions
    
    return df



def get_xgboost_data(input_file, user_fraction=1):
    print(f"Reading data from {input_file}...")
    
    # 1. Load Data
    df = pd.read_csv(input_file, compression='infer')
    original_len = len(df)
    if user_fraction < 1.0:
        print(f"Shrinking dataset to {user_fraction*100}% of users for a fast trial...")
        unique_users = df['user_id'].unique()
        
        # Set a random seed so your trial is reproducible!
        np.random.seed(42) 
        sampled_users = np.random.choice(
            unique_users, 
            size=int(len(unique_users) * user_fraction), 
            replace=False
        )
        
        # Filter the dataframe to only include our chosen users
        df = df[df['user_id'].isin(sampled_users)].copy()
        print(f"Trial Data ready: Shrunk from {original_len} rows to {len(df)} rows.")
    
    df = generate_mcm_features(df)
    # cache_file = "mcm_cached_dataset.pkl"
    
    # # If we already ran MCM once, just load the saved file!
    # if os.path.exists(cache_file) :
    #     print("🟢 Found cached MCM data! Loading instantly...")
    #     df = pd.read_pickle(cache_file)
    # else:
    #     print(f"🟡 No cache found. Reading raw data from {input_file}...")

    #     # Run the heavy 30-minute MCM math
    #     df = generate_mcm_features(df)
    #     print("💾 Saving MCM data to cache...")
    #     df.to_pickle(cache_file)


    df['pos_tag'] = df['lexeme_string'].str.extract(r'<([^>]+)>').fillna('unknown')
    df['historical_accuracy'] = np.where(
        df['history_seen'] > 0, 
        df['history_correct'] / df['history_seen'], 
        0.0
    )
    
    # 5. Global User Accuracy
    user_acc = df.groupby('user_id').apply(
        lambda x: x['history_correct'].sum() / (x['history_seen'].sum() + 1e-5)
    ).reset_index(name='user_global_accuracy')
    df = df.merge(user_acc, on='user_id', how='left')
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')

    
    # 2. Vectorized Target Variable (y)
    y = df['p_recall'].clip(lower=0.0001, upper=0.9999)



    # 3. Vectorized Feature Engineering (X)
    X = pd.DataFrame()

    #Time features
    X['hour_of_day'] = df['timestamp'].dt.hour
    X['day_of_week'] = df['timestamp'].dt.dayofweek

    #Delta represents the time since last practice in days, which is critical for the forgetting curve
    #DECIDE WHICH ONE TO KEEP: raw delta or log delta (log can help with skew but may lose interpretability)
    X['time_lag_days'] = df['delta'] / (60 * 60 * 24)
    X['log_delta'] = np.log1p(df['delta'] / (60 * 60 * 24))

    # Combine languages into a single feature
    X['lang'] = df['ui_language'] + "->" + df['learning_language']

    #MCM predictions model
    X['mcm_predicted_p'] = df['mcm_predicted_p']


    #Accuracy features
    X['historical_accuracy'] = df['historical_accuracy']            # Micro: Their skill on this word
    X['user_global_accuracy'] = df['user_global_accuracy']      # Macro: Their overall app skill
    X['right'] = np.sqrt(1 + df['history_correct'])             #Raw success count (sqrt to reduce skew)
    X['wrong'] = np.sqrt(1 + (df['history_seen'] - df['history_correct']))      #Raw failure count (sqrt authors mentioned it works better this way )
                                       
    # 4. The XGBoost Categorical Magic
    X['lang'] = X['lang'].astype('category')
    #X['lexeme_string'] = df['lexeme_string'].astype('category')
    X['pos_tag'] = df['pos_tag'].astype('category')



    print("Data processing complete! Splitting data...")
    
    # 5. Fast 90/10 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_xgboost_baseline(X_train, X_test, y_train, y_test):
    print("Initializing XGBoost Regressor (Phase 2 Baseline)...")
    
    # 1. Target Transformation: Calculate 'h' for training
    # We use X_train['time_lag_days'] for 't'
    h_train = -X_train['time_lag_days'] / np.log2(y_train)
    h_train = np.clip(h_train, MIN_HALF_LIFE, MAX_HALF_LIFE)
    
    h_test = -X_test['time_lag_days'] / np.log2(y_test)
    h_test = np.clip(h_test, MIN_HALF_LIFE, MAX_HALF_LIFE)
    
    # 1. The Model Architecture
    # tree_method="hist" is required for native categorical support
    training_features = [col for col in X_train.columns ]
    # We set a high n_estimators but use early_stopping to prevent overfitting
    model = xgb.XGBRegressor(
        tree_method="hist", 
        enable_categorical=True,
        n_estimators=200000,
        learning_rate=0.001,
        max_depth=6,
        early_stopping_rounds=50,
        random_state=42
    )

    print("Training model... (Monitor validation loss to prevent overfitting)")
    
    # 2. The Training Loop
    # We pass the test set as the eval_set so the model can track its out-of-sample error
    model.fit(
        X_train[training_features], h_train,
        eval_set=[(X_train[training_features], h_train), (X_test[training_features], h_test)],
        verbose=50 # Prints update every 50 trees
    )

    # 2. Predict h
    h_pred = model.predict(X_test[training_features])
    h_pred = np.clip(h_pred, MIN_HALF_LIFE, MAX_HALF_LIFE)
    
    # 3. Transform h back to probability of recall (p)
    p_pred = 2.0 ** (-X_test['time_lag_days'] / h_pred)
    p_pred = np.clip(p_pred, 0.0001, 0.9999)

    # 4. Evaluation Metrics
    mae_h = mean_absolute_error(h_test, h_pred)
    spearman_h, _ = spearmanr(h_test, h_pred)
    
    # --- Probability of Recall (p) Metrics ---
    mae_p = mean_absolute_error(y_test, p_pred)
    spearman_p,_ = spearmanr(y_test, p_pred)

    print("-" * 30)
    print("🏆 PHASE 2 RESULTS 🏆")
    print(f"  MAE (Days):           {mae_h:.4f}")
    print(f"  Spearman Correlation: {spearman_h:.4f}")
    print("-" * 40)
    print("RECALL PROBABILITY (p) PREDICTION:")
    print(f"  MAE:                  {mae_p:.4f}")
    print(f"  Spearman Correlation: {spearman_p:.4f}")
    print("-" * 30)
    
    return model

def generate_pitch_deck_visuals(model, X_train, X_test):
    print("Initializing SHAP TreeExplainer...")
    
    # SHAP requires the underlying booster for XGBoost categorical data
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a random sample of the test set 
    # (Using 10,000 rows to keep computation fast during the hackathon)
    X_sample = X_test.sample(n=10000, random_state=42)
    shap_values = explainer(X_sample)

    # ---------------------------------------------------------
    # GRAPH 1: The "What Matters Most" Slide (Summary Plot)
    # ---------------------------------------------------------
    print("Generating Feature Importance Graph...")
    plt.figure(figsize=(10, 6))
    plt.title("What Drives Human Forgetting? (SHAP Feature Importance)")
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("slide1_feature_importance.png", dpi=300)
    plt.clf()

    # ---------------------------------------------------------
    # GRAPH 2: The "When to Study" Slide (Dependence Plot)
    # ---------------------------------------------------------
    print("Generating Circadian Rhythm Insight Graph...")
    # This shows how the hour of the day impacts the memory half-life
    plt.figure(figsize=(8, 6))
    plt.title("The Synchrony Effect: Time of Day vs. Memory Retention")
    shap.dependence_plot("hour_of_day", shap_values.values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("slide2_circadian_rhythm.png", dpi=300)
    plt.clf()
    
    print("✅ Visuals saved! Check your directory for the PNG files.")


argparser = argparse.ArgumentParser(description='Fit a SpacedRepetitionModel to data.')
argparser.add_argument('-b', action="store_true", default=False, help='omit bias feature')
argparser.add_argument('-l', action="store_true", default=False, help='omit lexeme features')
argparser.add_argument('-t', action="store_true", default=False, help='omit half-life term')
argparser.add_argument('-m', action="store", dest="method", default='hlr', help="hlr, lr, leitner, pimsleur")
argparser.add_argument('-x', action="store", dest="max_lines", type=float, default=None, help="maximum number of lines to read (for dev)")
argparser.add_argument('input_file', action="store", help='log file for training')


if __name__ == "__main__":



    args = argparser.parse_args()
    print(args)

    X_train, X_test, y_train, y_test = get_xgboost_data(args.input_file,args.max_lines)
    baseline_model = train_xgboost_baseline(X_train, X_test, y_train, y_test)
    baseline_model.save_model("xgboost_baseline0.001.json")
    generate_pitch_deck_visuals(baseline_model, X_train, X_test)


