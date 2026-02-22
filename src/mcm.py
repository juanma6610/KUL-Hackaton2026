"""
src/mcm.py
==========
Multiscale Context Model (MCM) — cognitive memory model for spaced repetition.

Provides:
  - MultiscaleContextModel  : per-(user, word) memory state object
  - generate_mcm_features() : vectorized batch prediction over a DataFrame
"""

import numpy as np
import pandas as pd

MCM_CACHE_FILE = "mcm_cached_dataset.pkl"


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
