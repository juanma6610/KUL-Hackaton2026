# src/data_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

def get_xgboost_data(processed_file_path="data/processed_features.parquet", user_fraction=1.0):
    print(f"⚡ Loading pre-processed data from {processed_file_path}...")
    df = pd.read_parquet(processed_file_path)
    
    if user_fraction < 1.0:
        print(f"Shrinking dataset to {user_fraction*100}% of users for a fast trial...")
        unique_users = df['user_id'].unique()
        np.random.seed(42) 
        sampled_users = np.random.choice(
            unique_users, 
            size=int(len(unique_users) * user_fraction), 
            replace=False
        )
        df = df[df['user_id'].isin(sampled_users)].copy()
        print(f"Trial Data ready: Shrunk from {original_len} rows to {len(df)} rows.")

    # # MCM with caching
    # cache_file = MCM_CACHE_FILE
    # if use_cache and os.path.exists(cache_file):
    #     print("🟢 Found cached MCM data! Loading instantly...")
    #     df = pd.read_pickle(cache_file)
    # else:
    #     print("🟡 Running MCM feature generation (this may take a while)...")
    #     df = generate_mcm_features(df)
    #     if use_cache:
    #         print("💾 Saving MCM data to cache...")
    #         df.to_pickle(cache_file)


    # Target variable
    y = df['p_recall'].clip(lower=0.0001, upper=0.9999)

    # Feature matrix
    X = pd.DataFrame()

    # Time features
    X['hour_of_day'] = df['timestamp'].dt.hour
    X['day_of_week'] = df['timestamp'].dt.dayofweek

    #Delta represents the time since last practice in days, which is critical for the forgetting curve
    #DECIDE WHICH ONE TO KEEP: raw delta or log delta (log can help with skew but may lose interpretability)
    X['time_lag_days'] = df['delta'] / (60 * 60 * 24)
    #X['log_delta'] = np.log1p(df['delta'] / (60 * 60 * 24))

    # Combine languages into a single feature
    X['lang'] = df['ui_language'] + "->" + df['learning_language']

    # MCM predictions
    X['mcm_predicted_p'] = df['mcm_predicted_p']

    # Accuracy features
    X['historical_accuracy'] = df['historical_accuracy'] # Micro: Their skill on this word
    X['user_global_accuracy'] = df['user_global_accuracy'] # Macro: Their overall app skill
    X['right'] = np.sqrt(1 + df['history_correct'])  #Raw success count (sqrt to reduce skew)
    X['wrong'] = np.sqrt(1 + (df['history_seen'] - df['history_correct']))  #Raw failure count (sqrt authors mentioned it works better)

    # --- NEW: Full grammatical feature set from enrich_chunk ---
    # pos_label: descriptive POS ("noun", "verb_lexical", "adjective", …)
    X['pos_label'] = df['pos_label'].fillna('unknown')
    X['tense'] = df['tense'].fillna('unknown')
    X['person'] = df['person'].fillna('unknown')
    X['grammatical_number'] = df['number'].fillna('unknown')   # enrich_chunk uses 'number'
    X['gender'] = df['gender'].fillna('unknown')
    X['case'] = df['case'].fillna('unknown')
    X['definiteness'] = df['definiteness'].fillna('unknown')
    X['degree'] = df['degree'].fillna('unknown')

    # Cast all categoricals for native XGBoost handling
    cat_cols = ['lang', 'pos_label', 'tense', 'person',
                'grammatical_number', 'gender', 'case', 'definiteness', 'degree']
    for col in cat_cols:
        X[col] = X[col].astype('category')

    
    print("Feature engineering complete! Splitting data based on users...")


    # Stratified split
    groups = df['user_id'].values
    gkf = GroupKFold(n_splits=10)
    train_idx, test_idx = next(gkf.split(X, y, groups=groups))
    
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]