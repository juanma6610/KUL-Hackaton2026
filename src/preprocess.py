# scripts/preprocess_data.py
import pandas as pd
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.build_duo_data import enrich_chunk
from src.mcm import generate_mcm_features 

RAW_DATA_PATH = "data/SpacedRepetitionData.csv"
PROCESSED_DATA_PATH = "data/processed_features.parquet"

def build_gold_dataset():
    print("📥 Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    print("🧠 Generating Multiscale Context Model (MCM) features...")
    df = generate_mcm_features(df)
    
    print("📝 Running NLP grammatical enrichment (enrich_chunk)...")
    df = enrich_chunk(df)
    
    print("🧮 Calculating historical accuracies...")
    # Historical accuracy features
    df['historical_accuracy'] = np.where(
        df['history_seen'] > 0, 
        df['history_correct'] / df['history_seen'], 
        0.0
    )

    # Global User Accuracy
    user_acc = df.groupby('user_id').apply(
        lambda x: x['history_correct'].sum() / (x['history_seen'].sum() + 1e-5)
    ).reset_index(name='user_global_accuracy')
    df = df.merge(user_acc, on='user_id', how='left')
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
    
    print(f"💾 Saving highly compressed Parquet to {PROCESSED_DATA_PATH}...")
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_parquet(PROCESSED_DATA_PATH, index=False)
    print("✅ Preprocessing complete! Model training will now take seconds.")

if __name__ == "__main__":
    build_gold_dataset()