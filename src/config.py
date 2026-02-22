import math

# Time constraints
MIN_HALF_LIFE = 15.0 / (24 * 60)  # 15 minutes
MAX_HALF_LIFE = 274.0             # 9 months
LN2 = math.log(2.0)

# File paths
MCM_CACHE_FILE = "mcm_cached_dataset.pkl"
MODEL_SAVE_PATH = "models/xgboost_baseline0.001.json"