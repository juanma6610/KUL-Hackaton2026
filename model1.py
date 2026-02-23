"""
Optimization of a spaced repetition model (half-life regression) on a large-scale language learning dataset. See README for details.
"""

import argparse
import math
import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import shap
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler

# Make src/ importable regardless of where the script is launched from
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.build_duo_data import enrich_chunk  # canonical grammatical feature extractor

optuna.logging.set_verbosity(optuna.logging.WARNING)

# various constraints on parameters and outputs
MIN_HALF_LIFE = 15.0 / (24 * 60)    # 15 minutes
MAX_HALF_LIFE = 274.                # 9 months
LN2 = math.log(2.)

# Cache file for MCM features
MCM_CACHE_FILE = "mcm_cached_dataset.pkl"




def get_xgboost_data(input_file, user_fraction=1, use_cache=True):
    print(f"Reading data from {input_file}...")
    
    




    # Target v



model, X_train, X_test)
