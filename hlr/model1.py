"""
Optimization of a spaced repetition model (half-life regression) on a large-scale language learning dataset. See README for details.
"""

import argparse
import csv
import gzip
import math
import os
import random
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import shap
import matplotlib.pyplot as plt
from sys import intern
import optuna

from collections import defaultdict, namedtuple


# various constraints on parameters and outputs
MIN_HALF_LIFE = 15.0 / (24 * 60)    # 15 minutes
MAX_HALF_LIFE = 274.                # 9 months
LN2 = math.log(2.)



def get_xgboost_data(input_file, omit_bias=False, omit_lexemes=False, max_lines=None):
    print(f"Reading data from {input_file}...")
    
    # 1. Load Data
    if max_lines is not None:
        df = pd.read_csv(input_file, compression='infer', nrows=max_lines)
    else:
        df = pd.read_csv(input_file, compression='infer')

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
    # Replaces the pclip() function by clipping the entire column at once
    y = df['p_recall'].clip(lower=0.0001, upper=0.9999)

    # 3. Vectorized Feature Engineering (X)
    X = pd.DataFrame()


    X['hour_of_day'] = df['timestamp'].dt.hour
    X['day_of_week'] = df['timestamp'].dt.dayofweek

    X['time_lag_days'] = df['delta'] / (60 * 60 * 24)
    X['right'] = np.sqrt(1 + df['history_correct'])
    X['wrong'] = np.sqrt(1 + (df['history_seen'] - df['history_correct']))
    df['log_delta'] = np.log1p(df['delta'] / (60 * 60 * 24))
    # Combine languages into a single feature
    X['lang'] = df['ui_language'] + "->" + df['learning_language']

    X['historical_accuracy'] = df['historical_accuracy']
    X['log_delta'] = df['log_delta']
    
    # 4. The XGBoost Categorical Magic
    # Cast strings to pandas categorical dtype. XGBoost will handle the rest!
    X['lang'] = X['lang'].astype('category')
    #X['lexeme_string'] = df['lexeme_string'].astype('category')
    X['pos_tag'] = df['pos_tag'].astype('category')

    print("Data processing complete! Splitting data...")
    
    # 5. Fast 90/10 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_xgboost_baseline(X_train, X_test, y_train, y_test):
    print("Initializing XGBoost Regressor (Phase 2 Baseline)...")

    # Duolingo's official clipping constraints from their 2016 code
    MIN_HALF_LIFE = 15.0 / (24 * 60)  # 15 minutes in days
    MAX_HALF_LIFE = 274.0             # 9 months in days
    
    # 1. Target Transformation: Calculate 'h' for training
    # We use X_train['time_lag_days'] for 't'
    h_train = -X_train['time_lag_days'] / np.log2(y_train)
    h_train = np.clip(h_train, MIN_HALF_LIFE, MAX_HALF_LIFE)
    
    h_test = -X_test['time_lag_days'] / np.log2(y_test)
    h_test = np.clip(h_test, MIN_HALF_LIFE, MAX_HALF_LIFE)
    
    # 1. The Model Architecture
    # tree_method="hist" is required for native categorical support
    # We set a high n_estimators but use early_stopping to prevent overfitting
    model = xgb.XGBRegressor(
        tree_method="hist", 
        enable_categorical=True,
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        early_stopping_rounds=50,
        random_state=42
    )

    print("Training model... (Monitor validation loss to prevent overfitting)")
    
    # 2. The Training Loop
    # We pass the test set as the eval_set so the model can track its out-of-sample error
    model.fit(
        X_train, h_train,
        eval_set=[(X_train, h_train), (X_test, h_test)],
        verbose=50 # Prints update every 50 trees
    )

    # 2. Predict h
    h_pred = model.predict(X_test)
    h_pred = np.clip(h_pred, MIN_HALF_LIFE, MAX_HALF_LIFE)
    
    # 3. Transform h back to probability of recall (p)
    p_pred = 2.0 ** (-X_test['time_lag_days'] / h_pred)
    p_pred = np.clip(p_pred, 0.0001, 0.9999)

    # 4. Evaluate against the ORIGINAL p_recall (Apples-to-Apples with Duolingo)
    mae = mean_absolute_error(y_test, p_pred)

    # 4. Evaluation Metrics
    mae = mean_absolute_error(y_test, p_pred)
    #spearman_corr, _ = spearmanr(y_test, p_pred)

    print("-" * 30)
    print("🏆 PHASE 2 RESULTS 🏆")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    #print(f"Spearman Correlation:      {spearman_corr:.4f}")
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
argparser.add_argument('-x', action="store", dest="max_lines", type=int, default=None, help="maximum number of lines to read (for dev)")
argparser.add_argument('input_file', action="store", help='log file for training')


if __name__ == "__main__":



    args = argparser.parse_args()
    print(args)

    X_train, X_test, y_train, y_test = get_xgboost_data(args.input_file)
    baseline_model = train_xgboost_baseline(X_train, X_test, y_train, y_test)
    baseline_model.save_model("xgboost_baseline.json")
    generate_pitch_deck_visuals(baseline_model, X_train, X_test)

    

    # # model diagnostics
    # sys.stderr.write('method = "%s"\n' % args.method)
    # if args.b:
    #     sys.stderr.write('--> omit_bias\n')
    # if args.l:
    #     sys.stderr.write('--> omit_lexemes\n')
    # if args.t:
    #     sys.stderr.write('--> omit_h_term\n')

    # # read data set
    # trainset, testset = get_xgboost_data(args.input_file, args.method, args.b, args.l, args.max_lines)
    # sys.stderr.write('|train| = %d\n' % len(trainset))
    # sys.stderr.write('|test|  = %d\n' % len(testset))

    # # train model & print preliminary evaluation info
    # model = SpacedRepetitionModel(method=args.method, omit_h_term=args.t)
    # model.train(trainset)
    # model.eval(testset, 'test')

    # # write out model weights and predictions
    # filebits = [args.method] + \
    #     [k for k, v in sorted(vars(args).items()) if v is True] + \
    #     [os.path.splitext(os.path.basename(args.input_file).replace('.gz', ''))[0]]
    # if args.max_lines is not None:
    #     filebits.append(str(args.max_lines))
    # filebase = '.'.join(filebits)
    # if not os.path.exists('results/'):
    #     os.makedirs('results/')
    # model.dump_weights('results/'+filebase+'.weights')
    # model.dump_predictions('results/'+filebase+'.preds', testset)
    # model.dump_detailed_predictions('results/'+filebase+'.detailed', testset)
