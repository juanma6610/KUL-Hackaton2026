from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb

app = Flask(__name__)
CORS(app)

model = xgb.XGBRegressor()
model.load_model("../models/xgboost_baseline0.001.json")

@app.route('/predict', methods=['POST'])
def old_models():
    # --- MODEL 1: Pimsleur (1967) ---
    # Weights: x_n (practices) = 2.4, x_b (bias) = -16.5
    # Notice it uses history_seen (total practices), not just correct answers!
    pimsleur_log_h = (2.4 * history_seen) - 16.5

    # Clip the half-life so the math doesn't explode into infinity or zero
    pimsleur_h = np.clip(2 ** pimsleur_log_h, 0.0001, 274.0) 
    pimsleur_p = 2 ** (-time_lag_days / pimsleur_h)


    # --- MODEL 2: Leitner (1972) ---
    # Weights: x_correct = 1, x_incorrect = -1
    # Every correct answer doubles the interval, every wrong answer halves it.
    leitner_log_h = (1.0 * history_correct) + (-1.0 * history_wrong)

    leitner_h = np.clip(2 ** leitner_log_h, 0.0001, 274.0)
    leitner_p = 2 ** (-time_lag_days / leitner_h)

        # --- MODEL 3: Half-Life Regression (Duolingo, 2016) ---
    # This uses the specific weights YOU trained on the dataset earlier!

    hlr_weight_correct = 0.0020
    hlr_weight_wrong = -0.1891
    hlr_bias = 7.3341

def predict():
    data = request.json
    
    df = pd.DataFrame({
        'hour_of_day': [data['hour_of_day']],
        'day_of_week': [3],
        'time_lag_days': [data['time_lag_days']],
        'log_delta': [np.log1p(data['time_lag_days'])],
        'lang': [data['lang']],
        'mcm_predicted_p': [data['mcm_predicted_p']],
        'historical_accuracy': [data['historical_accuracy']],
        'user_global_accuracy': [0.80],
        'right': [np.sqrt(1 + data['history_correct'])],
        'wrong': [np.sqrt(1 + data['history_seen'] - data['history_correct'])],
        'pos_label': [data['pos_label']],
        'tense': [data['tense']],
        'person': [data['person']],
        'grammatical_number': [data['grammatical_number']],
        'gender': [data['gender']],
        'case': [data['case']],
        'definiteness': [data['definiteness']],
        'degree': [data['degree']],
    })
    
    for col in ['lang', 'pos_label', 'tense', 'person', 'grammatical_number', 'gender', 'case', 'definiteness', 'degree']:
        df[col] = df[col].astype('category')
    
    try:
        features = model.get_booster().feature_names
        for f in features:
            if f not in df.columns:
                df[f] = 0
        df = df[features]
    except:
        pass
    
    h = np.clip(model.predict(df)[0], 15.0 / (24 * 60), 274.0)
    return jsonify({'h_pred': float(h)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
