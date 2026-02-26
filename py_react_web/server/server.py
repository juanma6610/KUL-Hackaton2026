from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from pathlib import Path

app = Flask(__name__)
CORS(app)

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "xgboost_full_features_no_word_len.json"
model = xgb.XGBRegressor()
model.load_model(str(MODEL_PATH))
explainer = shap.TreeExplainer(model)

CATEGORICAL_COLS = ['lang', 'pos_label', 'tense', 'person', 'grammatical_number', 'gender', 'case', 'definiteness', 'degree']

def build_dataframe(data):
    history_wrong = data['history_seen'] - data['history_correct']
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
        'wrong': [np.sqrt(1 + history_wrong)],
        'pos_label': [data['pos_label']],
        'tense': [data['tense']],
        'person': [data['person']],
        'grammatical_number': [data['grammatical_number']],
        'gender': [data['gender']],
        'case': [data['case']],
        'definiteness': [data['definiteness']],
        'degree': [data['degree']],
    })
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype('category')
    return df

def clip_halflife(value):
    return float(np.clip(value, 0.01, 274.0))

def predict_xgboost(df):
    try:
        features = model.get_booster().feature_names
        for f in features:
            if f not in df.columns:
                df[f] = 0
        df = df[features]
    except Exception as e:
        print(f"Feature alignment warning: {e}")
    return clip_halflife(model.predict(df)[0])

def predict_pimsleur(history_seen):
    return clip_halflife(2 ** (2.4 * history_seen - 16.5))

def predict_leitner(history_correct, history_wrong):
    return clip_halflife(2 ** (history_correct - history_wrong))

def predict_hlr(history_correct, history_wrong):
    return clip_halflife(2 ** (0.0020 * history_correct - 0.1891 * history_wrong + 7.3341))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = build_dataframe(data)
    history_wrong = data['history_seen'] - data['history_correct']
    
    predictions = {
        'h_xgboost': round(predict_xgboost(df), 1),
        'h_pimsleur': round(predict_pimsleur(data['history_seen']), 1),
        'h_leitner': round(predict_leitner(data['history_correct'], history_wrong), 1),
        'h_hlr': round(predict_hlr(data['history_correct'], history_wrong), 1),
    }
    
    shap_vals = explainer.shap_values(df)
    shap_dict = {col: float(shap_vals[0][i]) for i, col in enumerate(df.columns)}
    predictions['shap_values'] = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    
    return jsonify(predictions)

@app.route('/user_stats', methods=['GET'])
def user_stats():
    return jsonify({
        'total_words': 245,
        'accuracy': 0.78,
        'streak_days': 12,
        'weak_categories': [
            {'name': 'Verbs (Past)', 'count': 23},
            {'name': 'Adjectives', 'count': 18},
            {'name': 'Pronouns', 'count': 15}
        ],
        'time_distribution': [{'hour': i, 'count': int(np.random.randint(10, 50))} for i in range(24)],
        'accuracy_trend': [{'day': i, 'accuracy': float(0.6 + np.random.random() * 0.3)} for i in range(30)]
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
