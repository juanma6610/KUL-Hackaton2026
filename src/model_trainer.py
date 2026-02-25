import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from src.config import MIN_HALF_LIFE, MAX_HALF_LIFE

optuna.logging.set_verbosity(optuna.logging.WARNING)

def train_xgboost_baseline(X_train, X_test, y_train, y_test, use_optuna=False, n_trials=300):
    print("Initializing XGBoost Regressor (Phase 2 Baseline)...")
    
    # Target Transformation: Calculate 'h' for training
    h_train = -X_train['time_lag_days'] / np.log2(y_train)
    h_train = np.clip(h_train, MIN_HALF_LIFE, MAX_HALF_LIFE)
    
    h_test = -X_test['time_lag_days'] / np.log2(y_test)
    h_test = np.clip(h_test, MIN_HALF_LIFE, MAX_HALF_LIFE)

    training_features = list(X_train.columns)

    if use_optuna:
        print(f"🔎 Running Optuna hyperparameter search ({n_trials} trials)...")

        def objective(trial):
            params = {
                "tree_method": "hist",
                "enable_categorical": True,
                "n_estimators": trial.suggest_int("n_estimators", 200, 8000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "random_state": 42,
                "early_stopping_rounds": 50,
            }
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train[training_features], h_train,
                eval_set=[(X_test[training_features], h_test)],
                verbose=False
            )
            h_pred = model.predict(X_test[training_features])
            h_pred = np.clip(h_pred, MIN_HALF_LIFE, MAX_HALF_LIFE)
            return mean_absolute_error(h_test, h_pred)

        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_params = study.best_params
        best_params.update({"tree_method": "hist", "enable_categorical": True, "random_state": 42})
        print(f"\n✅ Best params: {best_params}")

        model = xgb.XGBRegressor(**best_params)
        model.fit(
            X_train[training_features], h_train,
            eval_set=[(X_train[training_features], h_train), (X_test[training_features], h_test)],
            verbose=50
        )
    else:
        model = xgb.XGBRegressor(
            tree_method="hist", 
            enable_categorical=True,
            n_estimators=200000,
            learning_rate=0.001   ,
            max_depth=6,
            early_stopping_rounds=50,
            random_state=42
        )
        model.fit(
            X_train[training_features], h_train,
            eval_set=[(X_train[training_features], h_train), (X_test[training_features], h_test)],
            verbose=50 # Prints update every 50 trees
        )

    # Predict h
    h_pred = model.predict(X_test[training_features])
    h_pred = np.clip(h_pred, MIN_HALF_LIFE, MAX_HALF_LIFE)
    
    # Transform h back to probability of recall (p)
    p_pred = 2.0 ** (-X_test['time_lag_days'] / h_pred)
    p_pred = np.clip(p_pred, 0.0001, 0.9999)

    # Evaluation Metrics
    mae_h = mean_absolute_error(h_test, h_pred)
    spearman_h, _ = spearmanr(h_test, h_pred)
    mae_p = mean_absolute_error(y_test, p_pred)
    spearman_p, _ = spearmanr(y_test, p_pred)

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