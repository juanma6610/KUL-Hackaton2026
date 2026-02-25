import argparse
from src.data_pipeline import get_xgboost_data
from src.model_trainer import train_xgboost_baseline
from src.visuals import generate_pitch_deck_visuals
from src.config import MODEL_SAVE_PATH

def main():
    parser = argparse.ArgumentParser(description='Run Cram-Buster ML Pipeline.')
    parser.add_argument('input_file', action="store", help='Dataset CSV file')
    parser.add_argument('-x', dest="user_fraction", type=float, default=1.0, help="Fraction of users to sample")
    parser.add_argument('--optuna', action="store_true", default=False, help="Run Optuna search")
    parser.add_argument('--trials', dest="n_trials", type=int, default=30, help="Optuna trials")
    parser.add_argument('--no-cache', action="store_true", default=False, help="Ignore MCM cache")
    args = parser.parse_args()

    print("🚀 Starting Training Pipeline...")
    print(args)
    
    use_cache = not args.no_cache
    
    # 1. Data Pipeline
    X_train, X_test, y_train, y_test = get_xgboost_data(args.input_file, args.user_fraction)
    
    # 2. Model Training
    model = train_xgboost_baseline(X_train, X_test, y_train, y_test, args.optuna, args.n_trials)
    
    # 3. Save Artifacts
    model.save_model(MODEL_SAVE_PATH)
    print(f"💾 Model saved to {MODEL_SAVE_PATH}")

    # 3.5 Dump official evaluation CSV
    # dump_predictions_to_csv(X_test, y_test, model)
    
    # 4. Generate Visuals
    generate_pitch_deck_visuals(model, X_train, X_test)
    print("✅ Pipeline Complete.")

if __name__ == "__main__":
    main()