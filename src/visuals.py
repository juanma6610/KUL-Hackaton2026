import shap
import matplotlib.pyplot as plt

def generate_pitch_deck_visuals(model, X_train, X_test):
    print("Initializing SHAP TreeExplainer...")
    
    # SHAP requires the underlying booster for XGBoost categorical data
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a random sample of the test set 
    X_sample = X_test.sample(n=min(10000, len(X_test)), random_state=42)
    shap_values = explainer(X_sample)

    # GRAPH 1: The "What Matters Most" Slide (Summary Plot)
    print("Generating Feature Importance Graph...")
    plt.figure(figsize=(10, 6))
    plt.title("What Drives Human Forgetting? (SHAP Feature Importance)")
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("slide1_feature_importance.png", dpi=300)
    plt.clf()

    # GRAPH 2: The "When to Study" Slide (Dependence Plot)
    print("Generating Circadian Rhythm Insight Graph...")
    plt.figure(figsize=(8, 6))
    plt.title("The Synchrony Effect: Time of Day vs. Memory Retention")
    shap.dependence_plot("hour_of_day", shap_values.values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("slide2_circadian_rhythm.png", dpi=300)
    plt.clf()
    
    print("✅ Visuals saved! Check your directory for the PNG files.")
