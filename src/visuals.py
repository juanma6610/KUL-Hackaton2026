import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_halflife_correlation(h_true, h_pred_baseline, h_pred_new):
    """
    Generates a side-by-side scatter plot comparing the baseline model 
    to your XGBoost model on a logarithmic scale.
    """
    print("🎨 Generating the correlation visual...")
    
    # 1. Convert everything to log10 space for plotting
    # We add a tiny epsilon (1e-5) to prevent log(0) errors
    log_h_true = np.log10(h_true + 1e-5)
    log_h_base = np.log10(h_pred_baseline + 1e-5)
    log_h_new = np.log10(h_pred_new + 1e-5)
    
    # 2. Set up a beautiful, high-contrast aesthetic
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle("Predicting True Memory Half-Life ($h$)", fontsize=20, fontweight='bold', y=1.05)
    
    # 3. Plot 1: The Baseline Model
    sns.scatterplot(x=log_h_true, y=log_h_base, ax=axes[0], 
                    alpha=0.3, color="#FF6B6B", s=15, edgecolor="none")
    axes[0].set_title("Baseline Model (Standard HLR)", fontsize=16, pad=15)
    axes[0].set_xlabel("True $\log_{10}(h)$ (Empirical)", fontsize=14)
    axes[0].set_ylabel("Predicted $\log_{10}(h)$", fontsize=14)
    
    # 4. Plot 2: Your XGBoost Model
    sns.scatterplot(x=log_h_true, y=log_h_new, ax=axes[1], 
                    alpha=0.3, color="#4ECDC4", s=15, edgecolor="none")
    axes[1].set_title("Our Optimized Model (XGBoost + MCM)", fontsize=16, pad=15)
    axes[1].set_xlabel("True $\log_{10}(h)$ (Empirical)", fontsize=14)
    
    # 5. Add the "Perfect Prediction" Reference Lines (y = x)
    min_val = min(log_h_true.min(), -4)
    max_val = max(log_h_true.max(), 3)
    
    for ax in axes:
        ax.plot([min_val, max_val], [min_val, max_val], 
                color="black", linestyle="--", linewidth=2, alpha=0.8, label="Perfect Prediction ($y=x$)")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.legend(loc="upper left", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("half_life_correlation_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved as 'half_life_correlation_comparison.png' at 300 DPI!")

# --- How to use it ---
# h_true = The actual empirical half-life calculated from your test set
# h_base = The predictions from the baseline model
# h_new = The predictions from your new master XGBoost model
#
# plot_halflife_correlation(h_true, h_base, h_new)

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
