"""
==============================================================================
PCA on MCM Model — Numerical Features Only
==============================================================================

PCA on only the 9 continuous numerical features.
Categorical features (POS, tense, language, etc.) are used as coloring
variables in the scatter plots — not as input to PCA.

This avoids the one-hot dilution problem and produces strong PCs.

Usage:
    python pca_mcm_numerical.py

INPUT:  pca_input.parquet
OUTPUT: pca_mcm_num_main.png, pca_mcm_num_scree.png, pca_mcm_num_loadings.png,
        pca_mcm_num_loadings.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# STEP 1: LOAD AND SAMPLE
# =============================================================================

print("=" * 70)
print("  Loading data")
print("=" * 70)

df = pd.read_parquet('datasets/pca_input2.parquet')
#df['total_attempts_sqrt'] = np.sqrt(df['right']**2 + df['wrong']**2-2)
print(f"  Loaded: {len(df):,} rows × {df.shape[1]} columns")

SAMPLE_SIZE = 8_000_000
np.random.seed(42)
idx = np.random.choice(len(df), size=SAMPLE_SIZE, replace=False)
df = df.iloc[idx].reset_index(drop=True)
print(f"  Sampled: {len(df):,} rows")


# =============================================================================
# STEP 2: PCA ON NUMERICAL FEATURES ONLY
# =============================================================================

numerical_cols = [
    'mcm_predicted_p', 'historical_accuracy', 'user_global_accuracy',
    'right', 'wrong', 
    'time_lag_days', #'log_delta',
    'hour_of_day', 'day_of_week', 'word_length'
    #,'total_attempts_sqrt'
]

categorical_cols = [
    'pos_label', 'tense', 'person', 'grammatical_number',
    'gender', 'case', 'definiteness', 'degree', 'lang',
]

X = df[numerical_cols].fillna(0).values

print(f"\n  PCA on {len(numerical_cols)} numerical features:")
for col in numerical_cols:
    print(f"    {col:30s}  mean={df[col].mean():.4f}  std={df[col].std():.4f}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Results
cumvar = np.cumsum(pca.explained_variance_ratio_)
n50 = np.argmax(cumvar >= 0.5) + 1
n80 = np.argmax(cumvar >= 0.8) + 1

print(f"\n  {'PC':<6} {'Individual':>12} {'Cumulative':>12}")
print(f"  {'-' * 32}")
for i in range(len(numerical_cols)):
    ev = pca.explained_variance_ratio_[i]
    bar = "█" * int(ev * 50)
    print(f"  PC{i+1:<3d} {ev:>11.4f}  {cumvar[i]:>11.4f}  {bar}")

print(f"\n  Components for 50%: {n50}")
print(f"  Components for 80%: {n80}")


# =============================================================================
# STEP 3: LOADINGS
# =============================================================================

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(numerical_cols))],
    index=numerical_cols,
)

for pc in ['PC1', 'PC2', 'PC3']:
    ev = pca.explained_variance_ratio_[int(pc[-1]) - 1]
    print(f"\n  --- {pc} ({ev:.1%}) ---")
    for feat, val in loadings[pc].abs().sort_values(ascending=False).items():
        actual = loadings.loc[feat, pc]
        print(f"    {feat:30s}  {actual:+.3f}")

loadings.to_csv('pca_results/pca_mcm_num_loadings.csv')
print("\n  Saved: pca_mcm_num_loadings.csv")

# Add PC coordinates to dataframe for plotting
for i in range(min(5, len(numerical_cols))):
    df[f'PC{i+1}'] = X_pca[:, i]


# =============================================================================
# STEP 4: MAIN SCATTER PLOTS (2x3) — colored by categorical variables
# =============================================================================

ev1 = pca.explained_variance_ratio_[0]
ev2 = pca.explained_variance_ratio_[1]

fig, axes = plt.subplots(2, 3, figsize=(22, 14))

# --- (A) POS ---
ax = axes[0, 0]
# Macro-group POS for cleaner visualization
pos_macro = {
    'noun': 'Noun', 'proper_noun': 'Noun',
    'verb_lexical': 'Verb', 'verb_ser': 'Verb', 'verb_haver': 'Verb',
    'verb_modal': 'Verb', 'verb_auxiliary': 'Verb', 'verb_do': 'Verb',
    'adjective': 'AdjAdv', 'adverb': 'AdjAdv', 'numeral': 'AdjAdv', 'ordinal': 'AdjAdv',
    'determiner': 'FuncWord', 'preposition': 'FuncWord', 'pronoun': 'FuncWord',
    'post_preposition': 'FuncWord', 'pre_determiner': 'FuncWord',
    'conjunction_coordinating': 'FuncWord', 'conjunction_subordinating': 'FuncWord',
    'conjunction_adverbial': 'FuncWord', 'pre_adverb': 'FuncWord',
    'interjection': 'Other', 'sentence_marker': 'Other',
}
df['pos_macro'] = df['pos_label'].map(pos_macro).fillna('Other')

colors_pos = {'Noun': '#e41a1c', 'Verb': '#377eb8', 'AdjAdv': '#4daf4a',
              'FuncWord': '#ff7f00', 'Other': '#999999'}
for cat, color in colors_pos.items():
    mask = df['pos_macro'] == cat
    if mask.sum() > 0:
        ax.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
                   c=color, label=cat, alpha=0.3, s=5, edgecolors='none')
ax.set_xlabel(f'PC1 ({ev1:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({ev2:.1%})', fontsize=11)
ax.set_title('Colored by Part of Speech', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, markerscale=5)
ax.grid(True, alpha=0.3)

# --- (B) Language ---
ax = axes[0, 1]
lang_colors = plt.cm.get_cmap('tab10', df['lang'].nunique())
for i, lang in enumerate(sorted(df['lang'].unique())):
    mask = df['lang'] == lang
    ax.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
               c=[lang_colors(i)], label=lang, alpha=0.3, s=5, edgecolors='none')
ax.set_xlabel(f'PC1 ({ev1:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({ev2:.1%})', fontsize=11)
ax.set_title('Colored by Language Pair', fontsize=13, fontweight='bold')
ax.legend(fontsize=7, markerscale=5, loc='best')
ax.grid(True, alpha=0.3)

# --- (C) MCM Predicted P ---
ax = axes[0, 2]
sc = ax.scatter(df['PC1'], df['PC2'], c=df['mcm_predicted_p'],
                cmap='RdYlGn', alpha=0.3, s=5, edgecolors='none', vmin=0, vmax=1)
plt.colorbar(sc, ax=ax, label='MCM P(recall)')
ax.set_xlabel(f'PC1 ({ev1:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({ev2:.1%})', fontsize=11)
ax.set_title('Colored by MCM Predicted Recall', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# --- (D) Historical Accuracy ---
ax = axes[1, 0]
sc2 = ax.scatter(df['PC1'], df['PC2'], c=df['historical_accuracy'],
                 cmap='RdYlGn', alpha=0.3, s=5, edgecolors='none', vmin=0, vmax=1)
plt.colorbar(sc2, ax=ax, label='Historical Accuracy')
ax.set_xlabel(f'PC1 ({ev1:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({ev2:.1%})', fontsize=11)
ax.set_title('Colored by Historical Accuracy', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# --- (E) Tense ---
ax = axes[1, 1]
# Macro-group tense
tense_macro = {
    'present_indicative': 'Present', 'present': 'Present',
    'past_imperfect_indicative': 'Past', 'past_simple': 'Past',
    'past': 'Past', 'past_participle': 'Past', 'preterite': 'Past',
    'future_indicative': 'FutCond', 'conditional': 'FutCond',
    'present_subjunctive': 'SubjImp', 'imperative': 'SubjImp',
    'infinitive': 'NonFinite', 'gerund': 'NonFinite', 'present_participle': 'NonFinite',
    'unknown': 'None',
}
df['tense_macro'] = df['tense'].map(tense_macro).fillna('Other')

colors_tense = {'Present': '#e41a1c', 'Past': '#377eb8', 'FutCond': '#4daf4a',
                'SubjImp': '#ff7f00', 'NonFinite': '#984ea3', 'None': '#dddddd', 'Other': '#999999'}
# Plot 'None' first as background
for tense in ['None', 'Present', 'Past', 'FutCond', 'SubjImp', 'NonFinite', 'Other']:
    mask = df['tense_macro'] == tense
    if mask.sum() > 0:
        alpha = 0.1 if tense == 'None' else 0.4
        size = 3 if tense == 'None' else 8
        ax.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
                   c=colors_tense[tense], label=tense, alpha=alpha, s=size, edgecolors='none')
ax.set_xlabel(f'PC1 ({ev1:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({ev2:.1%})', fontsize=11)
ax.set_title('Colored by Tense', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, markerscale=4)
ax.grid(True, alpha=0.3)

# --- (F) Loadings Biplot ---
ax = axes[1, 2]
for feat in numerical_cols:
    x, y = loadings.loc[feat, 'PC1'], loadings.loc[feat, 'PC2']
    ax.arrow(0, 0, x * 2.5, y * 2.5, head_width=0.04, head_length=0.02,
             fc='steelblue', ec='steelblue', alpha=0.8, linewidth=2)
    # Shorten labels
    label = feat.replace('_', '\n') if len(feat) > 15 else feat
    ax.text(x * 2.8, y * 2.8, label, fontsize=7, ha='center', va='center', fontweight='bold')
ax.set_xlabel('PC1 loadings', fontsize=11)
ax.set_ylabel('PC2 loadings', fontsize=11)
ax.set_title('Loadings Biplot', fontsize=13, fontweight='bold')
ax.axhline(y=0, color='grey', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='grey', linestyle='--', alpha=0.3)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.grid(True, alpha=0.2)

plt.suptitle(f'PCA on MCM Numerical Features Only\n'
             f'({len(df):,} samples × {len(numerical_cols)} features)',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('pca_results/pca_mcm_num_main.png', dpi=150, bbox_inches='tight')
print("  Saved: pca_mcm_num_main.png")


# ---- Scree Plot ----
fig2, ax2 = plt.subplots(figsize=(10, 5))
x_pos = np.arange(1, len(numerical_cols) + 1)
ax2.bar(x_pos, pca.explained_variance_ratio_, color='steelblue', alpha=0.7, label='Individual')
ax2.plot(x_pos, cumvar, 'ro-', label='Cumulative')
ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='50%')
ax2.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='80%')
ax2.set_xlabel('Principal Component', fontsize=12)
ax2.set_ylabel('Explained Variance Ratio', fontsize=12)
ax2.set_title(f'Scree Plot — MCM Numerical Features\n'
              f'(50% at PC{n50}, 80% at PC{n80})', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'PC{i}' for i in x_pos])
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('pca_results/pca_mcm_num_scree.png', dpi=150, bbox_inches='tight')
print("  Saved: pca_mcm_num_scree.png")


# ---- Loadings Heatmap ----
fig3, ax3 = plt.subplots(figsize=(10, 6))
im = ax3.imshow(loadings.values, cmap='RdBu_r', aspect='auto', vmin=-0.7, vmax=0.7)
ax3.set_xticks(range(len(numerical_cols)))
ax3.set_xticklabels([f'PC{i+1}' for i in range(len(numerical_cols))], fontsize=10)
ax3.set_yticks(range(len(numerical_cols)))
ax3.set_yticklabels(numerical_cols, fontsize=10)
plt.colorbar(im, ax=ax3, label='Loading', shrink=0.8)
ax3.set_title('Feature Loadings Heatmap', fontsize=14, fontweight='bold')
for i in range(len(numerical_cols)):
    for j in range(len(numerical_cols)):
        val = loadings.values[i, j]
        if abs(val) > 0.15:
            color = 'white' if abs(val) > 0.35 else 'black'
            ax3.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
plt.tight_layout()
plt.savefig('pca_results/pca_mcm_num_loadings.png', dpi=150, bbox_inches='tight')
print("  Saved: pca_mcm_num_loadings.png")

print("\nDONE!")
