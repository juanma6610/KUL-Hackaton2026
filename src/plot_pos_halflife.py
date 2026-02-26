"""
plot_pos_halflife.py
---------------------
1. Loads the full duo_data.csv
2. Estimates per-row half-life from the HLR formula:  h = -delta / log2(p_recall)
3. Builds per-user features and clusters users into 4 groups (K-Means)
4. Produces a violin plot:  x = POS label,  y = log10(half-life),  hue = cluster
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'duo_data.csv')
OUT_PATH  = os.path.join(BASE_DIR, 'results',  'pos_halflife_by_cluster.png')
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

RANDOM_STATE = 42
N_CLUSTERS   = 4

# ── 1. Load only the columns we need ─────────────────────────────────────────
print("Loading data …")
COLS = ['user_id', 'lexeme_id', 'pos_label',
        'p_recall', 'delta',
        'history_seen', 'history_correct', 'session_seen', 'session_correct']

df = pd.read_csv(DATA_PATH, usecols=COLS)
print(f"  rows: {len(df):,}  |  users: {df['user_id'].nunique():,}")

# ── 2. Estimate half-life per row (HLR: p = 2^{-Δ/h} ⟹ h = -Δ / log2(p)) ──
valid = (df['p_recall'] > 0) & (df['p_recall'] < 1) & (df['delta'] > 0)
df = df[valid].copy()
df['half_life'] = -df['delta'] / np.log2(df['p_recall'])
# cap extreme values (top 1 %)
cap = df['half_life'].quantile(0.99)
df = df[df['half_life'] <= cap]
df['log_h'] = np.log10(df['half_life'].clip(lower=1e-3))

print(f"  rows after filtering: {len(df):,}")

# ── 3. Per-user features for clustering ───────────────────────────────────────
print("Building per-user features …")
def cv(x):
    return x.std() / (x.mean() + 1e-9)

user_stats = df.groupby('user_id').agg(
    mean_log_h      = ('log_h',       'mean'),
    mean_accuracy   = ('p_recall',    'mean'),
    mean_delta      = ('delta',       'mean'),
    n_reviews       = ('delta',       'count'),
    cv_delta        = ('delta',       cv),
).reset_index()

user_stats['log_mean_delta'] = np.log10(user_stats['mean_delta'].clip(lower=1))
user_stats['log_n_reviews']  = np.log10(user_stats['n_reviews'].clip(lower=1))
user_stats['log_cv_delta']   = np.log10(user_stats['cv_delta'].clip(lower=1e-4))

CLUSTER_FEATS = ['mean_log_h', 'mean_accuracy',
                 'log_mean_delta', 'log_n_reviews', 'log_cv_delta']

user_stats = user_stats.dropna(subset=CLUSTER_FEATS).reset_index(drop=True)
X = user_stats[CLUSTER_FEATS].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. K-Means (MiniBatch for speed) ─────────────────────────────────────────
print("Clustering …")
km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE,
                     n_init=20, batch_size=10_000)
user_stats['cluster'] = km.fit_predict(X_scaled)

# ── 5. Auto-label clusters by half-life rank ──────────────────────────────────
hl_rank = user_stats.groupby('cluster')['mean_log_h'].mean().rank()
acc_rank = user_stats.groupby('cluster')['mean_accuracy'].mean().rank()

# Heuristic: high HL + high acc → Natural Learners
#            low HL + low  acc → Struggling Learners
#            low HL + high acc → Fast Forgetters (practice a lot, still forget)
#            high HL + low acc → Slow but Steady

def label_cluster(c):
    h = hl_rank[c]
    a = acc_rank[c]
    if h >= 3 and a >= 3:
        return 'Natural Learners'
    elif h <= 2 and a <= 2:
        return 'Struggling Learners'
    elif h >= 3 and a <= 2:
        return 'Slow but Steady'
    else:
        return 'Disengaged Fast Forgetters'

cluster_names = {c: label_cluster(c) for c in range(N_CLUSTERS)}
user_stats['cluster_name'] = user_stats['cluster'].map(cluster_names)

print("Cluster sizes:")
print(user_stats['cluster_name'].value_counts())

# ── 6. Join cluster labels onto full data ─────────────────────────────────────
print("Joining cluster labels …")
df = df.merge(user_stats[['user_id', 'cluster_name']], on='user_id', how='left')

# ── 7. Aggregate: mean log_h per (lexeme_id, pos_label, cluster_name) ─────────
print("Aggregating per lexeme …")
lex = (df.groupby(['lexeme_id', 'pos_label', 'cluster_name'])['log_h']
         .mean()
         .reset_index()
         .rename(columns={'log_h': 'mean_log_h'}))

# Keep only POS labels with reasonable coverage (≥ 50 lexemes across all clusters)
pos_counts = lex.groupby('pos_label')['lexeme_id'].nunique()
pos_keep   = pos_counts[pos_counts >= 50].index
lex        = lex[lex['pos_label'].isin(pos_keep)]

# Order POS by median half-life
pos_order = (lex.groupby('pos_label')['mean_log_h']
               .median()
               .sort_values()
               .index.tolist())

print(f"POS labels plotted: {pos_order}")

# ── 8. Plot ───────────────────────────────────────────────────────────────────
print("Plotting …")
PALETTE = {
    'Natural Learners':            '#2ca02c',   # green
    'Struggling Learners':         '#d62728',   # red
    'Slow but Steady':             '#1f77b4',   # blue
    'Disengaged Fast Forgetters':  '#ff7f0e',   # orange
}

fig, ax = plt.subplots(figsize=(14, 7))

sns.violinplot(
    data=lex,
    x='pos_label', y='mean_log_h',
    hue='cluster_name',
    order=pos_order,
    palette=PALETTE,
    inner='quartile',
    cut=0,
    ax=ax,
)

ax.set_xlabel('Part-of-Speech Label', fontsize=13)
ax.set_ylabel('Half-Life  (log₁₀ days)', fontsize=13)
ax.set_title('Lexeme Half-Life by POS Label and Learner Cluster', fontsize=15, fontweight='bold')
ax.tick_params(axis='x', rotation=30)
ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda y, _: f'{10**y:.1f}d')
)
ax.legend(title='Cluster', fontsize=10, title_fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"Saved → {OUT_PATH}")
plt.show()
