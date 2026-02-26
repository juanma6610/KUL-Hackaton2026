"""
==============================================================================
Learner Cluster Visualizations — Duolingo Theme
==============================================================================

Run locally after running ideas.py (needs features_clustered DataFrame).

Usage:
    # After running clustering in ideas.py, add this at the bottom:
    exec(open('cluster_visualizations.py').read())
    
    # OR: save features_clustered to CSV and load here:
    # features_clustered = pd.read_csv('features_clustered.csv')

Produces:
    cluster_main.png     — 4-panel: PCA scatter, radar, pie, boxplots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DUOLINGO COLOR PALETTE
# =============================================================================

# Duolingo's brand green and complementary colors
DUO_GREEN = '#58CC02'       # Primary Duolingo green
DUO_GREEN_DARK = '#2B8000'  # Darker green
DUO_BG = '#FAFAFA'          # Light background

# Cluster colors — themed but distinct
CLUSTER_COLORS = {
    'Natural Learners':           '#58CC02',  # Duolingo green — the best learners
    'Solid Performers':           '#1CB0F6',  # Duolingo blue
    'Struggling Learners':        '#FF4B4B',  # Duolingo red
    'Disengaged Fast Forgetters': '#FF9600',  # Duolingo orange
}

CLUSTER_ORDER = ['Natural Learners', 'Solid Performers',
                 'Struggling Learners', 'Disengaged Fast Forgetters']

CLUSTER_ICONS = {
    'Natural Learners':           '★',
    'Solid Performers':           '●',
    'Struggling Learners':        '▼',
    'Disengaged Fast Forgetters': '◆',
}

CLUSTER_DESCRIPTIONS = {
    'Natural Learners':           'High retention, high accuracy,\nefficient spacing',
    'Solid Performers':           'Good retention, consistent\npractice patterns',
    'Struggling Learners':        'Low retention despite\nhigh practice volume',
    'Disengaged Fast Forgetters': 'Fast forgetting with\nirregular, bursty practice',
}


# =============================================================================
# LOAD DATA (adapt path as needed)
# =============================================================================

# Option 1: If features_clustered is already in memory (from ideas.py), skip this.
# Option 2: Load from CSV:
# features_clustered = pd.read_csv('features_clustered.csv')

# Check that cluster_name exists
if 'cluster_name' not in features_clustered.columns:
    cluster_names = {
        0.0: 'Disengaged Fast Forgetters',
        1.0: 'Struggling Learners',
        2.0: 'Solid Performers',
        3.0: 'Natural Learners'
    }
    features_clustered['cluster_name'] = features_clustered['cluster'].map(cluster_names)

df = features_clustered.dropna(subset=['cluster_name']).copy()
print(f"Loaded {len(df):,} learners across {df['cluster_name'].nunique()} clusters")
print(df['cluster_name'].value_counts())


# =============================================================================
# FIGURE: MAIN 4-PANEL VISUALIZATION
# =============================================================================

fig = plt.figure(figsize=(22, 18), facecolor=DUO_BG)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# ---- (A) PCA SCATTER WITH CLUSTER SEPARATION ----
ax = fig.add_subplot(gs[0, 0], facecolor=DUO_BG)

cluster_feats = ['log_h', 'mean_accuracy', 'log_mean_delta',
                 'log_n_reviews', 'log_cv_delta']
X = df[cluster_feats].fillna(0).values
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

for cluster in CLUSTER_ORDER:
    mask = df['cluster_name'].values == cluster
    if mask.sum() > 0:
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=CLUSTER_COLORS[cluster], label=cluster,
                   alpha=0.5, s=25, edgecolors='white', linewidth=0.3)

# Add cluster centroids
for cluster in CLUSTER_ORDER:
    mask = df['cluster_name'].values == cluster
    if mask.sum() > 0:
        cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
        ax.scatter(cx, cy, c=CLUSTER_COLORS[cluster],
                   s=200, edgecolors='black', linewidth=2, zorder=10, marker='D')
        ax.annotate(CLUSTER_ICONS[cluster], (cx, cy),
                    fontsize=14, ha='center', va='center',
                    fontweight='bold', color='white', zorder=11)

ev1 = pca.explained_variance_ratio_[0]
ev2 = pca.explained_variance_ratio_[1]
ax.set_xlabel(f'PC1 ({ev1:.1%} variance)', fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({ev2:.1%} variance)', fontsize=12, fontweight='bold')
ax.set_title('Learner Clusters in PCA Space', fontsize=15, fontweight='bold',
             color=DUO_GREEN_DARK, pad=15)
ax.legend(fontsize=10, framealpha=0.9, edgecolor=DUO_GREEN,
          loc='best', markerscale=2)
ax.grid(True, alpha=0.15)


# ---- (B) RADAR / SPIDER CHART ----
ax_radar = fig.add_subplot(gs[0, 1], polar=True, facecolor=DUO_BG)

radar_feats = ['log_h', 'mean_accuracy', 'log_mean_delta',
               'log_n_reviews', 'log_cv_delta']
radar_labels = ['Half-Life\n(retention)', 'Accuracy', 'Review\nGap',
                'Practice\nVolume', 'Spacing\nIrregularity']

# Normalize each feature to 0-1 across clusters
cluster_means = df.groupby('cluster_name')[radar_feats].mean()
cluster_norm = (cluster_means - cluster_means.min()) / \
               (cluster_means.max() - cluster_means.min())

angles = np.linspace(0, 2 * np.pi, len(radar_feats), endpoint=False).tolist()
angles += angles[:1]

for cluster in CLUSTER_ORDER:
    if cluster in cluster_norm.index:
        values = cluster_norm.loc[cluster].tolist()
        values += values[:1]
        ax_radar.plot(angles, values, 'o-', linewidth=2.5,
                      color=CLUSTER_COLORS[cluster], label=cluster,
                      markersize=6)
        ax_radar.fill(angles, values, alpha=0.1, color=CLUSTER_COLORS[cluster])

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(radar_labels, fontsize=10, fontweight='bold')
ax_radar.set_ylim(0, 1.1)
ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_radar.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8, color='gray')
ax_radar.set_title('Cluster Feature Profiles', fontsize=15, fontweight='bold',
                   color=DUO_GREEN_DARK, pad=25)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9,
                framealpha=0.9, edgecolor=DUO_GREEN)


# ---- (C) PIE CHART WITH PERCENTAGES ----
ax_pie = fig.add_subplot(gs[1, 0], facecolor=DUO_BG)

counts = df['cluster_name'].value_counts()
# Reorder to match CLUSTER_ORDER
sizes = [counts.get(c, 0) for c in CLUSTER_ORDER]
colors = [CLUSTER_COLORS[c] for c in CLUSTER_ORDER]
total = sum(sizes)
labels = [f'{c}\n({s:,} learners · {100*s/total:.1f}%)'
          for c, s in zip(CLUSTER_ORDER, sizes)]

explode = [0.03] * len(CLUSTER_ORDER)

wedges, texts, autotexts = ax_pie.pie(
    sizes, labels=None, colors=colors, autopct='%1.1f%%',
    startangle=90, explode=explode,
    pctdistance=0.75,
    wedgeprops=dict(edgecolor='white', linewidth=2.5),
    textprops=dict(fontsize=12, fontweight='bold'),
)

# Style the percentage text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(13)

# Add legend instead of labels for cleaner look
ax_pie.legend(wedges, labels, loc='center left', bbox_to_anchor=(-0.3, 0.5),
              fontsize=10, framealpha=0.9, edgecolor=DUO_GREEN)
ax_pie.set_title('Learner Distribution', fontsize=15, fontweight='bold',
                 color=DUO_GREEN_DARK, pad=15)


# ---- (D) BOXPLOTS: KEY METRICS BY CLUSTER ----
ax_box = fig.add_subplot(gs[1, 1], facecolor=DUO_BG)

# Show half-life distribution per cluster (the key metric)
box_data = [df[df['cluster_name'] == c]['log_h'].dropna().values
            for c in CLUSTER_ORDER]

bp = ax_box.boxplot(box_data,
                    labels=[c.replace(' ', '\n') for c in CLUSTER_ORDER],
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

for patch, cluster in zip(bp['boxes'], CLUSTER_ORDER):
    patch.set_facecolor(CLUSTER_COLORS[cluster])
    patch.set_alpha(0.7)
    patch.set_edgecolor(CLUSTER_COLORS[cluster])
    patch.set_linewidth(2)

# Add means as diamonds
means = [np.mean(d) for d in box_data]
ax_box.scatter(range(1, len(means) + 1), means,
               marker='D', color='black', s=60, zorder=5, label='Mean')

ax_box.set_ylabel('log₁₀(Half-Life, days)', fontsize=12, fontweight='bold')
ax_box.set_title('Memory Retention by Cluster', fontsize=15, fontweight='bold',
                 color=DUO_GREEN_DARK, pad=15)
ax_box.grid(True, alpha=0.15, axis='y')
ax_box.legend(fontsize=10)
ax_box.tick_params(axis='x', labelsize=10)


# ---- MAIN TITLE ----
fig.suptitle('Duolingo Learner Memory Profiles\n'
             'Four Distinct Patterns of Forgetting',
             fontsize=20, fontweight='bold', color=DUO_GREEN_DARK, y=0.98)

plt.savefig('cluster_main.png', dpi=150, bbox_inches='tight',
            facecolor=DUO_BG)
print("\nSaved: cluster_main.png")


# =============================================================================
# BONUS: STANDALONE PIE CHART (cleaner for slides)
# =============================================================================

fig2, ax2 = plt.subplots(figsize=(10, 8), facecolor=DUO_BG)

wedges2, texts2, autotexts2 = ax2.pie(
    sizes, labels=None, colors=colors, autopct='%1.1f%%',
    startangle=90, explode=[0.04] * len(CLUSTER_ORDER),
    pctdistance=0.8,
    wedgeprops=dict(edgecolor='white', linewidth=3, width=0.6),  # donut style
    textprops=dict(fontsize=14, fontweight='bold'),
)

for autotext in autotexts2:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(14)

# Center text
ax2.text(0, 0, f'{total:,}\nlearners', ha='center', va='center',
         fontsize=18, fontweight='bold', color=DUO_GREEN_DARK)

# Legend with descriptions
legend_labels = [f'{CLUSTER_ICONS[c]}  {c}\n     {CLUSTER_DESCRIPTIONS[c]}'
                 for c in CLUSTER_ORDER]
ax2.legend(wedges2, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5),
           fontsize=11, framealpha=0.95, edgecolor=DUO_GREEN,
           handlelength=2, handleheight=2.5)

ax2.set_title('Who Are Duolingo\'s Learners?',
              fontsize=20, fontweight='bold', color=DUO_GREEN_DARK, pad=20)

plt.savefig('cluster_pie.png', dpi=150, bbox_inches='tight', facecolor=DUO_BG)
print("Saved: cluster_pie.png")


# =============================================================================
# BONUS: STANDALONE RADAR (for slides)
# =============================================================================

fig3, ax3 = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True),
                          facecolor=DUO_BG)

for cluster in CLUSTER_ORDER:
    if cluster in cluster_norm.index:
        values = cluster_norm.loc[cluster].tolist()
        values += values[:1]
        ax3.plot(angles, values, 'o-', linewidth=3,
                 color=CLUSTER_COLORS[cluster], label=cluster,
                 markersize=8)
        ax3.fill(angles, values, alpha=0.12, color=CLUSTER_COLORS[cluster])

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(radar_labels, fontsize=12, fontweight='bold')
ax3.set_ylim(0, 1.15)
ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
ax3.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9, color='gray')
ax3.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11,
           framealpha=0.95, edgecolor=DUO_GREEN)
ax3.set_title('Learner Cluster Profiles',
              fontsize=18, fontweight='bold', color=DUO_GREEN_DARK, pad=25)

plt.savefig('cluster_radar.png', dpi=150, bbox_inches='tight', facecolor=DUO_BG)
print("Saved: cluster_radar.png")

print("\n✅ All cluster visualizations saved!")
