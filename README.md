# KUL-Hackaton2026 — Project Readme (Updated)

This repository collects code, notebooks and data for experiments with Duolingo-style spaced repetition models (Half-Life Regression, Multiscale Context Model, and an XGBoost baseline) and tooling to run, evaluate and visualize them.

Summary of new / updated features (detailed and specific)
- Multiscale Context Model (MCM)
  - Implementation: [`hlr.model1.MultiscaleContextModel`](hlr/model1.py)
  - Key constructor params: `mu=0.01`, `nu=1.05`, `xi=0.9`, `N=100`, `eps_r=9.0`. The model instantiates exponentially spaced time scales tau and normalized weights gamma for N integrators.
  - Methods: `decay(t)`, `get_strengths()`, `predict(t)` and `study(t, recalled)` — see [`hlr/model1.py`](hlr/model1.py) for full implementation and inline docstrings.
  - Usage in pipeline: produced per-record MCM recall predictions via [`hlr.model1.generate_mcm_features`](hlr/model1.py), which is vectorized (NumPy arrays) and maintains per (user,lexeme) state for speed.

- XGBoost Phase‑2 Baseline (half-life regression target)
  - Data preparation: [`hlr.model1.get_xgboost_data`](hlr/model1.py)
    - Reads `data/SpacedRepetitionData.csv` ([data/SpacedRepetitionData.csv](data/SpacedRepetitionData.csv)).
    - Feature engineering includes:
      - Temporal features: `hour_of_day`, `day_of_week`, `time_lag_days`, `log_delta`
      - Language pair: `lang` (e.g., `ui_language->learning_language`)
      - MCM baseline: `mcm_predicted_p`
      - Accuracy and counts: `historical_accuracy`, `user_global_accuracy`, `right = sqrt(1 + history_correct)`, `wrong = sqrt(1 + (history_seen - history_correct))`
      - Morphology: `pos_tag` extracted from `lexeme_string`
    - Casts categorical columns for native XGBoost categorical handling (`lang`, `pos_tag`).
    - Performs fast 90/10 train/test split.
  - Target transform (train): half-life computed as
    - $h = -\dfrac{t}{\log_2 p}$ where $p$ is the observed recall and $t$ is `time_lag_days` (see code in [`hlr.model1.train_xgboost_baseline`](hlr/model1.py)).
    - Clipped to [`hlr.model1.MIN_HALF_LIFE`](hlr/model1.py) and [`hlr.model1.MAX_HALF_LIFE`](hlr/model1.py).
  - Model architecture and training:
    - Model: `xgb.XGBRegressor` with explicit params: `tree_method="hist"`, `enable_categorical=True`, `n_estimators=1000`, `learning_rate=0.01`, `max_depth=6`, `early_stopping_rounds=50`, `random_state=42` — see [`hlr.model1.train_xgboost_baseline`](hlr/model1.py).
    - Fit uses `eval_set=[(X_train, h_train), (X_test, h_test)]` and `verbose=50`.
  - Predictions:
    - Predicts half-life `h_pred`, clipped to `[MIN_HALF_LIFE, MAX_HALF_LIFE]`.
    - Converts back to probability: $p = 2^{-t/h}$ (code: `p_pred = 2.0 ** (-X_test['time_lag_days'] / h_pred)`).
    - Post-clipping of probabilities to `[0.0001, 0.9999]`.
  - Metrics and reporting:
    - MAE for half-life (days): `mae_h = mean_absolute_error(h_test, h_pred)`
    - Spearman correlation for half-life and p: `spearmanr(...)`
    - MAE for probability predictions and printed Phase 2 summary — see output example in [`hlr/xgboost.ipynb`](hlr/xgboost.ipynb) and the helper [`hlr.model1.train_xgboost_baseline`](hlr/model1.py).

- Visualization / Pitch-deck asset generation
  - SHAP feature importance and dependence plots are produced by [`hlr.model1.generate_pitch_deck_visuals`](hlr/model1.py) and in the notebook [`hlr/xgboost.ipynb`](hlr/xgboost.ipynb).
  - Example saved files: `slide1_feature_importance.png`, `slide2_circadian_rhythm.png` (hour_of_day dependence) — see the SHAP code block in [`hlr/xgboost.ipynb`](hlr/xgboost.ipynb).

- Streamlit demo app
  - GUI: [`app.py`](app.py)
    - Loads saved XGBoost model with `@st.cache_resource` via [`app.load_model`](app.py) which calls `xgb.XGBRegressor().load_model("xgboost_baseline.json")`.
    - Input panel includes categorical dropdowns for `lang`, sliders and numeric inputs for `time_lag_days`, `history_seen`, `history_correct`, `pos_tag`, etc.
    - Prepares input DataFrame matching the training features and performs dtype casting for categorical features (`lang`, `pos_tag`) before prediction.
    - Prediction flow: compute `h_pred`, clip, convert to `p_pred = 2^{-t/h_pred}`, display metrics and interactive forgetting curve with Plotly (`plotly.graph_objects`).
    - Enforces basic input logic in `enforce_math_logic()` to keep `history_correct <= history_seen`.
  - Model used by app: saved model files `xgboost_baseline.json` and `xgboost_baseline0.001.json` (produced by training scripts/notebooks) — check [`hlr/model1.py`](hlr/model1.py) and [`hlr/xgboost.ipynb`](hlr/xgboost.ipynb).



- Classic HLR / SpacedRepetitionModel scripts (trainable baselines)
  - Two implementations:
    - Legacy script: [`hlr/experiment.py`](hlr/experiment.py) — Python reimplementation of Duolingo HLR with command-line usage.
    - Root-level `experiment.py` (original from repo): [`experiment.SpacedRepetitionModel`](experiment.py).
  - CLI flags in both: `-b` (omit bias), `-l` (omit lexeme features), `-t` (omit half-life term), `-m <method>` (hlr|lr|leitner|pimsleur), `-x <max_lines>` for dev truncation. See parsing logic in [`experiment.py`](experiment.py) and [`hlr/model1.py` argparse block] (script entrypoint).
 

- Evaluation tooling (R)
  - R evaluation script: [`hlr/evaluation.r`](hlr/evaluation.r) exposing `sr_evaluate(preds_file)` which computes:
    - MAE significance test (Welch t-test),
    - AUC via `pROC::roc`,
    - Wilcoxon rank-sum test,
    - Spearman half-life correlation.
  - See included commented example outputs in [`hlr/evaluation.r`](hlr/evaluation.r).

- Notebooks (EDA and experiments)
  - Exploratory data analysis and baseline fitting: [`edaSRD.ipynb`](edaSRD.ipynb) and [`duolingo-exploratory-data-analysis.ipynb`](duolingo-exploratory-data-analysis.ipynb).
  - XGBoost training, diagnostics, SHAP visuals and saved assets: [`hlr/xgboost.ipynb`](hlr/xgboost.ipynb).

- Misc / infra
  - Dataset: [data/SpacedRepetitionData.csv](data/SpacedRepetitionData.csv)
  - Requirements: [requirements.txt](requirements.txt)
  - Project-level README for the HLR original repo: [hlr/README.md](hlr/README.md)
  - Results folder usage: training and evaluation writes are placed under `results/` by the CLI entrypoints (see `experiment.py` and [`hlr/experiment.py`](hlr/experiment.py)).

How to run (concise steps)
1. Prepare environment:
   - Install deps: `pip install -r requirements.txt` ([requirements.txt](requirements.txt)).
2. Generate MCM features and train XGBoost (two options):
   - Notebook: open [`hlr/xgboost.ipynb`](hlr/xgboost.ipynb) and run cells; or
   - Script: run `python hlr/model1.py data/SpacedRepetitionData.csv` (script entrypoint in [`hlr/model1.py`](hlr/model1.py)) to call `get_xgboost_data` and `train_xgboost_baseline`.
3. Run the demo app:
   - Ensure `xgboost_baseline.json` exists (produced by training).
   - Launch: `streamlit run app.py` ([app.py](app.py)).
4. Train HLR baselines:
   - `python experiment.py -m hlr data/SpacedRepetitionData.csv` ([experiment.py](experiment.py)) or use [`hlr/experiment.py`](hlr/experiment.py) variant.
5. Evaluate predictions:
   - Use `Rscript -e "source('hlr/evaluation.r'); sr_evaluate('results/<preds_file>')"` or run interactively in R using [`hlr/evaluation.r`](hlr/evaluation.r).

Important equations and clipping
- Probability decay used by model: $p = 2^{-t/h}$ (used to transform predicted half-life to recall probability).
- Half-life target transform: $h = -\dfrac{t}{\log_2 p}$ (clipped to min / max half-life constants in [`hlr/model1.py`](hlr/model1.py) and [`experiment.py`](experiment.py)).
- Constants: [`hlr.model1.MIN_HALF_LIFE`](hlr/model1.py), [`hlr.model1.MAX_HALF_LIFE`](hlr/model1.py) and their equivalents in `experiment.py` ([experiment.py](experiment.py), [`hlr/experiment.py`](hlr/experiment.py)).

Where to look for code
- Core MCM + XGBoost pipeline: [`hlr/model1.py`](hlr/model1.py) — primary entrypoint for feature generation, training and visualization.
- Notebook walkthrough & reproducible run: [`hlr/xgboost.ipynb`](hlr/xgboost.ipynb).
- Streamlit demo: [`app.py`](app.py).
- Legacy HLR scripts & CLI: [`experiment.py`](experiment.py) and [`hlr/experiment.py`](hlr/experiment.py).
- Evaluation script (R): [`hlr/evaluation.r`](hlr/evaluation.r).
- Data: [data/SpacedRepetitionData.csv](data/SpacedRepetitionData.csv).
- Dataset descriptions: [datasets/README_2020StapleSharedTaskData.txt](datasets/README_2020StapleSharedTaskData.txt).

---

## Dataset Pipeline (`src/`)

These two scripts process the raw Duolingo learning-trace dataset into a structured, morphologically-enriched CSV ready for modelling.

### Prerequisites

```bash
pip install pandas numpy
```

You also need the raw Duolingo dataset file **`learning_traces.13m.csv`** (~13 M rows).
Place it in the same directory from which you run the scripts, or adjust the `INPUT_CSV` constant in `build_duo_data.py`.

> `duo_data.csv` is listed in `.gitignore` because it is ~1.8 GB — regenerate it locally after cloning.

---

### `src/build_duo_data.py` — Full dataset builder

Reads `learning_traces.13m.csv` in memory-efficient chunks (default 500 000 rows), parses the `lexeme_string` column into grammatical features using vectorised pandas regex, and streams the result to `duo_data.csv`.

**Run:**

```bash
# run from the directory containing learning_traces.13m.csv
python src/build_duo_data.py
```

Progress is printed per chunk:

```
Reading  : learning_traces.13m.csv  (chunks of 500,000)
Writing  : duo_data.csv

  chunk    1 | rows      500,000 | chunk 4.2s | total  4s | ~118,000 rows/s
  ...
Done! 12,854,284 rows → 'duo_data.csv'  (112s total)
```

Tune memory usage by changing `CHUNK_SIZE` at the top of the file.

**Columns added to the output** (on top of all original columns):

| Column | Description |
|---|---|
| `surface_form` | The inflected word form (before `/`) |
| `lemma` | Dictionary / base form (between `/` and first `<`) |
| `pos_label` | Part of speech (`verb_lexical`, `noun`, `adjective`, …) |
| `tense` | Tense/mood (`present_indicative`, `past_participle`, `infinitive`, …) |
| `person` | Grammatical person (`1st_person`, `2nd_person`, `3rd_person`) |
| `number` | `singular` / `plural` / `singular_or_plural` |
| `gender` | `masculine` / `feminine` / `neuter` / `masculine_or_feminine` |
| `case` | `nominative` / `accusative` / `dative` / `genitive` / … |
| `definiteness` | `definite` / `indefinite` / `demonstrative` / `possessive` / … |
| `degree` | Adjective degree: `comparative` / `superlative` |
| `pronoun_type` | `reflexive` / `personal` / `object` / `subject` / … |
| `adj_declension` | German-style declension: `strong` / `weak` / `uninflected` |

---

### `src/lexeme_parser.py` — Row-level parser & diagnostics

A more detailed, row-by-row parser. Use it to inspect a small sample or import its functions into notebooks and model scripts.

**Run standalone** (prints a diagnostic summary on 3 000 rows):

```bash
# run from the directory containing learning_traces.13m.csv
python src/lexeme_parser.py
```

Output includes: POS distribution, per-POS subcategory breakdown, tense/mood counts for verbs, gender distribution for nouns and adjectives, and 15 sample parsed rows.

**Import into your own code:**

```python
from src.lexeme_parser import parse_lexeme, enrich_dataframe
import pandas as pd

# Parse a single lexeme string
result = parse_lexeme("lernt/lernen<vblex><pri><p3><sg>")
# → {'surface_form': 'lernt', 'lemma': 'lernen', 'pos_label': 'verb_lexical',
#    'tense': 'present_indicative', 'person': '3rd_person', 'number': 'singular', …}

# Enrich a DataFrame that contains a 'lexeme_string' column
df = pd.read_csv("learning_traces.13m.csv", nrows=5000)
df_enriched = enrich_dataframe(df)
print(df_enriched[["lexeme_string", "pos_label", "tense", "gender"]].head())
```

**Lexeme string format** (Apertium convention):

```
surface_form/lemma<POS><tense><person><number>…
```

Example: `estudia/estudiar<vblex><pri><p3><sg>`
→ surface=`estudia`, lemma=`estudiar`, POS=`verb_lexical`, tense=`present_indicative`, person=`3rd_person`, number=`singular`

---

### Which script to use

| Goal | Script |
|---|---|
| Build the full `duo_data.csv` from scratch | `src/build_duo_data.py` |
| Inspect / debug lexeme parsing on a small sample | `src/lexeme_parser.py` (standalone) |
| Use the parser inside a notebook or model script | `from src.lexeme_parser import parse_lexeme, enrich_dataframe` |
