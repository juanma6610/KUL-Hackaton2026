# 🦉 Duolingo Memory Engine — KUL-Hackaton2026

This repository contains an advanced memory modeling pipeline for Duolingo-style spaced repetition. It implements a hybrid architecture combining the **Multiscale Context Model (MCM)** from cognitive science with **XGBoost** for high-precision recall prediction and linguistic feature integration.

---

## 📂 Project Structure

The project is organized into a modular package format for production-ready experimentation.

### 🚀 Entry Points
- **[`train.py`](train.py)**: The primary CLI entry point for the training pipeline.
- **[`app.py`](app.py)**: The interactive Streamlit demo app with a Duolingo-inspired theme and SHAP explainability.

### 📦 Core Package (`src/`)
- **[`mcm.py`](src/mcm.py)**: Implementation of the **Multiscale Context Model**. Maintains stateful memory simulations.
- **[`lexeme_parser.py`](src/lexeme_parser.py)**: Sophisticated NLP parser for Duolingo lexeme strings. Extracts 15+ morphological categories.
- **[`data_pipeline.py`](src/data_pipeline.py)**: Handles feature engineering and prepares the training/testing matrices.
- **[`model_trainer.py`](src/model_trainer.py)**: High-performance XGBoost training logic with **Optuna** support.
- **[`visuals.py`](src/visuals.py)**: Generates SHAP importance and circadian rhythm plots.
- **[`preprocess.py`](src/preprocess.py)**: Utility to build the "Gold Dataset" (Parquet) from raw CSVs for lightning-fast training.
- **[`build_duo_data.py`](src/build_duo_data.py)**: Vectorized batch enricher for the full 13M row dataset.
- **[`config.py`](src/config.py)**: Centralized configuration and file paths.

### 🏛️ Legacy & Research
- **[`hlr/`](hlr/)**: Contains legacy Half-Life Regression scripts and evaluation utilities.
- **[`eda/`](eda/)**: Notebooks for Exploratory Data Analysis.

---

## 📊 Dataset Pipeline

To fully utilize the pipeline, you need the raw Duolingo dataset (e.g., `learning_traces.13m.csv`). By default, the scripts expect this file in the `data/` directory.

> [!NOTE]
> `duo_data.csv` and `processed_features.parquet` are git-ignored due to their size (~1.8 GB). You should regenerate them locally using the steps below.

### 1. Build the Enriched Dataset
`src/build_duo_data.py` reads the raw traces in memory-efficient chunks, parses the `lexeme_string` using vectorized regex, and produces an enriched CSV.

```bash
# Run the batch builder
python src/build_duo_data.py
```

**Linguistic Features Extracted:**

| Column | Description |
| :--- | :--- |
| `surface_form` | The actual word seen by the student (e.g., *lernt*). |
| `lemma` | The dictionary base form (e.g., *lernen*). |
| `pos_label` | Part of Speech (Noun, Verb, Adjective, etc.). |
| `tense` | Grammatical tense (Present, Past, Future, etc.). |
| `person` | 1st, 2nd, or 3rd person. |
| `number` | Singular or Plural. |
| `gender` | Masculine, Feminine, or Neuter. |
| `case` | Nominative, Accusative, Dative, etc. |
| `definiteness` | Definite, Indefinite, etc. |
| `degree` | Adjective degree (Comparative, Superlative). |
| `pronoun_type` | Reflexive, Personal, Object, etc. |
| `adj_declension` | Strong, Weak, or Uninflected. |

### 2. Pre-process for Speed (Optional)
Use `preprocess.py` to convert the CSV into a Parquet "Gold Dataset". This significantly accelerates the training phase.

```bash
python src/preprocess.py
```

## 🛠️ How to Run

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Training the Model
**Standard Training:**
```bash
python train.py data/SpacedRepetitionData.csv
```

**Optuna Hyperparameter Search:**
```bash
python train.py data/SpacedRepetitionData.csv --optuna --trials 100
```

### 3. Launch the App
```bash
streamlit run app.py
```

---

## 🧠 Model Methodology

The engine uses a two-stage hybrid approach to model human memory decay:

1.  **Cognitive Stage (MCM)**: Simulates the underlying strength of a student's memory based on their practice history across multiple time scales.
2.  **Machine Learning Stage (XGBoost)**: Refines the cognitive prediction by factoring in linguistic complexity, circadian rhythms, and global user performance.

### Features Included
-   **Cognitive**: MCM recall probability baseline.
-   **Linguistic Depth**: POS labels, tense, person, number, gender, case, definiteness, and degree.
-   **Temporal**: Hour of day and day of week (rhythm effects).
-   **History**: Historical accuracy, total success/failures (sqrt transformed).

### Model Validity & Explainability
-   **Chronological Validity**: GroupKFold splitting ensures the model is evaluated on unseen users, preventing data leakage.
-   **Metrics**: Spearman Correlation (on $h$) and MAE (Mean Absolute Error on recall probability $p$).
-   **Explainability**: Full SHAP integration allows us to visualize exactly why a student is likely to forget a specific word.

## 📊 Results & Artifacts
- **Models**: Saved in the `models/` directory.
- **Visuals**: SHAP analysis plots are saved automatically after training for use in presentations.

---

## 📜 Credits
Developed for **KU Leuven MAI Hackathon 2026**.
Based on the Duolingo Spaced Repetition Dataset.
