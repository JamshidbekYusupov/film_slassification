# 🎬 Animation Technique Classification — Machine Learning Project

> Classifying animated films by production technique using ensemble machine learning methods, explainable AI, and rich feature engineering on Wikipedia-scraped data.

---

## 📁 Project Structure

```
├── Data/
│   ├── basic_eng/
│   ├── Engineered_data/
│   └── Raw_Data/
├── Logging/
│   ├── bagging.log
│   ├── data_loader.log
│   ├── engineering.log
│   ├── feature_analysis.log
│   ├── overfitting.log
│   ├── stacking.log
│   ├── tuning.log
│   └── voting.log
├── Metrics/
├── Models/
├── myvenv/
├── Notebook/
│   ├── analysis.ipynb
│   ├── dataloader.ipynb
│   ├── engineering.ipynb
│   └── shap.ipynb
├── Scraping/
├── source/
├── src/
├── .gitignore
├── README.md
└── Requirements.txt
```

---

## 🌐 Data Source

Data was scraped from Wikipedia's lists of animated films:
🔗 [https://en.wikipedia.org/wiki/Lists_of_animated_films](https://en.wikipedia.org/wiki/Lists_of_animated_films)

The scraper collects film metadata including titles, release dates, production studios, techniques, and more across decades of animation history.

---

## ⚙️ Feature Engineering

Raw data was processed and enriched using **Pandas** and **NumPy**:

- **Datetime conversion** — release date columns converted to proper datetime types
- **Temporal extraction** — year, month, and other time-based features derived from date fields
- **Column pruning** — unnecessary and redundant features dropped to reduce noise
- **Type normalization** — categorical and numerical fields standardised for model compatibility

---

## 🔍 Feature Analysis

A thorough analysis was conducted to understand the structure and quality of features:

| Analysis Type | Description |
|---|---|
| **Pearson Correlation** | Measured linear relationships between features and the target |
| **Input vs Input Correlation** | Identified multicollinearity between independent variables |
| **Skewness Analysis** | Detected distributional asymmetry in numerical features |
| **Redundant Feature Detection** | Identified and removed features with low informational value |

Feature selection was subsequently performed using **tree-based methods**, leveraging feature importance scores from decision tree models.

---

## 🤖 Models & Ensemble Methods

### 🌲 Bagging — Random Forest Classifier
Bootstrap aggregation with a Random Forest classifier to reduce variance and improve generalization.

### 🗳️ Voting Classifier
- **Hard Voting** — majority class label wins
- **Soft Voting** — aggregated predicted probabilities used for final decision

### 🏗️ Stacking Classifier
| Role | Model |
|---|---|
| Base Learner 1 | Logistic Regression |
| Base Learner 2 | Support Vector Machine (SVM) |
| Base Learner 3 | Decision Tree |
| Meta Model | Random Forest |

---

## 🎛️ Hyperparameter Tuning

Tuning was performed using **GridSearchCV** wrapped in a **scikit-learn Pipeline**:

- Systematic grid search over hyperparameter space
- Cross-validated evaluation at each parameter combination
- Best model persisted to the `Models/` directory for reuse

---

## 📊 Model Performance

Results from the tuned Random Forest classifier and Bagging methods with RF:

| **Metric** | **RF Tuning Score** | **Bagging Scores** | **Delta (Δ)**|
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.7551 | 0.7428 | +1.23% |
| **Precision** | 0.7335 | 0.728 | +0.55% |
| **Recall** | 0.7551 | 0.742 | +1.31% |
| **F1 Score** | 0.7375 | 0.712 | +2.55% |

---

## ⚠️ Bias-Variance Analysis

The project includes a dedicated investigation into model generalisation behaviour:

- **Overfitting scenario** — evaluated model performance when training loss >> validation loss
- **Underfitting scenario** — evaluated model performance when the model is too simple to learn the data
- Findings logged to `Logging/overfitting.log`

---

## 🔬 Explainable AI (XAI) — SHAP Analysis

Model predictions were interpreted using **SHAP (SHapley Additive exPlanations)**:

- **Waterfall Plot** — contribution of each feature to a single prediction
- **Summary Plot** — global feature importance with directional impact across all predictions

SHAP analysis is implemented in `Notebook/shap.ipynb`.

---

## 🛠️ Requirements

Install all dependencies with:

```bash
pip install -r Requirements.txt
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd animation-classification

# 2. Create and activate virtual environment
python -m venv myvenv
source myvenv/bin/activate  # Windows: myvenv\Scripts\activate

# 3. Install dependencies
pip install -r Requirements.txt

# 4. Run notebooks in order
#    dataloader.ipynb → engineering.ipynb → analysis.ipynb → shap.ipynb
```

---

## 📓 Notebooks Overview

| Notebook | Purpose |
|---|---|
| `dataloader.ipynb` | Scraping, loading, and initial data inspection |
| `engineering.ipynb` | Feature engineering and preprocessing pipeline |
| `analysis.ipynb` | Feature analysis, model training, and ensemble experiments |
| `shap.ipynb` | SHAP-based explainability and visualisation |