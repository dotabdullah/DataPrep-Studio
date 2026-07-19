# 🧪 DataPrep Studio

A Streamlit desktop web app for cleaning datasets and training ML models before
serious work in scikit-learn — upload, analyze, clean, visualize, train, export.

## Features (current, v4)

| Page | What it does |
|---|---|
| 📂 Upload & Preview | Load CSV/Excel, preview rows, see quick stats |
| 🔍 Analyze Dataset | Dtypes, missing values, duplicates, per-column detail |
| 🧹 Clean Data | Missing value handling with an optional auto-suggested strategy per column (based on missing %, dtype, and skewness); duplicate removal; drop/rename columns; type conversion; outlier detection (IQR/Z-score); encoding (Label/One-Hot — original category labels are remembered for later); **Value Labels** (attach friendly names like "Male"/"Female" to a column that's already numerically coded, without transforming the data); scaling (Standard/MinMax); text cleanup; dimensionality reduction (PCA); undo; and a downloadable **cleaning recipe** that replays every step — reusing the same fitted encoders/scalers/labels — on a new file in one click |
| 📊 Statistics & Visualization | `describe()`, histograms, boxplots, correlation heatmap, scatter plots, categorical breakdowns, and a one-click downloadable **HTML profiling report** combining all of the above |
| 🤖 Model Training | Discover which features matter most before training (optional, model-aware suggestion); Linear/Logistic Regression, Decision Trees, Random Forests, SVM, Naive Bayes, KNN, KMeans clustering; optional model suggestion via cross-validation; optional **hyperparameter tuning (Grid Search)**; optional **cross-validation scoring**; classification report + confusion matrix; predict on new input values using the original category names for any encoded *or* value-labeled column; **batch prediction from an uploaded file**; feature importance; elbow plots; export as Pickle or Joblib |
| 💾 Export Clean Data | Download cleaned data as CSV or Excel, view the log of cleaning steps applied |

Every chart across the app has its own **⬇️ download button** (PNG).

## Version history

### v1 — Foundation
- Upload CSV/Excel, preview data
- Analyze dataset: dtypes, missing values, duplicates, per-column detail
- Clean data: missing values, duplicates, drop/rename columns, type conversion, outlier handling, Label/One-Hot encoding, scaling, text cleanup, undo
- Statistics & visualization: summary stats, histograms, boxplots, correlation heatmap, scatter plots
- Model training: Linear/Logistic Regression, Decision Trees, Random Forests, KMeans clustering
- Export cleaned data as CSV/Excel

### v2 — More models, smarter training
- Added SVM, Naive Bayes, and KNN to the model lineup
- Optional model suggestion: quick cross-validation benchmark across all models for a task, ranked in a table/chart
- Classification report (precision/recall/F1/support) alongside confusion matrix
- Predict-on-new-data form for single inputs
- Export trained models as Pickle **or** Joblib
- Download button on every chart in the app
- Optional feature importance suggestion before training (which columns actually matter for the target)

### v3 — Consistency between cleaning and prediction
- **Encoding memory**: Label/One-Hot encoded columns now show their original category names (e.g. "Karachi") in the prediction form instead of raw numbers
- **Missing-value strategy auto-suggestion**: analyzes each column's missing %, dtype, and skewness, and recommends drop/mean/median/mode — accept all at once or handle manually

### v4 — Power tools
- **Hyperparameter tuning**: optional Grid Search per model, one click to apply the best-found settings
- **Cross-validation scoring**: optional, more robust accuracy estimate alongside the single train/test split
- **PCA / dimensionality reduction**: new Clean Data tab, with a scree plot to help choose how many components to keep
- **Batch prediction**: upload a whole file of new rows and get predictions for all of them at once, correctly translating any encoded or value-labeled columns
- **Cleaning recipes**: every cleaning action is recorded as a replayable step; download a `.recipe` file and replay it on a new file later, reusing the exact same fitted encoders/scalers/PCA/labels — not just refitting fresh ones
- **HTML profiling report**: one-click, one-file report combining column overview, missing values, stats, correlation, distributions, and categorical breakdowns
- **Value Labels**: for columns that are *already* numeric but represent categories (e.g. `Sex` coded 0/1) — attach friendly display names without touching the underlying data, so the prediction form and batch prediction show "Male"/"Female" instead of raw codes

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Project structure

```
app.py                      # entry point + sidebar navigation
modules/
  state.py                  # session-state helpers (working df, history/undo, recipe sync)
  data_upload.py            # Upload & Preview page
  analysis.py                # Analyze Dataset page
  cleaning.py                # Clean Data page (missing values, encoding, value labels, PCA, recipe UI)
  statistics_viz.py          # Statistics & Visualization page (incl. profiling report)
  modeling.py                 # Model Training page (incl. tuning, CV, batch prediction)
  export.py                  # Export Clean Data page
  recipe.py                   # cleaning-recipe recording/replay logic
  report.py                   # HTML profiling report generator
  plot_utils.py               # shared chart-download helper
requirements.txt
```

## Notes

- The **original uploaded data** is kept untouched in memory (`raw_df`) — all
  cleaning happens on a working copy (`df`), and you can undo any cleaning
  step from the Clean Data page.
- Model Training expects fully numeric features — encode categorical columns
  and handle missing values on the Clean Data page first.
- Trained models can be downloaded as `.pkl` or `.joblib` files for reuse elsewhere.
- Cleaning recipes are `.recipe` files (joblib-serialized) that bundle both the
  step sequence and any fitted encoders/scalers/PCA/value-labels — replaying one
  on a new file reuses those exact fitted objects rather than refitting from
  scratch, which matters if you've already trained a model on the original encoding.
- **Value Labels** vs. **Label Encoding**: use Encoding when your column holds
  text that needs converting to numbers; use Value Labels when your column is
  already numeric but you want a friendly name attached for display purposes only.
