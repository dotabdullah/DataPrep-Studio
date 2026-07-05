# 🧪 DataPrep Studio

A Streamlit desktop web app for cleaning datasets and training ML models before
serious work in scikit-learn — upload, analyze, clean, visualize, train, export.

## Features

| Page | What it does |
|---|---|
| 📂 Upload & Preview | Load CSV/Excel, preview rows, see quick stats |
| 🔍 Analyze Dataset | Dtypes, missing values, duplicates, per-column detail |
| 🧹 Clean Data | Missing value handling, duplicate removal, drop/rename columns, type conversion, outlier detection (IQR/Z-score), encoding (Label/One-Hot), scaling (Standard/MinMax), text cleanup, **undo** |
| 📊 Statistics & Visualization | `describe()`, histograms, boxplots, correlation heatmap, scatter plots, categorical breakdowns |
| 🤖 Model Training | Linear/Logistic Regression, Decision Trees, Random Forests, **SVM, Naive Bayes, KNN**, KMeans clustering — **optional model suggestion via cross-validation**, classification report + confusion matrix, **predict on new input values**, feature importance, elbow plots, **export as Pickle or Joblib** |
| 💾 Export Clean Data | Download cleaned data as CSV or Excel, view the log of cleaning steps applied |

Every chart across the app has its own **⬇️ download button** (PNG).

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Project structure

```
app.py                      # entry point + sidebar navigation
modules/
  state.py                  # session-state helpers (working df, history/undo)
  data_upload.py            # Upload & Preview page
  analysis.py                # Analyze Dataset page
  cleaning.py                # Clean Data page
  statistics_viz.py          # Statistics & Visualization page
  modeling.py                 # Model Training page
  export.py                  # Export Clean Data page
requirements.txt
```

## Notes

- The **original uploaded data** is kept untouched in memory (`raw_df`) — all
  cleaning happens on a working copy (`df`), and you can undo any cleaning
  step from the Clean Data page.
- Model Training expects fully numeric features — encode categorical columns
  and handle missing values on the Clean Data page first.
- Trained models can be downloaded as `.pkl` files for reuse elsewhere.
