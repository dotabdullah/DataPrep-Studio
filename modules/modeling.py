import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import io
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    silhouette_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from modules.plot_utils import download_chart_button


REGRESSION_MODELS = {
    "Linear Regression": LinearRegression,
    "Decision Tree Regressor": DecisionTreeRegressor,
    "Random Forest Regressor": RandomForestRegressor,
    "Support Vector Machine (SVR)": SVR,
    "K-Nearest Neighbors Regressor": KNeighborsRegressor,
}
CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree Classifier": DecisionTreeClassifier,
    "Random Forest Classifier": RandomForestClassifier,
    "Support Vector Machine (SVC)": SVC,
    "Naive Bayes (Gaussian)": GaussianNB,
    "K-Nearest Neighbors Classifier": KNeighborsClassifier,
}
# Models whose constructor does NOT accept a random_state kwarg
NO_RANDOM_STATE = {
    "Linear Regression", "Support Vector Machine (SVR)",
    "K-Nearest Neighbors Regressor", "K-Nearest Neighbors Classifier",
    "Naive Bayes (Gaussian)",
}


def render():
    st.header("🤖 Model Training")
    df = st.session_state.df
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.info("💡 Tip: encode categorical columns and handle missing values on the **Clean Data** "
            "page before training a model — most sklearn models require fully numeric input.")

    task = st.radio("Task type", ["Supervised: Regression", "Supervised: Classification", "Unsupervised: Clustering"],
                     horizontal=False)

    if task in ("Supervised: Regression", "Supervised: Classification"):
        _supervised(df, task)
    else:
        _clustering(df, num_cols)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _make_model(model_name, model_dict, params, random_state):
    cls = model_dict[model_name]
    kwargs = dict(params)
    if model_name not in NO_RANDOM_STATE:
        kwargs["random_state"] = int(random_state)
    return cls(**kwargs)


def _benchmark_models(work_df, features, target, model_dict, task):
    """Quick cross-validation across all candidate models for this task. Returns a ranked DataFrame."""
    X = work_df[features]
    y = work_df[target]
    if len(X) > 3000:
        sampled = work_df.sample(3000, random_state=42)
        X, y = sampled[features], sampled[target]

    scoring = "r2" if task == "Supervised: Regression" else "accuracy"
    n_splits = min(5, max(2, len(X) // 5))

    rows, failed = [], []
    for name, cls in model_dict.items():
        try:
            kwargs = {} if name in NO_RANDOM_STATE else {"random_state": 42}
            if name == "Logistic Regression":
                kwargs["max_iter"] = 1000
            mdl = cls(**kwargs)
            scores = cross_val_score(mdl, X, y, cv=n_splits, scoring=scoring)
            mean_score = scores.mean()
            if np.isnan(mean_score):
                failed.append(name)
            else:
                rows.append({"Model": name, "Score": mean_score})
        except Exception:
            failed.append(name)

    result_df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    return result_df, failed


def _rank_features_supervised(df, candidate_features, target, task):
    """Quick Random Forest importance ranking — works as a rough guide for either regression or classification."""
    numeric_candidates = [c for c in candidate_features if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_candidates:
        return pd.DataFrame(columns=["Feature", "Importance"])

    work = df[numeric_candidates + [target]].dropna()
    if len(work) < 10:
        return pd.DataFrame(columns=["Feature", "Importance"])

    X, y = work[numeric_candidates], work[target]
    try:
        if task == "Supervised: Regression":
            if not pd.api.types.is_numeric_dtype(y):
                return pd.DataFrame(columns=["Feature", "Importance"])
            model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        model.fit(X, y)
        result = pd.DataFrame({"Feature": numeric_candidates, "Importance": model.feature_importances_})
        return result.sort_values("Importance", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])


def _rank_features_clustering(df, candidate_features):
    """No target exists for clustering, so 'importance' isn't well defined the usual way.
    Instead, rank numeric columns by how much they drive variation in the data (via PCA loadings)."""
    numeric_candidates = [c for c in candidate_features if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_candidates) < 2:
        return pd.DataFrame(columns=["Feature", "Influence"])

    work = df[numeric_candidates].dropna()
    if len(work) < 10:
        return pd.DataFrame(columns=["Feature", "Influence"])

    try:
        X = StandardScaler().fit_transform(work)
        n_components = min(2, X.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X)
        loadings = np.abs(pca.components_)               # (n_components, n_features)
        weights = pca.explained_variance_ratio_           # (n_components,)
        influence = loadings.T @ weights                  # (n_features,)
        result = pd.DataFrame({"Feature": numeric_candidates, "Influence": influence})
        return result.sort_values("Influence", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["Feature", "Influence"])


def _model_download_section(model, model_name):
    st.subheader("Export Trained Model")
    export_format = st.radio("File format", ["Pickle (.pkl)", "Joblib (.joblib)"],
                              horizontal=True, key="export_fmt")
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    buf = io.BytesIO()
    if export_format.startswith("Pickle"):
        pickle.dump(model, buf)
        ext = "pkl"
    else:
        joblib.dump(model, buf)
        ext = "joblib"
    buf.seek(0)
    st.download_button(
        f"⬇️ Download trained model (.{ext})",
        data=buf, file_name=f"{safe_name}.{ext}",
        mime="application/octet-stream", key="dl_model_file"
    )


# ----------------------------------------------------------------------------
# Supervised (Regression / Classification)
# ----------------------------------------------------------------------------

def _supervised(df, task):
    st.subheader("1. Select Features & Target")
    all_cols = df.columns.tolist()
    target = st.selectbox("Target column (what you want to predict)", all_cols)
    feature_options = [c for c in all_cols if c != target]

    features_key = f"features_select_{task}"
    if features_key not in st.session_state:
        st.session_state[features_key] = feature_options
    else:
        st.session_state[features_key] = [f for f in st.session_state[features_key] if f in feature_options]

    with st.expander("🔍 Discover which features matter most (optional)"):
        st.caption("Runs a quick Random Forest to estimate how useful each candidate column is for predicting "
                   "**" + target + "**. This is a suggestion, not a rule — accept it, tweak it, or ignore it "
                   "and pick features manually below. Only numeric candidate columns can be ranked; "
                   "encode categorical ones first if you want them included.")
        if st.button("Rank feature importance", key=f"rank_features_btn_{task}"):
            with st.spinner("Ranking features..."):
                st.session_state[f"feature_importance_{task}"] = _rank_features_supervised(
                    df, feature_options, target, task
                )
        if f"feature_importance_{task}" in st.session_state:
            imp_df = st.session_state[f"feature_importance_{task}"]
            if imp_df.empty:
                st.warning("Couldn't rank features — make sure candidate columns are numeric and there's "
                           "enough non-missing data.")
            else:
                st.dataframe(imp_df, use_container_width=True, hide_index=True)
                fig, ax = plt.subplots(figsize=(6, max(2, len(imp_df) * 0.4)))
                ax.barh(imp_df["Feature"], imp_df["Importance"], color="#3d5a80")
                ax.invert_yaxis()
                ax.set_xlabel("Estimated importance")
                ax.set_title("Feature Importance (suggestion)")
                st.pyplot(fig)
                download_chart_button(fig, "feature_importance_suggestion.png", key=f"dl_feat_rank_{task}")
                plt.close(fig)

                top_n = st.slider("How many top features to use?", 1, len(imp_df),
                                   min(5, len(imp_df)), key=f"top_n_{task}")
                if st.button(f"✅ Use top {top_n} suggested feature(s)", key=f"use_feat_suggest_{task}"):
                    st.session_state[features_key] = imp_df["Feature"].head(top_n).tolist()
                    st.rerun()

    features = st.multiselect("Feature columns", feature_options, key=features_key)

    if not features:
        st.warning("Select at least one feature column.")
        return

    non_numeric_features = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric_features:
        st.error(f"These feature columns are not numeric: {non_numeric_features}. "
                  f"Encode them first on the Clean Data page.")
        return

    if task == "Supervised: Classification":
        if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 20:
            st.warning("Target looks continuous (many unique numeric values). "
                       "Classification expects discrete class labels — consider Regression instead, "
                       "or encode the target into categories first.")
    else:
        if not pd.api.types.is_numeric_dtype(df[target]):
            st.error("Regression requires a numeric target column. Encode or choose a numeric column.")
            return

    work_df = df[features + [target]].dropna()
    if len(work_df) < len(df):
        st.caption(f"Dropped {len(df) - len(work_df)} row(s) with missing values in selected columns for training.")

    st.subheader("2. Train / Test Split")
    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
    random_state = st.number_input("Random seed", value=42, step=1)

    st.subheader("3. Choose Model")
    model_dict = REGRESSION_MODELS if task == "Supervised: Regression" else CLASSIFICATION_MODELS
    model_names = list(model_dict.keys())
    model_select_key = f"model_select_{task}"
    if model_select_key not in st.session_state or st.session_state[model_select_key] not in model_names:
        st.session_state[model_select_key] = model_names[0]

    with st.expander("🔍 Not sure which model to use? Get a suggestion (optional)"):
        st.caption("Runs quick cross-validation across all available models for this task and ranks them "
                   "by average score. This is a rough guide, not a guarantee — you can still pick manually.")
        if st.button("Run comparison", key=f"suggest_btn_{task}"):
            with st.spinner("Testing models..."):
                result_df, failed = _benchmark_models(work_df, features, target, model_dict, task)
                st.session_state[f"suggestion_{task}"] = result_df
                st.session_state[f"suggestion_failed_{task}"] = failed

        if f"suggestion_{task}" in st.session_state:
            result_df = st.session_state[f"suggestion_{task}"]
            if result_df.empty:
                st.error("None of the models could be evaluated on this data.")
            else:
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                fig, ax = plt.subplots(figsize=(6, max(2, len(result_df) * 0.5)))
                ax.barh(result_df["Model"], result_df["Score"], color="#3d5a80")
                ax.invert_yaxis()
                ax.set_xlabel("R² Score" if task == "Supervised: Regression" else "Accuracy")
                ax.set_title("Model Comparison (cross-validation)")
                st.pyplot(fig)
                download_chart_button(fig, "model_comparison.png", key=f"dl_comparison_{task}")
                plt.close(fig)

                failed = st.session_state.get(f"suggestion_failed_{task}", [])
                if failed:
                    st.caption(f"Couldn't evaluate: {', '.join(failed)} (often due to small class sizes).")

                best_name = result_df.iloc[0]["Model"]
                if st.button(f"✅ Use suggested model: {best_name}", key=f"use_suggest_{task}"):
                    st.session_state[model_select_key] = best_name
                    st.rerun()

    model_name = st.selectbox("Model", model_names, key=model_select_key)

    params = {}
    if "Tree" in model_name or "Forest" in model_name:
        max_depth = st.slider("Max depth (0 = unlimited)", 0, 30, 5)
        params["max_depth"] = None if max_depth == 0 else max_depth
        if "Forest" in model_name:
            params["n_estimators"] = st.slider("Number of trees", 10, 300, 100, step=10)
    if "Support Vector Machine" in model_name:
        c1, c2 = st.columns(2)
        params["kernel"] = c1.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        params["C"] = c2.slider("Regularization (C)", 0.01, 10.0, 1.0)
        if model_name == "Support Vector Machine (SVC)":
            params["probability"] = st.checkbox(
                "Enable probability estimates (slower to train)", value=False,
                help="Needed if you want class probabilities when predicting new data."
            )
    if "K-Nearest Neighbors" in model_name:
        approx_train_size = max(1, int(len(work_df) * (1 - test_size)))
        max_k = max(1, min(30, approx_train_size - 1)) if approx_train_size > 1 else 1
        params["n_neighbors"] = st.slider("Number of neighbors (K)", 1, max_k, min(5, max_k))
    if model_name == "Logistic Regression":
        params["max_iter"] = 1000

    if st.button("🚀 Train Model", type="primary"):
        X = work_df[features]
        y = work_df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )

        model = _make_model(model_name, model_dict, params, random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.session_state.trained_model = model
        st.session_state.model_info = {
            "task": task, "model_name": model_name, "features": features, "target": target,
            "y_test": y_test, "y_pred": y_pred,
            "class_labels": sorted(y.unique().tolist()) if task == "Supervised: Classification" else None,
            "feature_means": X.mean().to_dict(),
            "n_train": len(X_train), "n_test": len(X_test),
        }
        st.success(f"Trained **{model_name}** on {len(X_train)} rows, tested on {len(X_test)} rows.")
        st.rerun()

    # Results persist across reruns (e.g. clicking Predict) as long as the setup still matches
    info = st.session_state.get("model_info", {})
    if (st.session_state.get("trained_model") is not None and info.get("task") == task
            and info.get("features") == features and info.get("target") == target):
        _render_results(task)


def _render_results(task):
    info = st.session_state.model_info
    model = st.session_state.trained_model
    y_test, y_pred = info["y_test"], info["y_pred"]
    features, model_name = info["features"], info["model_name"]

    st.subheader("4. Results")
    st.caption(f"Model: **{model_name}** — trained on {info['n_train']} rows, tested on {info['n_test']} rows.")

    if task == "Supervised: Regression":
        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
        c2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")
        c3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.6, color="#3d5a80")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", label="Perfect prediction")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)
        download_chart_button(fig, "actual_vs_predicted.png", key="dl_actual_vs_pred")
        plt.close(fig)
    else:
        class_labels = info["class_labels"]
        average = "binary" if len(class_labels) == 2 else "weighted"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        c2.metric("Precision", f"{precision_score(y_test, y_pred, average=average, zero_division=0):.4f}")
        c3.metric("Recall", f"{recall_score(y_test, y_pred, average=average, zero_division=0):.4f}")
        c4.metric("F1 Score", f"{f1_score(y_test, y_pred, average=average, zero_division=0):.4f}")

        st.write("**Classification Report** (per class)")
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

        st.write("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 5))
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        ConfusionMatrixDisplay(cm, display_labels=class_labels).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        download_chart_button(fig, "confusion_matrix.png", key="dl_confusion_matrix")
        plt.close(fig)

    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance")
        imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.35)))
        ax.barh(imp.index, imp.values, color="#98c1d9")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
        download_chart_button(fig, "feature_importance.png", key="dl_feat_importance")
        plt.close(fig)
    elif hasattr(model, "coef_"):
        coefs = np.atleast_2d(model.coef_)
        if coefs.shape[0] == 1 and coefs.shape[1] == len(features):
            st.subheader("Model Coefficients")
            coef_series = pd.Series(coefs[0], index=features).sort_values()
            fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.35)))
            ax.barh(coef_series.index, coef_series.values, color="#ee6c4d")
            ax.set_title("Coefficients")
            st.pyplot(fig)
            download_chart_button(fig, "coefficients.png", key="dl_coefficients")
            plt.close(fig)
        else:
            st.caption("Coefficients not shown (multi-class model has one coefficient set per class).")

    if "Decision Tree" in model_name:
        with st.expander("Visualize Decision Tree (top levels)"):
            fig, ax = plt.subplots(figsize=(14, 8))
            plot_tree(model, feature_names=features, filled=True, max_depth=3, fontsize=8, ax=ax)
            st.pyplot(fig)
            download_chart_button(fig, "decision_tree.png", key="dl_dec_tree")
            plt.close(fig)

    st.subheader("5. Predict on New Data")
    with st.form("predict_form"):
        st.caption("Enter feature values to get a prediction from the trained model above.")
        input_vals = {}
        cols = st.columns(2)
        for i, feat in enumerate(features):
            default_val = float(info["feature_means"].get(feat, 0.0))
            input_vals[feat] = cols[i % 2].number_input(feat, value=round(default_val, 4),
                                                          key=f"predict_input_{feat}")
        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        input_df = pd.DataFrame([input_vals])[features]
        pred = model.predict(input_df)[0]
        st.success(f"**Predicted {info['target']}:** {pred}")
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(input_df)[0]
                proba_df = pd.DataFrame({"Class": model.classes_, "Probability": proba}) \
                    .sort_values("Probability", ascending=False)
                st.write("**Class probabilities:**")
                st.dataframe(proba_df, use_container_width=True, hide_index=True)
            except Exception:
                pass

    _model_download_section(model, model_name)


# ----------------------------------------------------------------------------
# Unsupervised: Clustering
# ----------------------------------------------------------------------------

def _clustering(df, num_cols):
    st.subheader("1. Select Features")

    features_key = "cluster_features_select"
    default_features = num_cols[:min(4, len(num_cols))]
    if features_key not in st.session_state:
        st.session_state[features_key] = default_features
    else:
        st.session_state[features_key] = [f for f in st.session_state[features_key] if f in num_cols]

    with st.expander("🔍 Discover which features matter most (optional)"):
        st.caption("Clustering has no target column, so 'importance' isn't defined the same way it is for "
                   "regression or classification. This instead ranks numeric columns by how much they drive "
                   "variation in the data (via PCA) — columns with higher influence tend to have a bigger "
                   "effect on which cluster a row lands in. Accept it, tweak it, or pick manually below.")
        if st.button("Rank feature influence", key="rank_cluster_features_btn"):
            with st.spinner("Ranking features..."):
                st.session_state["cluster_feature_influence"] = _rank_features_clustering(df, num_cols)
        if "cluster_feature_influence" in st.session_state:
            inf_df = st.session_state["cluster_feature_influence"]
            if inf_df.empty:
                st.warning("Couldn't rank features — need at least 2 numeric columns with enough non-missing data.")
            else:
                st.dataframe(inf_df, use_container_width=True, hide_index=True)
                fig, ax = plt.subplots(figsize=(6, max(2, len(inf_df) * 0.4)))
                ax.barh(inf_df["Feature"], inf_df["Influence"], color="#98c1d9")
                ax.invert_yaxis()
                ax.set_xlabel("Estimated influence")
                ax.set_title("Feature Influence on Clustering (PCA-based)")
                st.pyplot(fig)
                download_chart_button(fig, "cluster_feature_influence.png", key="dl_cluster_feat_influence")
                plt.close(fig)

                top_n = st.slider("How many top features to use?", 2, len(inf_df),
                                   min(4, len(inf_df)), key="cluster_top_n")
                if st.button(f"✅ Use top {top_n} suggested feature(s)", key="use_cluster_feat_suggest"):
                    st.session_state[features_key] = inf_df["Feature"].head(top_n).tolist()
                    st.rerun()

    features = st.multiselect("Feature columns (numeric)", num_cols, key=features_key)
    if len(features) < 2:
        st.warning("Select at least 2 numeric feature columns.")
        return

    work_df = df[features].dropna()
    scale = st.checkbox("Standardize features before clustering (recommended)", value=True)
    X = StandardScaler().fit_transform(work_df) if scale else work_df.values

    st.subheader("2. Choose K (number of clusters)")
    k = st.slider("Number of clusters (K)", 2, min(10, max(2, len(work_df) - 1)), min(3, max(2, len(work_df) - 1)))

    if st.checkbox("Show elbow plot to help choose K"):
        inertias = []
        k_range = range(2, min(11, len(work_df)))
        for kk in k_range:
            km = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(X)
            inertias.append(km.inertia_)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(list(k_range), inertias, marker="o", color="#3d5a80")
        ax.set_xlabel("K")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method")
        st.pyplot(fig)
        download_chart_button(fig, "elbow_plot.png", key="dl_elbow_plot")
        plt.close(fig)

    if st.button("🚀 Run KMeans Clustering", type="primary"):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        work_df_result = work_df.copy()
        work_df_result["cluster"] = labels

        st.session_state.trained_model = model
        st.session_state.model_info = {"task": "Clustering", "features": features}
        st.session_state.cluster_result = work_df_result
        st.session_state.cluster_labels = labels
        st.session_state.cluster_X = X
        st.rerun()

    if "cluster_result" in st.session_state and st.session_state.model_info.get("features") == features:
        work_df_result = st.session_state.cluster_result
        labels = st.session_state.cluster_labels
        X = st.session_state.cluster_X

        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
        st.metric("Silhouette Score (higher is better, max 1.0)", f"{sil:.4f}")

        st.subheader("Cluster Sizes")
        st.dataframe(work_df_result["cluster"].value_counts().rename("count"), use_container_width=True)

        if len(features) >= 2:
            c1, c2 = st.columns(2)
            x_ax = c1.selectbox("X axis", features, index=0, key="cluster_x_axis")
            y_ax = c2.selectbox("Y axis", features, index=1, key="cluster_y_axis")
            fig, ax = plt.subplots(figsize=(6, 5))
            scatter = ax.scatter(work_df_result[x_ax], work_df_result[y_ax], c=labels, cmap="tab10", alpha=0.7)
            ax.set_xlabel(x_ax)
            ax.set_ylabel(y_ax)
            ax.set_title("Cluster Visualization")
            legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
            ax.add_artist(legend1)
            st.pyplot(fig)
            download_chart_button(fig, "cluster_visualization.png", key="dl_cluster_viz")
            plt.close(fig)

        st.subheader("Clustered Data (preview)")
        st.dataframe(work_df_result.head(50), use_container_width=True)

        csv = work_df_result.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download clustered data (CSV)", data=csv,
                            file_name="clustered_data.csv", mime="text/csv", key="dl_clustered_csv")

        _model_download_section(st.session_state.trained_model, "KMeans")
