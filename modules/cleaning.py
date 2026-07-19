import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from modules.state import push_history, undo_last
from modules.plot_utils import download_chart_button
from modules.recipe import record_step, export_recipe_bytes, load_recipe_bytes, apply_recipe


def _suggest_missing_strategy(df: pd.DataFrame, col: str) -> dict:
    """Recommend a missing-value strategy for a single column based on missing %,
    dtype, and (for numeric columns) skewness."""
    missing_pct = df[col].isna().mean() * 100

    if missing_pct > 50:
        return {"method": "Drop column",
                "reason": f"{missing_pct:.1f}% missing — too much to reliably fill in."}

    if pd.api.types.is_numeric_dtype(df[col]):
        if missing_pct <= 5:
            return {"method": "Drop rows",
                     "reason": f"Only {missing_pct:.1f}% missing — safe to drop those rows without losing much data."}
        skew = df[col].skew()
        if pd.isna(skew):
            return {"method": "Fill with median",
                     "reason": f"{missing_pct:.1f}% missing — median is a safe default here."}
        if abs(skew) > 1:
            return {"method": "Fill with median",
                     "reason": f"{missing_pct:.1f}% missing, skewed distribution (skew={skew:.2f}) — "
                               f"median is more robust to outliers than the mean."}
        return {"method": "Fill with mean",
                 "reason": f"{missing_pct:.1f}% missing, roughly symmetric distribution (skew={skew:.2f})."}

    # categorical / text column
    if missing_pct <= 5:
        return {"method": "Drop rows",
                 "reason": f"Only {missing_pct:.1f}% missing — safe to drop those rows without losing much data."}
    return {"method": "Fill with mode",
             "reason": f"{missing_pct:.1f}% missing — mode (most frequent value) is the standard choice "
                       f"for categorical data."}


def _apply_suggested_missing(df: pd.DataFrame, suggestions: dict) -> pd.DataFrame:
    """Apply each column's suggested strategy. Drops columns first, then rows, then fills what remains."""
    new_df = df.copy()

    drop_cols = [c for c, s in suggestions.items() if s["method"] == "Drop column"]
    if drop_cols:
        new_df = new_df.drop(columns=drop_cols)

    dropna_cols = [c for c, s in suggestions.items() if s["method"] == "Drop rows" and c in new_df.columns]
    if dropna_cols:
        new_df = new_df.dropna(subset=dropna_cols)

    for c, s in suggestions.items():
        if c not in new_df.columns:
            continue
        if s["method"] == "Fill with mean":
            new_df[c] = new_df[c].fillna(new_df[c].mean())
        elif s["method"] == "Fill with median":
            new_df[c] = new_df[c].fillna(new_df[c].median())
        elif s["method"] == "Fill with mode":
            mode_val = new_df[c].mode()
            if not mode_val.empty:
                new_df[c] = new_df[c].fillna(mode_val.iloc[0])

    return new_df


def render():
    st.header("🧹 Clean Data")
    df = st.session_state.df

    with st.expander("📋 Cleaning Recipe — save these steps and reapply them on a new file"):
        st.caption("Every action you take below (missing values, encoding, scaling, PCA, etc.) is recorded "
                   "as a replayable step. Download the recipe now, and next time you upload a new file with "
                   "the same columns, come back here and replay it to redo all this cleaning in one click — "
                   "including reusing the exact same encoders/scalers, so results stay consistent with any "
                   "model you've already trained.")
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            st.write(f"**Steps recorded this session:** {len(st.session_state.recipe)}")
            if st.session_state.recipe:
                steps_display = pd.DataFrame([
                    {"#": i + 1, "Step": s["type"]} for i, s in enumerate(st.session_state.recipe)
                ])
                st.dataframe(steps_display, width='stretch', hide_index=True)
                recipe_bytes = export_recipe_bytes(
                    st.session_state.recipe, st.session_state.encoders, st.session_state.onehot_columns,
                    st.session_state.value_labels
                )
                base_name = (st.session_state.filename or "dataset").rsplit(".", 1)[0]
                st.download_button("⬇️ Download recipe", data=recipe_bytes,
                                    file_name=f"{base_name}_recipe.recipe",
                                    mime="application/octet-stream", key="dl_recipe")
            else:
                st.caption("No cleaning steps recorded yet — actions you take below will show up here.")
        with rec_col2:
            st.write("**Replay a saved recipe on this file's original data:**")
            uploaded_recipe = st.file_uploader("Upload a .recipe file", type=["recipe"], key="recipe_uploader")
            if uploaded_recipe is not None:
                if st.button("▶️ Replay recipe now", type="primary"):
                    try:
                        payload = load_recipe_bytes(uploaded_recipe.getvalue())
                        new_df, new_encoders, new_onehot, new_value_labels, warnings_list = apply_recipe(
                            st.session_state.raw_df, payload
                        )
                        st.session_state.df = new_df
                        st.session_state.encoders = new_encoders
                        st.session_state.onehot_columns = new_onehot
                        st.session_state.value_labels = new_value_labels
                        st.session_state.recipe = payload.get("steps", [])
                        st.session_state.history = []
                        st.success(f"Replayed {len(payload.get('steps', []))} step(s) on the original data — "
                                   f"now {new_df.shape[0]} rows × {new_df.shape[1]} columns.")
                        if warnings_list:
                            with st.expander(f"⚠️ {len(warnings_list)} warning(s) during replay"):
                                for w in warnings_list[:50]:
                                    st.write("- " + w)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Couldn't replay this recipe: {e}")

    top_l, top_r = st.columns([3, 1])
    with top_r:
        if st.button("↩️ Undo last action", width='stretch'):
            action = undo_last()
            if action:
                st.success(f"Undid: {action}")
                st.rerun()
            else:
                st.info("Nothing to undo.")
    with top_l:
        if st.session_state.history:
            st.caption(f"History: {len(st.session_state.history)} action(s) can be undone.")

    tabs = st.tabs([
        "Missing Values", "Duplicates", "Columns", "Data Types",
        "Outliers", "Encoding", "Value Labels", "Scaling", "Text Cleanup", "Dimensionality Reduction (PCA)"
    ])

    # ---------------- Missing Values ----------------
    with tabs[0]:
        st.subheader("Handle Missing Values")
        cols_with_na = df.columns[df.isna().any()].tolist()
        if not cols_with_na:
            st.success("No missing values to handle! ✅")
        else:
            with st.expander("🔍 Suggested strategy per column (optional)", expanded=True):
                st.caption("A quick analysis of each column with missing data, recommending a method based on "
                           "how much is missing, the data type, and (for numeric columns) how skewed the values "
                           "are. Accept all suggestions at once, or ignore this and choose manually below.")
                suggestions = {c: _suggest_missing_strategy(df, c) for c in cols_with_na}
                sugg_df = pd.DataFrame([
                    {"Column": c, "Missing %": f"{df[c].isna().mean() * 100:.1f}%",
                     "Suggested Method": s["method"], "Why": s["reason"]}
                    for c, s in suggestions.items()
                ])
                st.dataframe(sugg_df, width='stretch', hide_index=True)
                if st.button("✅ Apply all suggested strategies", type="primary"):
                    drop_cols = [c for c, s in suggestions.items() if s["method"] == "Drop column"]
                    dropna_cols = [c for c, s in suggestions.items() if s["method"] == "Drop rows"]
                    mean_cols = [c for c, s in suggestions.items() if s["method"] == "Fill with mean"]
                    median_cols = [c for c, s in suggestions.items() if s["method"] == "Fill with median"]
                    mode_cols = [c for c, s in suggestions.items() if s["method"] == "Fill with mode"]
                    steps = []
                    if drop_cols:
                        steps.append(record_step("drop_columns", columns=drop_cols))
                    if dropna_cols:
                        steps.append(record_step("dropna_rows", columns=dropna_cols))
                    if mean_cols:
                        steps.append(record_step("fillna", columns=mean_cols, method="mean"))
                    if median_cols:
                        steps.append(record_step("fillna", columns=median_cols, method="median"))
                    if mode_cols:
                        steps.append(record_step("fillna", columns=mode_cols, method="mode"))
                    push_history("Applied suggested missing-value strategies", steps)
                    st.session_state.df = _apply_suggested_missing(df, suggestions)
                    st.success("Applied suggested strategies to all columns with missing values.")
                    st.rerun()

            st.divider()
            st.write("**Or choose a method manually:**")
            strategy_scope = st.radio("Apply to", ["Selected column(s)", "Entire dataset"], horizontal=True)
            method = st.selectbox(
                "Method",
                ["Drop rows with missing values", "Drop columns with missing values",
                 "Fill with mean", "Fill with median", "Fill with mode",
                 "Fill with constant value", "Forward fill (ffill)", "Backward fill (bfill)"]
            )

            target_cols = cols_with_na
            if strategy_scope == "Selected column(s)":
                target_cols = st.multiselect("Column(s)", cols_with_na, default=cols_with_na[:1])

            fill_value = None
            if method == "Fill with constant value":
                fill_value = st.text_input("Constant value to fill with", value="0")

            if st.button("Apply missing value strategy", type="primary"):
                method_to_step = {
                    "Drop rows with missing values": ("dropna_rows", {"columns": target_cols}),
                    "Drop columns with missing values": ("drop_columns", {"columns": target_cols}),
                    "Fill with mean": ("fillna", {"columns": target_cols, "method": "mean"}),
                    "Fill with median": ("fillna", {"columns": target_cols, "method": "median"}),
                    "Fill with mode": ("fillna", {"columns": target_cols, "method": "mode"}),
                    "Fill with constant value": ("fillna", {"columns": target_cols, "method": "constant", "value": fill_value}),
                    "Forward fill (ffill)": ("fillna", {"columns": target_cols, "method": "ffill"}),
                    "Backward fill (bfill)": ("fillna", {"columns": target_cols, "method": "bfill"}),
                }
                step_type, step_kwargs = method_to_step[method]
                push_history(f"Missing values: {method} on {target_cols}", record_step(step_type, **step_kwargs))
                new_df = df.copy()
                try:
                    if method == "Drop rows with missing values":
                        new_df = new_df.dropna(subset=target_cols)
                    elif method == "Drop columns with missing values":
                        new_df = new_df.drop(columns=target_cols)
                    elif method == "Fill with mean":
                        for c in target_cols:
                            if pd.api.types.is_numeric_dtype(new_df[c]):
                                new_df[c] = new_df[c].fillna(new_df[c].mean())
                            else:
                                st.warning(f"Skipped '{c}': not numeric.")
                    elif method == "Fill with median":
                        for c in target_cols:
                            if pd.api.types.is_numeric_dtype(new_df[c]):
                                new_df[c] = new_df[c].fillna(new_df[c].median())
                            else:
                                st.warning(f"Skipped '{c}': not numeric.")
                    elif method == "Fill with mode":
                        for c in target_cols:
                            mode_val = new_df[c].mode()
                            if not mode_val.empty:
                                new_df[c] = new_df[c].fillna(mode_val.iloc[0])
                    elif method == "Fill with constant value":
                        for c in target_cols:
                            try:
                                if pd.api.types.is_numeric_dtype(new_df[c]):
                                    new_df[c] = new_df[c].fillna(float(fill_value))
                                else:
                                    new_df[c] = new_df[c].fillna(fill_value)
                            except ValueError:
                                new_df[c] = new_df[c].fillna(fill_value)
                    elif method == "Forward fill (ffill)":
                        new_df[target_cols] = new_df[target_cols].ffill()
                    elif method == "Backward fill (bfill)":
                        new_df[target_cols] = new_df[target_cols].bfill()

                    st.session_state.df = new_df
                    st.success("Missing value strategy applied.")
                    st.rerun()
                except Exception as e:
                    st.session_state.history.pop()  # revert the push since it failed
                    st.error(f"Error: {e}")

    # ---------------- Duplicates ----------------
    with tabs[1]:
        st.subheader("Remove Duplicate Rows")
        dup_count = df.duplicated().sum()
        st.metric("Duplicate rows", int(dup_count))
        subset_cols = st.multiselect("Consider only these columns when detecting duplicates (optional)",
                                      df.columns.tolist())
        keep_option = st.radio("Which duplicate to keep", ["first", "last", "none (drop all)"], horizontal=True)
        keep_val = False if keep_option == "none (drop all)" else keep_option

        if st.button("Remove duplicates", type="primary", disabled=(dup_count == 0)):
            keep_recorded = None if keep_val is False else keep_val
            push_history("Removed duplicate rows",
                          record_step("drop_duplicates", subset=subset_cols or None,
                                      keep=keep_recorded if keep_recorded else False))
            new_df = df.drop_duplicates(subset=subset_cols if subset_cols else None, keep=keep_val)
            st.session_state.df = new_df
            st.success(f"Removed {len(df) - len(new_df)} duplicate row(s).")
            st.rerun()

    # ---------------- Columns ----------------
    with tabs[2]:
        st.subheader("Drop or Rename Columns")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Drop columns**")
            drop_cols = st.multiselect("Select columns to drop", df.columns.tolist())
            if st.button("Drop selected columns", disabled=not drop_cols):
                push_history(f"Dropped columns: {drop_cols}", record_step("drop_columns", columns=drop_cols))
                st.session_state.df = df.drop(columns=drop_cols)
                st.success(f"Dropped: {', '.join(drop_cols)}")
                st.rerun()
        with c2:
            st.write("**Rename a column**")
            col_to_rename = st.selectbox("Column", df.columns.tolist(), key="rename_select")
            new_name = st.text_input("New name", value=col_to_rename)
            if st.button("Rename column", disabled=(new_name.strip() == "" or new_name == col_to_rename)):
                push_history(f"Renamed '{col_to_rename}' to '{new_name}'",
                             record_step("rename_column", old_name=col_to_rename, new_name=new_name))
                st.session_state.df = df.rename(columns={col_to_rename: new_name})
                st.success(f"Renamed '{col_to_rename}' → '{new_name}'")
                st.rerun()

    # ---------------- Data Types ----------------
    with tabs[3]:
        st.subheader("Convert Data Types")
        col = st.selectbox("Column", df.columns.tolist(), key="dtype_select")
        st.caption(f"Current dtype: `{df[col].dtype}`")
        new_type = st.selectbox("Convert to", ["int64", "float64", "str (object)", "category", "datetime"])
        if st.button("Convert type", type="primary"):
            push_history(f"Converted '{col}' to {new_type}", record_step("convert_dtype", column=col, new_type=new_type))
            new_df = df.copy()
            try:
                if new_type == "int64":
                    new_df[col] = pd.to_numeric(new_df[col], errors="coerce").astype("Int64")
                elif new_type == "float64":
                    new_df[col] = pd.to_numeric(new_df[col], errors="coerce").astype("float64")
                elif new_type == "str (object)":
                    new_df[col] = new_df[col].astype(str)
                elif new_type == "category":
                    new_df[col] = new_df[col].astype("category")
                elif new_type == "datetime":
                    new_df[col] = pd.to_datetime(new_df[col], errors="coerce")
                st.session_state.df = new_df
                st.success(f"Converted '{col}' to {new_type}. Values that couldn't convert became missing (NaN).")
                st.rerun()
            except Exception as e:
                st.session_state.history.pop()
                st.error(f"Conversion failed: {e}")

    # ---------------- Outliers ----------------
    with tabs[4]:
        st.subheader("Detect & Handle Outliers")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            st.info("No numeric columns available for outlier detection.")
        else:
            col = st.selectbox("Numeric column", num_cols, key="outlier_select")
            method = st.radio("Detection method", ["IQR (1.5×)", "Z-score (>3)"], horizontal=True)

            series = df[col].dropna()
            if method == "IQR (1.5×)":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_mask = (df[col] < lower) | (df[col] > upper)
            else:
                mean, std = series.mean(), series.std()
                z = (df[col] - mean) / std
                outlier_mask = z.abs() > 3
                lower, upper = mean - 3 * std, mean + 3 * std

            st.write(f"Bounds: **[{lower:.4f}, {upper:.4f}]** — {int(outlier_mask.sum())} outlier(s) detected")

            import matplotlib.pyplot as plt
            from modules.plot_utils import download_chart_button
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.boxplot(series, vert=False)
            ax.set_title(f"Boxplot: {col}")
            st.pyplot(fig)
            download_chart_button(fig, f"boxplot_{col}.png", key=f"dl_outlier_box_{col}")
            plt.close(fig)

            action = st.radio("Action", ["Remove outlier rows", "Cap to bounds (winsorize)"], horizontal=True)
            if st.button("Apply outlier treatment", type="primary", disabled=(outlier_mask.sum() == 0)):
                step_method = "iqr" if method == "IQR (1.5×)" else "zscore"
                step_action = "remove" if action == "Remove outlier rows" else "cap"
                push_history(f"Outlier treatment on '{col}' ({action})",
                             record_step("outlier", column=col, method=step_method, action=step_action))
                new_df = df.copy()
                if action == "Remove outlier rows":
                    new_df = new_df[~outlier_mask]
                else:
                    new_df[col] = new_df[col].clip(lower=lower, upper=upper)
                st.session_state.df = new_df
                st.success("Outlier treatment applied.")
                st.rerun()

    # ---------------- Encoding ----------------
    with tabs[5]:
        st.subheader("Encode Categorical Variables")
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        if not cat_cols:
            st.info("No categorical columns to encode.")
        else:
            col = st.selectbox("Column to encode", cat_cols, key="encode_select")
            n_unique = df[col].nunique()
            st.caption(f"'{col}' has {n_unique} unique values.")
            method = st.radio("Encoding method", ["Label Encoding", "One-Hot Encoding"], horizontal=True)

            if method == "One-Hot Encoding" and n_unique > 20:
                st.warning("This column has many unique values — one-hot encoding will create many new columns.")

            if st.button("Apply encoding", type="primary"):
                step_type = "label_encode" if method == "Label Encoding" else "onehot_encode"
                push_history(f"Encoded '{col}' with {method}", record_step(step_type, column=col))
                new_df = df.copy()
                if method == "Label Encoding":
                    le = LabelEncoder()
                    new_df[col] = le.fit_transform(new_df[col].astype(str))
                    st.session_state.encoders[col] = le
                    st.session_state.onehot_columns.pop(col, None)
                else:
                    categories = df[col].dropna().astype(str).unique().tolist()
                    dummies = pd.get_dummies(new_df[col].astype(str), prefix=col, dtype=int)
                    new_df = pd.concat([new_df.drop(columns=[col]), dummies], axis=1)
                    st.session_state.onehot_columns[col] = {
                        "dummy_columns": dummies.columns.tolist(),
                        "category_to_column": {cat: f"{col}_{cat}" for cat in categories},
                    }
                    st.session_state.encoders.pop(col, None)
                st.session_state.df = new_df
                st.success(f"Encoded '{col}' using {method}.")
                st.rerun()

    # ---------------- Value Labels ----------------
    with tabs[6]:
        st.subheader("Value Labels")
        st.caption("For columns that are **already numeric but represent categories** — e.g. 'Sex' coded as "
                   "0/1, or 'Status' coded 0/1/2. This doesn't change your data at all; it just remembers a "
                   "friendly name for each value so the Model Training prediction form can show \"Male\" / "
                   "\"Female\" instead of asking you to type 0 or 1. (Use the **Encoding** tab instead if your "
                   "column currently holds text like \"Male\"/\"Female\" that needs converting to numbers.)")

        num_cols_vl = df.select_dtypes(include=np.number).columns.tolist()
        eligible_cols = [c for c in num_cols_vl if 2 <= df[c].dropna().nunique() <= 15]
        if not eligible_cols:
            st.info("No numeric columns with a small number of unique values (2–15) found to label.")
        else:
            col = st.selectbox("Column to label", eligible_cols, key="value_label_col_select")
            unique_vals = sorted(df[col].dropna().unique().tolist())
            existing = st.session_state.value_labels.get(col, {})

            st.write(f"'{col}' has {len(unique_vals)} unique value(s) — give each one a name:")
            label_inputs = {}
            n_cols_ui = min(3, len(unique_vals))
            cols_ui = st.columns(n_cols_ui)
            for i, val in enumerate(unique_vals):
                default_label = existing.get(val, str(val))
                label_inputs[val] = cols_ui[i % n_cols_ui].text_input(
                    f"Label for {val}", value=default_label, key=f"vl_input_{col}_{val}"
                )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("💾 Save labels", type="primary"):
                    if len(set(label_inputs.values())) != len(label_inputs):
                        st.error("Each label must be unique.")
                    elif any(not v.strip() for v in label_inputs.values()):
                        st.error("Labels can't be empty.")
                    else:
                        st.session_state.value_labels[col] = label_inputs
                        st.session_state.recipe.append(record_step("value_labels", column=col, labels=label_inputs))
                        st.success(f"Saved labels for '{col}': {label_inputs}")
                        st.rerun()
            with c2:
                if col in st.session_state.value_labels:
                    if st.button(f"🗑️ Remove labels for '{col}'"):
                        del st.session_state.value_labels[col]
                        st.session_state.recipe = [
                            s for s in st.session_state.recipe
                            if not (s["type"] == "value_labels" and s["column"] == col)
                        ]
                        st.rerun()

        if st.session_state.value_labels:
            st.divider()
            st.write("**Currently labeled columns:**")
            for c, labels in st.session_state.value_labels.items():
                st.write(f"- **{c}**: {labels}")

    # ---------------- Scaling ----------------
    with tabs[7]:
        st.subheader("Feature Scaling")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            st.info("No numeric columns available for scaling.")
        else:
            cols_to_scale = st.multiselect("Column(s) to scale", num_cols)
            method = st.radio("Scaler", ["StandardScaler (mean=0, std=1)", "MinMaxScaler (0 to 1)"], horizontal=True)
            if st.button("Apply scaling", type="primary", disabled=not cols_to_scale):
                scaler_key = "scaler_" + "_".join(cols_to_scale)
                scale_method = "standard" if "Standard" in method else "minmax"
                push_history(f"Scaled {cols_to_scale} with {method}",
                             record_step("scale", columns=cols_to_scale, method=scale_method, scaler_key=scaler_key))
                new_df = df.copy()
                scaler = StandardScaler() if "Standard" in method else MinMaxScaler()
                new_df[cols_to_scale] = scaler.fit_transform(new_df[cols_to_scale])
                st.session_state.df = new_df
                st.session_state.encoders[scaler_key] = scaler
                st.success(f"Scaled: {', '.join(cols_to_scale)}")
                st.rerun()

    # ---------------- Text Cleanup ----------------
    with tabs[8]:
        st.subheader("Text Cleanup")
        text_cols = df.select_dtypes(include="object").columns.tolist()
        if not text_cols:
            st.info("No text columns detected.")
        else:
            cols_to_clean = st.multiselect("Column(s)", text_cols)
            ops = st.multiselect(
                "Operations (applied in order)",
                ["Strip whitespace", "Lowercase", "Uppercase", "Remove extra internal spaces"],
                default=["Strip whitespace"]
            )
            if st.button("Apply text cleanup", type="primary", disabled=not cols_to_clean):
                op_map = {"Strip whitespace": "strip", "Lowercase": "lower",
                          "Uppercase": "upper", "Remove extra internal spaces": "collapse_spaces"}
                step_ops = [op_map[o] for o in ops]
                push_history(f"Text cleanup {ops} on {cols_to_clean}",
                             record_step("text_cleanup", columns=cols_to_clean, ops=step_ops))
                new_df = df.copy()
                for c in cols_to_clean:
                    series = new_df[c].astype(str)
                    for op in ops:
                        if op == "Strip whitespace":
                            series = series.str.strip()
                        elif op == "Lowercase":
                            series = series.str.lower()
                        elif op == "Uppercase":
                            series = series.str.upper()
                        elif op == "Remove extra internal spaces":
                            series = series.str.replace(r"\s+", " ", regex=True)
                    new_df[c] = series
                st.session_state.df = new_df
                st.success(f"Cleaned text in: {', '.join(cols_to_clean)}")
                st.rerun()

    with tabs[9]:
        st.subheader("Dimensionality Reduction (PCA)")
        num_cols_pca = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols_pca) < 2:
            st.info("Need at least 2 numeric columns to run PCA.")
        else:
            st.caption("PCA combines correlated numeric columns into a smaller set of 'principal components' "
                       "that capture most of the original variation — useful for simplifying a model's inputs "
                       "or speeding up training when you have many numeric features.")
            cols_for_pca = st.multiselect("Numeric columns to reduce", num_cols_pca, default=num_cols_pca,
                                           key="pca_cols_select")
            if len(cols_for_pca) < 2:
                st.info("Select at least 2 numeric columns.")
            else:
                scale_pca = st.checkbox("Standardize before PCA (recommended)", value=True, key="pca_scale")
                work_pca = df[cols_for_pca].dropna()
                if len(work_pca) < len(df):
                    st.caption(f"{len(df) - len(work_pca)} row(s) with missing values in these columns "
                               f"will be excluded from the reduction.")
                max_components = max(1, min(len(cols_for_pca), work_pca.shape[0] - 1, 10))

                pca_scaler = StandardScaler() if scale_pca else None
                X_pca_input = pca_scaler.fit_transform(work_pca) if pca_scaler else work_pca.values

                if st.checkbox("Show explained variance (scree plot) to help choose the number of components",
                                key="pca_show_scree"):
                    pca_full = PCA(n_components=max_components, random_state=42)
                    pca_full.fit(X_pca_input)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    comp_idx = range(1, max_components + 1)
                    ax.bar(comp_idx, pca_full.explained_variance_ratio_, color="#3d5a80", label="Individual")
                    ax.plot(comp_idx, np.cumsum(pca_full.explained_variance_ratio_), color="#ee6c4d",
                            marker="o", label="Cumulative")
                    ax.set_xlabel("Component")
                    ax.set_ylabel("Explained variance ratio")
                    ax.set_title("PCA Explained Variance")
                    ax.legend()
                    st.pyplot(fig)
                    download_chart_button(fig, "pca_scree_plot.png", key="dl_pca_scree")
                    plt.close(fig)

                n_components = st.slider("Number of components to keep", 1, max_components,
                                          min(2, max_components), key="pca_n_components")
                drop_originals = st.checkbox("Drop original columns after reducing (recommended)", value=True,
                                              key="pca_drop_originals")

                if st.button("Apply PCA", type="primary"):
                    pca_key = f"pca_{'_'.join(cols_for_pca)[:40]}"
                    push_history(f"Applied PCA on {cols_for_pca} → {n_components} component(s)",
                                 record_step("pca", columns=cols_for_pca, n_components=n_components,
                                             drop_originals=drop_originals, pca_key=pca_key))
                    pca = PCA(n_components=n_components, random_state=42)
                    transformed = pca.fit_transform(X_pca_input)
                    comp_cols = [f"PC{i + 1}" for i in range(n_components)]
                    comp_df = pd.DataFrame(transformed, columns=comp_cols, index=work_pca.index)

                    new_df = df.copy()
                    for c in comp_cols:
                        new_df[c] = comp_df[c]
                    if drop_originals:
                        new_df = new_df.drop(columns=cols_for_pca)

                    st.session_state.encoders[pca_key] = pca
                    if pca_scaler is not None:
                        st.session_state.encoders[f"{pca_key}_scaler"] = pca_scaler

                    st.session_state.df = new_df
                    st.success(f"Added {n_components} component(s): {', '.join(comp_cols)} — together they "
                               f"explain {pca.explained_variance_ratio_.sum() * 100:.1f}% of the original variance.")
                    st.rerun()

    st.divider()
    st.caption(f"Current shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
