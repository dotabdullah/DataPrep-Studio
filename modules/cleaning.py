import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from modules.state import push_history, undo_last


def render():
    st.header("🧹 Clean Data")
    df = st.session_state.df

    top_l, top_r = st.columns([3, 1])
    with top_r:
        if st.button("↩️ Undo last action", use_container_width=True):
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
        "Outliers", "Encoding", "Scaling", "Text Cleanup"
    ])

    # ---------------- Missing Values ----------------
    with tabs[0]:
        st.subheader("Handle Missing Values")
        cols_with_na = df.columns[df.isna().any()].tolist()
        if not cols_with_na:
            st.success("No missing values to handle! ✅")
        else:
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
                push_history(f"Missing values: {method} on {target_cols}")
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
            push_history("Removed duplicate rows")
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
                push_history(f"Dropped columns: {drop_cols}")
                st.session_state.df = df.drop(columns=drop_cols)
                st.success(f"Dropped: {', '.join(drop_cols)}")
                st.rerun()
        with c2:
            st.write("**Rename a column**")
            col_to_rename = st.selectbox("Column", df.columns.tolist(), key="rename_select")
            new_name = st.text_input("New name", value=col_to_rename)
            if st.button("Rename column", disabled=(new_name.strip() == "" or new_name == col_to_rename)):
                push_history(f"Renamed '{col_to_rename}' to '{new_name}'")
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
            push_history(f"Converted '{col}' to {new_type}")
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
                push_history(f"Outlier treatment on '{col}' ({action})")
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
                push_history(f"Encoded '{col}' with {method}")
                new_df = df.copy()
                if method == "Label Encoding":
                    le = LabelEncoder()
                    new_df[col] = le.fit_transform(new_df[col].astype(str))
                    st.session_state.encoders[col] = le
                else:
                    dummies = pd.get_dummies(new_df[col], prefix=col, dtype=int)
                    new_df = pd.concat([new_df.drop(columns=[col]), dummies], axis=1)
                st.session_state.df = new_df
                st.success(f"Encoded '{col}' using {method}.")
                st.rerun()

    # ---------------- Scaling ----------------
    with tabs[6]:
        st.subheader("Feature Scaling")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            st.info("No numeric columns available for scaling.")
        else:
            cols_to_scale = st.multiselect("Column(s) to scale", num_cols)
            method = st.radio("Scaler", ["StandardScaler (mean=0, std=1)", "MinMaxScaler (0 to 1)"], horizontal=True)
            if st.button("Apply scaling", type="primary", disabled=not cols_to_scale):
                push_history(f"Scaled {cols_to_scale} with {method}")
                new_df = df.copy()
                scaler = StandardScaler() if "Standard" in method else MinMaxScaler()
                new_df[cols_to_scale] = scaler.fit_transform(new_df[cols_to_scale])
                st.session_state.df = new_df
                st.session_state.encoders["scaler_" + "_".join(cols_to_scale)] = scaler
                st.success(f"Scaled: {', '.join(cols_to_scale)}")
                st.rerun()

    # ---------------- Text Cleanup ----------------
    with tabs[7]:
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
                push_history(f"Text cleanup {ops} on {cols_to_clean}")
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

    st.divider()
    st.caption(f"Current shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
