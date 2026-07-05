import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules.plot_utils import download_chart_button


def render():
    st.header("🔍 Analyze Dataset")
    df = st.session_state.df

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Missing Values", "Duplicates", "Column Details"]
    )

    with tab1:
        st.subheader("Shape & Data Types")
        c1, c2 = st.columns(2)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])

        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype": df.dtypes.astype(str).values,
            "Non-Null Count": df.notna().sum().values,
            "Null Count": df.isna().sum().values,
            "Unique Values": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        st.write(f"**Numeric columns ({len(num_cols)}):** {', '.join(num_cols) if num_cols else 'None'}")
        st.write(f"**Categorical / text columns ({len(cat_cols)}):** {', '.join(cat_cols) if cat_cols else 'None'}")

    with tab2:
        st.subheader("Missing Values")
        missing = df.isna().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        miss_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Count": missing.values,
            "Missing %": missing_pct.values
        }).sort_values("Missing Count", ascending=False)

        st.dataframe(miss_df, use_container_width=True, hide_index=True)

        cols_with_missing = miss_df[miss_df["Missing Count"] > 0]
        if not cols_with_missing.empty:
            fig, ax = plt.subplots(figsize=(8, max(2, len(cols_with_missing) * 0.4)))
            ax.barh(cols_with_missing["Column"], cols_with_missing["Missing %"], color="#e07a5f")
            ax.set_xlabel("Missing %")
            ax.set_title("Missing Values by Column")
            ax.invert_yaxis()
            st.pyplot(fig)
            download_chart_button(fig, "missing_values.png", key="dl_missing_values")
            plt.close(fig)
        else:
            st.success("No missing values in the dataset! ✅")

    with tab3:
        st.subheader("Duplicate Rows")
        dup_count = df.duplicated().sum()
        st.metric("Duplicate rows found", int(dup_count))
        if dup_count > 0:
            st.write("Preview of duplicated rows:")
            st.dataframe(df[df.duplicated(keep=False)].sort_values(by=df.columns[0]),
                         use_container_width=True)
            st.caption("Go to the **Clean Data** page to remove duplicates.")
        else:
            st.success("No duplicate rows found! ✅")

    with tab4:
        st.subheader("Column Details")
        col = st.selectbox("Select a column to inspect", df.columns)
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Dtype:** {df[col].dtype}")
            st.write(f"**Unique values:** {df[col].nunique()}")
            st.write(f"**Missing values:** {df[col].isna().sum()}")
            if pd.api.types.is_numeric_dtype(df[col]):
                st.write(f"**Min:** {df[col].min()}")
                st.write(f"**Max:** {df[col].max()}")
                st.write(f"**Mean:** {df[col].mean():.4f}")
                st.write(f"**Std Dev:** {df[col].std():.4f}")
        with c2:
            st.write("**Top value counts:**")
            st.dataframe(df[col].value_counts().head(10).rename("count"), use_container_width=True)
