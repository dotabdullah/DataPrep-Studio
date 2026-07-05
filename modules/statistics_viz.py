import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modules.plot_utils import download_chart_button


def render():
    st.header("📊 Statistics & Visualization")
    df = st.session_state.df
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    tabs = st.tabs(["Summary Statistics", "Distributions", "Correlation", "Categorical Breakdown"])

    with tabs[0]:
        st.subheader("Descriptive Statistics (numeric columns)")
        if num_cols:
            st.dataframe(df[num_cols].describe().T, use_container_width=True)
        else:
            st.info("No numeric columns found.")

        st.subheader("Descriptive Statistics (categorical columns)")
        if cat_cols:
            st.dataframe(df[cat_cols].describe().T, use_container_width=True)
        else:
            st.info("No categorical columns found.")

    with tabs[1]:
        st.subheader("Distribution of a Column")
        if not num_cols:
            st.info("No numeric columns available.")
        else:
            col = st.selectbox("Choose numeric column", num_cols)
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#3d5a80")
                ax.set_title(f"Histogram: {col}")
                st.pyplot(fig)
                download_chart_button(fig, f"histogram_{col}.png", key=f"dl_hist_{col}")
                plt.close(fig)
            with c2:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.boxplot(x=df[col].dropna(), ax=ax, color="#ee6c4d")
                ax.set_title(f"Boxplot: {col}")
                st.pyplot(fig)
                download_chart_button(fig, f"boxplot_{col}.png", key=f"dl_box_{col}")
                plt.close(fig)

    with tabs[2]:
        st.subheader("Correlation Heatmap")
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns to compute correlation.")
        else:
            method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True)
            corr = df[num_cols].corr(method=method)
            fig, ax = plt.subplots(figsize=(max(6, len(num_cols) * 0.7), max(5, len(num_cols) * 0.6)))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
            st.pyplot(fig)
            download_chart_button(fig, "correlation_heatmap.png", key="dl_corr_heatmap")
            plt.close(fig)

            st.subheader("Scatter Plot Between Two Columns")
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X axis", num_cols, index=0)
            y_col = c2.selectbox("Y axis", num_cols, index=min(1, len(num_cols) - 1))
            hue_col = c3.selectbox("Color by (optional)", ["None"] + cat_cols)
            fig, ax = plt.subplots(figsize=(6, 5))
            if hue_col != "None":
                sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
            else:
                sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            ax.set_title(f"{y_col} vs {x_col}")
            st.pyplot(fig)
            download_chart_button(fig, f"scatter_{x_col}_vs_{y_col}.png", key="dl_scatter")
            plt.close(fig)

    with tabs[3]:
        st.subheader("Categorical Column Breakdown")
        if not cat_cols:
            st.info("No categorical columns found.")
        else:
            col = st.selectbox("Choose categorical column", cat_cols)
            counts = df[col].value_counts().head(20)
            fig, ax = plt.subplots(figsize=(7, max(3, len(counts) * 0.35)))
            sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#98c1d9")
            ax.set_xlabel("Count")
            ax.set_title(f"Top values in '{col}'")
            st.pyplot(fig)
            download_chart_button(fig, f"category_counts_{col}.png", key=f"dl_cat_{col}")
            plt.close(fig)
            st.dataframe(counts.rename("count"), use_container_width=True)
