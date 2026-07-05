import streamlit as st
import pandas as pd
from modules.state import set_new_dataset, has_data


def _read_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        # let user tweak parsing options for tricky CSVs
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a .csv, .xlsx or .xls file.")


def render():
    st.header("📂 Upload Dataset")
    st.write("Upload a CSV or Excel file to get started. This becomes your working dataset "
             "for analysis, cleaning, statistics, and model training.")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"]
        )
    with col2:
        sep = st.text_input("CSV delimiter (optional)", value=",", help="Only used for CSV files")
        sheet_name = st.text_input("Excel sheet name (optional)", value="",
                                    help="Leave blank to use the first sheet")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file, sep=sep if sep else ",")
            else:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name if sheet_name else 0)

            if st.button("✅ Load this dataset", type="primary"):
                set_new_dataset(df, uploaded_file.name)
                st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]} rows × {df.shape[1]} columns")
                st.rerun()

            st.caption("Quick look before loading:")
            st.dataframe(df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Could not read this file: {e}")

    st.divider()

    if has_data():
        df = st.session_state.df
        st.subheader(f"Current working dataset: `{st.session_state.filename}`")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", df.shape[0])
        m2.metric("Columns", df.shape[1])
        m3.metric("Missing cells", int(df.isna().sum().sum()))
        m4.metric("Duplicate rows", int(df.duplicated().sum()))

        with st.expander("Preview data", expanded=True):
            total_rows = len(df)
            if total_rows <= 5:
                n = total_rows
                st.caption(f"Showing all {total_rows} row(s) — too few rows for a slider.")
            else:
                n = st.slider("Rows to preview", 5, min(100, total_rows), min(10, total_rows))
            st.dataframe(df.head(n), use_container_width=True)

        if st.button("🗑️ Clear dataset and start over"):
            for key in ["raw_df", "df", "filename", "history", "trained_model", "model_info"]:
                st.session_state[key] = None if key != "history" else []
            st.rerun()
    else:
        st.info("No dataset loaded yet. Upload a file above to begin.")
