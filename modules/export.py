import streamlit as st
import io
import pandas as pd


def render():
    st.header("💾 Export Cleaned Data")
    df = st.session_state.df
    raw_df = st.session_state.raw_df

    c1, c2 = st.columns(2)
    c1.metric("Original shape", f"{raw_df.shape[0]} × {raw_df.shape[1]}")
    c2.metric("Cleaned shape", f"{df.shape[0]} × {df.shape[1]}")

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Download")
    base_name = (st.session_state.filename or "dataset").rsplit(".", 1)[0]

    col1, col2 = st.columns(2)
    with col1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download as CSV",
            data=csv_bytes,
            file_name=f"{base_name}_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Cleaned Data")
        st.download_button(
            "⬇️ Download as Excel (.xlsx)",
            data=buffer.getvalue(),
            file_name=f"{base_name}_cleaned.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    if st.session_state.history:
        with st.expander("📝 Cleaning steps applied this session"):
            for i, (action, _) in enumerate(st.session_state.history, 1):
                st.write(f"{i}. {action}")
