import streamlit as st
from modules.state import init_state, has_data
from modules import data_upload, analysis, cleaning, statistics_viz, modeling, export

st.set_page_config(
    page_title="DataPrep Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

PAGES = {
    "📂 Upload & Preview": data_upload,
    "🔍 Analyze Dataset": analysis,
    "🧹 Clean Data": cleaning,
    "📊 Statistics & Visualization": statistics_viz,
    "🤖 Model Training": modeling,
    "💾 Export Clean Data": export,
}

with st.sidebar:
    st.title("🧪 DataPrep Studio")
    st.caption("Clean your data. Explore it. Train a model. Export the result.")
    st.divider()

    page_name = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

    st.divider()
    if has_data():
        df = st.session_state.df
        st.success(f"**{st.session_state.filename}**")
        st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
    else:
        st.info("No dataset loaded yet.")

    st.divider()
    st.caption("Built with Streamlit, pandas, scikit-learn & matplotlib.")

# Route to the selected page
page = PAGES[page_name]

if page_name != "📂 Upload & Preview" and not has_data():
    st.warning("⚠️ Please upload a dataset first on the **Upload & Preview** page.")
    st.stop()

page.render()
