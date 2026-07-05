"""Shared helper so every chart in the app can offer a PNG download."""
import io
import streamlit as st


def download_chart_button(fig, filename: str, key: str, label: str = "⬇️ Download chart (PNG)"):
    """Render a download button for a matplotlib figure. Call before plt.close(fig)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.download_button(label, data=buf, file_name=filename, mime="image/png", key=key)
