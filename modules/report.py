"""Generates a single self-contained HTML data profiling report (stats tables + embedded charts)."""
import base64
import io
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _df_to_html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="tbl", border=0)


def generate_html_report(df: pd.DataFrame, filename: str) -> str:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- Overview table ---
    overview_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Missing": df.isna().sum().values,
        "Missing %": (df.isna().mean() * 100).round(1).values,
        "Unique": [df[c].nunique() for c in df.columns],
    })

    sections = []
    sections.append(f"""
    <h1>Data Profiling Report</h1>
    <p class="meta">File: <b>{filename}</b> &nbsp;|&nbsp; Generated: {generated_at} &nbsp;|&nbsp;
    Shape: <b>{df.shape[0]} rows × {df.shape[1]} columns</b></p>
    """)

    sections.append("<h2>Column Overview</h2>" + _df_to_html_table(overview_df))

    dup_count = int(df.duplicated().sum())
    sections.append(f"<h2>Duplicate Rows</h2><p>{dup_count} duplicate row(s) found.</p>")

    # --- Missing values chart ---
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(7, max(2, len(missing) * 0.35)))
        ax.barh(missing.index.astype(str), missing.values, color="#e07a5f")
        ax.invert_yaxis()
        ax.set_xlabel("Missing count")
        ax.set_title("Missing Values by Column")
        img = _fig_to_base64(fig)
        sections.append(f'<h2>Missing Values</h2><img src="data:image/png;base64,{img}"/>')

    # --- Descriptive statistics ---
    if num_cols:
        sections.append("<h2>Numeric Summary Statistics</h2>" +
                         _df_to_html_table(df[num_cols].describe().T.reset_index().rename(columns={"index": "Column"})))
    if cat_cols:
        sections.append("<h2>Categorical Summary Statistics</h2>" +
                         _df_to_html_table(df[cat_cols].describe().T.reset_index().rename(columns={"index": "Column"})))

    # --- Correlation heatmap ---
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(max(6, len(num_cols) * 0.7), max(5, len(num_cols) * 0.6)))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
        img = _fig_to_base64(fig)
        sections.append(f'<h2>Correlation Heatmap</h2><img src="data:image/png;base64,{img}"/>')

    # --- Histograms for numeric columns (cap to keep report a reasonable size) ---
    if num_cols:
        cols_to_plot = num_cols[:12]
        n = len(cols_to_plot)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for i, col in enumerate(cols_to_plot):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#3d5a80")
            axes[i].set_title(col)
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.tight_layout()
        img = _fig_to_base64(fig)
        note = f" (showing first {len(cols_to_plot)} of {len(num_cols)})" if len(num_cols) > 12 else ""
        sections.append(f'<h2>Distributions{note}</h2><img src="data:image/png;base64,{img}"/>')

    # --- Top categorical breakdowns ---
    if cat_cols:
        cols_to_plot = cat_cols[:6]
        for col in cols_to_plot:
            counts = df[col].value_counts().head(10)
            if counts.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, max(2.5, len(counts) * 0.35)))
            sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax, color="#98c1d9")
            ax.set_title(f"Top values: {col}")
            ax.set_xlabel("Count")
            img = _fig_to_base64(fig)
            sections.append(f'<h3>{col}</h3><img src="data:image/png;base64,{img}"/>')

    body = "\n".join(sections)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Data Profiling Report — {filename}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px auto; max-width: 960px;
         color: #1d2129; line-height: 1.5; }}
  h1 {{ font-size: 28px; margin-bottom: 4px; }}
  h2 {{ font-size: 20px; margin-top: 40px; border-bottom: 2px solid #3d5a80; padding-bottom: 6px; }}
  h3 {{ font-size: 16px; margin-top: 24px; }}
  .meta {{ color: #555; font-size: 14px; }}
  table.tbl {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }}
  table.tbl th {{ background: #3d5a80; color: white; text-align: left; padding: 6px 10px; }}
  table.tbl td {{ padding: 6px 10px; border-bottom: 1px solid #eee; }}
  table.tbl tr:nth-child(even) {{ background: #f7f9fb; }}
  img {{ max-width: 100%; margin-top: 12px; border: 1px solid #eee; border-radius: 4px; }}
</style>
</head>
<body>
{body}
</body>
</html>"""
    return html
