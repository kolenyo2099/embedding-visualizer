import streamlit as st
import pandas as pd
import numpy as np
import json
from core.state import AppState
from core.utils import format_memory


def render_data_explorer(state: AppState) -> None:
    """
    Renders the data explorer UI, including cluster analysis, embedding statistics, and raw data preview.

    Args:
        state (AppState): The application state object.
    """
    result = state.result
    if result.df is None:
        st.info("No processed data available yet.")
        return

    df = result.df
    st.header("📈 Data Explorer")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Points", len(df))
    with col2:
        st.metric("Clusters", df["cluster"].nunique())
    with col3:
        st.metric("Noise Points", int((df["cluster"] == -1).sum()))

    if "thumbnail" in df.columns:
        st.subheader("🖼️ Image Preview")
        preview_count = min(len(df), 12)
        cols = st.columns(min(6, preview_count))
        for i, (_, row) in enumerate(df.head(preview_count).iterrows()):
            col = cols[i % len(cols)]
            with col:
                if row.get("thumbnail"):
                    col.image(
                        f"data:image/png;base64,{row['thumbnail']}",
                        caption=row.get("label", f"Item {i}")[:40],
                        use_column_width=True,
                    )
                else:
                    col.markdown(f"**{row.get('label', f'Item {i}')}**")
                    col.caption(row.get("hover_text", "")[:80])

    st.subheader("🎯 Cluster Analysis")
    cluster_stats = df.groupby("cluster_label").size().reset_index(name="count")
    cluster_stats = cluster_stats.sort_values("count", ascending=False)
    cluster_stats["percentage"] = (cluster_stats["count"] / len(df) * 100).round(1)
    st.dataframe(cluster_stats, use_container_width=True, hide_index=True)

    if result.embeddings is not None:
        st.subheader("🔢 Embedding Statistics")
        embeddings = result.embeddings
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Embedding Shape", f"{embeddings.shape[0]}×{embeddings.shape[1]}")
        with col2:
            st.metric("Mean Norm", f"{np.mean(np.linalg.norm(embeddings, axis=1)):.3f}")
        with col3:
            st.metric("Std Dev", f"{np.std(embeddings):.3f}")
        with col4:
            st.metric("Memory Usage", format_memory(embeddings.nbytes / (1024 * 1024)))

    st.subheader("📋 Raw Data Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("📥 Download Results")
    col1, col2 = st.columns(2)
    with col1:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📊 Download Processed Data (CSV)",
            data=csv_data,
            file_name="processed_embeddings.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        summary_data = {
            "processing_info": state.processing_info,
            "data_stats": {
                "total_points": len(df),
                "clusters": df["cluster"].nunique(),
                "noise_points": int((df["cluster"] == -1).sum()),
                "columns": list(df.columns),
            },
        }
        json_data = json.dumps(summary_data, indent=2, default=str).encode("utf-8")
        st.download_button(
            label="📄 Download Summary (JSON)",
            data=json_data,
            file_name="processing_summary.json",
            mime="application/json",
            use_container_width=True,
        )
