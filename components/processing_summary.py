import streamlit as st
import pandas as pd
from core.state import AppState
from core.utils import format_time


def render_summary(state: AppState) -> None:
    """
    Renders the processing summary, including total time, data stats, and a detailed timing breakdown.

    Args:
        state (AppState): The application state object.
    """
    st.markdown("### 📊 Processing Summary")

    processing_info = state.processing_info
    data_stats = processing_info.get("data_stats", {})
    total_time = processing_info.get("total_time", 0.0)

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    if data_stats.get("modality") == "image":
        with summary_col1:
            st.metric("Total Time", format_time(total_time))
            st.metric("Images", data_stats.get("total_points", 0))
        with summary_col2:
            st.metric("Clusters", data_stats.get("clusters", 0))
            st.metric("Noise Images", data_stats.get("noise_points", 0))
        with summary_col3:
            st.metric("Embedding Dim", f"{data_stats.get('embedding_dim', 0)}D")
            st.metric("Processing Speed", f"{data_stats.get('total_points', 0) / max(total_time, 1e-6):.1f} img/sec")
    else:
        with summary_col1:
            st.metric("Total Time", format_time(total_time))
            st.metric("Data Points", data_stats.get("total_points", 0))
        with summary_col2:
            st.metric("Clusters Found", data_stats.get("clusters", 0))
            st.metric("Noise Points", data_stats.get("noise_points", 0))
        with summary_col3:
            st.metric("Embedding Dim", f"{data_stats.get('embedding_dim', 0)}D")
            st.metric("Processing Speed", f"{data_stats.get('total_points', 0) / max(total_time, 1e-6):.1f} items/sec")

    with st.expander("⏱️ Detailed Timing Breakdown"):
        processing_steps = processing_info.get("steps", [])
        timing_df = pd.DataFrame(processing_steps)
        if not timing_df.empty:
            timing_df["percentage"] = (
                timing_df["time"] / timing_df["time"].sum() * 100
            ).round(1)
            timing_df["Time (formatted)"] = timing_df["time"].apply(format_time)
            st.dataframe(
                timing_df[["step", "Time (formatted)", "percentage", "details"]]
                .rename(
                    columns={
                        "step": "Step",
                        "percentage": "Percentage",
                        "details": "Details",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
