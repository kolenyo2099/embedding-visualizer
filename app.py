import streamlit as st
from core.state import get_app_state, reset_processed_state
from core.caching import list_saved_embeddings
from components.sidebar import render_sidebar
from components.visualization import render_visualization
from components.semantic_search import render_semantic_search
from components.data_explorer import render_data_explorer
from components.processing_summary import render_summary
from components.image_metadata_editor import render_image_metadata_editor


def render_header() -> None:
    """
    Renders the main header of the application.
    """
    st.title("🎯 Semantic Embedding Explorer")
    st.markdown(
        """
        Visualize and explore text **or image** data using state-of-the-art embeddings and interactive clustering.
        Choose your modality, upload your sources, and explore semantic relationships in your content.
        """
    )


def render_welcome_message(state) -> None:
    """
    Renders a welcome message and instructions for new users.

    Args:
        state (AppState): The application state object.
    """
    st.info("👆 Upload data in the sidebar to get started!")
    with st.expander("ℹ️ How to use this app"):
        st.markdown(
            """
            ### Step-by-step guide:

            1. **Upload Data**: Use the sidebar to provide a CSV or images
            2. **Configure Columns**: Select label and link columns where applicable
            3. **Choose Clustering**: Pick automatic HDBSCAN or fixed KMeans
            4. **Process**: Click the process button and wait for completion
            5. **Explore**:
               - Use **Semantic Search** to find similar nodes
               - View the **Visualization** to explore clusters
               - Check **Data Explorer** for statistics and downloads
            """
        )


def render_main_content(state) -> None:
    """
    Renders the main content of the application, including the data editor, summary, and tabs.

    Args:
        state (AppState): The application state object.
    """
    if state.modality == "Images" and state.image_records and not state.processed:
        render_image_metadata_editor(state)

    result = state.result
    if not state.processed or result.df is None:
        render_welcome_message(state)
        return

    render_summary(state)

    df = result.df
    required_columns = ["label", "x", "y", "cluster_label", "hover_text", "link"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"❌ Processing incomplete. Missing columns: {', '.join(missing_columns)}")
        reset_processed_state(state)
        return

    tab1, tab2, tab3 = st.tabs(["🔍 Semantic Search", "📊 Visualization", "📈 Data Explorer"])
    with tab1:
        render_semantic_search(state)
    with tab2:
        render_visualization(state)
    with tab3:
        render_data_explorer(state)


def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(page_title="Semantic Embedding Explorer", page_icon="🎯", layout="wide")
    state = get_app_state()
    saved_embeddings = list_saved_embeddings()
    render_header()
    render_sidebar(state, saved_embeddings)
    render_main_content(state)


if __name__ == "__main__":
    main()
