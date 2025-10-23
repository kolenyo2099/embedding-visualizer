import streamlit as st
import numpy as np
import time
from core.state import AppState, SearchState
from core.embeddings import load_siglip_model, encode_text_with_siglip, encode_images_with_siglip
from core.utils import create_image_record


def render_semantic_search(state: AppState) -> None:
    """
    Renders the semantic search UI and handles the search logic.

    Args:
        state (AppState): The application state object.
    """
    result = state.result
    if result.df is None or result.embeddings is None:
        st.info("Run processing to enable semantic search.")
        return

    df = result.df
    data_stats = state.processing_info.get("data_stats", {})
    current_modality = result.modality or data_stats.get("modality", "text")

    st.header("🔍 Semantic Search")
    st.markdown("Search for nodes semantically similar to your query.")

    if current_modality == "image":
        col1, col2 = st.columns([3, 1])
        with col2:
            top_k = st.number_input(
                "Top Results",
                min_value=1,
                max_value=max(1, len(df)),
                value=min(state.search_state.top_k, len(df)) if state.search_state.top_k <= len(df) else min(12, len(df)),
                help="Number of most similar images to highlight",
            )
        with col1:
            query_mode = st.radio(
                "Query Type",
                options=["Text", "Image"],
                horizontal=True,
                index=(0 if state.search_state.mode != "Image" else 1),
                help="Search using a descriptive caption or by uploading a query image",
            )
            if query_mode == "Text":
                search_query = st.text_input(
                    "Describe the image you're searching for",
                    placeholder="e.g., 'sunset over mountains'",
                    value=state.search_state.query if state.search_state.mode == "Text" else "",
                )
                query_image_bytes = None
            else:
                query_file = st.file_uploader(
                    "Upload query image",
                    type=[ext.replace(".", "") for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]],
                    key="image_query_upload",
                )
                search_query = ""
                query_image_bytes = query_file.read() if query_file else None
        trigger_search = st.button("🔎 Search", type="primary")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "Search Query",
                placeholder="e.g., 'economic impact', 'policy response'...",
                value=state.search_state.query if state.search_state.mode == "Text" else "",
                help="Enter a search query to find semantically similar documents",
            )
        with col2:
            top_k = st.number_input(
                "Top Results",
                min_value=5,
                max_value=100,
                value=state.search_state.top_k if state.search_state.top_k else 20,
                help="Number of most similar documents to highlight",
            )
        query_mode = "Text"
        query_image_bytes = None
        trigger_search = st.button("🔎 Search", type="primary")

    if trigger_search:
        if current_modality == "image" and query_mode == "Image" and not query_image_bytes:
            st.warning("Please upload a query image before searching.")
        elif query_mode == "Text" and not search_query:
            st.warning("Please enter a search query.")
        else:
            search_placeholder = st.empty()
            with search_placeholder.container():
                st.markdown("### 🔍 Searching for Similar Content")
                search_progress = st.progress(0)
                search_status = st.empty()
                search_details = st.empty()
                search_start = time.time()

                device = "cuda" if torch.cuda.is_available() else (
                    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
                )

                if current_modality == "image":
                    search_status.text("🤖 Loading SigLIP encoder...")
                    model, processor = load_siglip_model(
                        model_name="google/siglip-base-patch16-512",
                        device=device,
                        token=state.hf_token or None,
                    )
                    search_progress.progress(15)

                    if query_mode == "Text":
                        search_status.text("🔤 Encoding text query...")
                        search_details.text(f"Caption: '{search_query}'")
                        query_embedding = encode_text_with_siglip([search_query], model, processor, device)[0]
                    else:
                        search_status.text("🖼️ Encoding query image...")
                        record = create_image_record("query", data_bytes=query_image_bytes, label="query-image")
                        query_embedding_arr, _ = encode_images_with_siglip(
                            [record], model, processor, device, batch_size=1
                        )
                        query_embedding = query_embedding_arr[0]
                else:
                    search_status.text("🔤 Encoding search query...")
                    search_details.text(f"Query: '{search_query}'")
                    search_progress.progress(25)
                    if state.model is None:
                        st.error("Embedding model is not available. Please reprocess the dataset.")
                        return
                    query_embedding = state.model.encode([search_query], normalize_embeddings=True)[0]

                search_status.text("📊 Calculating similarities...")
                search_details.text(f"Comparing with {len(result.embeddings)} items")
                search_progress.progress(60)
                similarities = np.dot(result.embeddings, query_embedding)

                search_status.text("🎯 Ranking results...")
                search_details.text(f"Finding top {top_k} matches")
                search_progress.progress(80)

                top_indices = np.argsort(similarities)[-int(top_k):][::-1]
                top_scores = similarities[top_indices]
                results_df = df.iloc[top_indices].copy()
                results_df["similarity"] = top_scores
                results_df["rank"] = range(1, len(results_df) + 1)

                search_time = time.time() - search_start
                avg_similarity = float(np.mean(top_scores)) if len(top_scores) else 0.0
                similarity_range = (
                    f"{top_scores.min():.3f} - {top_scores.max():.3f}" if len(top_scores) else "N/A"
                )

                search_progress.progress(100)
                search_status.text("✅ Search complete!")
                search_details.text(f"Found {len(results_df)} results in {search_time:.2f}s")
                time.sleep(0.5)
                search_placeholder.empty()

            state.search_state = SearchState(
                query=search_query,
                top_k=int(top_k),
                mode=query_mode,
                results_df=results_df,
                similarity_threshold=0.0,
                query_image_bytes=query_image_bytes,
                search_time=search_time,
                avg_similarity=avg_similarity,
                similarity_range=similarity_range,
            )

    results_df = state.search_state.results_df
    if results_df is not None and not results_df.empty:
        st.markdown("### 📊 Search Results Summary")
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

        if current_modality == "image" and state.search_state.mode == "Image":
            with summary_col1:
                st.metric("Query Type", "Image Upload")
        else:
            with summary_col1:
                st.metric("Query Length", len(state.search_state.query))
        with summary_col2:
            st.metric("Search Time", f"{state.search_state.search_time:.2f}s")
        with summary_col3:
            st.metric("Avg Similarity", f"{state.search_state.avg_similarity:.3f}")
        with summary_col4:
            st.metric("Similarity Range", state.search_state.similarity_range or "N/A")

        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=state.search_state.similarity_threshold,
            step=0.01,
            help="Filter results by minimum similarity score",
        )
        state.search_state.similarity_threshold = threshold
        filtered_results = results_df[results_df["similarity"] >= threshold].copy()

        if filtered_results.empty:
            st.info("No results above the selected threshold.")
        else:
            if current_modality == "image":
                st.subheader(f"🎯 Top {len(filtered_results)} Image Results")
                cols = st.columns(min(4, len(filtered_results)))
                for idx, (_, row) in enumerate(filtered_results.iterrows()):
                    col = cols[idx % len(cols)]
                    with col:
                        if row.get("thumbnail"):
                            col.image(
                                f"data:image/png;base64,{row['thumbnail']}",
                                caption=f"#{int(row['rank'])} · {row.get('label', 'Image')}"[:60],
                                use_column_width=True,
                            )
                        else:
                            col.markdown(f"**#{int(row['rank'])} · {row.get('label', 'Image')}**")
                        col.caption(f"Similarity: {row['similarity']:.3f}")
            else:
                st.subheader(f"🎯 Top {len(filtered_results)} Results")
                st.dataframe(
                    filtered_results[[
                        "rank",
                        "label",
                        state.result.text_column if state.result.text_column in filtered_results.columns else "hover_text",
                        "similarity",
                        "cluster_label",
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.info("Enter a query and click search to view results.")
