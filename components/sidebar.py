from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd
import hashlib
import os
import zipfile
from io import BytesIO
from core.state import AppState, reset_processed_state
from core.utils import create_image_record, _clean_metadata_value


@dataclass
class TextProcessingConfig:
    """
    Configuration for text data processing.
    """
    model_name: str
    text_column: str
    label_column: str
    link_column: Optional[str]
    clustering_method: str
    n_clusters: Optional[int]
    min_cluster_size: Optional[int]
    use_cached: bool
    selected_cache: Optional[Dict[str, Any]]
    save_embeddings: bool


@dataclass
class ImageProcessingConfig:
    """
    Configuration for image data processing.
    """
    clustering_method: str
    n_clusters: Optional[int]
    min_cluster_size: Optional[int]
    use_cached: bool
    selected_cache: Optional[Dict[str, Any]]
    save_embeddings: bool


def render_sidebar(state: AppState, saved_embeddings: List[Dict[str, Any]]) -> None:
    """
    Renders the sidebar UI, including data modality selection and processing controls.

    Args:
        state (AppState): The application state object.
        saved_embeddings (List[Dict[str, Any]]): A list of saved embeddings.
    """
    st.sidebar.header("⚙️ Configuration")
    render_system_info()
    state.hf_token = st.sidebar.text_input(
        "Hugging Face Token",
        value=state.hf_token,
        type="password",
        help="Optional token required for some gated models",
    )
    render_modality_selector(state)

    if state.modality == "Text (CSV)":
        config, should_process = text_processing_sidebar(state, saved_embeddings)
        if should_process and config:
            from core.processing import process_text_data
            process_text_data(state, config)
    else:
        config, should_process = image_processing_sidebar(state, saved_embeddings)
        if should_process and config:
            from core.processing import process_image_data
            process_image_data(state, config)


def render_system_info() -> None:
    """
    Displays system information (CPU, RAM, device) in the sidebar.
    """
    try:
        import psutil
        import torch
        memory = psutil.virtual_memory()
        if torch.backends.mps.is_available():
            device = "MPS (Apple Silicon)"
        elif torch.cuda.is_available():
            device = "CUDA"
        else:
            device = "CPU"
        st.sidebar.caption(
            f"**System:** {psutil.cpu_count()} cores · RAM {memory.total / (1024 ** 3):.1f}GB · "
            f"Available {memory.available / (1024 ** 3):.1f}GB · {device}"
        )
    except ImportError:
        pass


def render_modality_selector(state: AppState) -> None:
    """
    Renders the radio button to select the data modality (Text or Images).

    Args:
        state (AppState): The application state object.
    """
    modality_options = ["Text (CSV)", "Images"]
    selected_modality = st.sidebar.radio(
        "Data Modality",
        modality_options,
        index=modality_options.index(state.modality),
        help="Switch between processing text data from CSV files or image collections",
    )
    if selected_modality != state.modality:
        state.modality = selected_modality
        reset_processed_state(state)


def text_processing_sidebar(state: AppState, saved_embeddings: List[Dict[str, Any]]):
    """
    Renders the sidebar controls for text data processing.

    Args:
        state (AppState): The application state object.
        saved_embeddings (List[Dict[str, Any]]): A list of saved embeddings.

    Returns:
        Tuple[Optional[TextProcessingConfig], bool]: The processing configuration and a boolean indicating whether to start processing.
    """
    handle_text_upload(state)
    if state.raw_df is None or state.raw_df.empty:
        st.sidebar.info("👆 Upload a CSV file to configure processing")
        return None, False

    df = state.raw_df
    st.sidebar.subheader("📊 Dataset Preview")
    st.sidebar.caption(f"Rows: {len(df)} · Columns: {len(df.columns)}")

    text_embeddings = [emb for emb in saved_embeddings if emb.get("metadata", {}).get("modality", "text") == "text"]

    with st.sidebar.form("text_processing_form"):
        text_column = st.selectbox(
            "Text Column",
            options=df.columns.tolist(),
            index=df.columns.get_loc(state.result.text_column) if state.result.text_column in df.columns else 0,
            help="Select the column containing the text to embed",
        )

        label_options = ["Index"] + df.columns.tolist()
        label_default = state.processing_info.get("label_column", "Index")
        label_index = label_options.index(label_default) if label_default in label_options else 0
        label_column = st.selectbox(
            "Label Column",
            options=label_options,
            index=label_index,
            help="Labels are shown in the visualization and search results",
        )

        link_options = [None] + df.columns.tolist()
        link_default = state.processing_info.get("link_column")
        link_index = link_options.index(link_default) if link_default in link_options else 0
        link_column = st.selectbox(
            "Link Column (optional)",
            options=link_options,
            index=link_index,
            help="Optional column containing URLs for each row",
        )

        clustering_method = st.radio(
            "Clustering Method",
            options=["HDBSCAN (Automatic)", "KMeans (Fixed)"],
            help="Choose how to group similar items",
        )

        if clustering_method == "KMeans (Fixed)":
            n_clusters = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=min(30, len(df)),
                value=min(10, len(df)),
                help="Specify the number of clusters",
            )
            min_cluster_size = None
        else:
            min_cluster_size = st.slider(
                "Minimum Cluster Size",
                min_value=5,
                max_value=max(5, min(200, len(df))),
                value=min(30, len(df)),
                help="Minimum number of items to form a cluster",
            )
            n_clusters = None

        use_cached = st.checkbox(
            "📁 Use Cached Embeddings",
            value=False,
            help="Reuse previously computed embeddings from disk",
        )

        selected_cache = None
        if use_cached and text_embeddings:
            cache_labels = [
                f"{emb['model_name']} | {emb['timestamp']} | {emb['num_texts']} items"
                for emb in text_embeddings
            ]
            cache_index = st.selectbox(
                "Select Cached Embedding",
                options=list(range(len(cache_labels))),
                format_func=lambda idx: cache_labels[idx],
            )
            selected_cache = text_embeddings[cache_index]
        elif use_cached and not text_embeddings:
            st.warning("⚠️ No cached embeddings found")

        model_name = st.selectbox(
            "Embedding Model",
            options=[
                "google/embeddinggemma-300m",
                "nomic-ai/nomic-embed-text-v1.5",
                "BAAI/bge-base-en-v1.5",
                "sentence-transformers/all-mpnet-base-v2",
            ],
            help="Choose the embedding model",
        )

        save_embeddings_flag = st.checkbox(
            "💾 Save Embeddings for Later",
            value=True,
            help="Persist computed embeddings to disk",
        )

        process_requested = st.form_submit_button("🚀 Process Data", type="primary", use_container_width=True)

    if not process_requested:
        return None, False

    config = TextProcessingConfig(
        model_name=model_name,
        text_column=text_column,
        label_column=label_column,
        link_column=link_column,
        clustering_method=clustering_method,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        use_cached=use_cached,
        selected_cache=selected_cache,
        save_embeddings=save_embeddings_flag,
    )
    return config, True


def image_processing_sidebar(state: AppState, saved_embeddings: List[Dict[str, Any]]):
    """
    Renders the sidebar controls for image data processing.

    Args:
        state (AppState): The application state object.
        saved_embeddings (List[Dict[str, Any]]): A list of saved embeddings.

    Returns:
        Tuple[Optional[ImageProcessingConfig], bool]: The processing configuration and a boolean indicating whether to start processing.
    """
    image_source_sidebar(state)

    if not state.image_records:
        st.sidebar.info("👆 Upload images or provide a CSV with image URLs to get started!")
        return None, False

    st.sidebar.info(f"📦 {len(state.image_records)} images ready for embedding")

    st.sidebar.subheader("🎨 Clustering Settings")
    clustering_method = st.sidebar.radio(
        "Clustering Method",
        options=["HDBSCAN (Automatic)", "KMeans (Fixed)"],
        help="Choose how to group images",
    )

    if clustering_method == "KMeans (Fixed)":
        n_clusters = st.sidebar.slider(
            "Number of Clusters",
            min_value=2,
            max_value=min(30, len(state.image_records)),
            value=min(10, len(state.image_records)),
            help="Fixed number of clusters",
        )
        min_cluster_size = None
    else:
        min_cluster_size = st.sidebar.slider(
            "Minimum Cluster Size",
            min_value=5,
            max_value=max(5, min(200, len(state.image_records))),
            value=min(25, len(state.image_records)),
            help="Minimum number of images required to form a cluster",
        )
        n_clusters = None

    image_saved_embeddings = [
        emb for emb in saved_embeddings if emb.get("metadata", {}).get("modality") == "image"
    ]

    use_cached = st.sidebar.checkbox(
        "📁 Use Cached Embeddings",
        value=False,
        help="Reuse previously computed image embeddings",
    )

    selected_cache = None
    if use_cached and image_saved_embeddings:
        cache_labels = []
        for emb in image_saved_embeddings:
            meta = emb.get("metadata", {})
            preprocess = meta.get("preprocessing", {}).get("description", "SigLIP")
            cache_labels.append(
                f"{emb['model_name']} | {preprocess} | {emb['timestamp']} ({emb['num_texts']} images)"
            )
        cache_index = st.sidebar.selectbox(
            "Select Cached Embedding",
            options=list(range(len(cache_labels))),
            format_func=lambda idx: cache_labels[idx],
        )
        selected_cache = image_saved_embeddings[cache_index]
    elif use_cached and not image_saved_embeddings:
        st.sidebar.warning("⚠️ No cached image embeddings found")

    save_embeddings_flag = st.sidebar.checkbox(
        "💾 Save Image Embeddings for Later",
        value=True,
        help="Persist computed embeddings to disk",
    )

    process_requested = st.sidebar.button("🚀 Process Images", type="primary", use_container_width=True)

    if not process_requested:
        return None, False

    config = ImageProcessingConfig(
        clustering_method=clustering_method,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        use_cached=use_cached,
        selected_cache=selected_cache,
        save_embeddings=save_embeddings_flag,
    )
    return config, True


def handle_text_upload(state: AppState) -> None:
    """
    Handles the CSV file upload and updates the application state.

    Args:
        state (AppState): The application state object.
    """
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"], key="text_csv_uploader")
    if uploaded_file is None:
        if state.raw_df is None:
            return
    else:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        if state.uploaded_file_hash != file_hash:
            try:
                df = pd.read_csv(BytesIO(file_bytes))
                state.raw_df = df
                state.uploaded_file_hash = file_hash
                reset_processed_state(state, clear_model=False)
                st.sidebar.success(f"✅ Loaded dataset with {len(df)} rows")
            except Exception as exc:
                st.sidebar.error(f"Failed to parse CSV: {exc}")
                return


def image_source_sidebar(state: AppState) -> None:
    """
    Renders the UI for selecting the image data source (upload, ZIP, or CSV).

    Args:
        state (AppState): The application state object.
    """
    st.sidebar.subheader("🖼️ Image Source")
    image_source = st.sidebar.radio(
        "Choose how to provide images",
        options=["Upload Images", "Upload Folder (ZIP)", "CSV with Image URLs"],
        help="Supply individual image files, a compressed folder, or a CSV that points to image URLs",
    )

    image_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    if image_source == "Upload Images":
        uploaded_images = st.sidebar.file_uploader(
            "Upload image files",
            type=[ext.replace(".", "") for ext in image_extensions],
            accept_multiple_files=True,
            key="image_file_uploader",
            help="Select one or more images to embed",
        )
        if uploaded_images:
            source_id = tuple((img.name, img.size) for img in uploaded_images)
            if state.image_source_id != source_id:
                records = []
                for file in uploaded_images:
                    data = file.read()
                    if not data:
                        continue
                    label = os.path.splitext(file.name)[0]
                    records.append(create_image_record(file.name, data_bytes=data, label=label))
                set_image_records(
                    state,
                    records,
                    source_id,
                    {"source": "uploaded_files", "count": len(records)},
                )

    elif image_source == "Upload Folder (ZIP)":
        uploaded_zip = st.sidebar.file_uploader(
            "Upload ZIP archive",
            type=["zip"],
            key="image_zip_uploader",
            help="Provide a compressed folder containing images",
        )
        if uploaded_zip is not None:
            source_id = (uploaded_zip.name, uploaded_zip.size)
            if state.image_source_id != source_id:
                records: List[Dict[str, Any]] = []
                try:
                    with zipfile.ZipFile(uploaded_zip) as zf:
                        for member in zf.namelist():
                            if not member.lower().endswith(image_extensions):
                                continue
                            base = os.path.basename(member)
                            if not base:
                                continue
                            with zf.open(member) as f:
                                data = f.read()
                            label = os.path.splitext(base)[0]
                            records.append(create_image_record(base, data_bytes=data, label=label))
                except Exception as exc:
                    st.sidebar.error(f"Failed to extract ZIP: {exc}")
                    records = []
                if records:
                    set_image_records(
                        state,
                        records,
                        source_id,
                        {"source": "zip_archive", "count": len(records)},
                    )

    else:
        url_csv = st.sidebar.file_uploader(
            "Upload CSV with image URLs",
            type=["csv"],
            key="image_url_csv",
            help="CSV should include at least one column with image URLs",
        )
        if url_csv is not None:
            try:
                url_df = pd.read_csv(url_csv)
            except Exception as exc:
                st.sidebar.error(f"Failed to parse CSV: {exc}")
                return
            if url_df.empty:
                st.sidebar.warning("CSV appears to be empty")
            else:
                url_column = st.sidebar.selectbox("Image URL Column", options=url_df.columns.tolist())
                label_column_choice = st.sidebar.selectbox(
                    "Label Column (optional)",
                    options=[None] + url_df.columns.tolist(),
                )
                link_column_choice = st.sidebar.selectbox(
                    "Link Column (optional)",
                    options=[None] + url_df.columns.tolist(),
                )
                caption_column_choice = st.sidebar.selectbox(
                    "Caption Column (optional)",
                    options=[None] + url_df.columns.tolist(),
                )
                limit = st.sidebar.number_input(
                    "Max rows to load",
                    min_value=1,
                    max_value=len(url_df),
                    value=min(len(url_df), 500),
                    help="Limit number of images to fetch",
                )

                records: List[Dict[str, Any]] = []
                for idx, row in url_df.head(limit).iterrows():
                    url_value = row.get(url_column)
                    url = _clean_metadata_value(url_value, default="", allow_empty=False)
                    if not url:
                        continue
                    default_label = f"Image {idx + 1}"
                    if label_column_choice:
                        label = _clean_metadata_value(
                            row.get(label_column_choice),
                            default=default_label,
                            allow_empty=False,
                        )
                    else:
                        label = default_label
                    if caption_column_choice:
                        caption = _clean_metadata_value(
                            row.get(caption_column_choice), default="", allow_empty=True
                        )
                    else:
                        caption = ""
                    if link_column_choice:
                        link_value = _clean_metadata_value(
                            row.get(link_column_choice), default="#", allow_empty=False
                        )
                    else:
                        link_value = "#"
                    records.append(
                        create_image_record(
                            identifier=f"row_{idx}",
                            label=label,
                            link=link_value or "#",
                            caption=caption,
                            url=url,
                        )
                    )
                if records:
                    signature = (
                        url_csv.name,
                        url_csv.size,
                        url_column,
                        label_column_choice,
                        link_column_choice,
                        caption_column_choice,
                        limit,
                    )
                    set_image_records(
                        state,
                        records,
                        signature,
                        {
                            "source": "url_csv",
                            "count": len(records),
                            "url_column": url_column,
                        },
                    )
                else:
                    st.sidebar.warning("No valid URLs found in the selected column")


def set_image_records(state: AppState, records: List[Dict[str, Any]], source_id: Any, metadata: Dict[str, Any]) -> None:
    """
    Updates the application state with the loaded image records.

    Args:
        state (AppState): The application state object.
        records (List[Dict[str, Any]]): A list of image records.
        source_id (Any): A unique identifier for the image source.
        metadata (Dict[str, Any]): Additional metadata about the image source.
    """
    if not records:
        return
    state.image_records = records
    state.image_source_id = source_id
    state.image_metadata = metadata
    reset_processed_state(state)
    st.sidebar.success(f"✅ Loaded {len(records)} images")
