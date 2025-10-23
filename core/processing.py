from typing import List, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from core.state import AppState, ProcessingResult, SearchState
from core.caching import save_embeddings, load_embeddings
from core.embeddings import reduce_dimensionality, encode_images_with_siglip, load_siglip_model
from core.clustering import perform_clustering
from core.utils import format_time, create_thumbnail_base64, load_pil_image, record_signature
from components.sidebar import TextProcessingConfig, ImageProcessingConfig


def process_text_data(state: AppState, config: TextProcessingConfig) -> None:
    """
    Orchestrates the entire text data processing pipeline, from loading data to generating visualizations.

    Args:
        state (AppState): The application state object.
        config (TextProcessingConfig): The configuration for the text processing pipeline.
    """
    if state.raw_df is None:
        st.error("No data available for processing.")
        return

    df = state.raw_df.copy()
    processing_placeholder = st.empty()

    with processing_placeholder.container():
        st.markdown("### 🔄 Processing Data")
        start_time = time.time()
        processing_steps: List[Dict[str, Any]] = []

        with st.spinner("🧹 Cleaning and preparing data..."):
            step_start = time.time()
            original_count = len(df)
            df_clean = df.dropna(subset=[config.text_column]).reset_index(drop=True)
            cleaned_count = len(df_clean)

            if config.label_column == "Index":
                df_clean["label"] = df_clean.index.astype(str)
            else:
                df_clean["label"] = df_clean[config.label_column].astype(str)

            if config.link_column:
                df_clean["link"] = df_clean[config.link_column].fillna("#")
            else:
                df_clean["link"] = "#"

            step_time = time.time() - step_start
            processing_steps.append(
                {
                    "step": "Data Cleaning",
                    "time": step_time,
                    "details": f"Processed {original_count} → {cleaned_count} rows",
                }
            )

        progress_bar = st.progress(0)
        status_text = st.empty()
        details_text = st.empty()

        status_text.text("🤖 Loading embedding model...")
        progress_bar.progress(10)

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        model_kwargs: Dict[str, Any] = {"device": device, "trust_remote_code": True}
        if state.hf_token:
            model_kwargs["use_auth_token"] = state.hf_token

        step_start = time.time()
        load_details = ""
        try:
            needs_new_model = True
            if state.model is not None:
                model_name_attr = getattr(state.model, "name_or_path", None)
                if model_name_attr == config.model_name:
                    needs_new_model = False
            if needs_new_model:
                model = SentenceTransformer(config.model_name, **model_kwargs)
                state.model = model
                load_details = f"Loaded model on {device.upper()}"
            else:
                model = state.model
                load_details = f"Reusing loaded model on {device.upper()}"
        except Exception as model_error:
            message = str(model_error)
            if "401" in message or "authentication" in message.lower():
                st.error(f"❌ Authentication failed for model {config.model_name}")
                st.error("Please check your Hugging Face token and ensure you have accepted the model's terms of use.")
                st.error("Get your token at: https://huggingface.co/settings/tokens")
                processing_placeholder.empty()
                st.stop()
            elif "gated" in message.lower() or "private" in message.lower():
                st.error(f"❌ Model {config.model_name} requires authentication")
                st.error("Please enter a valid Hugging Face token in the sidebar.")
                processing_placeholder.empty()
                st.stop()
            else:
                st.error(f"❌ Failed to load model {config.model_name}: {message}")
                processing_placeholder.empty()
                st.stop()
            return

        step_time = time.time() - step_start
        processing_steps.append(
            {
                "step": "Model Loading",
                "time": step_time,
                "details": load_details + (" (authenticated)" if state.hf_token else ""),
            }
        )
        details_text.text(load_details)

        texts_to_embed = df_clean[config.text_column].tolist()
        total_texts = len(texts_to_embed)

        embeddings = None
        if config.use_cached and config.selected_cache is not None:
            status_text.text("📁 Loading cached embeddings...")
            progress_bar.progress(20)
            step_start = time.time()
            cached_data = load_embeddings(config.selected_cache["filepath"])
            if cached_data and cached_data.get("num_texts") == total_texts:
                embeddings = cached_data.get("embeddings")
                step_time = time.time() - step_start
                processing_steps.append(
                    {
                        "step": "Load Cached Embeddings",
                        "time": step_time,
                        "details": f"Loaded {len(embeddings)} embeddings from cache",
                    }
                )
                details_text.text(
                    f"Loaded cached embeddings ({cached_data.get('embedding_dim', '?')}D)"
                )
            else:
                st.warning("⚠️ Cached embeddings don't match current data size. Creating new embeddings...")
                embeddings = None

        if embeddings is None:
            status_text.text("🔤 Creating text embeddings...")
            progress_bar.progress(25)
            step_start = time.time()
            embedding_progress = st.progress(0)
            batch_size = 64
            num_batches = (total_texts + batch_size - 1) // batch_size
            collected_embeddings: List[np.ndarray] = []
            for batch_idx, start in enumerate(range(0, total_texts, batch_size), start=1):
                end = start + batch_size
                batch_texts = texts_to_embed[start:end]
                batch_embeddings = model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                collected_embeddings.extend(batch_embeddings)
                embedding_progress.progress(min(batch_idx / num_batches, 1.0))
                status_text.text(
                    f"🔤 Creating embeddings... Batch {batch_idx}/{num_batches}"
                )
                details_text.text(f"Processed {min(end, total_texts)}/{total_texts} texts")
            embeddings = np.array(collected_embeddings)
            embedding_progress.empty()
            step_time = time.time() - step_start
            processing_steps.append(
                {
                    "step": "Text Embeddings",
                    "time": step_time,
                    "details": f"Created {len(embeddings)} embeddings ({embeddings.shape[1]}D)",
                }
            )
        progress_bar.progress(50)

        if embeddings is None or len(embeddings) == 0:
            st.error("No embeddings were created.")
            processing_placeholder.empty()
            return

        if config.save_embeddings:
            status_text.text("💾 Saving embeddings to cache...")
            save_path, _ = save_embeddings(
                embeddings,
                texts_to_embed,
                config.model_name,
                metadata={"text_column": config.text_column, "modality": "text"},
            )
            if save_path:
                processing_steps.append(
                    {
                        "step": "Save Embeddings",
                        "time": 0.0,
                        "details": f"Saved to embeddings_cache/{save_path.name}",
                    }
                )
                details_text.text(f"Saved embeddings to {save_path.name}")

        try:
            status_text.text("🎨 Performing clustering...")
            progress_bar.progress(60)
            step_start = time.time()
            df_clean["cluster"] = perform_clustering(
                embeddings, config.clustering_method, config.n_clusters, config.min_cluster_size
            )
            df_clean["cluster_label"] = df_clean["cluster"].apply(
                lambda x: "Noise" if x == -1 else f"Cluster {x}"
            )
            step_time = time.time() - step_start
            n_clusters_found = len(df_clean["cluster"].unique())
            noise_points = int((df_clean["cluster"] == -1).sum())
            processing_steps.append(
                {
                    "step": "Clustering",
                    "time": step_time,
                    "details": f"Found {n_clusters_found} clusters, {noise_points} noise points",
                }
            )
        except Exception as cluster_error:
            st.error(f"❌ Error during clustering: {cluster_error}")
            processing_placeholder.empty()
            st.stop()
            return

        progress_bar.progress(75)

        try:
            status_text.text("🗺️ Reducing dimensionality with UMAP...")
            details_text.text("Creating 2D coordinates for visualization")
            step_start = time.time()
            embeddings_2d = reduce_dimensionality(embeddings)
            df_clean["x"] = embeddings_2d[:, 0]
            df_clean["y"] = embeddings_2d[:, 1]
            df_clean["hover_text"] = df_clean[config.text_column].astype(str).str[:150] + "..."
            step_time = time.time() - step_start
            processing_steps.append(
                {
                    "step": "Dimensionality Reduction",
                    "time": step_time,
                    "details": f"UMAP: {embeddings.shape[1]}D → 2D",
                }
            )
        except Exception as umap_error:
            st.error(f"❌ Error during UMAP dimensionality reduction: {umap_error}")
            processing_placeholder.empty()
            st.stop()
            return

        progress_bar.progress(90)
        status_text.text("⚡ Finalizing results...")
        details_text.text("Preparing visualization data")

        state.result = ProcessingResult(
            df=df_clean,
            embeddings=embeddings,
            modality="text",
            text_column=config.text_column,
        )
        state.processed = True
        state.processing_info = {
            "total_time": time.time() - start_time,
            "steps": processing_steps,
            "data_stats": {
                "total_points": len(df_clean),
                "clusters": len(df_clean["cluster"].unique()),
                "noise_points": int((df_clean["cluster"] == -1).sum()),
                "embedding_dim": embeddings.shape[1],
                "model": config.model_name,
                "device": device.upper(),
                "modality": "text",
                "label_column": config.label_column,
                "link_column": config.link_column,
            },
        }
        state.search_state = SearchState()

        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        details_text.text(f"Processed {len(df_clean)} records in {format_time(state.processing_info['total_time'])}")

        st.sidebar.success("✅ Processing complete!")

    processing_placeholder.empty()


def process_image_data(state: AppState, config: ImageProcessingConfig) -> None:
    """
    Orchestrates the entire image data processing pipeline, from loading images to generating visualizations.

    Args:
        state (AppState): The application state object.
        config (ImageProcessingConfig): The configuration for the image processing pipeline.
    """
    if not state.image_records:
        st.error("No images available for processing.")
        return

    processing_placeholder = st.empty()
    with processing_placeholder.container():
        st.markdown("### 🔄 Processing Images")
        start_time = time.time()
        processing_steps: List[Dict[str, Any]] = []

        with st.spinner("🧹 Validating image records..."):
            step_start = time.time()
            valid_records: List[Dict[str, Any]] = []
            for record in state.image_records:
                try:
                    if record.get("bytes") is None and not record.get("url"):
                        continue
                    load_pil_image(record)
                    valid_records.append(record)
                except Exception:
                    continue
            if not valid_records:
                st.error("No valid images available for processing.")
                processing_placeholder.empty()
                st.stop()
                return
            step_time = time.time() - step_start
            processing_steps.append(
                {
                    "step": "Validation",
                    "time": step_time,
                    "details": f"Validated {len(valid_records)} images",
                }
            )

        current_signatures = [record_signature(rec) for rec in valid_records]

        progress_bar = st.progress(0)
        status_text = st.empty()
        details_text = st.empty()

        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
        status_text.text("🤖 Loading SigLIP model...")
        details_text.text(f"Device: {device.upper()}")
        progress_bar.progress(10)

        try:
            model, processor = load_siglip_model(
                model_name="google/siglip-base-patch16-512",
                device=device,
                token=state.hf_token or None,
            )
        except Exception as model_error:
            st.error(f"❌ Failed to load SigLIP model: {model_error}")
            processing_placeholder.empty()
            st.stop()
            return

        embeddings = None
        if config.use_cached and config.selected_cache is not None:
            status_text.text("📁 Loading cached embeddings...")
            progress_bar.progress(25)
            cached_data = load_embeddings(config.selected_cache["filepath"])
            meta = (cached_data or {}).get("metadata", {})
            cached_signatures = meta.get("item_signatures", [])
            if cached_data and set(cached_signatures) == set(current_signatures):
                embeddings = cached_data.get("embeddings")
                processing_steps.append(
                    {
                        "step": "Load Cached Embeddings",
                        "time": 0.0,
                        "details": f"Loaded {len(embeddings)} cached vectors",
                    }
                )
            else:
                st.warning("⚠️ Cached embeddings do not match current images. Recomputing...")
                embeddings = None

        if embeddings is None:
            status_text.text("🖼️ Encoding images with SigLIP...")
            progress_bar.progress(40)
            step_start = time.time()
            embeddings, encoded_records = encode_images_with_siglip(
                valid_records,
                model,
                processor,
                device,
            )
            if embeddings.size == 0:
                st.error("❌ Unable to encode images")
                processing_placeholder.empty()
                st.stop()
                return
            valid_records = encoded_records
            current_signatures = [record_signature(rec) for rec in valid_records]
            step_time = time.time() - step_start
            processing_steps.append(
                {
                    "step": "Image Embeddings",
                    "time": step_time,
                    "details": f"Encoded {len(valid_records)} images to {embeddings.shape[1]}D vectors",
                }
            )

            if config.save_embeddings:
                status_text.text("💾 Saving embeddings to cache...")
                meta = {
                    "modality": "image",
                    "item_signatures": current_signatures,
                    "preprocessing": {
                        "description": "SigLIP Base Patch16 512",
                        "image_normalization": "SigLIP default",
                    },
                }
                save_path, _ = save_embeddings(
                    embeddings,
                    current_signatures,
                    "google/siglip-base-patch16-512",
                    metadata=meta,
                )
                if save_path:
                    processing_steps.append(
                        {
                            "step": "Save Embeddings",
                            "time": 0.0,
                            "details": f"Saved cache as {save_path.name}",
                        }
                    )
        progress_bar.progress(55)

        status_text.text("🎯 Clustering embeddings...")
        step_start = time.time()
        cluster_labels = perform_clustering(
            embeddings, config.clustering_method, config.n_clusters, config.min_cluster_size
        )
        step_time = time.time() - step_start
        n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_points = int(np.sum(cluster_labels == -1))
        processing_steps.append(
            {
                "step": "Clustering",
                "time": step_time,
                "details": f"Found {n_clusters_found} clusters with {noise_points} noise images",
            }
        )

        progress_bar.progress(70)

        status_text.text("🗺️ Running dimensionality reduction...")
        details_text.text("Projecting embeddings to 2D space")
        step_start = time.time()
        umap_coords = reduce_dimensionality(embeddings)
        step_time = time.time() - step_start
        processing_steps.append(
            {
                "step": "UMAP Reduction",
                "time": step_time,
                "details": "Reduced embeddings to 2D",
            }
        )

        progress_bar.progress(85)

        status_text.text("🗂️ Preparing visualization data...")
        step_start = time.time()
        thumbnails: List[str] = []
        captions: List[str] = []
        for record in valid_records:
            try:
                image = load_pil_image(record)
                thumb_b64 = create_thumbnail_base64(image)
            except Exception:
                thumb_b64 = ""
            thumbnails.append(thumb_b64)
            caption_text = _clean_metadata_value(record.get("caption"), default="", allow_empty=True)
            if not caption_text:
                caption_text = _clean_metadata_value(
                    record.get("label"), default=record.get("id", ""), allow_empty=False
                )
            captions.append(caption_text)

        df_clean = pd.DataFrame(
            {
                "label": [rec["label"] for rec in valid_records],
                "link": [rec.get("link", "#") or "#" for rec in valid_records],
                "caption": captions,
                "thumbnail": thumbnails,
                "source_id": [rec["id"] for rec in valid_records],
            }
        )
        df_clean["x"] = umap_coords[:, 0]
        df_clean["y"] = umap_coords[:, 1]
        df_clean["cluster"] = cluster_labels
        df_clean["cluster_label"] = df_clean["cluster"].apply(lambda x: f"Cluster {x}" if x != -1 else "Noise")
        df_clean["hover_text"] = df_clean["caption"]
        df_clean["media_type"] = "image"

        step_time = time.time() - step_start
        processing_steps.append(
            {
                "step": "Prepare DataFrame",
                "time": step_time,
                "details": "Added visualization metadata",
            }
        )

        progress_bar.progress(95)

        status_text.text("⚡ Finalizing results...")
        total_time = time.time() - start_time

        state.result = ProcessingResult(
            df=df_clean,
            embeddings=embeddings,
            modality="image",
            text_column="caption",
        )
        state.processed = True
        state.processing_info = {
            "total_time": total_time,
            "steps": processing_steps,
            "data_stats": {
                "total_points": len(df_clean),
                "clusters": int(n_clusters_found),
                "noise_points": int(noise_points),
                "embedding_dim": embeddings.shape[1],
                "model": "google/siglip-base-patch16-512",
                "device": device.upper(),
                "modality": "image",
            },
        }
        state.search_state = SearchState()
        state.image_metadata = {**state.image_metadata, "images": len(df_clean)}

        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        details_text.text(f"Processed {len(df_clean)} images in {format_time(total_time)}")
        st.sidebar.success("✅ Processing complete!")

    processing_placeholder.empty()
