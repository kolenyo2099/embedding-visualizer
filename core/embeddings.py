from typing import List, Dict, Any, Optional
import streamlit as st
import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
import umap
from core.utils import chunk_list, load_pil_image


@st.cache_resource(show_spinner=False)
def load_siglip_model(
    model_name: str = "google/siglip-base-patch16-512",
    device: str = "cpu",
    token: Optional[str] = None,
):
    """
    Loads the SigLIP model and processor from Hugging Face, with caching.

    Args:
        model_name (str): The name of the SigLIP model to load.
        device (str): The device to load the model on ('cpu', 'cuda', 'mps').
        token (Optional[str]): A Hugging Face authentication token for gated models.

    Returns:
        Tuple[AutoModel, AutoProcessor]: The loaded model and processor.
    """
    processor_kwargs = {"trust_remote_code": True}
    model_kwargs = {"trust_remote_code": True}
    if token:
        processor_kwargs["use_auth_token"] = token
        model_kwargs["use_auth_token"] = token
    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
    if device == "cuda" and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32
    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    return model, processor


def encode_images_with_siglip(
    records: List[Dict[str, Any]],
    model,
    processor,
    device: str,
    batch_size: int = 8,
):
    """
    Encodes a list of image records into embeddings using the SigLIP model.

    Args:
        records (List[Dict[str, Any]]): A list of image records.
        model: The loaded SigLIP model.
        processor: The loaded SigLIP processor.
        device (str): The device to use for encoding.
        batch_size (int): The batch size for encoding.

    Returns:
        Tuple[np.ndarray, List[Dict[str, Any]]]: The computed embeddings and the list of valid records that were successfully encoded.
    """
    embeddings = []
    valid_records = []
    for batch in chunk_list(records, batch_size):
        pil_images = []
        filtered_batch = []
        for record in batch:
            try:
                image = load_pil_image(record)
                pil_images.append(image)
                filtered_batch.append(record)
            except Exception:
                continue
        if not pil_images:
            continue
        processed = processor(images=pil_images, return_tensors="pt")
        pixel_values = processed.get("pixel_values")
        if pixel_values is None:
            continue
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            image_embeds = model.get_image_features(pixel_values=pixel_values)
            image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
        embeddings.append(image_embeds.cpu().numpy())
        valid_records.extend(filtered_batch)
    if not embeddings:
        return np.array([]), []
    return np.vstack(embeddings), valid_records


def encode_text_with_siglip(
    texts: List[str],
    model,
    processor,
    device: str,
    batch_size: int = 16,
):
    """
    Encodes a list of texts into embeddings using the SigLIP model.

    Args:
        texts (List[str]): A list of text strings to encode.
        model: The loaded SigLIP model.
        processor: The loaded SigLIP processor.
        device (str): The device to use for encoding.
        batch_size (int): The batch size for encoding.

    Returns:
        np.ndarray: The computed text embeddings.
    """
    text_embeddings = []
    for batch in chunk_list(texts, batch_size):
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            text_embeds = outputs.text_embeds
            text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)
        text_embeddings.append(text_embeds.cpu().numpy())
    if not text_embeddings:
        return np.array([])
    return np.vstack(text_embeddings)


def reduce_dimensionality(embeddings: np.ndarray) -> np.ndarray:
    """
    Reduces the dimensionality of embeddings to 2D using UMAP.

    Args:
        embeddings (np.ndarray): The high-dimensional embeddings.

    Returns:
        np.ndarray: The 2D embeddings.
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)
