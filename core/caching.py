from typing import Any, Dict, List, Optional, Tuple
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
import streamlit as st
import numpy as np


def get_embeddings_dir() -> Path:
    """
    Ensures the 'embeddings_cache' directory exists and returns its path.

    Returns:
        Path: The path to the embeddings cache directory.
    """
    embeddings_dir = Path("embeddings_cache")
    embeddings_dir.mkdir(exist_ok=True)
    return embeddings_dir


def generate_embedding_hash(items: Any, model_name: str, modality: str = "text") -> str:
    """
    Generates a unique hash for a set of items and a model name to be used as a cache key.

    Args:
        items (Any): The items being embedded (e.g., a list of strings or an array of image signatures).
        model_name (str): The name of the embedding model.
        modality (str): The data modality ('text' or 'image').

    Returns:
        str: The MD5 hash of the content.
    """
    if isinstance(items, np.ndarray):
        items_repr = hashlib.md5(items.tobytes()).hexdigest()
        length = items.shape[0]
    else:
        if isinstance(items, list):
            sample = items[:100] if len(items) > 100 else items
            length = len(items)
        else:
            sample = [items]
            length = 1
        sample_strings = [str(item)[:200] for item in sample]
        items_repr = "|".join(sample_strings)
    content = f"{items_repr}_{model_name}_{length}_{modality}"
    return hashlib.md5(content.encode()).hexdigest()


def save_embeddings(
    embeddings: np.ndarray,
    items: Any,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Saves computed embeddings to a pickle file in the cache directory.

    Args:
        embeddings (np.ndarray): The embeddings to save.
        items (Any): The original items that were embedded.
        model_name (str): The name of the model used to generate the embeddings.
        metadata (Optional[Dict[str, Any]]): Additional metadata to store with the embeddings.

    Returns:
        Tuple[Optional[Path], Optional[str]]: The path to the saved file and the embedding hash, or (None, None) on failure.
    """
    try:
        embeddings_dir = get_embeddings_dir()
        modality = (metadata or {}).get("modality", "text")
        embedding_hash = generate_embedding_hash(items, model_name, modality=modality)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embeddings_{embedding_hash}_{timestamp}.pkl"
        filepath = embeddings_dir / filename
        data = {
            "embeddings": embeddings,
            "model_name": model_name,
            "num_texts": len(items) if hasattr(items, "__len__") else None,
            "embedding_dim": embeddings.shape[1] if hasattr(embeddings, "shape") else None,
            "timestamp": timestamp,
            "metadata": metadata or {},
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        return filepath, embedding_hash
    except Exception as exc:
        st.warning(f"Failed to save embeddings: {exc}")
        return None, None


def load_embeddings(filepath: Path):
    """
    Loads embeddings from a specified pickle file.

    Args:
        filepath (Path): The path to the embedding file.

    Returns:
        Any: The loaded data, or None on failure.
    """
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as exc:
        st.error(f"Failed to load embeddings: {exc}")
        return None


def list_saved_embeddings() -> List[Dict[str, Any]]:
    """
    Lists all saved embeddings in the cache directory, sorted by timestamp.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing info about a cached embedding file.
    """
    embeddings_dir = get_embeddings_dir()
    embeddings_files = list(embeddings_dir.glob("embeddings_*.pkl"))
    embeddings_info: List[Dict[str, Any]] = []
    for filepath in embeddings_files:
        try:
            data = load_embeddings(filepath)
            if data:
                embeddings_info.append(
                    {
                        "filepath": filepath,
                        "filename": filepath.name,
                        "model_name": data.get("model_name", "Unknown"),
                        "num_texts": data.get("num_texts", 0),
                        "embedding_dim": data.get("embedding_dim", 0),
                        "timestamp": data.get("timestamp", "Unknown"),
                        "size_mb": filepath.stat().st_size / (1024 * 1024),
                        "metadata": data.get("metadata", {}),
                    }
                )
        except Exception:
            continue
    return sorted(embeddings_info, key=lambda x: x["timestamp"], reverse=True)
