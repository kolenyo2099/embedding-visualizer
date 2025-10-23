from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
import streamlit as st
@dataclass
class ProcessingResult:
    """
    Data class to hold the results of the data processing pipeline.

    Attributes:
        df (Optional[pd.DataFrame]): DataFrame containing the processed data, including coordinates and cluster labels.
        embeddings (Optional[np.ndarray]): The computed embeddings for the dataset.
        modality (Optional[str]): The modality of the data, either 'text' or 'image'.
        text_column (Optional[str]): The name of the text column used for embedding.
    """
    df: Optional[pd.DataFrame] = None
    embeddings: Optional[np.ndarray] = None
    modality: Optional[str] = None
    text_column: Optional[str] = None


@dataclass
class SearchState:
    """
    Data class to manage the state of the semantic search feature.

    Attributes:
        query (str): The search query entered by the user.
        top_k (int): The number of top results to display.
        mode (str): The search mode, either 'Text' or 'Image'.
        results_df (Optional[pd.DataFrame]): DataFrame containing the search results.
        similarity_threshold (float): The similarity threshold for filtering results.
        query_image_bytes (Optional[bytes]): The bytes of the query image.
        search_time (float): The time taken to perform the search.
        avg_similarity (float): The average similarity of the search results.
        similarity_range (str): The range of similarity scores.
    """
    query: str = ""
    top_k: int = 20
    mode: str = "Text"
    results_df: Optional[pd.DataFrame] = None
    similarity_threshold: float = 0.0
    query_image_bytes: Optional[bytes] = None
    search_time: float = 0.0
    avg_similarity: float = 0.0
    similarity_range: str = ""


class AppState:
    """
    A proxy object that encapsulates the application's state within Streamlit's session store.

    This class provides a clean and organized way to manage session state by defining
    default values and ensuring that the state is properly initialized.
    """

    DEFAULTS: Dict[str, Any] = {
        "modality": "Text (CSV)",
        "processed": False,
        "raw_df": None,
        "uploaded_file_hash": None,
        "result": lambda: ProcessingResult(),
        "model": None,
        "processing_info": lambda: {},
        "hf_token": "",
        "node_size": 1.0,
        "search_result_size_multiplier": 3.0,
        "search_state": lambda: SearchState(),
        "image_records": lambda: [],
        "image_metadata": lambda: {},
        "image_source_id": None,
        "text_cache_choice": None,
        "image_cache_choice": None,
    }

    __slots__ = ("_state",)

    def __init__(self, backing_state: Dict[str, Any]):
        object.__setattr__(self, "_state", backing_state)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._state[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self._state[name] = value

    def ensure_defaults(self) -> None:
        """
        Initializes the session state with default values if they are not already present.
        """
        for key, default in self.DEFAULTS.items():
            if key not in self._state:
                self._state[key] = default() if callable(default) else default


APP_STATE_KEY = "__app_state__"


def get_app_state() -> AppState:
    """
    Retrieves the application state from Streamlit's session state,
    ensuring it is properly initialized.

    Returns:
        AppState: The application state object.
    """
    backing_state = st.session_state.setdefault(APP_STATE_KEY, {})
    state = AppState(backing_state)
    state.ensure_defaults()
    return state


def reset_processed_state(state: AppState, *, clear_model: bool = False) -> None:
    """
    Resets the application state to its initial state, clearing any processed data.

    Args:
        state (AppState): The application state object.
        clear_model (bool): Whether to clear the loaded model.
    """
    state.processed = False
    state.result = ProcessingResult()
    state.processing_info = {}
    state.search_state = SearchState()
    if clear_model:
        state.model = None
