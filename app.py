import streamlit as st
import json
import time
import os
import hashlib
import pickle
import zipfile
from dataclasses import dataclass
from io import BytesIO
import base64
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def check_and_import_dependencies():
    """Check and import all required dependencies with helpful error messages."""
    missing_deps = []

    try:
        import pandas as pd
    except ImportError:
        missing_deps.append("pandas")
        pd = None

    try:
        import numpy as np
    except ImportError:
        missing_deps.append("numpy")
        np = None

    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
        torch = None

    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("Pillow")
        Image = None

    try:
        import torchvision.transforms as T
    except ImportError:
        missing_deps.append("torchvision")
        T = None

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing_deps.append("sentence-transformers")
        SentenceTransformer = None

    try:
        import umap
    except ImportError:
        missing_deps.append("umap-learn")
        umap = None

    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        missing_deps.append("scikit-learn")
        HDBSCAN = None

    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError:
        missing_deps.append("transformers")
        AutoModel = None
        AutoProcessor = None

    try:
        import requests
    except ImportError:
        missing_deps.append("requests")
        requests = None

    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False

    try:
        import streamlit.components.v1 as components
        COMPONENTS_AVAILABLE = True
    except ImportError:
        COMPONENTS_AVAILABLE = False
        components = None

    critical_deps = [
        "pandas",
        "numpy",
        "torch",
        "sentence-transformers",
        "umap-learn",
        "scikit-learn",
        "Pillow",
        "torchvision",
        "transformers",
        "requests",
    ]
    missing_critical = [dep for dep in missing_deps if dep in critical_deps]

    if missing_critical:
        error_msg = """
        ## ❌ Missing Critical Dependencies

        The following required packages are missing:
        """
        for dep in missing_critical:
            error_msg += f"\n- **{dep}**"

        error_msg += f"""

        ### 🛠️ How to Fix:

        1. **Install missing dependencies:**
        ```bash
        pip install {' '.join(missing_critical)}
        ```

        2. **Or install all requirements:**
        ```bash
        pip install -r requirements.txt
        ```

        3. **Or use the setup script:**
        ```bash
        ./setup_and_run.sh
        ```

        Please install the missing dependencies and restart the app.
        """

        st.error(error_msg)
        st.stop()

    return locals()


imports = check_and_import_dependencies()
pd = imports["pd"]
np = imports["np"]
torch = imports["torch"]
SentenceTransformer = imports["SentenceTransformer"]
umap = imports["umap"]
HDBSCAN = imports["HDBSCAN"]
components = imports["components"]
PSUTIL_AVAILABLE = imports["PSUTIL_AVAILABLE"]
COMPONENTS_AVAILABLE = imports["COMPONENTS_AVAILABLE"]
Image = imports["Image"]
T = imports["T"]
AutoModel = imports["AutoModel"]
AutoProcessor = imports["AutoProcessor"]
requests = imports["requests"]


@dataclass
class ProcessingResult:
    df: Optional[pd.DataFrame] = None
    embeddings: Optional[np.ndarray] = None
    modality: Optional[str] = None
    text_column: Optional[str] = None


@dataclass
class SearchState:
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
    """Proxy object that keeps application state inside Streamlit's session store."""

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
        except KeyError as exc:  # pragma: no cover - passthrough for attribute access
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self._state[name] = value

    def ensure_defaults(self) -> None:
        for key, default in self.DEFAULTS.items():
            if key not in self._state:
                self._state[key] = default() if callable(default) else default


APP_STATE_KEY = "__app_state__"


def get_app_state() -> AppState:
    backing_state = st.session_state.setdefault(APP_STATE_KEY, {})
    state = AppState(backing_state)
    state.ensure_defaults()
    return state


def reset_processed_state(state: AppState, *, clear_model: bool = False) -> None:
    state.processed = False
    state.result = ProcessingResult()
    state.processing_info = {}
    state.search_state = SearchState()
    if clear_model:
        state.model = None


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def format_memory(mb: float) -> str:
    if mb < 1024:
        return f"{mb:.1f}MB"
    return f"{mb / 1024:.1f}GB"


def chunk_list(items: List[Any], chunk_size: int):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _clean_metadata_value(value: Any, default: str = "", allow_empty: bool = True) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        candidate = value.strip()
    else:
        try:
            if pd.isna(value):
                return default
        except TypeError:
            pass
        candidate = str(value).strip()
    if not candidate and not allow_empty:
        return default
    return candidate if allow_empty or candidate else default


def create_image_record(
    identifier: str,
    data_bytes: Optional[bytes] = None,
    label: Optional[str] = None,
    link: Optional[str] = None,
    caption: Optional[str] = None,
    url: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": identifier,
        "bytes": data_bytes,
        "label": label or identifier,
        "link": link or "#",
        "caption": caption or "",
        "url": url,
    }


def record_signature(record: Dict[str, Any]) -> str:
    if record.get("bytes"):
        digest = hashlib.md5(record["bytes"]).hexdigest()
    else:
        digest = hashlib.md5((record.get("url") or record["id"]).encode()).hexdigest()
    return f"{record['id']}:{digest}"


def load_pil_image(record: Dict[str, Any]):
    data_bytes = record.get("bytes")
    if data_bytes is None and record.get("url"):
        response = requests.get(record["url"], timeout=15)
        response.raise_for_status()
        data_bytes = response.content
        record["bytes"] = data_bytes
    image = Image.open(BytesIO(data_bytes))
    return image.convert("RGB")


def create_thumbnail_base64(image, max_size=(160, 160)):
    preview = image.copy()
    preview.thumbnail(max_size)
    buffer = BytesIO()
    preview.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_embeddings_dir() -> Path:
    embeddings_dir = Path("embeddings_cache")
    embeddings_dir.mkdir(exist_ok=True)
    return embeddings_dir


def generate_embedding_hash(items: Any, model_name: str, modality: str = "text") -> str:
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
    except Exception as exc:  # pragma: no cover - best effort cache helper
        st.warning(f"Failed to save embeddings: {exc}")
        return None, None


def load_embeddings(filepath: Path):
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as exc:
        st.error(f"Failed to load embeddings: {exc}")
        return None


def list_saved_embeddings() -> List[Dict[str, Any]]:
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


@st.cache_resource(show_spinner=False)
def load_siglip_model(
    model_name: str = "google/siglip-base-patch16-512",
    device: str = "cpu",
    token: Optional[str] = None,
):
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


def generate_cosmograph_html(
    df: pd.DataFrame,
    *,
    text_column: str,
    node_size: float,
    search_result_size_multiplier: float,
    search_results: Optional[pd.DataFrame] = None,
) -> str:
    optional_cols = [col for col in ["thumbnail", "caption", "media_type", "source_id"] if col in df.columns]
    if text_column in df.columns:
        selected_cols = [
            "label",
            "x",
            "y",
            "cluster_label",
            "hover_text",
            text_column,
            "link",
        ] + optional_cols
        data_for_js = df[selected_cols].copy()
        rename_map = {"label": "id", "cluster_label": "cluster", text_column: "full_text"}
        data_for_js = data_for_js.rename(columns=rename_map)
    else:
        selected_cols = ["label", "x", "y", "cluster_label", "hover_text", "link"] + optional_cols
        data_for_js = df[selected_cols].copy()
        rename_map = {"label": "id", "cluster_label": "cluster"}
        data_for_js = data_for_js.rename(columns=rename_map)
        data_for_js["full_text"] = data_for_js["hover_text"]
    cluster_colors = {}
    base_palette = [
        "#4C78A8",
        "#F58518",
        "#E45756",
        "#72B7B2",
        "#54A24B",
        "#EECA3B",
        "#B279A2",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]
    unique_clusters = sorted(df["cluster_label"].unique())
    for idx, cluster in enumerate(unique_clusters):
        cluster_colors[str(cluster)] = base_palette[idx % len(base_palette)]
    if search_results is not None and not search_results.empty:
        highlighted_ids = set(search_results["label"].astype(str))
    else:
        highlighted_ids = set()
    data_for_js["is_search_result"] = data_for_js["id"].astype(str).isin(highlighted_ids)
    if search_results is not None and not search_results.empty:
        score_lookup = dict(zip(search_results["label"].astype(str), search_results["similarity"]))
        rank_lookup = dict(zip(search_results["label"].astype(str), search_results["rank"]))
        data_for_js["search_score"] = data_for_js["id"].astype(str).map(score_lookup)
        data_for_js["search_rank"] = data_for_js["id"].astype(str).map(rank_lookup)
    else:
        data_for_js["search_score"] = None
        data_for_js["search_rank"] = None
    if "thumbnail" in data_for_js.columns:
        data_for_js["thumbnail"] = data_for_js["thumbnail"].fillna("")
    json_data = {
        "data": data_for_js.to_dict(orient="records"),
        "colors": cluster_colors,
    }
    json_b64 = base64.b64encode(json.dumps(json_data).encode("utf-8")).decode("utf-8")
    # The HTML template is lengthy but provides the Cosmograph visualization and interactive search overlay.
    html_template = f"""
    <style>
    body {{
        margin: 0;
        background: #0f172a;
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: white;
    }}
    #graph-container {{
        width: 100%;
        height: 680px;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: radial-gradient(circle at top, rgba(148, 163, 184, 0.15), rgba(15, 23, 42, 0.95));
        position: relative;
    }}
    .legend {{
        position: absolute;
        top: 24px;
        right: 24px;
        background: rgba(15, 23, 42, 0.85);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        backdrop-filter: blur(12px);
        max-height: 320px;
        overflow-y: auto;
        width: 220px;
    }}
    .legend-title {{
        font-size: 13px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: rgba(226, 232, 240, 0.75);
        margin-bottom: 12px;
    }}
    .legend-item {{
        display: flex;
        align-items: center;
        margin-bottom: 8px;
        font-size: 13px;
        color: rgba(226, 232, 240, 0.9);
    }}
    .legend-color {{
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
        flex-shrink: 0;
        border: 1px solid rgba(148, 163, 184, 0.4);
    }}
    .modal {{
        display: none;
        position: fixed;
        z-index: 1000;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(8px);
    }}
    .modal-content {{
        background-color: #111827;
        margin: 4% auto;
        padding: 0;
        border-radius: 16px;
        width: min(720px, 90%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 30px 60px rgba(15, 23, 42, 0.45);
        color: #e2e8f0;
        overflow: hidden;
    }}
    .modal-header {{
        padding: 20px 28px 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(120deg, rgba(56, 189, 248, 0.08), rgba(147, 51, 234, 0.12));
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }}
    .modal-title {{
        margin: 0;
        font-size: 20px;
        font-weight: 600;
        letter-spacing: -0.01em;
    }}
    .close {{
        color: rgba(226, 232, 240, 0.6);
        font-size: 28px;
        cursor: pointer;
        padding: 8px;
        border-radius: 8px;
        transition: all 0.2s ease;
    }}
    .close:hover {{
        color: #f8fafc;
        background: rgba(59, 130, 246, 0.1);
    }}
    .modal-media {{
        padding: 24px 28px 12px;
    }}
    .modal-media img {{
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }}
    .modal-text {{
        padding: 0 28px 24px;
        line-height: 1.7;
        color: rgba(226, 232, 240, 0.88);
    }}
    .modal-footer {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 28px 24px;
        border-top: 1px solid rgba(148, 163, 184, 0.15);
        background: rgba(15, 23, 42, 0.65);
    }}
    .modal-link {{
        color: #38bdf8;
        text-decoration: none;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: all 0.2s ease;
    }}
    .modal-link:hover {{
        color: #0ea5e9;
        transform: translateX(3px);
    }}
    .cluster-badge {{
        background: rgba(59, 130, 246, 0.18);
        color: rgba(191, 219, 254, 0.95);
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 12px;
        margin-right: 8px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}
    .search-badge {{
        background: rgba(244, 114, 182, 0.15);
        color: rgba(251, 191, 36, 0.95);
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 12px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}
    .search-controls {{
        position: absolute;
        top: 24px;
        left: 24px;
        display: flex;
        gap: 8px;
        align-items: center;
        background: rgba(15, 23, 42, 0.9);
        padding: 10px 14px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        backdrop-filter: blur(12px);
    }}
    .search-input {{
        background: transparent;
        border: none;
        color: rgba(226, 232, 240, 0.95);
        font-size: 13px;
        outline: none;
        width: 200px;
    }}
    .search-button {{
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 8px 16px;
        cursor: pointer;
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.02em;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.35);
        transition: all 0.2s ease;
    }}
    .search-button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 16px 32px rgba(59, 130, 246, 0.45);
    }}
    .search-info {{
        position: absolute;
        bottom: 24px;
        left: 24px;
        background: rgba(15, 23, 42, 0.85);
        padding: 10px 16px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: rgba(226, 232, 240, 0.85);
        font-size: 13px;
        backdrop-filter: blur(12px);
    }}
    </style>
    <div id="graph-container">
        <div class="legend" id="legend"></div>
        <div class="search-controls">
            <input type="text" class="search-input" id="searchInput" placeholder="Search in graph...">
            <button class="search-button" onclick="performSearch()">🔍 Search</button>
        </div>
        <div class="search-info">🔍 Search-ready: type a query to highlight matches</div>
    </div>
    <div id="modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title" id="modalTitle"></h2>
                <span class="close" onclick="closeModal()">&times;</span>
            </div>
            <div id="badges"></div>
            <div id="modalMedia" class="modal-media"></div>
            <div class="modal-text" id="modalText"></div>
            <div class="modal-footer">
                <div></div>
                <a id="modalLink" class="modal-link" href="#" target="_blank">Open Link →</a>
            </div>
        </div>
    </div>
    <script type="importmap">
        {{ "imports": {{ "@cosmograph/cosmograph": "https://esm.sh/@cosmograph/cosmograph@1.4.2" }} }}
    </script>
    <script type="module">
        import {{ Cosmograph }} from '@cosmograph/cosmograph';
        let cosmograph;
        let DATA;
        let COLORS;
        try {{
            const b64Data = '{json_b64}';
            const jsonStr = atob(b64Data);
            const jsonData = JSON.parse(jsonStr);
            DATA = jsonData.data;
            COLORS = jsonData.colors;
            const nodes = DATA.map((d, i) => ({{ id: String(i), x: d.x, y: d.y }}));
            const container = document.getElementById('graph-container');
            const config = {{
                nodeColor: (node) => {{
                    const idx = parseInt(node.id);
                    const data = DATA[idx];
                    return data.is_search_result ? '#ff0000' : (COLORS[data.cluster] || '#999999');
                }},
                nodeSize: (node) => {{
                    const idx = parseInt(node.id);
                    const data = DATA[idx];
                    const baseSize = {node_size};
                    const multiplier = {search_result_size_multiplier};
                    return data.is_search_result ? (baseSize * multiplier) : baseSize;
                }},
                renderLinks: false,
                simulation: false,
                onClick: (node) => {{
                    if (node) {{
                        const idx = parseInt(node.id);
                        const d = DATA[idx];
                        showModal(d, idx);
                    }}
                }}
            }};
            cosmograph = new Cosmograph(container, config);
            cosmograph.setData(nodes, []);
            setTimeout(() => cosmograph.fitView(0.8), 500);
            setupLegend();
        }} catch (err) {{
            console.error('Error initializing visualization:', err);
            document.getElementById('graph-container').innerHTML =
                '<div style="color: white; padding: 20px; text-align: center;">' +
                '<h3>Error loading visualization</h3>' +
                '<p>' + err.message + '</p>' +
                '</div>';
        }}
        function setupLegend() {{
            const clusterCounts = {{}};
            DATA.forEach(d => {{ clusterCounts[d.cluster] = (clusterCounts[d.cluster] || 0) + 1; }});
            const legendHtml = Object.keys(COLORS).sort().map(cluster =>
                '<div class="legend-item"><div class="legend-color" style="background: ' +
                COLORS[cluster] + '"></div><span>' + cluster + ' (' + clusterCounts[cluster] + ')</span></div>'
            ).join('');
            document.getElementById('legend').innerHTML = '<div class="legend-title">Clusters</div>' + legendHtml;
        }}
        window.showModal = function(data, index) {{
            document.getElementById('modalTitle').textContent = data.id;
            const mediaContainer = document.getElementById('modalMedia');
            if (data.thumbnail) {{
                mediaContainer.innerHTML = `<img src="data:image/png;base64,${{data.thumbnail}}" alt="${{data.id}}" />`;
            }} else {{
                mediaContainer.innerHTML = '';
            }}
            const modalText = document.getElementById('modalText');
            modalText.textContent = data.full_text || data.hover_text || 'No additional text available';
            let badges = '<span class="cluster-badge">' + data.cluster + '</span>';
            if (data.is_search_result) {{
                badges += '<span class="search-badge">Search Result #' + data.search_rank +
                    ' (Score: ' + (data.search_score || 0).toFixed(3) + ')</span>';
            }}
            document.getElementById('badges').innerHTML = badges;
            const link = document.getElementById('modalLink');
            link.href = data.link || '#';
            link.style.display = data.link && data.link !== '#' ? 'block' : 'none';
            document.getElementById('modal').style.display = 'block';
        }}
        window.closeModal = function() {{
            document.getElementById('modal').style.display = 'none';
        }}
        window.onclick = function(event) {{
            if (event.target == document.getElementById('modal')) closeModal();
        }}
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') closeModal();
        }});
        window.performSearch = function() {{
            const query = document.getElementById('searchInput').value.toLowerCase().trim();
            if (!query) {{
                if (cosmograph) {{
                    cosmograph.setNodeColor((node) => {{
                        const idx = parseInt(node.id);
                        const data = DATA[idx];
                        return data.is_search_result ? '#ff0000' : (COLORS[data.cluster] || '#999999');
                    }});
                    cosmograph.setNodeSize((node) => {{
                        const idx = parseInt(node.id);
                        const data = DATA[idx];
                        const baseSize = {node_size};
                        const multiplier = {search_result_size_multiplier};
                        return data.is_search_result ? (baseSize * multiplier) : baseSize;
                    }});
                }}
                const searchInfo = document.querySelector('.search-info');
                if (searchInfo) {{
                    const highlighted = DATA.filter(d => d.is_search_result).length;
                    searchInfo.textContent = '🔍 ' + highlighted + ' search results highlighted in red';
                }}
                return;
            }}
            const matchingIndices = [];
            DATA.forEach((d, i) => {{
                const searchText = (d.full_text + ' ' + d.hover_text + ' ' + d.id).toLowerCase();
                if (searchText.includes(query)) {{
                    matchingIndices.push(i);
                }}
            }});
            if (cosmograph) {{
                cosmograph.setNodeColor((node) => {{
                    const idx = parseInt(node.id);
                    const data = DATA[idx];
                    if (matchingIndices.includes(idx)) {{
                        return '#ff6b35';
                    }}
                    return data.is_search_result ? '#ff0000' : (COLORS[data.cluster] || '#999999');
                }});
                cosmograph.setNodeSize((node) => {{
                    const idx = parseInt(node.id);
                    const data = DATA[idx];
                    const baseSize = {node_size};
                    const multiplier = {search_result_size_multiplier};
                    if (matchingIndices.includes(idx)) {{
                        return baseSize * 4;
                    }}
                    return data.is_search_result ? (baseSize * multiplier) : baseSize;
                }});
            }}
            const searchInfo = document.querySelector('.search-info');
            if (searchInfo) {{
                searchInfo.textContent = '🔍 ' + matchingIndices.length + ' matches for "' + query + '"';
            }}
        }}
        document.getElementById('searchInput').addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') {{
                performSearch();
            }}
        }});
    </script>
    """
    return html_template

@dataclass
class TextProcessingConfig:
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
    clustering_method: str
    n_clusters: Optional[int]
    min_cluster_size: Optional[int]
    use_cached: bool
    selected_cache: Optional[Dict[str, Any]]
    save_embeddings: bool


def render_header() -> None:
    st.title("🎯 Semantic Embedding Explorer")
    st.markdown(
        """
        Visualize and explore text **or image** data using state-of-the-art embeddings and interactive clustering.
        Choose your modality, upload your sources, and explore semantic relationships in your content.
        """
    )


def render_system_info() -> None:
    if not PSUTIL_AVAILABLE:
        return
    import psutil  # type: ignore

    try:
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
    except Exception:
        pass


def render_modality_selector(state: AppState) -> None:
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


def handle_text_upload(state: AppState) -> None:
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


def text_processing_sidebar(state: AppState, saved_embeddings: List[Dict[str, Any]]):
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

def set_image_records(state: AppState, records: List[Dict[str, Any]], source_id: Any, metadata: Dict[str, Any]) -> None:
    if not records:
        return
    state.image_records = records
    state.image_source_id = source_id
    state.image_metadata = metadata
    reset_processed_state(state)
    st.sidebar.success(f"✅ Loaded {len(records)} images")


def image_source_sidebar(state: AppState) -> None:
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

    else:  # CSV with Image URLs
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


def image_processing_sidebar(state: AppState, saved_embeddings: List[Dict[str, Any]]):
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

def render_image_metadata_editor(state: AppState) -> None:
    if state.modality != "Images":
        return
    if not state.image_records:
        return
    st.subheader("📝 Image Metadata")
    editor_df = pd.DataFrame(
        {
            "Identifier": [rec["id"] for rec in state.image_records],
            "Label": [rec.get("label", rec["id"]) for rec in state.image_records],
            "Caption": [rec.get("caption", "") for rec in state.image_records],
            "Link": [rec.get("link", "#") for rec in state.image_records],
        }
    )
    edited_df = st.data_editor(
        editor_df,
        num_rows="dynamic",
        use_container_width=True,
        disabled=["Identifier"],
        key="image_metadata_editor",
    )
    if edited_df.empty:
        return
    updated_records = []
    for record, row in zip(state.image_records, edited_df.to_dict("records")):
        record["label"] = _clean_metadata_value(row.get("Label"), default=record["label"], allow_empty=False)
        record["caption"] = _clean_metadata_value(row.get("Caption"), default="", allow_empty=True)
        record["link"] = _clean_metadata_value(row.get("Link"), default="#", allow_empty=False) or "#"
        updated_records.append(record)
    state.image_records = updated_records

def process_text_data(state: AppState, config: TextProcessingConfig) -> None:
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
            if config.clustering_method == "HDBSCAN (Automatic)":
                details_text.text(f"Minimum cluster size: {config.min_cluster_size}")
                clusterer = HDBSCAN(
                    min_cluster_size=config.min_cluster_size or 10,
                    min_samples=10,
                    metric="euclidean",
                )
                df_clean["cluster"] = clusterer.fit_predict(embeddings)
            else:
                details_text.text(f"Creating {config.n_clusters} clusters")
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=config.n_clusters or 10, random_state=42, n_init=10)
                df_clean["cluster"] = kmeans.fit_predict(embeddings)
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
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            embeddings_2d = reducer.fit_transform(embeddings)
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
        if config.clustering_method == "KMeans (Fixed)":
            from sklearn.cluster import KMeans

            clusterer = KMeans(n_clusters=config.n_clusters or 10, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(embeddings)
            details_text.text(f"KMeans with k={config.n_clusters}")
        else:
            clusterer = HDBSCAN(
                min_cluster_size=config.min_cluster_size or 15,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            cluster_labels = clusterer.fit_predict(embeddings)
            details_text.text(f"HDBSCAN with min cluster size {config.min_cluster_size}")
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
        reducer = umap.UMAP(n_neighbors=12, min_dist=0.05, metric="cosine", random_state=42)
        umap_coords = reducer.fit_transform(embeddings)
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

def render_semantic_search(state: AppState) -> None:
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

def render_visualization(state: AppState) -> None:
    result = state.result
    if result.df is None:
        st.info("No visualization data available yet.")
        return

    st.header("📊 Visualization")
    st.markdown("Explore the semantic layout of your dataset. Use the controls to adjust node sizes.")

    col1, col2 = st.columns(2)
    with col1:
        state.node_size = st.slider(
            "Node Size",
            min_value=0.2,
            max_value=6.0,
            value=float(state.node_size),
            step=0.2,
            help="Adjust base node size in the visualization",
        )
    with col2:
        state.search_result_size_multiplier = st.slider(
            "Search Result Size Multiplier",
            min_value=1.0,
            max_value=6.0,
            value=float(state.search_result_size_multiplier),
            step=0.5,
            help="Scale nodes that appear in search results",
        )

    if not COMPONENTS_AVAILABLE:
        st.warning("Streamlit components are unavailable. Unable to render Cosmograph visualization.")
        return

    html = generate_cosmograph_html(
        result.df,
        text_column=result.text_column or "hover_text",
        node_size=state.node_size,
        search_result_size_multiplier=state.search_result_size_multiplier,
        search_results=state.search_state.results_df,
    )
    components.html(html, height=720, scrolling=False)

    st.caption("Tip: Use the search controls overlaying the visualization to highlight matches.")


def render_data_explorer(state: AppState) -> None:
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


def render_summary(state: AppState) -> None:
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


def render_welcome(state: AppState) -> None:
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


def render_main_content(state: AppState) -> None:
    if state.modality == "Images" and state.image_records and not state.processed:
        render_image_metadata_editor(state)

    result = state.result
    if not state.processed or result.df is None:
        render_welcome(state)
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


def render_sidebar(state: AppState, saved_embeddings: List[Dict[str, Any]]) -> None:
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
            process_text_data(state, config)
    else:
        config, should_process = image_processing_sidebar(state, saved_embeddings)
        if should_process and config:
            process_image_data(state, config)

def main() -> None:
    st.set_page_config(page_title="Semantic Embedding Explorer", page_icon="🎯", layout="wide")
    state = get_app_state()
    saved_embeddings = list_saved_embeddings()
    render_header()
    render_sidebar(state, saved_embeddings)
    render_main_content(state)


if __name__ == "__main__":
    main()
