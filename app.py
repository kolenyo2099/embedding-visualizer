import streamlit as st
import json
import time
import os
import hashlib
import pickle
import zipfile
from io import BytesIO
import base64
from pathlib import Path
from datetime import datetime

# Robust import handling with error checking
def check_and_import_dependencies():
    """Check and import all required dependencies with helpful error messages"""
    missing_deps = []
    
    # Core dependencies
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

    # ML dependencies
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
    
    # Optional dependencies
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
    
    # Check if critical dependencies are missing
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
        "requests"
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

# Import all dependencies
imports = check_and_import_dependencies()

# Extract imported modules for easier access
pd = imports['pd']
np = imports['np']
torch = imports['torch']
SentenceTransformer = imports['SentenceTransformer']
umap = imports['umap']
HDBSCAN = imports['HDBSCAN']
components = imports['components']
PSUTIL_AVAILABLE = imports['PSUTIL_AVAILABLE']
COMPONENTS_AVAILABLE = imports['COMPONENTS_AVAILABLE']
Image = imports['Image']
T = imports['T']
AutoModel = imports['AutoModel']
AutoProcessor = imports['AutoProcessor']
requests = imports['requests']

# Page configuration
st.set_page_config(
    page_title="Semantic Embedding Explorer",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processing_info' not in st.session_state:
    st.session_state.processing_info = {}
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'search_scores' not in st.session_state:
    st.session_state.search_scores = []
if 'hf_token' not in st.session_state:
    st.session_state.hf_token = ""
if 'uploaded_file_id' not in st.session_state:
    st.session_state.uploaded_file_id = None
if 'node_size' not in st.session_state:
    st.session_state.node_size = 1.0
if 'search_result_size_multiplier' not in st.session_state:
    st.session_state.search_result_size_multiplier = 3.0
if 'modality' not in st.session_state:
    st.session_state.modality = "Text (CSV)"
if 'image_records' not in st.session_state:
    st.session_state.image_records = []
if 'image_metadata' not in st.session_state:
    st.session_state.image_metadata = {}
if 'image_source_id' not in st.session_state:
    st.session_state.image_source_id = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'current_results_modality' not in st.session_state:
    st.session_state.current_results_modality = None


def clear_processed_results():
    """Remove any stored processed outputs from session state."""
    st.session_state.df = None
    st.session_state.embeddings = None
    st.session_state.processed = False
    st.session_state.processing_info = {}
    st.session_state.search_results = []
    st.session_state.search_scores = []
    st.session_state.processed_df = None
    st.session_state.current_results_modality = None
    st.session_state.image_metadata = {}

def get_system_info():
    """Get system information for display"""
    try:
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(memory.total / (1024**3), 1),
                'memory_available_gb': round(memory.available / (1024**3), 1),
                'device': 'MPS (Apple Silicon)' if torch.backends.mps.is_available() else ('CUDA' if torch.cuda.is_available() else 'CPU'),
                'torch_version': torch.__version__
            }
        else:
            return {
                'cpu_count': 'N/A',
                'memory_gb': 'N/A',
                'memory_available_gb': 'N/A',
                'device': 'MPS (Apple Silicon)' if torch.backends.mps.is_available() else ('CUDA' if torch.cuda.is_available() else 'CPU'),
                'torch_version': torch.__version__
            }
    except:
        return {
            'cpu_count': 'N/A', 
            'memory_gb': 'N/A', 
            'memory_available_gb': 'N/A',
            'device': 'CPU', 
            'torch_version': torch.__version__
        }

def format_time(seconds):
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def format_memory(mb):
    """Format memory in human readable format"""
    if mb < 1024:
        return f"{mb:.1f}MB"
    else:
        return f"{mb/1024:.1f}GB"

def get_embeddings_dir():
    """Get or create the embeddings directory"""
    embeddings_dir = Path("embeddings_cache")
    embeddings_dir.mkdir(exist_ok=True)
    return embeddings_dir

def generate_embedding_hash(items, model_name, modality="text"):
    """Generate a unique hash for the embedding configuration"""
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

def save_embeddings(embeddings, items, model_name, metadata=None):
    """Save embeddings to disk"""
    try:
        embeddings_dir = get_embeddings_dir()
        modality = (metadata or {}).get('modality', 'text')
        embedding_hash = generate_embedding_hash(items, model_name, modality=modality)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embeddings_{embedding_hash}_{timestamp}.pkl"
        filepath = embeddings_dir / filename

        # Save embeddings with metadata
        data = {
            'embeddings': embeddings,
            'model_name': model_name,
            'num_texts': len(items) if hasattr(items, '__len__') else None,
            'embedding_dim': embeddings.shape[1] if hasattr(embeddings, 'shape') else None,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        return filepath, embedding_hash
    except Exception as e:
        st.warning(f"Failed to save embeddings: {str(e)}")
        return None, None

def load_embeddings(filepath):
    """Load embeddings from disk"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        st.error(f"Failed to load embeddings: {str(e)}")
        return None

def list_saved_embeddings():
    """List all saved embeddings in the cache directory"""
    embeddings_dir = get_embeddings_dir()
    embeddings_files = list(embeddings_dir.glob("embeddings_*.pkl"))

    embeddings_info = []
    for filepath in embeddings_files:
        try:
            data = load_embeddings(filepath)
            if data:
                embeddings_info.append({
                    'filepath': filepath,
                    'filename': filepath.name,
                    'model_name': data.get('model_name', 'Unknown'),
                    'num_texts': data.get('num_texts', 0),
                    'embedding_dim': data.get('embedding_dim', 0),
                    'timestamp': data.get('timestamp', 'Unknown'),
                    'size_mb': filepath.stat().st_size / (1024 * 1024),
                    'metadata': data.get('metadata', {})
                })
        except:
            continue

    return sorted(embeddings_info, key=lambda x: x['timestamp'], reverse=True)


def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def create_image_record(identifier, data_bytes=None, label=None, link=None, caption=None, url=None):
    return {
        'id': identifier,
        'bytes': data_bytes,
        'label': label or identifier,
        'link': link or '#',
        'caption': caption or '',
        'url': url,
    }


def record_signature(record):
    if record.get('bytes'):
        digest = hashlib.md5(record['bytes']).hexdigest()
    else:
        digest = hashlib.md5((record.get('url') or record['id']).encode()).hexdigest()
    return f"{record['id']}:{digest}"


def load_pil_image(record):
    data_bytes = record.get('bytes')
    if data_bytes is None and record.get('url'):
        response = requests.get(record['url'], timeout=15)
        response.raise_for_status()
        data_bytes = response.content
        record['bytes'] = data_bytes

    image = Image.open(BytesIO(data_bytes))
    return image.convert('RGB')


def create_thumbnail_base64(image, max_size=(160, 160)):
    preview = image.copy()
    preview.thumbnail(max_size)
    buffer = BytesIO()
    preview.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _clean_metadata_value(value, default="", allow_empty=True):
    """Normalize text-like metadata values while treating NaNs as missing."""
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


@st.cache_resource(show_spinner=False)
def load_siglip_model(model_name="google/siglip-base-patch16-512", device="cpu", token=None):
    processor_kwargs = {'trust_remote_code': True}
    model_kwargs = {'trust_remote_code': True}

    if token:
        processor_kwargs['use_auth_token'] = token
        model_kwargs['use_auth_token'] = token

    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

    if device == 'cuda' and torch.cuda.is_available():
        model_kwargs['torch_dtype'] = torch.float16
    else:
        model_kwargs['torch_dtype'] = torch.float32

    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    return model, processor


def encode_images_with_siglip(records, model, processor, device, batch_size=8):
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

        processed = processor(images=pil_images, return_tensors='pt')
        pixel_values = processed.get('pixel_values')

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


def encode_text_with_siglip(texts, model, processor, device, batch_size=16):
    text_embeddings = []

    for batch in chunk_list(texts, batch_size):
        inputs = processor(text=batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            text_embeds = outputs.text_embeds
            text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)

        text_embeddings.append(text_embeds.cpu().numpy())

    if not text_embeddings:
        return np.array([])

    return np.vstack(text_embeddings)

def generate_cosmograph_html(df, search_results=[], search_scores=[], node_size=1.0, search_result_size_multiplier=3.0):
    """Generate the Cosmograph visualization HTML"""
    
    # Get the text column name from session state, or use 'hover_text' as fallback
    text_col = st.session_state.get('text_column', 'hover_text')
    
    optional_cols = [col for col in ['thumbnail', 'caption', 'media_type', 'source_id'] if col in df.columns]

    if text_col in df.columns:
        selected_cols = ['label', 'x', 'y', 'cluster_label', 'hover_text', text_col, 'link'] + optional_cols
        data_for_js = df[selected_cols].copy()
        rename_map = {
            'label': 'id',
            'cluster_label': 'cluster',
            text_col: 'full_text'
        }
        data_for_js = data_for_js.rename(columns=rename_map)
    else:
        selected_cols = ['label', 'x', 'y', 'cluster_label', 'hover_text', 'link'] + optional_cols
        data_for_js = df[selected_cols].copy()
        data_for_js = data_for_js.rename(columns={
            'label': 'id',
            'cluster_label': 'cluster'
        })
        data_for_js['full_text'] = data_for_js['hover_text']

    if 'full_text' not in data_for_js.columns:
        data_for_js['full_text'] = data_for_js['hover_text']
    
    # Convert to dict and sanitize data
    data_list = data_for_js.to_dict('records')
    
    # Sanitize data - replace NaN, Inf, and ensure all values are JSON-serializable
    for row in data_list:
        for key, value in row.items():
            if pd.isna(value):
                row[key] = ""  # Use empty string instead of None for text fields
            elif isinstance(value, (np.integer, np.floating)):
                if np.isnan(value) or np.isinf(value):
                    row[key] = 0.0  # Use 0 for numeric fields
                else:
                    row[key] = float(value)
            # json.dumps will handle string escaping, so don't manually escape
    
    # Add search highlighting info
    for i, row in enumerate(data_list):
        if i in search_results:
            idx = search_results.index(i)
            row['is_search_result'] = True
            row['search_rank'] = idx + 1
            row['search_score'] = float(search_scores[idx]) if search_scores and idx < len(search_scores) else 0.0
        else:
            row['is_search_result'] = False
    
    # Generate colors for clusters
    unique_clusters = sorted(df['cluster_label'].unique())
    color_palette = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
    ]
    cluster_colors = {cluster: color_palette[i % len(color_palette)] 
                      for i, cluster in enumerate(unique_clusters)}
    
    # Base64 encode the JSON to avoid ALL escaping issues
    import base64
    json_str = json.dumps({'data': data_list, 'colors': cluster_colors}, ensure_ascii=True)
    json_b64 = base64.b64encode(json_str.encode('utf-8')).decode('ascii')
    
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{ 
            width: 100%; 
            height: 100%; 
            margin: 0; 
            padding: 0; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            overflow: hidden; 
        }}
        #graph-container {{ 
            width: 100%; 
            height: 100vh; 
            position: relative; 
            background: #1a1a1a;
        }}
        .modal {{
            display: none; position: fixed; z-index: 1000; left: 0; top: 0;
            width: 100%; height: 100%; background: rgba(0,0,0,0.6);
        }}
        .modal-content {{
            background: white; margin: 5% auto; padding: 30px; border-radius: 10px;
            width: 80%; max-width: 900px; max-height: 80vh; overflow-y: auto;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .modal-header {{
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #0088cc;
        }}
        .modal-title {{ color: #333; font-size: 20px; font-weight: bold; }}
        .close {{
            font-size: 32px; font-weight: bold; color: #aaa;
            cursor: pointer; line-height: 20px;
        }}
        .close:hover {{ color: #000; }}
        .cluster-badge {{
            display: inline-block; padding: 6px 14px; background: #0088cc;
            color: white; border-radius: 20px; font-size: 14px; font-weight: bold; margin: 10px 5px;
        }}
        .search-badge {{
            display: inline-block; padding: 6px 14px; background: #ff6b6b;
            color: white; border-radius: 20px; font-size: 14px; font-weight: bold; margin: 10px 5px;
        }}
        .modal-text {{
            line-height: 1.8; font-size: 16px; padding: 20px;
            background: #f9f9f9; border-radius: 8px; border-left: 4px solid #0088cc;
            margin: 20px 0; max-height: 300px; overflow-y: auto;
            white-space: pre-wrap;
        }}
        .modal-media {{
            display: flex; justify-content: center; align-items: center;
            margin: 10px 0;
        }}
        .modal-media img {{
            max-width: 100%; max-height: 220px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        .modal-footer {{
            display: flex; justify-content: space-between; align-items: center;
            padding-top: 15px; border-top: 1px solid #ddd;
        }}
        .modal-link {{
            padding: 12px 28px; background: #0088cc; color: white;
            text-decoration: none; border-radius: 5px; font-weight: bold;
        }}
        .modal-link:hover {{ background: #006699; }}
        .legend {{
            position: absolute; bottom: 20px; left: 20px; background: white;
            padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            max-height: 300px; overflow-y: auto; z-index: 10; font-size: 12px;
        }}
        .legend-title {{ font-weight: bold; margin-bottom: 10px; }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }}
        .search-info {{
            position: absolute; top: 20px; left: 20px; background: #ff6b6b;
            color: white; padding: 15px; border-radius: 8px; font-weight: bold;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3); z-index: 10;
        }}
        .search-controls {{
            position: absolute; top: 20px; right: 20px; background: white;
            padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            z-index: 10; min-width: 250px;
        }}
        .search-input {{
            width: 100%; padding: 8px; margin-bottom: 10px; border: 1px solid #ddd;
            border-radius: 4px; font-size: 14px;
        }}
        .search-button {{
            width: 100%; padding: 8px; background: #0088cc; color: white;
            border: none; border-radius: 4px; cursor: pointer; font-weight: bold;
        }}
        .search-button:hover {{ background: #006699; }}
    </style>
</head>
<body>
    <div id="graph-container"></div>
    
    {f'''<div class="search-info">
        🔍 {len(search_results)} search results highlighted in red
    </div>''' if search_results else ''}
    
    <div class="legend" id="legend"></div>
    
    <div class="search-controls">
        <input type="text" class="search-input" id="searchInput" placeholder="Search in graph...">
        <button class="search-button" onclick="performSearch()">🔍 Search</button>
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
            // Decode base64 data - this completely bypasses all escaping issues
            const b64Data = '{json_b64}';
            const jsonStr = atob(b64Data);
            const jsonData = JSON.parse(jsonStr);
            
            DATA = jsonData.data;
            COLORS = jsonData.colors;
            
            console.log('Successfully loaded', DATA.length, 'data points');
            console.log('Available clusters:', Object.keys(COLORS));
            
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
                    const baseSize = {node_size};
                    const multiplier = {search_result_size_multiplier};
                    return DATA[idx].is_search_result ? (baseSize * multiplier) : baseSize;
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
            
            // Setup legend
            setupLegend();
            
        }} catch(err) {{
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
            if (data.media_type === 'image') {{
                modalText.textContent = data.full_text || data.hover_text || 'No caption provided';
            }} else {{
                modalText.textContent = data.full_text || data.hover_text || 'No text available';
            }}

            let badges = '<span class="cluster-badge">' + data.cluster + '</span>';
            if (data.is_search_result) {{
                badges += '<span class="search-badge">Search Result #' + data.search_rank + 
                          ' (Score: ' + data.search_score.toFixed(3) + ')</span>';
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
        
        // Enhanced search functionality with dynamic node sizes
        window.performSearch = function() {{
            const query = document.getElementById('searchInput').value.toLowerCase().trim();
            if (!query) {{
                // Reset to original state
                if (cosmograph) {{
                    cosmograph.setNodeColor((node) => {{
                        const idx = parseInt(node.id);
                        const data = DATA[idx];
                        return data.is_search_result ? '#ff0000' : (COLORS[data.cluster] || '#999999');
                    }});
                    cosmograph.setNodeSize((node) => {{
                        const idx = parseInt(node.id);
                        const baseSize = {node_size};
                        const multiplier = {search_result_size_multiplier};
                        return DATA[idx].is_search_result ? (baseSize * multiplier) : baseSize;
                    }});
                }}
                
                const searchInfo = document.querySelector('.search-info');
                if (searchInfo) {{
                    searchInfo.textContent = '🔍 ' + (DATA.filter(d => d.is_search_result).length) + ' search results highlighted in red';
                }}
                return;
            }}
            
            // Find matching nodes
            const matchingIndices = [];
            DATA.forEach((d, i) => {{
                const searchText = (d.full_text + ' ' + d.hover_text + ' ' + d.id).toLowerCase();
                if (searchText.includes(query)) {{
                    matchingIndices.push(i);
                }}
            }});
            
            // Update visualization
            if (cosmograph) {{
                cosmograph.setNodeColor((node) => {{
                    const idx = parseInt(node.id);
                    const data = DATA[idx];
                    if (matchingIndices.includes(idx)) {{
                        return '#ff6b35'; // Orange for search matches
                    }} else {{
                        return data.is_search_result ? '#ff0000' : (COLORS[data.cluster] || '#999999');
                    }}
                }});
                cosmograph.setNodeSize((node) => {{
                    const idx = parseInt(node.id);
                    const data = DATA[idx];
                    const baseSize = {node_size};
                    const multiplier = {search_result_size_multiplier};
                    if (matchingIndices.includes(idx)) {{
                        return baseSize * 4; // Larger for search matches
                    }} else {{
                        return data.is_search_result ? (baseSize * multiplier) : baseSize;
                    }}
                }});
            }}
            
            // Update search info
            const searchInfo = document.querySelector('.search-info');
            if (searchInfo) {{
                searchInfo.textContent = '🔍 ' + matchingIndices.length + ' matches for "' + query + '"';
            }}
        }}
        
        // Allow Enter key to search
        document.getElementById('searchInput').addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') {{
                performSearch();
            }}
        }});
    </script>
</body>
</html>"""
    
    return html_template


# Title and description
st.title("🎯 Semantic Embedding Explorer")
st.markdown("""
Visualize and explore text **or image** data using state-of-the-art embeddings and interactive clustering.
Choose your modality, upload your sources, and explore semantic relationships in your content.
""")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Modality selection
modality_options = ["Text (CSV)", "Images"]
selected_modality = st.sidebar.radio(
    "Data Modality",
    modality_options,
    index=modality_options.index(st.session_state.modality),
    help="Switch between processing text data from CSV files or image collections",
)

if selected_modality != st.session_state.modality:
    st.session_state.modality = selected_modality
    clear_processed_results()

    if selected_modality == "Text (CSV)":
        st.session_state.image_records = []
        st.session_state.image_source_id = None
    else:
        st.session_state.raw_df = None
        st.session_state.uploaded_file_id = None

# Hugging Face Token Input
with st.sidebar.expander("🔑 Hugging Face Authentication", expanded=False):
    hf_token = st.text_input(
        "Hugging Face Token",
        type="password",
        value=st.session_state.hf_token,
        help="Enter your Hugging Face token to access gated models. Get your token at: https://huggingface.co/settings/tokens",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    )
    
    # Update session state with token
    if hf_token != st.session_state.hf_token:
        st.session_state.hf_token = hf_token
    
    # Show token status
    if st.session_state.hf_token:
        if st.session_state.hf_token.startswith("hf_"):
            st.success("✅ Valid token format detected")
        else:
            st.warning("⚠️ Token should start with 'hf_'")
    else:
        st.info("ℹ️ Token optional for public models, required for gated models")

# System Info
system_info = get_system_info()
with st.sidebar.expander("💻 System Info", expanded=False):
    st.markdown(f"""
    **Device:** {system_info['device']}
    **CPU Cores:** {system_info['cpu_count']}
    **Memory:** {system_info['memory_gb']}GB total
    **Available:** {system_info['memory_available_gb']}GB
    **PyTorch:** {system_info['torch_version']}
    """)

saved_embeddings = list_saved_embeddings()

if st.session_state.modality == "Text (CSV)":
    # Step 1: File Upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="Upload a CSV file containing text data"
    )

    if uploaded_file is not None:
        # Load CSV
        try:
            # Create a unique ID for the uploaded file based on name and size
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
            # Only reload if it's a different file
            if st.session_state.uploaded_file_id != file_id:
                df = pd.read_csv(uploaded_file)
                st.session_state.raw_df = df
                clear_processed_results()
                st.session_state.uploaded_file_id = file_id

            # Use the dataframe from session state
            df = st.session_state.raw_df
        
            st.sidebar.success(f"✅ Loaded {len(df)} rows")
        
            # Step 2: Column Selection
            st.sidebar.subheader("📊 Column Selection")
        
            text_column = st.sidebar.selectbox(
                "Text Column (to embed)",
                options=df.columns.tolist(),
                help="Select the column containing text to embed"
            )
        
            label_column = st.sidebar.selectbox(
                "Label Column (for node titles)",
                options=['Index'] + df.columns.tolist(),
                help="Select the column to use as node labels in the modal"
            )
        
            # Optional: Link column
            has_link = st.sidebar.checkbox("I have a URL column", value=False)
            if has_link:
                link_column = st.sidebar.selectbox(
                    "Link Column (optional)",
                    options=[None] + df.columns.tolist(),
                    help="Column containing URLs to open when clicking nodes"
                )
            else:
                link_column = None
        
            # Step 3: Clustering Settings
            st.sidebar.subheader("🎨 Clustering Settings")
        
            clustering_method = st.sidebar.radio(
                "Clustering Method",
                options=["HDBSCAN (Automatic)", "KMeans (Fixed)"],
                help="HDBSCAN finds clusters automatically, KMeans requires specifying the number"
            )
        
            if clustering_method == "KMeans (Fixed)":
                n_clusters = st.sidebar.slider(
                    "Number of Clusters",
                    min_value=2,
                    max_value=20,
                    value=10,
                    help="How many clusters to create"
                )
            else:
                min_cluster_size = st.sidebar.slider(
                    "Minimum Cluster Size",
                    min_value=10,
                    max_value=200,
                    value=50,
                    help="Minimum points to form a cluster (higher = fewer, larger clusters)"
                )
        
        
            # Step 4: Embedding Settings
            st.sidebar.subheader("🤖 Embedding Settings")
        
            # Check for saved embeddings
            text_saved_embeddings = [
                emb for emb in saved_embeddings
                if emb.get('metadata', {}).get('modality', 'text') in ('text', 'Text', None)
            ]

            use_cached = st.sidebar.checkbox(
                "📁 Use Cached Embeddings",
                value=False,
                help="Load previously saved embeddings instead of creating new ones"
            )

            if use_cached and text_saved_embeddings:
                st.sidebar.markdown("**Available Cached Embeddings:**")

                # Create a selection for cached embeddings
                embedding_options = []
                for idx, emb_info in enumerate(text_saved_embeddings):
                    label = f"{emb_info['model_name']} | {emb_info['num_texts']} items | {emb_info['timestamp']}"
                    embedding_options.append(label)

                selected_cache_idx = st.sidebar.selectbox(
                    "Select Cached Embedding",
                    options=range(len(embedding_options)),
                    format_func=lambda x: embedding_options[x],
                    help="Choose from previously saved embeddings"
                )
            
                selected_cache = text_saved_embeddings[selected_cache_idx]
            
                # Display cache info
                with st.sidebar.expander("📊 Cache Details"):
                    st.markdown(f"""
                    **Model:** {selected_cache['model_name']}  
                    **Texts:** {selected_cache['num_texts']}  
                    **Dimensions:** {selected_cache['embedding_dim']}D  
                    **Size:** {selected_cache['size_mb']:.2f} MB  
                    **Created:** {selected_cache['timestamp']}
                    """)
            
                model_name = selected_cache['model_name']  # For compatibility
            else:
                if use_cached and not text_saved_embeddings:
                    st.sidebar.warning("⚠️ No cached embeddings found")
            
                model_name = st.sidebar.selectbox(
                    "Embedding Model",
                    options=[
                        "google/embeddinggemma-300m",
                        "nomic-ai/nomic-embed-text-v1.5", 
                        "BAAI/bge-base-en-v1.5",
                        "sentence-transformers/all-mpnet-base-v2"
                    ],
                    help="Choose the embedding model (EmbeddingGemma-300m is best under 500M params)"
                )
            
                save_embeddings_checkbox = st.sidebar.checkbox(
                    "💾 Save Embeddings for Later",
                    value=True,
                    help="Save created embeddings to disk for future use"
                )
        
            # Step 5: Process Button
            if st.sidebar.button("🚀 Process Data", type="primary", use_container_width=True):
                # Create a placeholder for processing details
                processing_placeholder = st.empty()
            
                with processing_placeholder.container():
                    st.markdown("### 🔄 Processing Data")
                
                    # Initialize timing
                    start_time = time.time()
                    processing_steps = []
                
                    # Clean data
                    with st.spinner("🧹 Cleaning and preparing data..."):
                        step_start = time.time()
                        original_count = len(df)
                        df_clean = df.dropna(subset=[text_column]).reset_index(drop=True)
                        cleaned_count = len(df_clean)
                    
                        # Create label column
                        if label_column == 'Index':
                            df_clean['label'] = df_clean.index.astype(str)
                        else:
                            df_clean['label'] = df_clean[label_column].astype(str)
                    
                        # Create link column
                        if link_column:
                            df_clean['link'] = df_clean[link_column].fillna('#')
                        else:
                            df_clean['link'] = '#'
                    
                        step_time = time.time() - step_start
                        processing_steps.append({
                            'step': 'Data Cleaning',
                            'time': step_time,
                            'details': f"Processed {original_count} → {cleaned_count} rows"
                        })
                
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    details_text = st.empty()
                
                    # Load model
                    status_text.text("🤖 Loading embedding model...")
                    details_text.text(f"Model: {model_name}")
                    progress_bar.progress(10)
                
                    step_start = time.time()
                    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
                
                    # Prepare model loading with token
                    model_kwargs = {
                        'device': device,
                        'trust_remote_code': True
                    }
                
                    # Add token if available
                    if st.session_state.hf_token:
                        model_kwargs['token'] = st.session_state.hf_token
                        details_text.text(f"Model: {model_name} (with authentication)")
                    else:
                        details_text.text(f"Model: {model_name} (no token)")
                
                    try:
                        model = SentenceTransformer(model_name, **model_kwargs)
                        st.session_state.model = model
                        step_time = time.time() - step_start
                        processing_steps.append({
                            'step': 'Model Loading',
                            'time': step_time,
                            'details': f"Device: {device.upper()} - Authenticated: {'Yes' if st.session_state.hf_token else 'No'}"
                        })
                    except Exception as model_error:
                        # Handle authentication errors
                        if "401" in str(model_error) or "authentication" in str(model_error).lower():
                            st.error(f"❌ Authentication failed for model {model_name}")
                            st.error("Please check your Hugging Face token and ensure you have accepted the model's terms of use.")
                            st.error("Get your token at: https://huggingface.co/settings/tokens")
                            processing_placeholder.empty()
                            st.stop()
                        elif "gated" in str(model_error).lower() or "private" in str(model_error).lower():
                            st.error(f"❌ Model {model_name} requires authentication")
                            st.error("Please enter a valid Hugging Face token in the sidebar.")
                            processing_placeholder.empty()
                            st.stop()
                        else:
                            st.error(f"❌ Failed to load model {model_name}: {str(model_error)}")
                            processing_placeholder.empty()
                            st.stop()
                
                    # Create or load embeddings
                    try:
                        texts_to_embed = df_clean[text_column].tolist()
                        total_texts = len(texts_to_embed)
                    
                        # Check if we should use cached embeddings
                        if use_cached and text_saved_embeddings:
                            status_text.text("📁 Loading cached embeddings...")
                            progress_bar.progress(20)
                        
                            step_start = time.time()
                        
                            # Load the selected cached embeddings
                            cached_data = load_embeddings(selected_cache['filepath'])
                        
                            if cached_data and cached_data['num_texts'] == total_texts:
                                embeddings = cached_data['embeddings']
                                st.session_state.embeddings = embeddings
                            
                                step_time = time.time() - step_start
                                processing_steps.append({
                                    'step': 'Load Cached Embeddings',
                                    'time': step_time,
                                    'details': f"Loaded {len(embeddings)} embeddings ({embeddings.shape[1]}D) from cache"
                                })
                            
                                status_text.text("✅ Cached embeddings loaded!")
                                details_text.text(f"Loaded {total_texts} embeddings in {step_time:.2f}s")
                            else:
                                st.warning("⚠️ Cached embeddings don't match current data size. Creating new embeddings...")
                                use_cached = False  # Fall through to create new embeddings
                    
                        if not use_cached or not text_saved_embeddings:
                            status_text.text("🔤 Creating text embeddings...")
                            progress_bar.progress(20)
                        
                            step_start = time.time()
                        
                            # Show embedding progress
                            embedding_progress = st.progress(0)
                            batch_size = 64
                            num_batches = (total_texts + batch_size - 1) // batch_size
                        
                            embeddings = []
                            for i in range(0, total_texts, batch_size):
                                batch_texts = texts_to_embed[i:i+batch_size]
                                batch_embeddings = model.encode(
                                    batch_texts,
                                    batch_size=batch_size,
                                    show_progress_bar=False,
                                    normalize_embeddings=True
                                )
                                embeddings.extend(batch_embeddings)
                            
                                # Update progress
                                batch_num = i // batch_size + 1
                                progress = min((batch_num / num_batches), 1.0)  # Progress from 0 to 1
                                embedding_progress.progress(progress)
                                status_text.text(f"🔤 Creating embeddings... Batch {batch_num}/{num_batches}")
                                details_text.text(f"Processed {min(i + batch_size, total_texts)}/{total_texts} texts")
                        
                            embeddings = np.array(embeddings)
                            st.session_state.embeddings = embeddings
                            embedding_progress.empty()
                        
                            step_time = time.time() - step_start
                            processing_steps.append({
                                'step': 'Text Embeddings',
                                'time': step_time,
                                'details': f"Created {len(embeddings)} embeddings ({embeddings.shape[1]}D)"
                            })
                        
                            # Save embeddings if requested
                        if save_embeddings_checkbox:
                            status_text.text("💾 Saving embeddings to cache...")
                            save_path, emb_hash = save_embeddings(
                                embeddings,
                                texts_to_embed,
                                model_name,
                                metadata={'text_column': text_column, 'modality': 'text'}
                            )
                            if save_path:
                                details_text.text(f"Saved to: {save_path.name}")
                                processing_steps.append({
                                    'step': 'Save Embeddings',
                                    'time': 0,
                                    'details': f"Saved to embeddings_cache/{save_path.name}"
                                })
                    
                        progress_bar.progress(50)
                    except Exception as embed_error:
                        st.error(f"❌ Error with embeddings: {str(embed_error)}")
                        st.error(f"Error type: {type(embed_error).__name__}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
                        processing_placeholder.empty()
                        st.stop()
                
                    # Clustering
                    try:
                        clustering_method_display = "HDBSCAN (Automatic)" if clustering_method == "HDBSCAN (Automatic)" else f"KMeans ({n_clusters} clusters)"
                        status_text.text(f"🎨 Performing clustering: {clustering_method_display}...")
                    
                        step_start = time.time()
                        if clustering_method == "HDBSCAN (Automatic)":
                            details_text.text(f"Minimum cluster size: {min_cluster_size}")
                            clusterer = HDBSCAN(
                                min_cluster_size=min_cluster_size,
                                min_samples=10,
                                metric='euclidean'
                            )
                            df_clean['cluster'] = clusterer.fit_predict(embeddings)
                        else:
                            details_text.text(f"Creating {n_clusters} clusters")
                            from sklearn.cluster import KMeans
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            df_clean['cluster'] = kmeans.fit_predict(embeddings)
                    
                        df_clean['cluster_label'] = df_clean['cluster'].apply(
                            lambda x: 'Noise' if x == -1 else f'Cluster {x}'
                        )
                    
                        step_time = time.time() - step_start
                        n_clusters_found = len(df_clean['cluster'].unique())
                        noise_points = (df_clean['cluster'] == -1).sum()
                        processing_steps.append({
                            'step': 'Clustering',
                            'time': step_time,
                            'details': f"Found {n_clusters_found} clusters, {noise_points} noise points"
                        })
                    
                        progress_bar.progress(70)
                    except Exception as cluster_error:
                        st.error(f"❌ Error during clustering: {str(cluster_error)}")
                        st.error(f"Error type: {type(cluster_error).__name__}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
                        processing_placeholder.empty()
                        st.stop()
                
                    # UMAP reduction
                    try:
                        status_text.text("🗺️ Reducing dimensionality with UMAP...")
                        details_text.text("Creating 2D coordinates for visualization")
                    
                        step_start = time.time()
                        reducer = umap.UMAP(
                            n_components=2,
                            n_neighbors=15,
                            min_dist=0.1,
                            metric='cosine',
                            random_state=42
                        )
                        embeddings_2d = reducer.fit_transform(embeddings)
                    
                        df_clean['x'] = embeddings_2d[:, 0]
                        df_clean['y'] = embeddings_2d[:, 1]
                    
                        # Create hover text
                        df_clean['hover_text'] = df_clean[text_column].str[:150] + '...'
                    
                        step_time = time.time() - step_start
                        processing_steps.append({
                            'step': 'Dimensionality Reduction',
                            'time': step_time,
                            'details': f"UMAP: {embeddings.shape[1]}D → 2D"
                        })
                    
                        progress_bar.progress(90)
                    except Exception as umap_error:
                        st.error(f"❌ Error during UMAP dimensionality reduction: {str(umap_error)}")
                        st.error(f"Error type: {type(umap_error).__name__}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
                        processing_placeholder.empty()
                        st.stop()
                
                    # Finalizing
                    status_text.text("⚡ Finalizing results...")
                    details_text.text("Preparing visualization data")
                
                    step_start = time.time()
                    st.session_state.df = df_clean
                    st.session_state.processed_df = df_clean
                    st.session_state.current_results_modality = 'text'
                    st.session_state.processed = True
                    st.session_state.text_column = text_column

                    # Store processing info
                    total_time = time.time() - start_time
                    st.session_state.processing_info = {
                        'total_time': total_time,
                        'steps': processing_steps,
                        'data_stats': {
                            'total_points': len(df_clean),
                            'clusters': n_clusters_found,
                            'noise_points': noise_points,
                            'embedding_dim': embeddings.shape[1],
                            'model': model_name,
                            'device': device.upper(),
                            'modality': 'text'
                        }
                    }

                    step_time = time.time() - step_start
                    processing_steps.append({
                        'step': 'Finalization',
                        'time': step_time,
                        'details': "Data saved to session"
                    })

                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    details_text.text(f"Total time: {format_time(total_time)}")

                    # Show processing summary
                    st.markdown("### 📊 Processing Summary")
                    summary_col1, summary_col2, summary_col3 = st.columns(3)

                    with summary_col1:
                        st.metric("Total Time", format_time(total_time))
                        st.metric("Data Points", len(df_clean))

                    with summary_col2:
                        st.metric("Clusters Found", n_clusters_found)
                        st.metric("Noise Points", noise_points)

                    with summary_col3:
                        st.metric("Embedding Dim", f"{embeddings.shape[1]}D")
                        st.metric("Processing Speed", f"{len(df_clean)/total_time:.1f} items/sec")
                
                    # Detailed timing breakdown
                    with st.expander("⏱️ Detailed Timing Breakdown"):
                        timing_df = pd.DataFrame(processing_steps)
                        timing_df['percentage'] = (timing_df['time'] / timing_df['time'].sum() * 100).round(1)
                        timing_df.columns = ['Step', 'Time (seconds)', 'Details', 'Percentage']
                        timing_df['Time (formatted)'] = timing_df['Time (seconds)'].apply(format_time)
                        st.dataframe(timing_df[['Step', 'Time (formatted)', 'Percentage', 'Details']], 
                                    use_container_width=True, hide_index=True)
                
                    # Clear the processing placeholder after a delay
                    time.sleep(3)
                    processing_placeholder.empty()
                
                    st.sidebar.success("✅ Processing complete!")
                    st.rerun()
        
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")



elif st.session_state.modality == "Images":
    st.sidebar.subheader("🖼️ Image Source")
    image_source = st.sidebar.radio(
        "Choose how to provide images",
        options=["Upload Images", "Upload Folder (ZIP)", "CSV with Image URLs"],
        help="Supply individual image files, a compressed folder, or a CSV that points to image URLs"
    )

    image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

    if image_source == "Upload Images":
        uploaded_images = st.sidebar.file_uploader(
            "Upload image files",
            type=[ext.replace('.', '') for ext in image_extensions],
            accept_multiple_files=True,
            help="Select one or more images to embed"
        )

        if uploaded_images:
            source_id = tuple((img.name, img.size) for img in uploaded_images)
            if st.session_state.image_source_id != source_id:
                records = []
                for file in uploaded_images:
                    data = file.read()
                    if not data:
                        continue
                    label = os.path.splitext(file.name)[0]
                    records.append(create_image_record(file.name, data_bytes=data, label=label))

                if records:
                    st.session_state.image_records = records
                    st.session_state.image_source_id = source_id
                    clear_processed_results()
                    st.sidebar.success(f"✅ Loaded {len(records)} images")
    elif image_source == "Upload Folder (ZIP)":
        uploaded_zip = st.sidebar.file_uploader(
            "Upload ZIP archive",
            type=['zip'],
            help="Provide a compressed folder containing images"
        )

        if uploaded_zip is not None:
            source_id = (uploaded_zip.name, uploaded_zip.size)
            if st.session_state.image_source_id != source_id:
                records = []
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
                except Exception as zip_error:
                    records = []
                    st.sidebar.error(f"Failed to extract ZIP: {zip_error}")

                if records:
                    st.session_state.image_records = records
                    st.session_state.image_source_id = source_id
                    clear_processed_results()
                    st.sidebar.success(f"✅ Loaded {len(records)} images from archive")
    else:
        url_csv = st.sidebar.file_uploader(
            "Upload CSV with image URLs",
            type=['csv'],
            help="CSV should include at least one column with image URLs"
        )

        if url_csv is not None:
            try:
                url_df = pd.read_csv(url_csv)
                if url_df.empty:
                    st.sidebar.warning("CSV appears to be empty")
                else:
                    url_column = st.sidebar.selectbox(
                        "Image URL Column",
                        options=url_df.columns.tolist()
                    )
                    label_column_choice = st.sidebar.selectbox(
                        "Label Column (optional)",
                        options=[None] + url_df.columns.tolist()
                    )
                    link_column_choice = st.sidebar.selectbox(
                        "Link Column (optional)",
                        options=[None] + url_df.columns.tolist()
                    )
                    caption_column_choice = st.sidebar.selectbox(
                        "Caption Column (optional)",
                        options=[None] + url_df.columns.tolist()
                    )
                    limit = st.sidebar.number_input(
                        "Max rows to load",
                        min_value=1,
                        max_value=len(url_df),
                        value=min(len(url_df), 500),
                        help="Limit number of images to fetch"
                    )

                    records = []
                    for idx, row in url_df.head(limit).iterrows():
                        url_value = row.get(url_column)
                        url = _clean_metadata_value(url_value, default="", allow_empty=False)
                        if not url:
                            continue

                        default_label = f"Image {idx + 1}"
                        if label_column_choice:
                            label = _clean_metadata_value(row.get(label_column_choice), default=default_label, allow_empty=False)
                        else:
                            label = default_label

                        if caption_column_choice:
                            caption = _clean_metadata_value(row.get(caption_column_choice), default="", allow_empty=True)
                        else:
                            caption = ""

                        if link_column_choice:
                            link_value = _clean_metadata_value(row.get(link_column_choice), default="#", allow_empty=False)
                        else:
                            link_value = "#"

                        records.append(create_image_record(
                            identifier=f"row_{idx}",
                            label=label,
                            link=link_value or '#',
                            caption=caption,
                            url=url
                        ))

                    if records:
                        meta_signature = (url_csv.name, url_csv.size, url_column, label_column_choice, link_column_choice, caption_column_choice, limit)
                        st.session_state.image_records = records
                        st.session_state.image_source_id = meta_signature
                        clear_processed_results()
                        st.sidebar.success(f"✅ Prepared {len(records)} image references")
            except Exception as csv_error:
                st.sidebar.error(f"Failed to parse CSV: {csv_error}")
                st.session_state.image_records = []
                clear_processed_results()

    image_records = st.session_state.image_records

    if image_records:
        st.sidebar.info(f"📦 {len(image_records)} images ready for embedding")

        editor_df = pd.DataFrame({
            'Identifier': [rec['id'] for rec in image_records],
            'Label': [rec['label'] for rec in image_records],
            'Caption': [rec['caption'] for rec in image_records],
            'Link': [rec['link'] for rec in image_records],
        })
        edited_df = st.data_editor(
            editor_df,
            num_rows="dynamic",
            use_container_width=True,
            disabled=['Identifier'],
            key='image_metadata_editor'
        )

        if not edited_df.empty:
            updated_records = []
            for record, row in zip(image_records, edited_df.to_dict('records')):
                record['label'] = _clean_metadata_value(row.get('Label'), default=record['label'], allow_empty=False)
                record['caption'] = _clean_metadata_value(row.get('Caption'), default="", allow_empty=True)
                record['link'] = _clean_metadata_value(row.get('Link'), default="#", allow_empty=False) or '#'
                updated_records.append(record)
            st.session_state.image_records = updated_records
            image_records = updated_records

        st.sidebar.subheader("🎨 Clustering Settings")
        image_clustering_method = st.sidebar.radio(
            "Clustering Method",
            options=["HDBSCAN (Automatic)", "KMeans (Fixed)"],
            help="Choose how to group images"
        )

        if image_clustering_method == "KMeans (Fixed)":
            image_n_clusters = st.sidebar.slider(
                "Number of Clusters",
                min_value=2,
                max_value=min(30, len(image_records)),
                value=min(10, len(image_records)),
                help="Fixed number of clusters"
            )
        else:
            image_min_cluster_size = st.sidebar.slider(
                "Minimum Cluster Size",
                min_value=5,
                max_value=max(5, min(200, len(image_records))),
                value=min(25, len(image_records)),
                help="Minimum number of images required to form a cluster"
            )

        st.sidebar.subheader("🤖 Embedding Settings")
        image_saved_embeddings = [
            emb for emb in saved_embeddings
            if emb.get('metadata', {}).get('modality') == 'image'
        ]

        image_use_cached = st.sidebar.checkbox(
            "📁 Use Cached Embeddings",
            value=False,
            help="Reuse previously computed image embeddings"
        )

        if image_use_cached and image_saved_embeddings:
            cache_labels = []
            for emb in image_saved_embeddings:
                meta = emb.get('metadata', {})
                preprocess = meta.get('preprocessing', {}).get('description', 'SigLIP')
                cache_labels.append(f"{emb['model_name']} | {preprocess} | {emb['timestamp']}")

            image_cache_idx = st.sidebar.selectbox(
                "Select Cached Embedding",
                options=range(len(cache_labels)),
                format_func=lambda x: cache_labels[x],
                help="Choose which cached embedding set to load"
            )
            image_selected_cache = image_saved_embeddings[image_cache_idx]

            with st.sidebar.expander("📊 Cache Details"):
                meta = image_selected_cache.get('metadata', {})
                st.markdown(f"""
                **Model:** {image_selected_cache['model_name']}
                **Images:** {image_selected_cache['num_texts']}
                **Dimensions:** {image_selected_cache['embedding_dim']}D
                **Modality:** {meta.get('modality', 'image')}
                """)
        else:
            if image_use_cached and not image_saved_embeddings:
                st.sidebar.warning("⚠️ No cached image embeddings found")
            image_selected_cache = None

        save_image_embeddings = st.sidebar.checkbox(
            "💾 Save Image Embeddings for Later",
            value=True,
            help="Persist computed embeddings to disk"
        )

        if st.sidebar.button("🚀 Process Images", type="primary", use_container_width=True):
            processing_placeholder = st.empty()

            with processing_placeholder.container():
                st.markdown("### 🔄 Processing Images")
                start_time = time.time()
                processing_steps = []

                with st.spinner("🧹 Validating image records..."):
                    step_start = time.time()
                    valid_records = []
                    for record in image_records:
                        try:
                            if record.get('bytes') is None and not record.get('url'):
                                continue
                            load_pil_image(record)
                            valid_records.append(record)
                        except Exception:
                            continue

                    if not valid_records:
                        st.error("No valid images available for processing.")
                        processing_placeholder.empty()
                        st.stop()

                    step_time = time.time() - step_start
                    processing_steps.append({
                        'step': 'Validation',
                        'time': step_time,
                        'details': f"Validated {len(valid_records)} images"
                    })

                current_signatures = [record_signature(rec) for rec in valid_records]

                progress_bar = st.progress(0)
                status_text = st.empty()
                details_text = st.empty()

                device = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
                status_text.text("🤖 Loading SigLIP model...")
                details_text.text(f"Device: {device.upper()}")
                progress_bar.progress(10)

                try:
                    model, processor = load_siglip_model(
                        model_name="google/siglip-base-patch16-512",
                        device=device,
                        token=st.session_state.hf_token or None
                    )
                except Exception as model_error:
                    st.error(f"❌ Failed to load SigLIP model: {model_error}")
                    processing_placeholder.empty()
                    st.stop()

                embeddings = None

                if image_use_cached and image_selected_cache:
                    status_text.text("📁 Loading cached embeddings...")
                    progress_bar.progress(25)
                    cached_data = load_embeddings(image_selected_cache['filepath'])
                    meta = (cached_data or {}).get('metadata', {})
                    cached_signatures = meta.get('item_signatures', [])
                    if cached_data and set(cached_signatures) == set(current_signatures):
                        embeddings = cached_data['embeddings']
                        st.session_state.embeddings = embeddings
                        processing_steps.append({
                            'step': 'Load Cached Embeddings',
                            'time': 0,
                            'details': f"Loaded {len(embeddings)} cached vectors"
                        })
                    else:
                        st.warning("⚠️ Cached embeddings do not match current images. Recomputing...")
                        image_use_cached = False

                if embeddings is None:
                    status_text.text("🖼️ Encoding images with SigLIP...")
                    progress_bar.progress(35)
                    step_start = time.time()
                    embeddings, encoded_records = encode_images_with_siglip(valid_records, model, processor, device)

                    if embeddings.size == 0:
                        st.error("❌ Unable to encode images")
                        processing_placeholder.empty()
                        st.stop()

                    valid_records = encoded_records
                    current_signatures = [record_signature(rec) for rec in valid_records]
                    st.session_state.embeddings = embeddings
                    step_time = time.time() - step_start
                    processing_steps.append({
                        'step': 'Image Embeddings',
                        'time': step_time,
                        'details': f"Encoded {len(valid_records)} images to {embeddings.shape[1]}D vectors"
                    })

                    if save_image_embeddings:
                        status_text.text("💾 Saving embeddings to cache...")
                        meta = {
                            'modality': 'image',
                            'item_signatures': current_signatures,
                            'preprocessing': {
                                'description': 'SigLIP Base Patch16 512',
                                'image_normalization': 'SigLIP default'
                            }
                        }
                        save_path, emb_hash = save_embeddings(
                            embeddings,
                            current_signatures,
                            "google/siglip-base-patch16-512",
                            metadata=meta
                        )
                        if save_path:
                            processing_steps.append({
                                'step': 'Save Embeddings',
                                'time': 0,
                                'details': f"Saved cache as {save_path.name}"
                            })

                progress_bar.progress(55)

                status_text.text("🗺️ Running dimensionality reduction...")
                details_text.text("Projecting embeddings to 2D space")
                step_start = time.time()
                reducer = umap.UMAP(n_neighbors=12, min_dist=0.05, metric='cosine', random_state=42)
                umap_coords = reducer.fit_transform(embeddings)
                step_time = time.time() - step_start
                processing_steps.append({
                    'step': 'UMAP Reduction',
                    'time': step_time,
                    'details': f"Computed coordinates for {len(umap_coords)} images"
                })

                progress_bar.progress(70)

                status_text.text("🎯 Clustering embeddings...")
                step_start = time.time()
                if image_clustering_method == "KMeans (Fixed)":
                    from sklearn.cluster import KMeans
                    clusterer = KMeans(n_clusters=image_n_clusters, random_state=42, n_init=10)
                    cluster_labels = clusterer.fit_predict(embeddings)
                    details_text.text(f"KMeans with k={image_n_clusters}")
                else:
                    clusterer = HDBSCAN(min_cluster_size=image_min_cluster_size, metric='euclidean', cluster_selection_method='eom')
                    cluster_labels = clusterer.fit_predict(embeddings)
                    details_text.text(f"HDBSCAN with min cluster size {image_min_cluster_size}")

                step_time = time.time() - step_start
                n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                noise_points = int(np.sum(cluster_labels == -1))
                processing_steps.append({
                    'step': 'Clustering',
                    'time': step_time,
                    'details': f"Found {n_clusters_found} clusters with {noise_points} noise images"
                })

                progress_bar.progress(85)

                status_text.text("🗂️ Preparing visualization data...")
                step_start = time.time()

                thumbnails = []
                captions = []
                for record in valid_records:
                    try:
                        image = load_pil_image(record)
                        thumb_b64 = create_thumbnail_base64(image)
                    except Exception:
                        thumb_b64 = ''
                    thumbnails.append(thumb_b64)

                    caption_text = _clean_metadata_value(record.get('caption'), default="", allow_empty=True)
                    if not caption_text:
                        caption_text = _clean_metadata_value(record.get('label'), default=record.get('id', ''), allow_empty=False)
                    captions.append(caption_text)

                df_clean = pd.DataFrame({
                    'label': [rec['label'] for rec in valid_records],
                    'link': [rec.get('link', '#') or '#' for rec in valid_records],
                    'caption': captions,
                    'thumbnail': thumbnails,
                    'source_id': [rec['id'] for rec in valid_records],
                })
                df_clean['x'] = umap_coords[:, 0]
                df_clean['y'] = umap_coords[:, 1]
                df_clean['cluster'] = cluster_labels
                df_clean['cluster_label'] = df_clean['cluster'].apply(lambda x: f"Cluster {x}" if x != -1 else "Noise")
                df_clean['hover_text'] = df_clean['caption']
                df_clean['media_type'] = 'image'

                step_time = time.time() - step_start
                processing_steps.append({
                    'step': 'Prepare DataFrame',
                    'time': step_time,
                    'details': "Added visualization metadata"
                })

                progress_bar.progress(95)

                status_text.text("⚡ Finalizing results...")
                total_time = time.time() - start_time
                st.session_state.df = df_clean
                st.session_state.embeddings = embeddings
                st.session_state.processed = True
                st.session_state.processed_df = df_clean
                st.session_state.current_results_modality = 'image'
                st.session_state.text_column = 'caption'
                st.session_state.processing_info = {
                    'total_time': total_time,
                    'steps': processing_steps,
                    'data_stats': {
                        'total_points': len(df_clean),
                        'clusters': n_clusters_found,
                        'noise_points': noise_points,
                        'embedding_dim': embeddings.shape[1],
                        'model': 'google/siglip-base-patch16-512',
                        'device': device.upper(),
                        'modality': 'image'
                    }
                }
                st.session_state.search_results = []
                st.session_state.search_scores = []
                st.session_state.image_metadata = {
                    'source': image_source,
                    'images': len(df_clean)
                }

                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                details_text.text(f"Processed {len(df_clean)} images in {format_time(total_time)}")

                st.sidebar.success("✅ Processing complete!")
                time.sleep(2)
                processing_placeholder.empty()
                st.rerun()
    else:
        st.info("👆 Upload images or provide a CSV with image URLs to get started!")

# Main content area
results_df = st.session_state.df
if results_df is None:
    results_df = st.session_state.get('processed_df')
    if results_df is not None:
        st.session_state.df = results_df

if st.session_state.processed and results_df is not None:
    df = results_df
    
    # Debug: Show what columns we have
    # st.write("DEBUG - Columns in df:", list(df.columns))
    
    # Verify that all required columns exist
    required_columns = ['label', 'x', 'y', 'cluster_label', 'hover_text', 'link']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Processing incomplete. Missing columns: {', '.join(missing_columns)}")
        st.error(f"Current columns: {', '.join(list(df.columns))}")
        st.info("💡 Please click the '🚀 Process Data' button in the sidebar to create embeddings and prepare the visualization.")
        # Reset stored results since data isn't actually processed
        clear_processed_results()
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Semantic Search", "📊 Visualization", "📈 Data Explorer"])
    
    with tab1:
        st.header("🔍 Semantic Search")
        st.markdown("Search for nodes semantically similar to your query.")
        
        data_stats = st.session_state.processing_info.get('data_stats', {})
        current_modality = data_stats.get('modality', 'text')

        if current_modality == 'image':
            col1, col2 = st.columns([3, 1])
            with col2:
                top_k = st.number_input(
                    "Top Results",
                    min_value=1,
                    max_value=max(1, len(df)),
                    value=min(12, max(1, len(df))),
                    help="Number of most similar images to highlight"
                )
            with col1:
                query_mode = st.radio(
                    "Query Type",
                    options=["Text", "Image"],
                    horizontal=True,
                    help="Search using a descriptive caption or by uploading a query image"
                )
                if query_mode == "Text":
                    search_query = st.text_input(
                        "Describe the image you're searching for",
                        placeholder="e.g., 'sunset over mountains', 'person riding a bike'",
                        key="image_text_query"
                    )
                    query_image_bytes = None
                else:
                    query_file = st.file_uploader(
                        "Upload query image",
                        type=[ext.replace('.', '') for ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']],
                        key="image_query_upload"
                    )
                    search_query = ""
                    query_image_bytes = query_file.read() if query_file else None

            trigger_search = st.button("🔎 Search", type="primary")

        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                search_query = st.text_input(
                    "Search Query",
                    placeholder="e.g., 'civilian casualties', 'economic impact', 'political statements'...",
                    help="Enter a search query to find semantically similar documents"
                )
            with col2:
                top_k = st.number_input(
                    "Top Results",
                    min_value=5,
                    max_value=100,
                    value=20,
                    help="Number of most similar documents to highlight"
                )
            query_image_bytes = None
            query_mode = "Text"
            trigger_search = st.button("🔎 Search", type="primary")

        if trigger_search:
            if current_modality == 'image' and query_mode == "Image" and not query_image_bytes:
                st.warning("Please upload a query image before searching.")
            elif current_modality != 'image' and not search_query:
                st.warning("Please enter a search query.")
            elif current_modality == 'image' and query_mode == "Text" and not search_query:
                st.warning("Please describe what you're looking for.")
            else:
                search_placeholder = st.empty()
                with search_placeholder.container():
                    st.markdown("### 🔍 Searching for Similar Content")
                    search_progress = st.progress(0)
                    search_status = st.empty()
                    search_details = st.empty()
                    search_start = time.time()

                    device = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')

                    if current_modality == 'image':
                        search_status.text("🤖 Loading SigLIP encoder...")
                        model, processor = load_siglip_model(
                            model_name="google/siglip-base-patch16-512",
                            device=device,
                            token=st.session_state.hf_token or None
                        )
                        search_progress.progress(15)

                        if query_mode == "Text":
                            search_status.text("🔤 Encoding text query...")
                            search_details.text(f"Caption: '{search_query}'")
                            query_embedding = encode_text_with_siglip([search_query], model, processor, device)[0]
                        else:
                            search_status.text("🖼️ Encoding query image...")
                            record = create_image_record('query', data_bytes=query_image_bytes, label='query-image')
                            query_embedding_arr, _ = encode_images_with_siglip([record], model, processor, device, batch_size=1)
                            query_embedding = query_embedding_arr[0]
                    else:
                        search_status.text("🔤 Encoding search query...")
                        search_details.text(f"Query: '{search_query}'")
                        search_progress.progress(25)
                        query_embedding = st.session_state.model.encode(
                            [search_query],
                            normalize_embeddings=True
                        )[0]

                    search_status.text("📊 Calculating similarities...")
                    search_details.text(f"Comparing with {len(st.session_state.embeddings)} items")
                    search_progress.progress(60)
                    similarities = np.dot(st.session_state.embeddings, query_embedding)

                    search_status.text("🎯 Ranking results...")
                    search_details.text(f"Finding top {top_k} matches")
                    search_progress.progress(80)
                    top_indices = np.argsort(similarities)[-top_k:][::-1]
                    top_scores = similarities[top_indices]

                    results_df = df.iloc[top_indices].copy()
                    results_df['similarity'] = top_scores
                    results_df['rank'] = range(1, len(results_df) + 1)

                    search_time = time.time() - search_start
                    avg_similarity = float(np.mean(top_scores)) if len(top_scores) else 0.0
                    similarity_range = f"{top_scores.min():.3f} - {top_scores.max():.3f}" if len(top_scores) else "N/A"

                    search_progress.progress(100)
                    search_status.text("✅ Search complete!")
                    search_details.text(f"Found {len(results_df)} results in {search_time:.2f}s")
                    time.sleep(1.0)
                    search_placeholder.empty()

                st.markdown("### 📊 Search Results Summary")
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

                if current_modality == 'image' and query_mode == "Image":
                    with summary_col1:
                        st.metric("Query Type", "Image Upload")
                else:
                    with summary_col1:
                        st.metric("Query Length", len(search_query))

                with summary_col2:
                    st.metric("Search Time", f"{search_time:.2f}s")
                with summary_col3:
                    st.metric("Avg Similarity", f"{avg_similarity:.3f}")
                with summary_col4:
                    st.metric("Similarity Range", similarity_range)

                st.subheader(f"🎯 Top {top_k} Results")
                threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    help="Filter results by minimum similarity score"
                )

                filtered_results = results_df[results_df['similarity'] >= threshold].copy()

                if filtered_results.empty:
                    st.warning("No results meet the similarity threshold. Try lowering the threshold.")
                else:
                    if current_modality == 'image':
                        st.write("Preview of matched images:")
                        cols = st.columns(min(4, len(filtered_results)))
                        for i, (_, row) in enumerate(filtered_results.iterrows()):
                            col = cols[i % len(cols)]
                            with col:
                                if row.get('thumbnail'):
                                    col.image(
                                        f"data:image/png;base64,{row['thumbnail']}",
                                        caption=f"#{row['rank']} • {row['label']}",
                                        use_column_width=True
                                    )
                                else:
                                    col.markdown(f"**#{row['rank']} • {row['label']}**")
                                if row.get('caption'):
                                    col.caption(row['caption'])
                    display_cols = ['rank', 'similarity', 'cluster_label', 'label']
                    if current_modality == 'image':
                        if 'caption' in filtered_results.columns:
                            display_cols.append('caption')
                    else:
                        display_cols.append('hover_text')

                    def color_similarity(val):
                        if val >= 0.8:
                            return 'background-color: #d4edda'
                        elif val >= 0.6:
                            return 'background-color: #fff3cd'
                        else:
                            return 'background-color: #f8d7da'

                    styled_df = filtered_results[display_cols].style.format({
                        'similarity': '{:.3f}',
                        'rank': '{}'
                    }).applymap(color_similarity, subset=['similarity'])

                    st.dataframe(styled_df, use_container_width=True, hide_index=True)

                    st.subheader("📈 Cluster Distribution of Results")
                    cluster_dist = filtered_results['cluster_label'].value_counts().reset_index()
                    cluster_dist.columns = ['Cluster', 'Count']
                    cluster_dist['Percentage'] = (cluster_dist['Count'] / len(filtered_results) * 100).round(1)
                    st.dataframe(cluster_dist, use_container_width=True, hide_index=True)

                    st.session_state.search_results = filtered_results.index.tolist()
                    st.session_state.search_scores = filtered_results['similarity'].tolist()

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("📊 View in Visualization", type="primary"):
                            st.success("🎯 Switch to the Visualization tab to see highlighted results!")
                    with col_b:
                        if st.button("🔄 Clear Search"):
                            st.session_state.search_results = []
                            st.session_state.search_scores = []
                            st.rerun()
                    with col_c:
                        if st.button("📥 Export Results"):
                            csv_data = filtered_results.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Search Results (CSV)",
                                data=csv_data,
                                file_name="search_results.csv",
                                mime="text/csv"
                            )
    with tab2:
        st.header("📊 Interactive Visualization")
        
        # Visualization Settings
        st.subheader("🎨 Visualization Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            node_size = st.slider(
                "Node Size",
                min_value=0.01,
                max_value=10.0,
                value=st.session_state.get('node_size', 1.0),
                step=0.01,
                help="Size of nodes in the visualization (try very small values like 0.01 for dense graphs)"
            )
        
        with col2:
            search_result_size_multiplier = st.slider(
                "Search Result Size Multiplier",
                min_value=1.0,
                max_value=5.0,
                value=st.session_state.get('search_result_size_multiplier', 3.0),
                step=0.1,
                help="How much larger to make search results compared to normal nodes"
            )
        
        # Save visualization settings to session state
        st.session_state.node_size = node_size
        st.session_state.search_result_size_multiplier = search_result_size_multiplier
        
        st.markdown("---")
        
        if not COMPONENTS_AVAILABLE:
            st.error("❌ Streamlit components not available. Please install streamlit-components:")
            st.code("pip install streamlit-components")
        else:
            html_content = generate_cosmograph_html(
                df,
                st.session_state.get('search_results', []),
                st.session_state.get('search_scores', []),
                node_size=node_size,
                search_result_size_multiplier=search_result_size_multiplier
            )
            components.html(html_content, height=800, scrolling=False)
    
    with tab3:
        st.header("📈 Data Explorer")
        
        # Processing Information
        if 'processing_info' in st.session_state and st.session_state.processing_info:
            with st.expander("⚡ Processing Information", expanded=True):
                info = st.session_state.processing_info
                stats = info['data_stats']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Processing Time", format_time(info['total_time']))
                    st.metric("Model Used", stats['model'].split('/')[-1][:15] + "...")
                
                with col2:
                    st.metric("Device", stats['device'])
                    st.metric("Embedding Dim", f"{stats['embedding_dim']}D")
                
                with col3:
                    st.metric("Total Points", stats['total_points'])
                    st.metric("Clusters Found", stats['clusters'])
                
                with col4:
                    st.metric("Noise Points", stats['noise_points'])
                    st.metric("Processing Speed", f"{stats['total_points']/info['total_time']:.1f} items/sec")
        
        # Data Statistics
        st.subheader("📊 Data Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Points", len(df))
        
        with col2:
            st.metric("Clusters", df['cluster'].nunique())
        
        with col3:
            noise_count = (df['cluster'] == -1).sum()
            st.metric("Noise Points", noise_count)

        if 'thumbnail' in df.columns:
            st.subheader("🖼️ Image Preview")
            preview_count = min(len(df), 12)
            cols = st.columns(min(6, preview_count))
            for i, (_, row) in enumerate(df.head(preview_count).iterrows()):
                col = cols[i % len(cols)]
                with col:
                    if row.get('thumbnail'):
                        col.image(
                            f"data:image/png;base64,{row['thumbnail']}",
                            caption=row.get('label', f'Item {i}')[:40],
                            use_column_width=True
                        )
                    else:
                        col.markdown(f"**{row.get('label', f'Item {i}')}**")
                        col.caption(row.get('hover_text', '')[:80])

        # Cluster statistics
        st.subheader("🎯 Cluster Analysis")
        cluster_stats = df.groupby('cluster_label').size().reset_index(name='count')
        cluster_stats = cluster_stats.sort_values('count', ascending=False)
        
        # Add percentage column
        cluster_stats['percentage'] = (cluster_stats['count'] / len(df) * 100).round(1)
        
        st.dataframe(cluster_stats, use_container_width=True, hide_index=True)
        
        # Embedding statistics
        if 'embeddings' in st.session_state and st.session_state.embeddings is not None:
            st.subheader("🔢 Embedding Statistics")
            embeddings = st.session_state.embeddings
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Embedding Shape", f"{embeddings.shape[0]}×{embeddings.shape[1]}")
            
            with col2:
                st.metric("Mean Norm", f"{np.mean(np.linalg.norm(embeddings, axis=1)):.3f}")
            
            with col3:
                st.metric("Std Dev", f"{np.std(embeddings):.3f}")
            
            with col4:
                st.metric("Memory Usage", format_memory(embeddings.nbytes / (1024*1024)))
        
        # Raw data preview
        st.subheader("📋 Raw Data Preview")
        st.dataframe(df, use_container_width=True)
        
        # Download options
        st.subheader("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Download Processed Data (CSV)",
                data=csv,
                file_name="processed_embeddings.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Create a summary JSON
            summary_data = {
                'processing_info': st.session_state.get('processing_info', {}),
                'data_stats': {
                    'total_points': len(df),
                    'clusters': df['cluster'].nunique(),
                    'noise_points': (df['cluster'] == -1).sum(),
                    'columns': list(df.columns)
                }
            }
            json_data = json.dumps(summary_data, indent=2, default=str).encode('utf-8')
            st.download_button(
                label="📄 Download Summary (JSON)",
                data=json_data,
                file_name="processing_summary.json",
                mime="application/json",
                use_container_width=True
            )

else:
    # Welcome screen
    st.info("👆 Upload a CSV file in the sidebar to get started!")
    
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        ### Step-by-step guide:
        
        1. **Upload CSV**: Click "Browse files" in the sidebar and select your CSV file
        2. **Select Columns**: 
           - Choose which column contains the text to embed
           - Choose which column to use as labels (node titles)
           - Optionally select a URL column for clickable links
        3. **Configure Clustering**: Choose automatic (HDBSCAN) or fixed (KMeans) clustering
        4. **Process**: Click the "Process Data" button and wait for completion
        5. **Explore**: 
           - Use **Semantic Search** to find similar nodes
           - View the **Visualization** to explore clusters
           - Check **Data Explorer** for statistics and downloads
        
        ### Features:
        - 🚀 GPU-accelerated embeddings using state-of-the-art models
        - 🎨 Automatic clustering and dimensionality reduction
        - 🔍 Semantic search to find similar content
        - 📊 Interactive Cosmograph visualization
        - 📥 Download processed results
        """)
