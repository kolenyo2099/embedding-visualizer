import streamlit as st
import json
import time
import os
import hashlib
import pickle
import base64
from pathlib import Path
from datetime import datetime
from io import BytesIO


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
        from transformers import (
            SiglipProcessor,
            SiglipTextModel,
            SiglipVisionModel,
        )
    except ImportError:
        missing_deps.append("transformers")
        SiglipProcessor = None
        SiglipTextModel = None
        SiglipVisionModel = None

    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except ImportError:
        missing_deps.append("torchvision")
        transforms = None
        InterpolationMode = None

    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("Pillow")
        Image = None
    
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
    critical_deps = ["pandas", "numpy", "torch", "sentence-transformers", "umap-learn", "scikit-learn"]
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
SiglipProcessor = imports['SiglipProcessor']
SiglipTextModel = imports['SiglipTextModel']
SiglipVisionModel = imports['SiglipVisionModel']
transforms = imports['transforms']
InterpolationMode = imports['InterpolationMode']
Image = imports['Image']
F = torch.nn.functional if torch else None

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
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processing_info' not in st.session_state:
    st.session_state.processing_info = {}
if 'hf_token' not in st.session_state:
    st.session_state.hf_token = ""
if 'uploaded_file_id' not in st.session_state:
    st.session_state.uploaded_file_id = None
if 'node_size' not in st.session_state:
    st.session_state.node_size = 1.0
if 'search_result_size_multiplier' not in st.session_state:
    st.session_state.search_result_size_multiplier = 3.0
if 'modality' not in st.session_state:
    st.session_state.modality = 'text'

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

def get_torch_device():
    """Determine the best available Torch device"""
    if not torch:
        return 'cpu'
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def load_modality_model(modality, model_name, hf_token=None):
    """Load embedding backbones for the requested modality"""
    device = get_torch_device()

    if modality == 'text':
        model_kwargs = {
            'device': device,
            'trust_remote_code': True
        }
        if hf_token:
            model_kwargs['token'] = hf_token

        model = SentenceTransformer(model_name, **model_kwargs)
        return {
            'device': device,
            'model_name': model_name,
            'modality': modality,
            'text_model': model,
            'preprocessing': None,
            'dtype': torch.float32,
        }

    if modality == 'image':
        missing = []
        if SiglipVisionModel is None or SiglipTextModel is None or SiglipProcessor is None:
            missing.append('transformers')
        if transforms is None or InterpolationMode is None:
            missing.append('torchvision')
        if Image is None:
            missing.append('Pillow')

        if missing:
            st.error("❌ Image embedding support requires additional dependencies.")
            st.info(
                "Install the following packages and restart: " +
                ", ".join(sorted(set(missing)))
            )
            st.stop()

        dtype = torch.float16 if device != 'cpu' else torch.float32

        model_kwargs = {
            'torch_dtype': dtype
        }
        if hf_token:
            model_kwargs['token'] = hf_token

        processor_kwargs = {}
        if hf_token:
            processor_kwargs['token'] = hf_token

        vision_model = SiglipVisionModel.from_pretrained(model_name, **model_kwargs)
        text_model = SiglipTextModel.from_pretrained(model_name, **model_kwargs)
        processor = SiglipProcessor.from_pretrained(model_name, **processor_kwargs)

        vision_model = vision_model.to(device)
        text_model = text_model.to(device)
        vision_model.eval()
        text_model.eval()

        # Prepare preprocessing pipeline
        image_mean = processor.image_processor.image_mean
        image_std = processor.image_processor.image_std
        image_size = 384
        if hasattr(processor.image_processor, 'size'):
            size_cfg = processor.image_processor.size
            if isinstance(size_cfg, dict):
                image_size = size_cfg.get('shortest_edge', size_cfg.get('height', image_size))
            elif isinstance(size_cfg, int):
                image_size = size_cfg

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ])

        preprocessing_config = {
            'size': [image_size, image_size],
            'mean': image_mean,
            'std': image_std,
            'interpolation': 'bicubic',
            'dtype': 'fp16' if dtype == torch.float16 else 'fp32',
        }

        return {
            'device': device,
            'model_name': model_name,
            'modality': modality,
            'vision_model': vision_model,
            'text_model': text_model,
            'processor': processor,
            'transform': transform,
            'preprocessing': preprocessing_config,
            'dtype': dtype,
        }

    raise ValueError(f"Unsupported modality: {modality}")

def load_image_from_identifier(identifier):
    """Load a PIL image from a CSV identifier."""
    if identifier is None:
        return None

    try:
        if isinstance(identifier, Image.Image):
            return identifier.convert('RGB')

        if hasattr(identifier, 'read'):
            return Image.open(identifier).convert('RGB')

        if isinstance(identifier, bytes):
            return Image.open(BytesIO(identifier)).convert('RGB')

        if isinstance(identifier, str):
            stripped = identifier.strip()
            if stripped.startswith('data:image') and ',' in stripped:
                _, b64_data = stripped.split(',', 1)
                image_bytes = base64.b64decode(b64_data)
                return Image.open(BytesIO(image_bytes)).convert('RGB')
            if os.path.exists(stripped):
                return Image.open(stripped).convert('RGB')
    except Exception:
        return None

    return None

def encode_image_identifiers(image_identifiers, resources, batch_size=16, progress_callback=None):
    """Encode image identifiers into normalized embeddings."""
    if not torch or not F:
        raise RuntimeError("PyTorch is required for image encoding")

    vision_model = resources['vision_model']
    transform = resources['transform']
    device = resources['device']
    dtype = resources['dtype']

    embeddings = [None] * len(image_identifiers)
    failed = []

    pixel_batch = []
    batch_indices = []

    def process_batch(batch_pixels, indices):
        if not batch_pixels:
            return
        batch_tensor = torch.stack(batch_pixels)
        batch_tensor = batch_tensor.to(device=device, dtype=dtype)
        with torch.no_grad():
            outputs = vision_model(pixel_values=batch_tensor)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                feats = outputs.pooler_output
            else:
                feats = outputs.last_hidden_state[:, 0]
            feats = F.normalize(feats, p=2, dim=-1)
        feats = feats.detach().cpu().numpy()
        for idx, feat in zip(indices, feats):
            embeddings[idx] = feat

    for idx, identifier in enumerate(image_identifiers):
        image = load_image_from_identifier(identifier)
        if image is None:
            failed.append(identifier)
            continue

        pixel = transform(image)
        pixel_batch.append(pixel)
        batch_indices.append(idx)

        if len(pixel_batch) >= batch_size:
            process_batch(pixel_batch, batch_indices)
            pixel_batch = []
            batch_indices = []
            if progress_callback:
                progress_callback(idx + 1, len(image_identifiers))

    if pixel_batch:
        process_batch(pixel_batch, batch_indices)
        if progress_callback:
            progress_callback(len(image_identifiers), len(image_identifiers))

    embedding_dim = None
    for emb in embeddings:
        if emb is not None:
            embedding_dim = len(emb)
            break

    if embedding_dim is None:
        raise ValueError("No valid images were processed. Check your image paths or data format.")

    for idx, emb in enumerate(embeddings):
        if emb is None:
            embeddings[idx] = np.zeros(embedding_dim, dtype=np.float32)

    return np.vstack(embeddings), failed

def get_embeddings_dir():
    """Get or create the embeddings directory"""
    embeddings_dir = Path("embeddings_cache")
    embeddings_dir.mkdir(exist_ok=True)
    return embeddings_dir

def generate_embedding_hash(items, model_name, metadata=None):
    """Generate a unique hash for the embedding configuration"""
    sample = items[:100] if len(items) > 100 else items
    try:
        sample_repr = json.dumps(sample, sort_keys=True, default=str)
    except TypeError:
        sample_repr = str(sample)

    hasher = hashlib.md5()
    hasher.update(sample_repr.encode())
    hasher.update(model_name.encode())
    hasher.update(str(len(items)).encode())

    if metadata:
        metadata_for_hash = {
            'modality': metadata.get('modality'),
            'preprocessing': metadata.get('preprocessing'),
        }
        identifiers = metadata.get('identifiers')
        if identifiers:
            identifiers_sample = identifiers[:100] if len(identifiers) > 100 else identifiers
            try:
                identifier_repr = json.dumps(identifiers_sample, sort_keys=True, default=str)
            except TypeError:
                identifier_repr = str(identifiers_sample)
            metadata_for_hash['identifiers_hash'] = hashlib.md5(identifier_repr.encode()).hexdigest()

        hasher.update(json.dumps(metadata_for_hash, sort_keys=True, default=str).encode())

    return hasher.hexdigest()

def save_embeddings(embeddings, items, model_name, metadata=None):
    """Save embeddings to disk"""
    try:
        embeddings_dir = get_embeddings_dir()
        embedding_hash = generate_embedding_hash(items, model_name, metadata)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embeddings_{embedding_hash}_{timestamp}.pkl"
        filepath = embeddings_dir / filename
        
        # Save embeddings with metadata
        data = {
            'embeddings': embeddings,
            'model_name': model_name,
            'num_texts': len(items),
            'num_items': len(items),
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
                metadata = data.get('metadata', {})
                embeddings_info.append({
                    'filepath': filepath,
                    'filename': filepath.name,
                    'model_name': data.get('model_name', 'Unknown'),
                    'num_texts': data.get('num_texts', data.get('num_items', 0)),
                    'num_items': data.get('num_items', data.get('num_texts', 0)),
                    'embedding_dim': data.get('embedding_dim', 0),
                    'timestamp': data.get('timestamp', 'Unknown'),
                    'size_mb': filepath.stat().st_size / (1024 * 1024),
                    'metadata': metadata,
                    'modality': metadata.get('modality', 'text'),
                })
        except:
            continue
    
    return sorted(embeddings_info, key=lambda x: x['timestamp'], reverse=True)

def generate_cosmograph_html(df, search_results=[], search_scores=[], node_size=1.0, search_result_size_multiplier=3.0):
    """Generate the Cosmograph visualization HTML"""
    
    # Get the text column name from session state, or use 'hover_text' as fallback
    text_col = st.session_state.get('text_column', 'hover_text')
    
    # Prepare data - check if text_col exists in df
    if text_col in df.columns:
        data_for_js = df[['label', 'x', 'y', 'cluster_label', 'hover_text', text_col, 'link']].copy()
        data_for_js.columns = ['id', 'x', 'y', 'cluster', 'hover_text', 'full_text', 'link']
    else:
        # Fallback: use hover_text as full_text
        data_for_js = df[['label', 'x', 'y', 'cluster_label', 'hover_text', 'link']].copy()
        data_for_js = data_for_js.rename(columns={
            'label': 'id',
            'cluster_label': 'cluster',
            'hover_text': 'hover_text'
        })
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
            document.getElementById('modalText').textContent = data.full_text || data.hover_text || 'No text available';
            
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
Visualize and explore text data using state-of-the-art embeddings and interactive clustering.
Upload a CSV, select columns, and explore semantic relationships in your data.
""")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

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
            st.session_state.df = df
            st.session_state.processed = False
            st.session_state.embeddings = None
            st.session_state.uploaded_file_id = file_id
        
        # Use the dataframe from session state
        df = st.session_state.df
        
        st.sidebar.success(f"✅ Loaded {len(df)} rows")

        # Step 2: Column Selection
        st.sidebar.subheader("📊 Column Selection")

        modality = st.sidebar.selectbox(
            "Input Modality",
            options=["text", "image"],
            index=0 if st.session_state.modality == 'text' else 1,
            format_func=lambda m: "📝 Text" if m == 'text' else "🖼️ Image",
            help="Choose whether to create text or image embeddings"
        )

        if modality != st.session_state.modality:
            st.session_state.modality = modality
            st.session_state.processed = False
            st.session_state.embeddings = None

        is_image_modality = modality == 'image'

        if is_image_modality:
            input_column = st.sidebar.selectbox(
                "Image Column (paths, URLs, or data URIs)",
                options=df.columns.tolist(),
                help="Select the column containing image references to embed"
            )
        else:
            input_column = st.sidebar.selectbox(
                "Text Column (to embed)",
                options=df.columns.tolist(),
                help="Select the column containing text to embed"
            )

        text_column = input_column

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
        saved_embeddings_all = list_saved_embeddings()
        current_modality = st.session_state.modality
        saved_embeddings = [info for info in saved_embeddings_all if info.get('modality', 'text') == current_modality]

        use_cached = st.sidebar.checkbox(
            "📁 Use Cached Embeddings",
            value=False,
            help="Load previously saved embeddings instead of creating new ones"
        )

        save_embeddings_checkbox = False

        if use_cached and saved_embeddings:
            st.sidebar.markdown("**Available Cached Embeddings:**")

            # Create a selection for cached embeddings
            embedding_options = []
            for idx, emb_info in enumerate(saved_embeddings):
                item_label = "items" if emb_info.get('num_items', emb_info.get('num_texts', 0)) != 1 else "item"
                label = f"[{emb_info.get('modality', 'text').title()}] {emb_info['model_name']} | {emb_info.get('num_items', emb_info.get('num_texts', 0))} {item_label} | {emb_info['timestamp']}"
                embedding_options.append(label)
            
            selected_cache_idx = st.sidebar.selectbox(
                "Select Cached Embedding",
                options=range(len(embedding_options)),
                format_func=lambda x: embedding_options[x],
                help="Choose from previously saved embeddings"
            )
            
            selected_cache = saved_embeddings[selected_cache_idx]
            
            # Display cache info
            with st.sidebar.expander("📊 Cache Details"):
                st.markdown(f"""
                **Model:** {selected_cache['model_name']}
                **Items:** {selected_cache.get('num_items', selected_cache.get('num_texts', 0))}
                **Dimensions:** {selected_cache['embedding_dim']}D
                **Size:** {selected_cache['size_mb']:.2f} MB
                **Created:** {selected_cache['timestamp']}
                """)

            model_name = selected_cache['model_name']  # For compatibility
        else:
            if use_cached and not saved_embeddings:
                st.sidebar.warning("⚠️ No cached embeddings found for this modality")

            if is_image_modality:
                model_name = "google/siglip-so400m-patch14-384"
                st.sidebar.info(
                    "Using google/siglip-so400m-patch14-384 for image embeddings (1152-D)."
                )
            else:
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

        save_embeddings_selected = save_embeddings_checkbox
        
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
                    df_clean[text_column] = df_clean[text_column].astype(str)
                    if is_image_modality:
                        df_clean[text_column] = df_clean[text_column].str.strip()
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

                try:
                    model_resources = load_modality_model(
                        modality,
                        model_name,
                        hf_token=st.session_state.hf_token or None,
                    )
                    device = model_resources['device']
                    dtype_label = model_resources.get('preprocessing', {}).get('dtype', 'fp32') if modality == 'image' else 'fp32'

                    auth_label = 'with authentication' if st.session_state.hf_token else 'no token'
                    details_text.text(f"Model: {model_name} ({device.upper()}, {auth_label})")

                    if modality == 'text':
                        st.session_state.model = model_resources['text_model']
                    else:
                        st.session_state.model = model_resources['vision_model']

                    step_time = time.time() - step_start
                    processing_steps.append({
                        'step': 'Model Loading',
                        'time': step_time,
                        'details': f"Device: {device.upper()} | Precision: {dtype_label} | Authenticated: {'Yes' if st.session_state.hf_token else 'No'}"
                    })
                except Exception as model_error:
                    message = str(model_error)
                    if "401" in message or "authentication" in message.lower():
                        st.error(f"❌ Authentication failed for model {model_name}")
                        st.error("Please check your Hugging Face token and ensure you have accepted the model's terms of use.")
                        st.error("Get your token at: https://huggingface.co/settings/tokens")
                    elif "gated" in message.lower() or "private" in message.lower():
                        st.error(f"❌ Model {model_name} requires authentication")
                        st.error("Please enter a valid Hugging Face token in the sidebar.")
                    else:
                        st.error(f"❌ Failed to load model {model_name}: {message}")
                    processing_placeholder.empty()
                    st.stop()
                
                # Create or load embeddings
                try:
                    items_to_embed = df_clean[text_column].tolist()
                    total_items = len(items_to_embed)
                    failed_items = []

                    metadata_context = {
                        'modality': modality,
                        'preprocessing': model_resources.get('preprocessing'),
                    }
                    if is_image_modality:
                        metadata_context['image_column'] = text_column
                        metadata_context['identifiers'] = items_to_embed
                    else:
                        metadata_context['text_column'] = text_column

                    # Check if we should use cached embeddings
                    if use_cached and saved_embeddings:
                        status_text.text("📁 Loading cached embeddings...")
                        progress_bar.progress(20)

                        step_start = time.time()

                        cached_data = load_embeddings(selected_cache['filepath'])

                        if cached_data:
                            cached_count = cached_data.get('num_items', cached_data.get('num_texts', 0))
                            cached_metadata = cached_data.get('metadata', {})
                            cached_modality = cached_metadata.get('modality', 'text')

                            if cached_modality != modality:
                                st.warning("⚠️ Cached embeddings are for a different modality. Creating new embeddings instead...")
                                use_cached = False
                                if not save_embeddings_selected:
                                    save_embeddings_selected = True
                            elif cached_count == total_items:
                                embeddings = cached_data['embeddings']
                                st.session_state.embeddings = embeddings

                                step_time = time.time() - step_start
                                processing_steps.append({
                                    'step': 'Load Cached Embeddings',
                                    'time': step_time,
                                    'details': f"Loaded {len(embeddings)} embeddings ({embeddings.shape[1]}D) from cache"
                                })

                                status_text.text("✅ Cached embeddings loaded!")
                                details_text.text(f"Loaded {total_items} embeddings in {step_time:.2f}s")
                            else:
                                st.warning("⚠️ Cached embeddings don't match current data size. Creating new embeddings...")
                                use_cached = False
                                if not save_embeddings_selected:
                                    save_embeddings_selected = True
                        else:
                            st.warning("⚠️ Failed to load cached embeddings. Creating new embeddings...")
                            use_cached = False
                            if not save_embeddings_selected:
                                save_embeddings_selected = True

                    if not use_cached or not saved_embeddings:
                        creation_label = "image" if is_image_modality else "text"
                        status_text.text(f"{'🖼️' if is_image_modality else '🔤'} Creating {creation_label} embeddings...")
                        progress_bar.progress(20)

                        step_start = time.time()

                        # Show embedding progress
                        embedding_progress = st.progress(0)

                        failed_items = []

                        if is_image_modality:
                            def image_progress(completed, total):
                                progress = 0 if total == 0 else min(completed / total, 1.0)
                                embedding_progress.progress(progress)
                                status_text.text(f"🖼️ Creating embeddings... {completed}/{total}")
                                details_text.text(f"Processed {completed}/{total} images")

                            embeddings, failed_items = encode_image_identifiers(
                                items_to_embed,
                                model_resources,
                                batch_size=16,
                                progress_callback=image_progress
                            )
                            if failed_items:
                                st.warning(f"⚠️ {len(failed_items)} images could not be loaded. Zero vectors were inserted.")
                        else:
                            text_model = model_resources['text_model']
                            batch_size = 64
                            num_batches = (total_items + batch_size - 1) // batch_size
                            embeddings = []

                            for i in range(0, total_items, batch_size):
                                batch_texts = items_to_embed[i:i+batch_size]
                                batch_embeddings = text_model.encode(
                                    batch_texts,
                                    batch_size=batch_size,
                                    show_progress_bar=False,
                                    normalize_embeddings=True
                                )
                                embeddings.extend(batch_embeddings)

                                batch_num = i // batch_size + 1
                                progress = min((batch_num / num_batches), 1.0)
                                embedding_progress.progress(progress)
                                status_text.text(f"🔤 Creating embeddings... Batch {batch_num}/{num_batches}")
                                details_text.text(f"Processed {min(i + batch_size, total_items)}/{total_items} texts")

                            embeddings = np.array(embeddings)

                        st.session_state.embeddings = embeddings
                        embedding_progress.empty()

                        step_time = time.time() - step_start
                        detail_suffix = ''
                        if is_image_modality and failed_items:
                            detail_suffix = f" | Failed loads: {len(failed_items)}"
                        processing_steps.append({
                            'step': 'Image Embeddings' if is_image_modality else 'Text Embeddings',
                            'time': step_time,
                            'details': f"Created {len(embeddings)} embeddings ({embeddings.shape[1]}D){detail_suffix}"
                        })

                        # Save embeddings if requested
                        if save_embeddings_selected:
                            status_text.text("💾 Saving embeddings to cache...")
                            save_path, emb_hash = save_embeddings(
                                embeddings,
                                items_to_embed,
                                model_name,
                                metadata=metadata_context
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
                    if is_image_modality:
                        df_clean['hover_text'] = df_clean['label']
                    else:
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
                st.session_state.processed = True
                st.session_state.text_column = text_column if not is_image_modality else 'hover_text'
                st.session_state.modality = modality
                
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
                        'modality': modality,
                        'failed_items': len(failed_items) if is_image_modality else 0
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
                    speed_unit = "images/sec" if is_image_modality else "texts/sec"
                    st.metric("Processing Speed", f"{len(df_clean)/total_time:.1f} {speed_unit}")
                
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

# Main content area
if st.session_state.processed and st.session_state.df is not None:
    df = st.session_state.df
    
    # Debug: Show what columns we have
    # st.write("DEBUG - Columns in df:", list(df.columns))
    
    # Verify that all required columns exist
    required_columns = ['label', 'x', 'y', 'cluster_label', 'hover_text', 'link']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Processing incomplete. Missing columns: {', '.join(missing_columns)}")
        st.error(f"Current columns: {', '.join(list(df.columns))}")
        st.info("💡 Please click the '🚀 Process Data' button in the sidebar to create embeddings and prepare the visualization.")
        # Reset the processed flag since data isn't actually processed
        st.session_state.processed = False
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Semantic Search", "📊 Visualization", "📈 Data Explorer"])
    
    with tab1:
        st.header("🔍 Semantic Search")
        current_modality = st.session_state.get('modality', 'text')
        if current_modality == 'image':
            st.info("Semantic text search is unavailable for image embeddings. Switch to text modality to enable search.")
        else:
            st.markdown("Search for nodes semantically similar to your query.")

            col1, col2 = st.columns([3, 1])

            with col1:
                search_query = st.text_input(
                    "Search Query",
                    placeholder="e.g., 'civilian casualties', 'economic impact', 'political statements'...",
                    help="Enter a search query to find semantically similar nodes"
                )

            with col2:
                top_k = st.number_input(
                    "Top Results",
                    min_value=5,
                    max_value=100,
                    value=20,
                    help="Number of most similar nodes to highlight"
                )

            if search_query and st.button("🔎 Search", type="primary"):
                # Create search placeholder
                search_placeholder = st.empty()

                with search_placeholder.container():
                    st.markdown("### 🔍 Searching for Similar Content")

                    # Search progress
                    search_progress = st.progress(0)
                    search_status = st.empty()
                    search_details = st.empty()

                    search_start = time.time()

                    # Step 1: Embed the query
                    search_status.text("🔤 Encoding search query...")
                    search_details.text(f"Query: '{search_query}'")
                    search_progress.progress(25)

                    query_embedding = st.session_state.model.encode(
                        [search_query],
                        normalize_embeddings=True
                    )[0]

                    # Step 2: Calculate similarities
                    search_status.text("📊 Calculating similarities...")
                    search_details.text(f"Comparing with {len(st.session_state.embeddings)} documents")
                    search_progress.progress(50)

                    similarities = np.dot(st.session_state.embeddings, query_embedding)

                    # Step 3: Find top results
                    search_status.text("🎯 Ranking results...")
                    search_details.text(f"Finding top {top_k} most similar documents")
                    search_progress.progress(75)

                    top_indices = np.argsort(similarities)[-top_k:][::-1]
                    top_scores = similarities[top_indices]

                    # Step 4: Prepare results
                    search_status.text("📋 Preparing results...")
                    search_progress.progress(90)

                    results_df = df.iloc[top_indices].copy()
                    results_df['similarity'] = top_scores
                    results_df['rank'] = range(1, len(results_df) + 1)

                    # Calculate search statistics
                    search_time = time.time() - search_start
                    avg_similarity = np.mean(top_scores)
                    similarity_range = f"{top_scores.min():.3f} - {top_scores.max():.3f}"

                    search_progress.progress(100)
                    search_status.text("✅ Search complete!")
                    search_details.text(f"Found {top_k} results in {search_time:.2f}s")

                    # Clear search placeholder after a delay
                    time.sleep(1.5)
                    search_placeholder.empty()

                # Display search summary
                st.markdown("### 📊 Search Results Summary")
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

                with summary_col1:
                    st.metric("Query Length", len(search_query))

                with summary_col2:
                    st.metric("Search Time", f"{search_time:.2f}s")

                with summary_col3:
                    st.metric("Avg Similarity", f"{avg_similarity:.3f}")

                with summary_col4:
                    st.metric("Similarity Range", similarity_range)

                # Display results
                st.subheader(f"🎯 Top {top_k} Results")

                # Add similarity threshold indicator
                threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    help="Filter results by minimum similarity score"
                )

                # Filter results by threshold
                filtered_results = results_df[results_df['similarity'] >= threshold].copy()

                if len(filtered_results) == 0:
                    st.warning("No results meet the similarity threshold. Try lowering the threshold.")
                else:
                    # Show results table with enhanced formatting
                    display_cols = ['rank', 'similarity', 'cluster_label', 'label', 'hover_text']

                    # Color-code similarity scores
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

                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Show cluster distribution of results
                    st.subheader("📈 Cluster Distribution of Results")
                    cluster_dist = filtered_results['cluster_label'].value_counts().reset_index()
                    cluster_dist.columns = ['Cluster', 'Count']
                    cluster_dist['Percentage'] = (cluster_dist['Count'] / len(filtered_results) * 100).round(1)

                    st.dataframe(cluster_dist, use_container_width=True, hide_index=True)

                    # Store results for visualization
                    filtered_indices = filtered_results.index.tolist()
                    st.session_state.search_results = filtered_indices
                    st.session_state.search_scores = filtered_results['similarity'].tolist()
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 View in Visualization", type="primary"):
                        st.success("🎯 Switch to the Visualization tab to see highlighted results!")
                
                with col2:
                    if st.button("🔄 Clear Search"):
                        st.session_state.search_results = []
                        st.session_state.search_scores = []
                        st.rerun()
                
                with col3:
                    if st.button("📥 Export Results"):
                        csv_data = filtered_results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Search Results (CSV)",
                            data=csv_data,
                            file_name=f"search_results_{search_query[:20].replace(' ', '_')}.csv",
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
            # Save data to a temporary JSON file
            import tempfile
            import os
            
            # Create temp file for data
            temp_dir = tempfile.gettempdir()
            data_file = os.path.join(temp_dir, 'cosmograph_data.json')
            
            # Prepare data for JSON export
            text_col = st.session_state.get('text_column', 'hover_text')
            if text_col in df.columns:
                data_for_js = df[['label', 'x', 'y', 'cluster_label', 'hover_text', text_col, 'link']].copy()
                data_for_js.columns = ['id', 'x', 'y', 'cluster', 'hover_text', 'full_text', 'link']
            else:
                data_for_js = df[['label', 'x', 'y', 'cluster_label', 'hover_text', 'link']].copy()
                data_for_js = data_for_js.rename(columns={
                    'label': 'id',
                    'cluster_label': 'cluster',
                    'hover_text': 'hover_text'
                })
                data_for_js['full_text'] = data_for_js['hover_text']
            
            # Add search results
            data_list = data_for_js.to_dict('records')
            search_results = st.session_state.get('search_results', [])
            search_scores = st.session_state.get('search_scores', [])
            
            for i, row in enumerate(data_list):
                # Clean data
                for key, value in row.items():
                    if pd.isna(value):
                        row[key] = ""
                    elif isinstance(value, (np.integer, np.floating)):
                        if np.isnan(value) or np.isinf(value):
                            row[key] = 0.0
                        else:
                            row[key] = float(value)
                
                # Add search info
                if i in search_results:
                    idx = search_results.index(i)
                    row['is_search_result'] = True
                    row['search_rank'] = idx + 1
                    row['search_score'] = float(search_scores[idx]) if search_scores and idx < len(search_scores) else 0.0
                else:
                    row['is_search_result'] = False
            
            # Generate colors
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
            
            # Get visualization settings from session state (use defaults if not available)
            node_size = st.session_state.get('node_size', 1.0)
            search_result_size_multiplier = st.session_state.get('search_result_size_multiplier', 3.0)
            
            # Generate HTML with dynamic node sizes using the function
            html_content = generate_cosmograph_html(
                df, 
                search_results, 
                search_scores, 
                node_size=node_size, 
                search_result_size_multiplier=search_result_size_multiplier
            )
            
            # Display using components
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
                    st.metric("Processing Speed", f"{stats['total_points']/info['total_time']:.1f} texts/sec")
        
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
