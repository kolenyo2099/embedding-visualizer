# 🎯 Semantic Embedding Explorer

A powerful Streamlit application for visualizing and exploring **text and image data** using state-of-the-art embeddings and interactive clustering.

## Features

✅ **File Upload** - Text CSVs, image files, ZIP archives, or CSVs with image URLs
✅ **Column Selection** - Choose text and label columns
✅ **Flexible Clustering** - HDBSCAN (automatic) or KMeans (fixed)
✅ **SOTA Embedding Models** - EmbeddingGemma, Nomic, BGE, and more
✅ **Embedding Cache** - Save and reuse embeddings for faster processing
✅ **Semantic Search** - Find similar nodes by meaning
✅ **Interactive Visualization** - Cosmograph with search highlighting
✅ **Data Explorer** - Statistics and download options
✅ **GPU Acceleration** - Automatic MPS/CUDA detection
✅ **SigLIP Image Embeddings** - Autodetect CPU/GPU for `google/siglip-base-patch16-512`

## Installation

### 🚀 Quick Start (macOS)

The easiest way to get started on macOS is to use the automated setup script:

```bash
./setup_and_run.sh
```

This script will:
- ✅ Check Python version (requires 3.8+)
- ✅ Create a virtual environment
- ✅ Install all dependencies
- ✅ Start the application

### 🔑 Hugging Face Authentication

Some advanced models require a Hugging Face token for access:

1. **Get your token** at: https://huggingface.co/settings/tokens
2. **Copy your token** (starts with `hf_`)
3. **Enter the token** in the sidebar under "🔑 Hugging Face Authentication"
4. **Access gated models** like EmbeddingGemma and other premium embeddings

**Token Requirements:**
- **Optional** for public models (BGE, Nomic, etc.)
- **Required** for gated/restricted models
- **Format:** Must start with `hf_` (e.g., `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`)

### 📋 Manual Installation

1. **Clone or download this project** to your local machine

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the app:**
```bash
streamlit run app.py
```

5. **Open in browser:** The app will automatically open at `http://localhost:8501`

## How to Use

### Step-by-step guide:

1. **Select Modality**: Choose **Text (CSV)** or **Images** from the sidebar.
2. **Provide Data**:
   - **Text**: Upload a CSV and choose text/label/link columns.
   - **Images**: Upload individual images, a ZIP archive, or a CSV of image URLs (with optional label/link/caption columns).
3. **Review Metadata**: Edit labels, captions, and links directly in the sidebar table (image mode) or via column selectors (text mode).
4. **Configure Clustering**: Choose automatic (HDBSCAN) or fixed (KMeans) clustering for either modality.
5. **Choose Embedding Strategy**:
   - **Create New**: Use Sentence-Transformer models for text or SigLIP for images.
   - **Use Cached**: Load previously saved embeddings (filter by modality automatically).
6. **Process**: Click the "Process" button and monitor progress, device selection, and timing breakdowns.
7. **Explore**:
   - Use **Semantic Search** for caption-based or image-based queries.
   - View the **Visualization** to explore clusters with thumbnails and captions.
   - Check **Data Explorer** for statistics, preview galleries, and downloads.

### 🖼️ Image Workflow

1. **Source Selection**: Choose between uploading individual images, a ZIP archive, or a CSV of URLs.
2. **Metadata Editing**: Update labels, optional captions, and outbound links inside the interactive data editor.
3. **SigLIP Embeddings**: The app downloads and caches `google/siglip-base-patch16-512` using Hugging Face Transformers, automatically detecting CUDA, MPS, or CPU execution.
4. **Caching**: Image embeddings are hashed by filename/URL so repeat runs are instantaneous.
5. **Visualization**: Hover cards include thumbnails, captions, and links; previews are available in the Data Explorer.
6. **Semantic Search**: Search with natural-language prompts (captioning) or upload a query image to find visually similar results.

### Features in Detail:

#### 💾 Embedding Cache System
The app automatically saves embeddings to the `embeddings_cache/` folder, providing significant time savings:

**Benefits:**
- ⚡ **10-100x faster** - Skip embedding creation for repeat processing
- 💰 **Resource efficient** - Reuse embeddings across different clustering configurations
- 📊 **Experiment friendly** - Try different clustering methods without re-embedding

**How to Use:**
1. **First Run**: Process your data normally with "💾 Save Embeddings for Later" checked
2. **Subsequent Runs**: Check "📁 Use Cached Embeddings" and select from saved embeddings
3. **Management**: The cache shows model name, text count, dimensions, and creation time

**Note:** Embeddings are saved in `embeddings_cache/` (automatically created and git-ignored)

#### 🔍 Semantic Search
- Enter natural language queries to find semantically similar **documents** (text mode)
- Describe an image or upload a query image to locate visually similar **thumbnails** (image mode)
- Results are highlighted in red in the visualization and include captions/links when available
- Shows similarity scores, rankings, and cluster summaries

#### 📊 Interactive Visualization
- Powered by Cosmograph for smooth, interactive exploration
- Click nodes to view full text in modal popups
- Color-coded clusters with legend
- Search results highlighted in red and enlarged

#### 📈 Data Explorer
- View cluster statistics and distributions
- Download processed data with embeddings and coordinates
- Examine raw data and filtering options

## Supported Embedding Models (October 2025 SOTA)

### Text Embeddings
- **google/embeddinggemma-300m** - Best under 500M params, SOTA efficient model
- **nomic-ai/nomic-embed-text-v1.5** - Excellent balance of performance and speed
- **BAAI/bge-base-en-v1.5** - Mid-size BGE model, strong retrieval performance
- **sentence-transformers/all-mpnet-base-v2** - Reliable baseline option

### Image Embeddings
- **google/siglip-base-patch16-512** - Vision-language model with 512-dim embeddings
  - ⚙️ **CPU**: Works out of the box; expect ~1s per image on modern laptops
  - ⚡ **GPU (6GB+ VRAM recommended)**: Dramatically faster batch inference
  - 💾 **RAM**: 8GB minimum, 16GB recommended for large ZIP archives
- Automatic caching stores image signatures, captions, and preprocessing metadata for reuse.

All models are optimized for semantic search and clustering tasks across modalities.

## System Requirements

- Python 3.8+
- 8GB+ RAM recommended (16GB for large image batches)
- GPU support (optional but recommended):
  - NVIDIA CUDA (6GB+ VRAM) for high-throughput text & image embeddings
  - Apple Metal Performance Shaders (MPS) for Mac
- Additional runtime dependencies (auto-checked by the app): `Pillow`, `torchvision`, `requests`, `transformers`

## CSV Format Requirements

### Text CSVs
Your CSV file should contain at least one text column. Example format:

```csv
text,label,url (optional)
"Sample text content here","Document 1","https://example.com/doc1"
"Another piece of text","Document 2","https://example.com/doc2"
```

### Image URL CSVs
Provide a column containing image URLs, with optional label/link/caption columns:

```csv
image_url,label,link,caption
"https://example.com/image1.png","Mountains","https://example.com","Sunset over snowy peaks"
"https://example.com/image2.jpg","Cyclist","","Person riding a bike at dusk"
```

## Performance Tips

- For datasets >10,000 rows, consider using a smaller model first
- HDBSCAN is better for discovering natural clusters
- KMeans is faster when you know the exact number of clusters needed
- GPU acceleration is automatically detected and used when available

## Troubleshooting

**Memory Issues**: 
- Reduce batch size in the code (line ~140)
- Use a smaller embedding model
- Process smaller chunks of your data

**Slow Performance**:
- Ensure GPU acceleration is working
- Try the MiniLM model for faster processing
- Reduce dataset size for testing

**Visualization Issues**:
- Refresh the page if the graph doesn't load
- Check browser console for errors
- Ensure internet connection for Cosmograph CDN

## License

This project is open source and available under the MIT License.
