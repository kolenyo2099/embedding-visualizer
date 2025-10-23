from typing import Optional
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import base64
import json
from core.state import AppState


def render_visualization(state: AppState) -> None:
    """
    Renders the interactive visualization of the clustered data.

    Args:
        state (AppState): The application state object.
    """
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

    html = generate_cosmograph_html(
        result.df,
        text_column=result.text_column or "hover_text",
        node_size=state.node_size,
        search_result_size_multiplier=state.search_result_size_multiplier,
        search_results=state.search_state.results_df,
    )
    components.html(html, height=720, scrolling=False)

    st.caption("Tip: Use the search controls overlaying the visualization to highlight matches.")


def generate_cosmograph_html(
    df: pd.DataFrame,
    *,
    text_column: str,
    node_size: float,
    search_result_size_multiplier: float,
    search_results: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generates the HTML and JavaScript for the Cosmograph visualization.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to visualize.
        text_column (str): The name of the column containing the full text.
        node_size (float): The base size of the nodes.
        search_result_size_multiplier (float): The multiplier for the size of search result nodes.
        search_results (Optional[pd.DataFrame]): A DataFrame containing the search results.

    Returns:
        str: The HTML string for the visualization.
    """
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
    with open("assets/styles.css", "r") as f:
        css_styles = f.read()

    html_template = f"""
    <style>
    {css_styles}
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
