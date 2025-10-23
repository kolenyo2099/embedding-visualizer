import streamlit as st
import pandas as pd
from core.state import AppState
from core.utils import _clean_metadata_value


def render_image_metadata_editor(state: AppState) -> None:
    """
    Renders the image metadata editor, allowing users to edit labels, captions, and links.

    Args:
        state (AppState): The application state object.
    """
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
