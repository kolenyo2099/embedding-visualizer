from typing import Any, Dict, List, Optional
import hashlib
import base64
from io import BytesIO
import pandas as pd
from PIL import Image
import requests


def format_time(seconds: float) -> str:
    """
    Formats a duration in seconds into a human-readable string (s, m, h).

    Args:
        seconds (float): The duration in seconds.

    Returns:
        str: The formatted duration string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def format_memory(mb: float) -> str:
    """
    Formats a memory size in megabytes into a human-readable string (MB, GB).

    Args:
        mb (float): The memory size in megabytes.

    Returns:
        str: The formatted memory string.
    """
    if mb < 1024:
        return f"{mb:.1f}MB"
    return f"{mb / 1024:.1f}GB"


def chunk_list(items: List[Any], chunk_size: int):
    """
    Splits a list into smaller chunks of a specified size.

    Args:
        items (List[Any]): The list to be chunked.
        chunk_size (int): The size of each chunk.

    Yields:
        List[Any]: A chunk of the original list.
    """
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _clean_metadata_value(value: Any, default: str = "", allow_empty: bool = True) -> str:
    """
    Cleans and standardizes metadata values, handling None, NaN, and whitespace.

    Args:
        value (Any): The metadata value to be cleaned.
        default (str): The default value to return if the cleaned value is empty.
        allow_empty (bool): Whether to allow empty strings as a valid value.

    Returns:
        str: The cleaned metadata value.
    """
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
    """
    Creates a standardized dictionary record for an image.

    Args:
        identifier (str): A unique identifier for the image.
        data_bytes (Optional[bytes]): The raw bytes of the image file.
        label (Optional[str]): A display label for the image.
        link (Optional[str]): An optional URL to link to.
        caption (Optional[str]): An optional caption for the image.
        url (Optional[str]): The URL of the image if it's remote.

    Returns:
        Dict[str, Any]: The structured image record.
    """
    return {
        "id": identifier,
        "bytes": data_bytes,
        "label": label or identifier,
        "link": link or "#",
        "caption": caption or "",
        "url": url,
    }


def record_signature(record: Dict[str, Any]) -> str:
    """
    Generates a unique signature for an image record based on its content or URL.

    Args:
        record (Dict[str, Any]): The image record.

    Returns:
        str: The MD5 hash signature.
    """
    if record.get("bytes"):
        digest = hashlib.md5(record["bytes"]).hexdigest()
    else:
        digest = hashlib.md5((record.get("url") or record["id"]).encode()).hexdigest()
    return f"{record['id']}:{digest}"


def load_pil_image(record: Dict[str, Any]):
    """
    Loads an image from a record, either from bytes or by downloading from a URL.

    Args:
        record (Dict[str, Any]): The image record.

    Returns:
        Image: The loaded PIL image object in RGB format.
    """
    data_bytes = record.get("bytes")
    if data_bytes is None and record.get("url"):
        response = requests.get(record["url"], timeout=15)
        response.raise_for_status()
        data_bytes = response.content
        record["bytes"] = data_bytes
    image = Image.open(BytesIO(data_bytes))
    return image.convert("RGB")


def create_thumbnail_base64(image, max_size=(160, 160)):
    """
    Creates a base64-encoded PNG thumbnail for a given PIL image.

    Args:
        image (Image): The PIL image.
        max_size (tuple): The maximum dimensions of the thumbnail.

    Returns:
        str: The base64-encoded thumbnail string.
    """
    preview = image.copy()
    preview.thumbnail(max_size)
    buffer = BytesIO()
    preview.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
