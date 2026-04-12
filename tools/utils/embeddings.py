import numpy as np
from google.genai import types

from tools.utils.config import get_gemini_client, EMBEDDING_MODEL, EMBEDDING_DIMS


def normalize_l2(vector):
    """L2-normalize a vector. Required for Gemini embeddings at 1536 dims."""
    arr = np.array(vector, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


def embed_content(contents):
    """Embed content using Gemini. Returns list of normalized embedding vectors.

    Args:
        contents: A single item or list — strings for text, types.Part for images/video,
                  or types.Content for multimodal aggregation.
    """
    client = get_gemini_client()
    config = types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMS)

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=contents,
        config=config,
    )

    return [normalize_l2(emb.values) for emb in result.embeddings]


def prepare_text_for_embedding(text, title=None, mode="document"):
    """Format text using Gemini's recommended structured prompt pattern.

    Args:
        text: The text content.
        title: Optional title for the content.
        mode: "document" for indexing, "query" for search queries.
    """
    if mode == "query":
        return f"task: search result | query: {text}"
    if title:
        return f"title: {title} | text: {text}"
    return text
