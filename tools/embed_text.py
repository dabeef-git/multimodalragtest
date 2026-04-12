"""Embed text content and upsert to Pinecone."""

import sys
import os
import json
import uuid
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.utils.config import get_pinecone_client, PINECONE_INDEX_NAME, TMP_DIR
from tools.utils.embeddings import embed_content, prepare_text_for_embedding

MAX_METADATA_TEXT = 39000  # Pinecone ~40KB metadata limit


def embed_single_text(text, title=None, description=None, source_id=None):
    """Embed a single text and upsert to Pinecone. Returns the vector ID."""
    formatted = prepare_text_for_embedding(text, title=title, mode="document")
    vectors = embed_content([formatted])

    vector_id = source_id or str(uuid.uuid4())
    metadata = {
        "content_type": "text",
        "source_path": "",
        "title": title or "",
        "description": description or "",
        "text": text[:MAX_METADATA_TEXT],
        "timestamp": datetime.now().isoformat(),
    }

    pc = get_pinecone_client()
    index = pc.Index(PINECONE_INDEX_NAME)
    index.upsert(vectors=[(vector_id, vectors[0], metadata)])

    print(f"Embedded text → {vector_id}")
    if title:
        print(f"  Title: {title}")

    result = {"vector_id": vector_id, "metadata": metadata}
    with open(os.path.join(TMP_DIR, "last_embed_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return vector_id


def embed_texts(items):
    """Batch-embed a list of text items.

    Args:
        items: List of dicts with keys: text, title (optional), description (optional).
    """
    formatted = [
        prepare_text_for_embedding(item["text"], title=item.get("title"), mode="document")
        for item in items
    ]
    vectors = embed_content(formatted)

    pc = get_pinecone_client()
    index = pc.Index(PINECONE_INDEX_NAME)

    records = []
    for i, item in enumerate(items):
        vector_id = item.get("source_id") or str(uuid.uuid4())
        metadata = {
            "content_type": "text",
            "source_path": "",
            "title": item.get("title", ""),
            "description": item.get("description", ""),
            "text": item["text"][:MAX_METADATA_TEXT],
            "timestamp": datetime.now().isoformat(),
        }
        records.append((vector_id, vectors[i], metadata))

    # Upsert in batches of 100
    for batch_start in range(0, len(records), 100):
        batch = records[batch_start : batch_start + 100]
        index.upsert(vectors=batch)

    print(f"Embedded {len(records)} text items.")
    return [r[0] for r in records]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed text into Pinecone")
    parser.add_argument("--text", required=True, help="Text content to embed")
    parser.add_argument("--title", default=None, help="Title for the content")
    parser.add_argument("--description", default=None, help="Description")
    parser.add_argument("--source-id", default=None, help="Custom vector ID")
    args = parser.parse_args()

    embed_single_text(args.text, title=args.title, description=args.description, source_id=args.source_id)
