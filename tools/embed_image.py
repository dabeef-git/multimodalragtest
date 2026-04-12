"""Embed image files and upsert to Pinecone."""

import sys
import os
import json
import uuid
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.genai import types
from tools.utils.config import get_pinecone_client, PINECONE_INDEX_NAME, TMP_DIR
from tools.utils.embeddings import embed_content

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def embed_single_image(image_path, title=None, description=None, source_id=None):
    """Embed a single image and upsert to Pinecone. Returns the vector ID."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = MIME_TYPES.get(ext)
    if not mime_type:
        raise ValueError(f"Unsupported image format: {ext}. Supported: {list(MIME_TYPES.keys())}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    vectors = embed_content([part])

    vector_id = source_id or str(uuid.uuid4())
    metadata = {
        "content_type": "image",
        "source_path": os.path.abspath(image_path),
        "title": title or os.path.basename(image_path),
        "description": description or "",
        "text": "",
        "timestamp": datetime.now().isoformat(),
        "file_size_bytes": len(image_bytes),
    }

    pc = get_pinecone_client()
    index = pc.Index(PINECONE_INDEX_NAME)
    index.upsert(vectors=[(vector_id, vectors[0], metadata)])

    print(f"Embedded image → {vector_id}")
    print(f"  File: {image_path}")
    if title:
        print(f"  Title: {title}")

    result = {"vector_id": vector_id, "metadata": metadata}
    with open(os.path.join(TMP_DIR, "last_embed_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return vector_id


def embed_images(image_paths, titles=None, descriptions=None):
    """Batch-embed multiple images.

    Args:
        image_paths: List of file paths.
        titles: Optional list of titles (same length as image_paths).
        descriptions: Optional list of descriptions.
    """
    parts = []
    for path in image_paths:
        ext = os.path.splitext(path)[1].lower()
        mime_type = MIME_TYPES.get(ext)
        if not mime_type:
            raise ValueError(f"Unsupported image format: {ext} for {path}")
        with open(path, "rb") as f:
            parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime_type))

    vectors = embed_content(parts)

    pc = get_pinecone_client()
    index = pc.Index(PINECONE_INDEX_NAME)

    records = []
    for i, path in enumerate(image_paths):
        vector_id = str(uuid.uuid4())
        metadata = {
            "content_type": "image",
            "source_path": os.path.abspath(path),
            "title": (titles[i] if titles else None) or os.path.basename(path),
            "description": (descriptions[i] if descriptions else None) or "",
            "text": "",
            "timestamp": datetime.now().isoformat(),
            "file_size_bytes": os.path.getsize(path),
        }
        records.append((vector_id, vectors[i], metadata))

    for batch_start in range(0, len(records), 100):
        batch = records[batch_start : batch_start + 100]
        index.upsert(vectors=batch)

    print(f"Embedded {len(records)} images.")
    return [r[0] for r in records]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed an image into Pinecone")
    parser.add_argument("--image-path", required=True, help="Path to image file")
    parser.add_argument("--title", default=None, help="Title for the image")
    parser.add_argument("--description", default=None, help="Description")
    parser.add_argument("--source-id", default=None, help="Custom vector ID")
    args = parser.parse_args()

    embed_single_image(args.image_path, title=args.title, description=args.description, source_id=args.source_id)
