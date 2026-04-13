"""Query Pinecone by embedding a text query or image and returning top-K results."""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.genai import types
from tools.utils.config import get_pinecone_client, PINECONE_INDEX_NAME, TMP_DIR
from tools.utils.embeddings import embed_content, prepare_text_for_embedding


def query_by_text(query, top_k=5, content_type_filter=None):
    """Query Pinecone with a text query. Returns list of matches."""
    formatted = prepare_text_for_embedding(query, mode="query")
    vectors = embed_content([formatted])
    return _query_pinecone(vectors[0], top_k, content_type_filter)


def query_by_image(image_path, top_k=5, content_type_filter=None):
    """Query Pinecone with an image. Returns list of matches."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp"}
    mime_type = mime_map.get(ext)
    if not mime_type:
        raise ValueError(f"Unsupported image format: {ext}")

    with open(image_path, "rb") as f:
        part = types.Part.from_bytes(data=f.read(), mime_type=mime_type)

    vectors = embed_content([part])
    return _query_pinecone(vectors[0], top_k, content_type_filter)


def _query_pinecone(vector, top_k, content_type_filter):
    pc = get_pinecone_client()
    index = pc.Index(PINECONE_INDEX_NAME)

    filter_dict = None
    if content_type_filter:
        filter_dict = {"content_type": {"$eq": content_type_filter}}

    results = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict,
    )

    matches = []
    print("=" * 60)
    print(f"  Top {top_k} results:")
    print("=" * 60)
    for i, match in enumerate(results.matches, 1):
        meta = match.metadata or {}
        print(f"  {i}. [{meta.get('content_type', '?')}] {meta.get('title', 'Untitled')}")
        print(f"     Score: {match.score:.4f}")
        if meta.get("source_path"):
            print(f"     Source: {meta['source_path']}")
        if meta.get("text"):
            preview = meta["text"][:200]
            print(f"     Text: {preview}...")
        print()

        matches.append({
            "id": match.id,
            "score": match.score,
            "metadata": meta,
        })

    try:
        with open(os.path.join(TMP_DIR, "last_query_results.json"), "w") as f:
            json.dump({"matches": matches}, f, indent=2, default=str)
    except OSError:
        pass

    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Pinecone embeddings")
    parser.add_argument("--query", default=None, help="Text query")
    parser.add_argument("--image-path", default=None, help="Image query (path to file)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--content-type-filter", default=None, choices=["text", "image", "video"],
                        help="Filter by content type")
    args = parser.parse_args()

    if not args.query and not args.image_path:
        parser.error("Provide --query or --image-path")

    if args.query:
        query_by_text(args.query, top_k=args.top_k, content_type_filter=args.content_type_filter)
    else:
        query_by_image(args.image_path, top_k=args.top_k, content_type_filter=args.content_type_filter)
