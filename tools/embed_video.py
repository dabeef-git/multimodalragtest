"""Embed video files and upsert to Pinecone."""

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
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}

MAX_VIDEO_SECONDS = 120
WARN_FILE_SIZE_MB = 20


def embed_single_video(video_path, title=None, description=None, source_id=None):
    """Embed a single video and upsert to Pinecone. Returns the vector ID."""
    ext = os.path.splitext(video_path)[1].lower()
    mime_type = MIME_TYPES.get(ext)
    if not mime_type:
        raise ValueError(f"Unsupported video format: {ext}. Supported: {list(MIME_TYPES.keys())}")

    file_size = os.path.getsize(video_path)
    if file_size > WARN_FILE_SIZE_MB * 1024 * 1024:
        print(f"  Warning: File is {file_size / 1024 / 1024:.1f}MB. Large files may be slow to upload.")

    print(f"  Note: Gemini processes max {MAX_VIDEO_SECONDS}s / 32 frames. Longer videos will be truncated.")

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    part = types.Part.from_bytes(data=video_bytes, mime_type=mime_type)
    vectors = embed_content([part])

    vector_id = source_id or str(uuid.uuid4())
    metadata = {
        "content_type": "video",
        "source_path": os.path.abspath(video_path),
        "title": title or os.path.basename(video_path),
        "description": description or "",
        "text": "",
        "timestamp": datetime.now().isoformat(),
        "file_size_bytes": file_size,
    }

    pc = get_pinecone_client()
    index = pc.Index(PINECONE_INDEX_NAME)
    index.upsert(vectors=[(vector_id, vectors[0], metadata)])

    print(f"Embedded video → {vector_id}")
    print(f"  File: {video_path}")
    if title:
        print(f"  Title: {title}")

    result = {"vector_id": vector_id, "metadata": metadata}
    with open(os.path.join(TMP_DIR, "last_embed_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return vector_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed a video into Pinecone")
    parser.add_argument("--video-path", required=True, help="Path to video file")
    parser.add_argument("--title", default=None, help="Title for the video")
    parser.add_argument("--description", default=None, help="Description")
    parser.add_argument("--source-id", default=None, help="Custom vector ID")
    args = parser.parse_args()

    embed_single_video(args.video_path, title=args.title, description=args.description, source_id=args.source_id)
