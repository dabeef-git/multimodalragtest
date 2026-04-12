"""Batch-embed all assets in the assets/ directory into Pinecone."""

import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.embed_image import embed_single_image
from tools.embed_video import embed_single_video
from tools.embed_document import embed_document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
DOC_EXTS = {".pdf", ".docx", ".doc"}


def embed_all():
    total = 0

    # Images
    image_dir = os.path.join(ASSETS_DIR, "images")
    if os.path.isdir(image_dir):
        images = [f for f in os.listdir(image_dir)
                  if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        if images:
            print(f"\n{'='*60}")
            print(f"  Embedding {len(images)} image(s)")
            print(f"{'='*60}\n")
            for fname in sorted(images):
                path = os.path.join(image_dir, fname)
                title = os.path.splitext(fname)[0]
                try:
                    embed_single_image(path, title=title)
                    total += 1
                except Exception as e:
                    print(f"  ERROR embedding {fname}: {e}")

    # Videos
    video_dir = os.path.join(ASSETS_DIR, "videos")
    if os.path.isdir(video_dir):
        videos = [f for f in os.listdir(video_dir)
                  if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
        if videos:
            print(f"\n{'='*60}")
            print(f"  Embedding {len(videos)} video(s)")
            print(f"{'='*60}\n")
            for fname in sorted(videos):
                path = os.path.join(video_dir, fname)
                title = os.path.splitext(fname)[0].replace("_", " ")
                try:
                    embed_single_video(path, title=title)
                    total += 1
                except Exception as e:
                    print(f"  ERROR embedding {fname}: {e}")

    # Documents
    text_dir = os.path.join(ASSETS_DIR, "text")
    if os.path.isdir(text_dir):
        docs = [f for f in os.listdir(text_dir)
                if os.path.splitext(f)[1].lower() in DOC_EXTS]
        if docs:
            print(f"\n{'='*60}")
            print(f"  Embedding {len(docs)} document(s)")
            print(f"{'='*60}\n")
            for fname in sorted(docs):
                path = os.path.join(text_dir, fname)
                title = os.path.splitext(fname)[0].replace("_", " ")
                try:
                    ids = embed_document(path, title=title)
                    total += len(ids) if ids else 0
                except Exception as e:
                    print(f"  ERROR embedding {fname}: {e}")

    print(f"\n{'='*60}")
    print(f"  Done! {total} total vectors upserted to Pinecone.")
    print(f"{'='*60}")


if __name__ == "__main__":
    embed_all()
