"""Create or verify the Pinecone serverless index for multimodal embeddings."""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.utils.config import (
    get_pinecone_client,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIMS,
    TMP_DIR,
)


def setup_index():
    pc = get_pinecone_client()

    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME in existing:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")
    else:
        print(f"Creating index '{PINECONE_INDEX_NAME}' ...")
        from pinecone import ServerlessSpec

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMS,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

        while not pc.describe_index(PINECONE_INDEX_NAME).status.get("ready"):
            print("  Waiting for index to be ready...")
            time.sleep(2)

        print("Index created and ready.")

    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    status = {
        "index_name": PINECONE_INDEX_NAME,
        "dimension": EMBEDDING_DIMS,
        "metric": "cosine",
        "total_vector_count": stats.total_vector_count,
        "namespaces": {k: v.vector_count for k, v in stats.namespaces.items()},
    }

    print("=" * 60)
    print(f"  Index:      {PINECONE_INDEX_NAME}")
    print(f"  Dimensions: {EMBEDDING_DIMS}")
    print(f"  Vectors:    {stats.total_vector_count}")
    print("=" * 60)

    with open(os.path.join(TMP_DIR, "pinecone_status.json"), "w") as f:
        json.dump(status, f, indent=2)

    return status


if __name__ == "__main__":
    setup_index()
