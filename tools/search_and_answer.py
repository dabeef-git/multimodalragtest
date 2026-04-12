"""RAG pipeline: query Pinecone for context, then generate an answer with OpenAI."""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.utils.config import get_openai_client, TMP_DIR
from tools.query_embeddings import query_by_text

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based on the provided context. "
    "If the context doesn't contain relevant information, say so. "
    "Cite which sources you used by referencing their titles."
)


def search_and_answer(question, top_k=5, model="gpt-4o"):
    """Retrieve relevant context from Pinecone and generate an answer with OpenAI."""
    print(f"Searching for: {question}")
    print()

    matches = query_by_text(question, top_k=top_k)

    # Build context from retrieved matches
    context_parts = []
    sources = []
    for match in matches:
        meta = match["metadata"]
        title = meta.get("title", "Untitled")
        content_type = meta.get("content_type", "unknown")

        if meta.get("text"):
            context_parts.append(f"[{content_type}] {title}: {meta['text']}")
        elif meta.get("description"):
            context_parts.append(f"[{content_type}] {title}: {meta['description']}")
        else:
            context_parts.append(f"[{content_type}] {title}: (no text content — {content_type} file at {meta.get('source_path', 'unknown path')})")

        sources.append({
            "id": match["id"],
            "title": title,
            "content_type": content_type,
            "score": match["score"],
            "source_path": meta.get("source_path", ""),
        })

    context = "\n\n---\n\n".join(context_parts)

    # Generate answer with OpenAI
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    answer = response.choices[0].message.content

    print("=" * 60)
    print("  Answer:")
    print("=" * 60)
    print(answer)
    print()

    result = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "model": model,
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(TMP_DIR, "last_rag_answer.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search and answer using RAG")
    parser.add_argument("--question", required=True, help="Your question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of context chunks")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model")
    args = parser.parse_args()

    search_and_answer(args.question, top_k=args.top_k, model=args.model)
