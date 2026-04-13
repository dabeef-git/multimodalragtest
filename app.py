"""Simple chat web app for testing the multimodal RAG pipeline."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, send_file, abort
from tools.utils.config import get_openai_client, get_pinecone_client, PINECONE_INDEX_NAME
from tools.utils.embeddings import embed_content, prepare_text_for_embedding
from tools.query_embeddings import query_by_text

app = Flask(__name__, template_folder="web", static_folder="web/static")

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based on the provided context. "
    "If the context doesn't contain relevant information, say so honestly. "
    "Cite which sources you used by referencing their titles. "
    "Keep answers clear and concise. "
    "Important: Images and videos from the context are automatically displayed inline below your response. "
    "Do NOT say you cannot show or display media. Simply describe what the images or videos contain based on the context, "
    "and the user will see them rendered below your answer."
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    top_k = data.get("top_k", 5)

    # Retrieve relevant context from Pinecone
    matches = query_by_text(question, top_k=top_k)

    # Build context
    context_parts = []
    sources = []
    for match in matches:
        meta = match["metadata"]
        title = meta.get("title", "Untitled")
        content_type = meta.get("content_type", "unknown")
        score = match["score"]

        if meta.get("text"):
            context_parts.append(f"[{content_type}] {title}: {meta['text'][:2000]}")
        elif meta.get("description"):
            context_parts.append(f"[{content_type}] {title}: {meta['description']}")
        else:
            context_parts.append(f"[{content_type}] {title}: ({content_type} file)")

        # Build a serveable URL for images and videos
        media_url = ""
        source_path = meta.get("source_path", "")
        if source_path and content_type in ("image", "video"):
            # Extract the relative path after "assets/" regardless of machine
            idx = source_path.find("/assets/")
            if idx != -1:
                rel = source_path[idx + len("/assets/"):]
                # Verify the file actually exists in our local assets dir
                local_path = os.path.join(ASSETS_DIR, rel)
                if os.path.isfile(local_path):
                    media_url = f"/media/{rel}"

        sources.append({
            "title": title,
            "content_type": content_type,
            "score": round(score, 4),
            "source_path": source_path,
            "page_range": meta.get("page_range", ""),
            "media_url": media_url,
        })

    context = "\n\n---\n\n".join(context_parts)

    # Generate answer with OpenAI
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    answer = response.choices[0].message.content

    return jsonify({
        "answer": answer,
        "sources": sources,
    })


@app.route("/media/<path:filepath>")
def serve_media(filepath):
    """Serve image/video files from the assets directory."""
    full_path = os.path.join(ASSETS_DIR, filepath)
    if not os.path.isfile(full_path):
        abort(404)
    # Security: ensure the resolved path is still within ASSETS_DIR
    if not os.path.realpath(full_path).startswith(os.path.realpath(ASSETS_DIR)):
        abort(403)
    return send_file(full_path)


@app.route("/api/stats", methods=["GET"])
def stats():
    pc = get_pinecone_client()
    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    return jsonify({
        "total_vectors": stats.total_vector_count,
        "index_name": PINECONE_INDEX_NAME,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
