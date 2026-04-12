"""Embed PDF and DOCX documents and upsert to Pinecone."""

import sys
import os
import json
import uuid
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.genai import types
from tools.utils.config import get_pinecone_client, PINECONE_INDEX_NAME, TMP_DIR
from tools.utils.embeddings import embed_content, prepare_text_for_embedding

MAX_METADATA_TEXT = 39000
MAX_PDF_PAGES_PER_EMBED = 6  # Gemini limit for PDF embedding


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return pages


def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    from docx import Document
    doc = Document(docx_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def embed_pdf_native(pdf_path, title=None, description=None):
    """Embed a PDF using Gemini's native PDF support (up to 6 pages per chunk)."""
    from PyPDF2 import PdfReader, PdfWriter
    import io

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    title = title or os.path.basename(pdf_path).replace(".pdf", "")

    # Also extract text for metadata
    all_text = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            all_text.append(t.strip())

    pc = get_pinecone_client()
    index = pc.Index(PINECONE_INDEX_NAME)
    vector_ids = []

    # Chunk PDF into groups of MAX_PDF_PAGES_PER_EMBED pages
    for chunk_start in range(0, total_pages, MAX_PDF_PAGES_PER_EMBED):
        chunk_end = min(chunk_start + MAX_PDF_PAGES_PER_EMBED, total_pages)

        # Create a sub-PDF for this chunk
        writer = PdfWriter()
        for i in range(chunk_start, chunk_end):
            writer.add_page(reader.pages[i])

        buf = io.BytesIO()
        writer.write(buf)
        pdf_bytes = buf.getvalue()

        part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
        vectors = embed_content([part])

        chunk_text = "\n".join(all_text[chunk_start:chunk_end])
        vector_id = str(uuid.uuid4())
        metadata = {
            "content_type": "document",
            "source_path": os.path.abspath(pdf_path),
            "title": title,
            "description": description or "",
            "text": chunk_text[:MAX_METADATA_TEXT],
            "timestamp": datetime.now().isoformat(),
            "page_range": f"{chunk_start + 1}-{chunk_end}",
            "total_pages": total_pages,
        }

        index.upsert(vectors=[(vector_id, vectors[0], metadata)])
        vector_ids.append(vector_id)
        print(f"  Embedded pages {chunk_start + 1}-{chunk_end} → {vector_id}")

    print(f"Embedded PDF: {title} ({total_pages} pages, {len(vector_ids)} chunks)")
    return vector_ids


def embed_docx(docx_path, title=None, description=None):
    """Embed a DOCX by extracting text and embedding it."""
    title = title or os.path.basename(docx_path).replace(".docx", "")
    text = extract_text_from_docx(docx_path)

    if not text.strip():
        print(f"  Warning: No text extracted from {docx_path}")
        return []

    # Chunk text into ~4000 char segments for better embedding quality
    chunks = _chunk_text(text, max_chars=4000)

    pc = get_pinecone_client()
    index = pc.Index(PINECONE_INDEX_NAME)
    vector_ids = []

    for i, chunk in enumerate(chunks):
        formatted = prepare_text_for_embedding(chunk, title=f"{title} (part {i + 1})", mode="document")
        vectors = embed_content([formatted])

        vector_id = str(uuid.uuid4())
        metadata = {
            "content_type": "document",
            "source_path": os.path.abspath(docx_path),
            "title": title,
            "description": description or "",
            "text": chunk[:MAX_METADATA_TEXT],
            "timestamp": datetime.now().isoformat(),
            "chunk_index": i + 1,
            "total_chunks": len(chunks),
        }

        index.upsert(vectors=[(vector_id, vectors[0], metadata)])
        vector_ids.append(vector_id)
        print(f"  Embedded chunk {i + 1}/{len(chunks)} → {vector_id}")

    print(f"Embedded DOCX: {title} ({len(chunks)} chunks)")
    return vector_ids


def _chunk_text(text, max_chars=4000):
    """Split text into chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text[:max_chars]]


def embed_document(doc_path, title=None, description=None):
    """Auto-detect document type and embed it."""
    ext = os.path.splitext(doc_path)[1].lower()
    if ext == ".pdf":
        return embed_pdf_native(doc_path, title=title, description=description)
    elif ext in (".docx", ".doc"):
        return embed_docx(doc_path, title=title, description=description)
    else:
        raise ValueError(f"Unsupported document format: {ext}. Supported: .pdf, .docx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed a PDF or DOCX into Pinecone")
    parser.add_argument("--doc-path", required=True, help="Path to document")
    parser.add_argument("--title", default=None, help="Title for the document")
    parser.add_argument("--description", default=None, help="Description")
    args = parser.parse_args()

    embed_document(args.doc_path, title=args.title, description=args.description)
