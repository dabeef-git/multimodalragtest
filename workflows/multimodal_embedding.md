# Multimodal Embedding Pipeline

## Objective

Embed text, images, and videos into a unified vector space using Google's Gemini Embedding 2 model, store them in Pinecone, and query them with semantic search or RAG (retrieve + OpenAI answer).

## Required Inputs

- **GOOGLE_API_KEY** — from [Google AI Studio](https://aistudio.google.com/apikey)
- **PINECONE_API_KEY** — from [Pinecone Console](https://app.pinecone.io/)
- **OPENAI_API_KEY** — from [OpenAI Platform](https://platform.openai.com/api-keys)
- All keys go in `.env`

## First-Time Setup

1. Get API keys from the links above and paste them into `.env`
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Create the Pinecone index:
   ```bash
   python3 tools/setup_pinecone.py
   ```

## Pre-flight Checklist

- [ ] `.env` has all three API keys filled in (no placeholder values)
- [ ] Virtual environment is active (`which python3` points to `.venv/`)
- [ ] Dependencies installed (`pip list | grep google-genai`)
- [ ] Pinecone index exists (`python3 tools/setup_pinecone.py` shows stats)

## Pipeline Steps

### 1. Embed Text
```bash
python3 tools/embed_text.py --text "Your content here" --title "Optional Title"
```
- Formats text with structured prompt for retrieval
- Embeds with `gemini-embedding-2-preview` at 1536 dims
- L2-normalizes and upserts to Pinecone
- Result saved to `.tmp/last_embed_result.json`

### 2. Embed Images
```bash
python3 tools/embed_image.py --image-path photo.png --title "Photo Title"
```
- Supported: PNG, JPEG, GIF, WebP
- Max 6 images per batch request (Gemini limit)

### 3. Embed Videos
```bash
python3 tools/embed_video.py --video-path clip.mp4 --title "Video Title"
```
- Supported: MP4, MOV, AVI, WebM, MKV
- **Gemini limit: 120 seconds, 32 frames sampled.** Longer videos are truncated.
- Audio tracks in videos are ignored by the embedding model

### 4. Semantic Search
```bash
python3 tools/query_embeddings.py --query "your search terms" --top-k 5
```
- Cross-modal: text queries find relevant images and videos too
- Filter by type: `--content-type-filter image`
- Image search: `--image-path query_image.png` (find similar images/videos)

### 5. RAG Answer
```bash
python3 tools/search_and_answer.py --question "What is X?" --top-k 5
```
- Retrieves top-K context from Pinecone, sends to OpenAI for answer generation
- Default model: `gpt-4o`. Override: `--model gpt-4o-mini`
- Answer + sources saved to `.tmp/last_rag_answer.json`

## Edge Cases & Recovery

| Issue | Cause | Fix |
|-------|-------|-----|
| `GOOGLE_API_KEY` error | Key missing or invalid | Check `.env`, regenerate key at AI Studio |
| Gemini rate limit (429) | Too many requests | Wait 60s, retry. For bulk: use smaller batches |
| Pinecone index already exists | Re-running setup | Safe — `setup_pinecone.py` detects existing index |
| Video too long | >120s | Trim video before embedding, or accept truncation |
| Large file upload slow | Video >20MB | Expected behavior. Consider compressing first |
| Pinecone metadata too large | Text >40KB | Text is auto-truncated to 39K chars |
| Empty query results | No matching content | Embed more content, or broaden search terms |
| OpenAI token limit | Context too long | Reduce `--top-k` to limit context size |

## Customization

- **Dimensions**: Change `EMBEDDING_DIMS` in `tools/utils/config.py` (128, 768, 1536, or 3072). Must recreate Pinecone index after changing.
- **Index name**: Change `PINECONE_INDEX_NAME` in `.env`
- **Pinecone region**: Change `PINECONE_CLOUD` and `PINECONE_REGION` in `.env`
