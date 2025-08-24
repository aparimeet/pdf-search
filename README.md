## FlowAutomate PDF Search (Hybrid RAG)

Search paragraphs, tables, and images extracted from PDFs with a secure FastAPI service. Combines lexical TF‑IDF, semantic embeddings, and CLIP image-text alignment (multimodal RAG). Per‑document indexing and retrieval.

### Architecture
- ETL: Normalize PDF/extracted JSON to chunk records (paragraph | table | image). Images are embedded as base64 in metadata for indexing, not returned by the API.
- Indexing: 
  - TF‑IDF vectors for lexical search
  - Optional Sentence‑Transformers for text embeddings (stored in Weaviate when configured)
  - Optional CLIP model for image embeddings (stored in Weaviate when configured)
- API: Ingest and Search endpoints with API key auth and simple rate limiting.

### Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export SEARCH_API_KEY="dev-secret-change-me"
```

Optional models (first run may download weights):
- TEXT_EMBEDDING_MODEL_NAME (default: sentence-transformers/all-MiniLM-L6-v2)
- IMAGE_EMBEDDING_MODEL_NAME (default: clip-ViT-B-32)

### Run
```bash
uvicorn app.api.main:app --reload --port 8000
```

### Demo data
```bash
python -m app.demo.generate_synthetic

# Ingest one of the generated docs
curl -s -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" -H "x-api-key: $SEARCH_API_KEY" \
  -d @data/raw/sample_doc_1.json
```

### Ingest a PDF
```bash
curl -s -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" -H "x-api-key: $SEARCH_API_KEY" \
  -d '{"doc_id":"generated_pdf","pdf_path":"generated_document.pdf"}'
```

### Search API
```bash
# Text-only
curl -s -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" -H "x-api-key: $SEARCH_API_KEY" \
  -d '{"query":"revenue growth for Q2","doc_id":"sample_doc_1","top_k":5,"modality":"text"}'

# Image-only (text query matched to images via CLIP)
curl -s -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" -H "x-api-key: $SEARCH_API_KEY" \
  -d '{"query":"header image","doc_id":"generated_pdf","top_k":5,"modality":"image"}'

# Multimodal (default)
curl -s -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" -H "x-api-key: $SEARCH_API_KEY" \
  -d '{"query":"matplotlib sine wave figure","doc_id":"generated_pdf","top_k":5}'
```

### Security
- Header `x-api-key` required (set `SEARCH_API_KEY`).
- Simple in-memory rate limiting (env: `RATE_LIMIT_MAX_REQUESTS`, `RATE_LIMIT_WINDOW_SECS`).

### Optional storage
- Local files under `data/` (TF‑IDF, embeddings) are still written for inspection.
- Weaviate (preferred): set env `WEAVIATE_URL` and optionally `WEAVIATE_API_KEY`. The service will upsert vectors on ingest and query them on search.