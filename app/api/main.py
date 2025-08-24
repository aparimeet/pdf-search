from __future__ import annotations

from typing import List

from fastapi import Depends, FastAPI, HTTPException

from app.api.security import get_api_key
from app.core.config import settings
from app.etl.ingest import ingest_request, load_chunks
from app.models.schemas import IngestRequest, SearchRequest, SearchResponse, SearchResult
from app.search.indexer import build_index, load_index, search as hybrid_search

app = FastAPI(title="FlowAutomate PDF Search")


@app.post("/ingest", tags=["ingest"])
def ingest(req: IngestRequest, api_key: str = Depends(get_api_key)) -> dict:
    """Ingest a document (extracted JSON or PDF) and build per-doc index."""
    chunks = ingest_request(req)
    # Build index per document
    build_index(req.doc_id, chunks)
    return {"doc_id": req.doc_id, "chunks": len(chunks)}


@app.post("/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest, api_key: str = Depends(get_api_key)) -> SearchResponse:
    """Search within a document. Supports modality: text | image | multimodal (default)."""
    if not req.doc_id:
        # For demo simplicity, require doc_id to scope search
        raise HTTPException(status_code=400, detail="doc_id is required for scoped search")
    # Ensure index exists
    load_index(req.doc_id)
    results = hybrid_search(req.doc_id, req.query, top_k=req.top_k, modality=req.modality)
    # Format
    out: List[SearchResult] = []
    for c, score in results:
        meta = dict(c.metadata or {})
        if "image_b64" in meta:
            # strip large payloads from response, keep a hint only
            meta.pop("image_b64", None)
            meta["has_image_data"] = True
        out.append(
            SearchResult(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                text=c.text,
                type=c.type,
                page=c.page,
                score=score,
                metadata=meta,
            )
        )
    return SearchResponse(results=out)
