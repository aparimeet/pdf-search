from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    type: str = Field(description="paragraph | table | image")
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    doc_id: str
    title: Optional[str] = None
    # Either provide extracted JSON or a PDF path. JSON is preferred.
    extracted_json: Optional[Dict[str, Any]] = None
    pdf_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None
    top_k: int = 10
    modality: Optional[str] = Field(default=None, description="text | image | multimodal")


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    type: str
    page: Optional[int] = None
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: List[SearchResult]
