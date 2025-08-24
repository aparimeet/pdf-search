"""Hybrid RAG indexer for text and images (PDF chunks).

Builds TF-IDF and optional embedding indexes, and supports FAISS acceleration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import BytesIO

import joblib
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

faiss = None  # FAISS removed; we use Weaviate for vector search

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

from app.core.config import settings
from app.search import vector_store
from app.models.schemas import Chunk


@dataclass
class HybridIndex:
    # paths
    tfidf_path: Path
    meta_path: Path
    embed_path: Optional[Path]
    faiss_path: Optional[Path]

    # runtime
    vectorizer: Optional[TfidfVectorizer] = None
    tfidf_matrix: Optional[np.ndarray] = None
    chunks: Optional[List[Chunk]] = None
    embed_matrix: Optional[np.ndarray] = None
    faiss_index: Optional[object] = None
    embed_model: Optional[SentenceTransformer] = None
    # image embeddings
    image_matrix: Optional[np.ndarray] = None
    image_indices: Optional[List[int]] = None
    image_faiss_index: Optional[object] = None


def _load_embed_model() -> Optional[SentenceTransformer]:
    """Load the text embedding model (Sentence-Transformers).

    Returns a model instance or None if unavailable/offline.
    """
    if SentenceTransformer is None:
        return None
    try:
        # Prefer explicit text model, fallback to image CLIP name
        model_name = getattr(settings, 'text_embedding_model_name', None) or getattr(settings, 'embedding_model_name', None) or 'sentence-transformers/all-MiniLM-L6-v2'
        return SentenceTransformer(model_name)
    except Exception:
        return None


def _load_image_embed_model() -> Optional[SentenceTransformer]:
    """Load the image-text model (CLIP) for image embeddings.

    Returns a model instance or None if unavailable/offline.
    """
    if SentenceTransformer is None:
        return None
    try:
        model_name = getattr(settings, 'image_embedding_model_name', None) or 'clip-ViT-B-32'
        return SentenceTransformer(model_name)
    except Exception:
        return None


def _prep_paths(doc_id: str) -> HybridIndex:
    base = settings.index_dir / doc_id
    base.mkdir(parents=True, exist_ok=True)
    return HybridIndex(
        tfidf_path=base / "tfidf.joblib",
        meta_path=base / "chunks.jsonl",
        embed_path=base / "embeddings.npy",
        faiss_path=None,
        # multimodal
    )


def build_index(doc_id: str, chunks: List[Chunk]) -> HybridIndex:
    """Build per-document hybrid index.

    - TF-IDF for lexical matching
    - Optional text embeddings (+FAISS)
    - Optional image embeddings (+FAISS)
    Persists artifacts under data/index/<doc_id>/
    """
    paths = _prep_paths(doc_id)

    texts = [c.text or "" for c in chunks]

    # TF-IDF (robust to very small corpora)
    num_docs = len(texts)
    max_df = 1.0 if num_docs <= 3 else 0.9
    stop_words = None if num_docs <= 3 else "english"
    vectorizer = TfidfVectorizer(
        max_df=max_df,
        min_df=1,
        ngram_range=(1, 2),
        stop_words=stop_words,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    joblib.dump({"vectorizer": vectorizer, "tfidf": tfidf_matrix}, paths.tfidf_path)

    # Persist chunks
    with paths.meta_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")

    # Embeddings (optional)
    embed_model = _load_embed_model()
    embed_matrix = None
    faiss_index = None
    if embed_model is not None:
        try:
            embed_matrix = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            np.save(paths.embed_path, embed_matrix)
            # Upsert into Weaviate (if configured)
            try:
                vector_store.ensure_schema()
                vector_store.upsert_text_embeddings(doc_id, chunks, embed_matrix)
            except Exception:
                pass
        except Exception:
            embed_matrix = None
            faiss_index = None

    # Image embeddings using CLIP (if available) via sentence-transformers
    image_matrix = None
    image_indices: List[int] = []
    image_model = _load_image_embed_model()
    if image_model is not None:
        try:
            images: List[Image.Image] = []
            for idx, c in enumerate(chunks):
                if c.type != "image" or not isinstance(c.metadata, dict):
                    continue
                # Prefer file path if available (written by ETL), else fall back to base64
                img = None
                try:
                    img_path = c.metadata.get("image_path")
                    if img_path:
                        img = Image.open(img_path).convert("RGB")
                except Exception:
                    img = None
                if img is None and c.metadata.get("image_b64"):
                    try:
                        import base64
                        raw = base64.b64decode(c.metadata["image_b64"])  # type: ignore[index]
                        img = Image.open(BytesIO(raw)).convert("RGB")
                    except Exception:
                        img = None
                if img is not None:
                    images.append(img)
                    image_indices.append(idx)
            if images:
                image_matrix = image_model.encode(images=images, convert_to_numpy=True, normalize_embeddings=True)
                np.save((paths.meta_path.parent / "image_embeddings.npy"), image_matrix)
                # Save mapping indices
                with (paths.meta_path.parent / "image_indices.json").open("w", encoding="utf-8") as f:
                    json.dump(image_indices, f)
                # Upsert into Weaviate (if configured)
                try:
                    vector_store.ensure_schema()
                    vector_store.upsert_image_embeddings(doc_id, chunks, image_matrix, image_indices)
                except Exception:
                    pass
        except Exception:
            image_matrix = None

    return HybridIndex(
        tfidf_path=paths.tfidf_path,
        meta_path=paths.meta_path,
        embed_path=paths.embed_path if embed_model is not None else None,
        faiss_path=paths.faiss_path if (embed_model is not None and faiss is not None) else None,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        chunks=chunks,
        embed_matrix=embed_matrix,
        faiss_index=faiss_index,
        embed_model=embed_model,
        image_matrix=image_matrix,
        image_indices=image_indices or None,
    )


def _load_chunks(path: Path) -> List[Chunk]:
    items: List[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(Chunk.model_validate_json(line))
    return items


def load_index(doc_id: str) -> HybridIndex:
    """Load a document's persisted hybrid index into memory."""
    paths = _prep_paths(doc_id)
    # Load tf-idf
    tfidf_obj = joblib.load(paths.tfidf_path)
    vectorizer: TfidfVectorizer = tfidf_obj["vectorizer"]
    tfidf_matrix = tfidf_obj["tfidf"]

    # Load chunks
    chunks = _load_chunks(paths.meta_path)

    # Load embeddings (if available)
    embed_matrix = None
    faiss_index = None
    embed_model = None
    if paths.embed_path.exists():
        try:
            embed_matrix = np.load(paths.embed_path)
            embed_model = _load_embed_model()
        except Exception:
            embed_matrix = None
    # FAISS removed

    # Load image embeddings and indices if present
    image_matrix = None
    image_indices = None
    image_faiss_index = None
    try:
        image_path = paths.meta_path.parent / "image_embeddings.npy"
        index_path = paths.meta_path.parent / "image_indices.json"
        if image_path.exists():
            image_matrix = np.load(image_path)
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                image_indices = json.load(f)
        # FAISS removed
    except Exception:
        image_matrix = None
        image_indices = None
        image_faiss_index = None

    return HybridIndex(
        tfidf_path=paths.tfidf_path,
        meta_path=paths.meta_path,
        embed_path=paths.embed_path if embed_matrix is not None else None,
        faiss_path=paths.faiss_path if faiss_index is not None else None,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        chunks=chunks,
        embed_matrix=embed_matrix,
        faiss_index=faiss_index,
        embed_model=embed_model,
        image_matrix=image_matrix,
        image_indices=image_indices,
        image_faiss_index=image_faiss_index,
    )


def search(doc_id: str, query: str, top_k: int = 10, modality: Optional[str] = None) -> List[Tuple[Chunk, float]]:
    """Search a document using hybrid scoring.

    modality: 'text' | 'image' | None (multimodal)
    Returns list of (Chunk, score) pairs.
    """
    idx = load_index(doc_id)
    assert idx.vectorizer is not None and idx.tfidf_matrix is not None and idx.chunks is not None

    # Lexical
    q_vec = idx.vectorizer.transform([query])
    lex_scores = cosine_similarity(q_vec, idx.tfidf_matrix).ravel()

    # Semantic (optional)
    sem_scores = np.zeros_like(lex_scores)
    if idx.embed_model is not None:
        try:
            # Compute query embedding once
            q_emb = idx.embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            # Use Weaviate vector search
            try:
                weav_hits = vector_store.search_text(doc_id, q_emb[0], top_k=min(top_k * 5, len(idx.chunks)))
            except Exception:
                weav_hits = []
            if weav_hits:
                # Map chunk_id -> index
                id_to_idx = {c.chunk_id: i for i, c in enumerate(idx.chunks)}
                for cid, s in weav_hits:
                    i = id_to_idx.get(cid)
                    if i is not None:
                        sem_scores[i] = max(sem_scores[i], float(s))
            elif idx.embed_matrix is not None:
                # Fallback: local dot product
                sims = (idx.embed_matrix @ q_emb[0])
                sem_scores = sims
        except Exception:
            sem_scores = np.zeros_like(lex_scores)

    # Fuzzy (tiny boost for near matches)
    fuzzy_boost = np.array([
        fuzz.token_set_ratio(query, c.text) / 100.0 for c in idx.chunks
    ]) * 0.05

    # Image alignment: text query vs image embeddings (CLIP)
    image_scores = np.zeros_like(lex_scores)
    if idx.image_matrix is not None and idx.image_indices:
        try:
            img_model = _load_image_embed_model()
            if img_model is None:
                raise RuntimeError("no image model")
            q_emb = img_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            # Use Weaviate vector search
            try:
                img_hits = vector_store.search_image(doc_id, q_emb[0], top_k=min(top_k * 5, len(idx.image_indices)))
            except Exception:
                img_hits = []
            if img_hits:
                # Map search results (chunk_id) back to chunk index
                id_to_idx = {c.chunk_id: i for i, c in enumerate(idx.chunks)}
                for cid, s in img_hits:
                    i = id_to_idx.get(cid)
                    if i is not None:
                        image_scores[i] = max(image_scores[i], float(s))
            else:
                sims = (idx.image_matrix @ q_emb[0])
                for local_i, chunk_idx in enumerate(idx.image_indices):
                    image_scores[chunk_idx] = float(sims[local_i])
        except Exception:
            pass

    if modality == "image":
        combined = image_scores
    elif modality == "text":
        combined = settings.lexical_weight * lex_scores + settings.semantic_weight * sem_scores + fuzzy_boost
    else:
        combined = (
            settings.lexical_weight * lex_scores
            + settings.semantic_weight * sem_scores
            + settings.image_weight * image_scores
            + fuzzy_boost
        )

    top_idx = np.argsort(-combined)[:top_k]
    results = [(idx.chunks[i], float(combined[i])) for i in top_idx]
    return results
