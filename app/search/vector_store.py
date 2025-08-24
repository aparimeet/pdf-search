from __future__ import annotations

"""Weaviate vector store wrapper for text and image chunks."""

from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    import weaviate
    from weaviate.classes.config import Property, DataType
    from weaviate.classes.init import AdditionalConfig, Auth
    from weaviate.classes.query import MetadataQuery
except Exception:  # pragma: no cover
    weaviate = None  # type: ignore

from app.core.config import settings
from app.models.schemas import Chunk


def _client() -> Optional[weaviate.WeaviateClient]:  # type: ignore[name-defined]
    if weaviate is None or not settings.weaviate_url:
        return None
    auth = None
    if settings.weaviate_api_key:
        auth = Auth.api_key(settings.weaviate_api_key)
    return weaviate.connect_to_weaviate(
        http_host=settings.weaviate_url,
        auth_credentials=auth,
        additional_config=AdditionalConfig(timeout=30),
    )


def ensure_schema() -> None:
    client = _client()
    if client is None:
        return
    try:
        # Text class
        if settings.weaviate_text_class not in [c.name for c in client.collections.list_all()]:
            client.collections.create(
                settings.weaviate_text_class,
                vectorizer_config={"none": {}},
                properties=[
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="type", data_type=DataType.TEXT),
                    Property(name="page", data_type=DataType.NUMBER),
                ],
            )
        # Image class
        if settings.weaviate_image_class not in [c.name for c in client.collections.list_all()]:
            client.collections.create(
                settings.weaviate_image_class,
                vectorizer_config={"none": {}},
                properties=[
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="type", data_type=DataType.TEXT),
                    Property(name="page", data_type=DataType.NUMBER),
                ],
            )
    finally:
        client.close()


def upsert_text_embeddings(doc_id: str, chunks: List[Chunk], embeddings: np.ndarray) -> None:
    client = _client()
    if client is None:
        return
    coll = client.collections.get(settings.weaviate_text_class)
    try:
        data = []
        for c, vec in zip(chunks, embeddings):
            if c.doc_id != doc_id:
                continue
            data.append({
                "properties": {
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "type": c.type,
                    "page": c.page or 0,
                },
                "vector": vec.astype(float).tolist(),
                "id": c.chunk_id,
            })
        if data:
            coll.data.insert_many(data)
    finally:
        client.close()


def upsert_image_embeddings(doc_id: str, chunks: List[Chunk], embeddings: np.ndarray, image_indices: List[int]) -> None:
    client = _client()
    if client is None:
        return
    coll = client.collections.get(settings.weaviate_image_class)
    try:
        data = []
        for local_i, vec in enumerate(embeddings):
            idx = image_indices[local_i]
            c = chunks[idx]
            if c.doc_id != doc_id:
                continue
            data.append({
                "properties": {
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "type": c.type,
                    "page": c.page or 0,
                },
                "vector": vec.astype(float).tolist(),
                "id": c.chunk_id,
            })
        if data:
            coll.data.insert_many(data)
    finally:
        client.close()


def search_text(doc_id: str, query_vec: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
    client = _client()
    if client is None:
        return []
    coll = client.collections.get(settings.weaviate_text_class)
    try:
        res = coll.query.hybrid(
            query="",
            vector=query_vec.astype(float).tolist(),
            limit=top_k,
            filters=weaviate.classes.query.Filter.by_property("doc_id").equal(doc_id),
            return_metadata=MetadataQuery(distance=True),
        )
        out: List[Tuple[str, float]] = []
        for o in res.objects:
            score = 1.0 - float(o.metadata.distance or 0.0)
            out.append((o.properties["chunk_id"], score))
        return out
    finally:
        client.close()


def search_image(doc_id: str, query_vec: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
    client = _client()
    if client is None:
        return []
    coll = client.collections.get(settings.weaviate_image_class)
    try:
        res = coll.query.hybrid(
            query="",
            vector=query_vec.astype(float).tolist(),
            limit=top_k,
            filters=weaviate.classes.query.Filter.by_property("doc_id").equal(doc_id),
            return_metadata=MetadataQuery(distance=True),
        )
        out: List[Tuple[str, float]] = []
        for o in res.objects:
            score = 1.0 - float(o.metadata.distance or 0.0)
            out.append((o.properties["chunk_id"], score))
        return out
    finally:
        client.close()


