from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, Iterable, List
import base64

from pypdf import PdfReader

from app.core.config import settings
from app.models.schemas import Chunk, IngestRequest


def _normalize_table(table_obj: Dict) -> str:
    # Expect table as list of rows or dict; join cells
    if not table_obj:
        return ""
    if isinstance(table_obj, dict) and "rows" in table_obj:
        rows = table_obj.get("rows", [])
    else:
        rows = table_obj
    try:
        return "\n".join(["\t".join(map(str, r)) for r in rows])
    except Exception:
        return json.dumps(table_obj)


def _normalize_image(image_obj: Dict) -> str:
    # For images, we may have captions/alt text extracted by upstream parser
    caption = image_obj.get("caption") if isinstance(image_obj, dict) else None
    return caption or "[image]"


def _from_extracted_json(doc_id: str, extracted: Dict) -> List[Chunk]:
    chunks: List[Chunk] = []
    paragraphs = extracted.get("paragraphs", [])
    tables = extracted.get("tables", [])
    images = extracted.get("images", [])

    for i, p in enumerate(paragraphs):
        text = p.get("text") if isinstance(p, dict) else str(p)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::p::{i}::{uuid.uuid4().hex[:8]}",
                doc_id=doc_id,
                text=text or "",
                type="paragraph",
                page=p.get("page") if isinstance(p, dict) else None,
                metadata={"source": "json"},
            )
        )

    for i, t in enumerate(tables):
        text = _normalize_table(t)
        page = t.get("page") if isinstance(t, dict) else None
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::t::{i}::{uuid.uuid4().hex[:8]}",
                doc_id=doc_id,
                text=text,
                type="table",
                page=page,
                metadata={"source": "json"},
            )
        )

    for i, img in enumerate(images):
        text = _normalize_image(img)
        page = img.get("page") if isinstance(img, dict) else None
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::i::{i}::{uuid.uuid4().hex[:8]}",
                doc_id=doc_id,
                text=text,
                type="image",
                page=page,
                metadata={"source": "json"},
            )
        )

    return chunks


def _extract_images_from_page(doc_id: str, page_obj, page_index: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    try:
        resources = page_obj.get("/Resources") or {}
        xobjects = resources.get("/XObject") or {}
        for name, xobj in (xobjects.items() if hasattr(xobjects, 'items') else []):
            obj = xobj.get_object() if hasattr(xobj, 'get_object') else xobj
            if obj.get("/Subtype") == "/Image":
                data = obj.get_data()
                b64 = base64.b64encode(data).decode("utf-8")
                meta = {"source": "pdf-image", "image_b64": b64}

                # Attempt to persist image to disk for inspection and downstream reuse
                try:
                    from io import BytesIO as _BytesIO  # local import to avoid hard dep at module load
                    from PIL import Image  # type: ignore
                    out_dir = settings.images_dir / doc_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    safe = str(name).replace("/", "_")
                    out_path = out_dir / f"page-{page_index + 1}-{safe}.png"
                    img = Image.open(_BytesIO(data)).convert("RGB")
                    img.save(out_path)
                    meta["image_path"] = str(out_path)
                except Exception:
                    # If Pillow or decoding fails, skip file persistence but keep base64
                    pass
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::i::{page_index}-{name}::{uuid.uuid4().hex[:8]}",
                        doc_id=doc_id,
                        text=f"[image] page={page_index+1}",
                        type="image",
                        page=page_index + 1,
                        metadata=meta,
                    )
                )
    except Exception:
        pass
    return chunks


def _from_pdf_text(doc_id: str, pdf_path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    reader = PdfReader(str(pdf_path))
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        for j, para in enumerate(text.split("\n\n")):
            if not para.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::p::{page_index}-{j}::{uuid.uuid4().hex[:8]}",
                    doc_id=doc_id,
                    text=para.strip(),
                    type="paragraph",
                    page=page_index + 1,
                    metadata={"source": "pdf-text"},
                )
            )
        # Extract images on the page
        chunks.extend(_extract_images_from_page(doc_id, page, page_index))
    return chunks


def ingest_request(req: IngestRequest) -> List[Chunk]:
    chunks: List[Chunk] = []
    if req.extracted_json:
        chunks = _from_extracted_json(req.doc_id, req.extracted_json)
    elif req.pdf_path:
        chunks = _from_pdf_text(req.doc_id, Path(req.pdf_path))
    else:
        raise ValueError("IngestRequest requires extracted_json or pdf_path")

    # Persist processed chunks per doc
    out_path = settings.processed_dir / f"{req.doc_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    return chunks


def load_chunks(doc_id: str) -> List[Chunk]:
    path = settings.processed_dir / f"{doc_id}.jsonl"
    if not path.exists():
        return []
    chunks: List[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunks.append(Chunk.model_validate_json(line))
    return chunks
