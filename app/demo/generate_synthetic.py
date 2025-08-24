from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from app.core.config import settings


def _make_doc(doc_id: str, title: str, sections: List[Dict]) -> Dict:
    return {
        "doc_id": doc_id,
        "title": title,
        "paragraphs": [
            {"text": s["text"], "page": i + 1} for i, s in enumerate(sections)
        ],
        "tables": [
            {
                "rows": [["Quarter", "Revenue", "Growth"], ["Q1", "$1.2M", "10%"], ["Q2", "${}", "{}%".format(1.5 + i * 0.2, 15 + i * 2)]],
                "page": 2,
            }
            for i in range(1)
        ],
        "images": [
            {"caption": "Figure 1: Revenue trend", "page": 3}
        ],
    }


def main() -> None:
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    doc1 = _make_doc(
        "sample_doc_1",
        "Quarterly Financial Summary",
        sections=[
            {"text": "This document summarizes the quarterly performance including revenue and growth by segment."},
            {"text": "In Q2, revenue growth accelerated due to marketing spend and new product launches."},
            {"text": "The table shows revenue figures and growth percentages for key quarters."},
        ],
    )

    doc2 = _make_doc(
        "sample_doc_2",
        "Product Technical Overview",
        sections=[
            {"text": "This technical document describes system architecture and deployment considerations."},
            {"text": "Latency improvements were achieved by caching and query optimizations."},
            {"text": "We also provide an overview of scaling strategies for increased load."},
        ],
    )

    for doc in [doc1, doc2]:
        out_path = settings.raw_dir / f"{doc['doc_id']}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"Wrote synthetic docs to {settings.raw_dir}")


if __name__ == "__main__":
    main()
