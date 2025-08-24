import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    raw_dir: Path = Path(os.getenv("RAW_DIR", str(Path("data/raw"))))
    processed_dir: Path = Path(os.getenv("PROCESSED_DIR", str(Path("data/processed"))))
    index_dir: Path = Path(os.getenv("INDEX_DIR", str(Path("data/index"))))
    images_dir: Path = Path(os.getenv("IMAGES_DIR", str(Path("data/images"))))

    api_key: str = os.getenv("SEARCH_API_KEY", "dev-secret-change-me")

    # Hybrid retrieval weights
    lexical_weight: float = float(os.getenv("LEXICAL_WEIGHT", "0.7"))
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.3"))
    image_weight: float = float(os.getenv("IMAGE_WEIGHT", "0.3"))

    # Embedding models
    text_embedding_model_name: str = os.getenv(
        "TEXT_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    image_embedding_model_name: str = os.getenv(
        "IMAGE_EMBEDDING_MODEL_NAME", "clip-ViT-B-32"
    )

    # Weaviate
    weaviate_url: str | None = os.getenv("WEAVIATE_URL")
    weaviate_api_key: str | None = os.getenv("WEAVIATE_API_KEY")
    weaviate_text_class: str = os.getenv("WEAVIATE_TEXT_CLASS", "PdfTextChunk")
    weaviate_image_class: str = os.getenv("WEAVIATE_IMAGE_CLASS", "PdfImageChunk")


settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.raw_dir.mkdir(parents=True, exist_ok=True)
settings.processed_dir.mkdir(parents=True, exist_ok=True)
settings.index_dir.mkdir(parents=True, exist_ok=True)
settings.images_dir.mkdir(parents=True, exist_ok=True)
