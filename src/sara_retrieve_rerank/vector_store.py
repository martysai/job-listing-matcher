from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from sara_retrieve_rerank.config import (
    BATCH_SIZE,
    COLLECTION_NAME,
    DEFAULT_CHROMA_DIR,
    EMBEDDING_MODEL,
    HNSW_SPACE,
)


def batched(items: Sequence, batch_size: int):
    """Yield items in fixed-size batches."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def create_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Create the embedding model used by Chroma."""
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vectorstore(
    embedding_model: str = EMBEDDING_MODEL,
    collection_name: str = COLLECTION_NAME,
    persist_directory: str | Path | None = DEFAULT_CHROMA_DIR,
    reset: bool = True,
) -> Chroma:
    """Create a Chroma vector store.

    By default this uses a persistent directory under `data/chroma`. Set
    `persist_directory=None` for an in-memory store like the original notebook.
    """
    embeddings = create_embeddings(embedding_model)

    kwargs = {}
    if persist_directory is not None:
        kwargs["persist_directory"] = str(persist_directory)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": HNSW_SPACE},
        **kwargs,
    )

    if reset:
        vectorstore.reset_collection()

    return vectorstore


def index_documents(
    vectorstore: Chroma,
    documents: Sequence[Document],
    batch_size: int = BATCH_SIZE,
) -> None:
    """Add vacancy documents to Chroma in batches."""
    for i, batch in enumerate(batched(documents, batch_size), start=1):
        ids = [doc.metadata["dataset_id"] for doc in batch]
        vectorstore.add_documents(batch, ids=ids)
        print(f"Added batch {i}: {len(batch)} vacancies")


def vectorstore_is_empty(vectorstore: Chroma) -> bool:
    """Best-effort emptiness check for a Chroma vector store.

    Used by the pipeline wrapper to decide whether to (re)build the index on
    startup or reuse the persisted one.
    """
    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        return True
    try:
        return int(collection.count()) == 0
    except Exception:
        return True
