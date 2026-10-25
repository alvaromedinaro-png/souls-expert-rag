"""Ingesta: Markdown de data/ -> limpieza -> chunking -> ChromaDB.

Uso:
    python -m src.ingest            # indexa lo que falte (reanudable)
    python -m src.ingest --reset    # borra la colección y reindexa todo
"""

import argparse
import hashlib
import re
import time

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.config import DATA_DIR, get_vectorstore

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 50  # chunks por petición de embeddings

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=True,
)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Separa la cabecera de metadatos (--- ... ---) del cuerpo."""
    match = re.match(r"^---\n(.*?)\n---\n", raw, re.DOTALL)
    if not match:
        return {}, raw
    meta = dict(line.split(": ", 1) for line in match.group(1).splitlines())
    return meta, raw[match.end():]


def load_documents() -> list[Document]:
    docs = []
    for path in sorted(DATA_DIR.rglob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        meta["file"] = path.relative_to(DATA_DIR).as_posix()
        docs.append(Document(page_content=body, metadata=meta))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Divide por secciones Markdown y luego por tamaño.

    Cada chunk empieza con "Página > Sección" para que el embedding sepa
    de qué personaje o mecánica habla aunque el texto no lo nombre.
    """
    chunks = []
    for doc in docs:
        for section in header_splitter.split_text(doc.page_content):
            headers = [section.metadata[h] for h in ("h1", "h2", "h3")
                       if h in section.metadata]
            breadcrumb = " > ".join(headers) or doc.metadata["title"]
            for i, piece in enumerate(text_splitter.split_text(section.page_content)):
                if len(piece) < 50:  # fragmentos sin contenido útil
                    continue
                meta = {**doc.metadata, "section": breadcrumb}
                chunk_id = hashlib.sha1(
                    f"{doc.metadata['file']}|{breadcrumb}|{i}|{piece}".encode()
                ).hexdigest()
                chunks.append(Document(
                    page_content=f"[{breadcrumb}]\n{piece}",
                    metadata=meta, id=chunk_id,
                ))
    return chunks


def add_with_retry(vectorstore, batch: list[Document], retries: int = 6) -> None:
    """Añade un lote reintentando si Gemini devuelve límite de cuota (429)."""
    for attempt in range(retries):
        try:
            vectorstore.add_documents(batch, ids=[d.id for d in batch])
            return
        except Exception as exc:
            if attempt == retries - 1 or "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc):
                raise
            wait = 15 * (attempt + 1)
            print(f"  límite de cuota, esperando {wait}s...")
            time.sleep(wait)


def ingest(reset: bool = False) -> None:
    vectorstore = get_vectorstore()
    if reset:
        vectorstore.reset_collection()

    docs = load_documents()
    chunks = split_documents(docs)
    existing = set(vectorstore.get(include=[])["ids"])
    pending = [c for c in chunks if c.id not in existing]
    print(f"{len(docs)} documentos -> {len(chunks)} chunks "
          f"({len(existing)} ya indexados, {len(pending)} pendientes)")

    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]
        add_with_retry(vectorstore, batch)
        print(f"  indexados {start + len(batch)}/{len(pending)}")

    print(f"Colección lista: {vectorstore._collection.count()} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true",
                        help="borra la colección antes de indexar")
    ingest(parser.parse_args().reset)
