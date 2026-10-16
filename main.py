from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import ensure_collection, get_client, search_embeddings
from preprocess import embed_text, load_model

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
MEDIA_DIR = ROOT / "data" / "cache"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


class ModelState:
    def __init__(self, processor: Any, model: Any, device: Any) -> None:
        self.processor = processor
        self.model = model
        self.device = device


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client()
    await ensure_collection(client)
    app.state.qdrant = client
    app.state.model = ModelState(*load_model())
    yield
    await client.close()


app = FastAPI(title="Biomedical Retrieval Engine", version="1.0.0", lifespan=lifespan)


@app.get("/")
async def ui() -> FileResponse:
    index = WEB_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index)


@app.post("/search")
async def search(request: SearchRequest) -> dict[str, list[dict]]:
    state = getattr(app.state, "model", None)
    client = getattr(app.state, "qdrant", None)
    if state is None or client is None:
        raise HTTPException(status_code=503, detail="Service is still starting")
    query_vector = embed_text(request.query, state.processor, state.model, state.device)
    results = await search_embeddings(client, query_vector, request.top_k)
    return {"results": results}


app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


async def index_directory(input_dir: str, limit: int | None = None) -> None:
    from database import ensure_collection, get_client, upsert_embeddings
    from preprocess import embed_directory, load_model

    processor, model, device = load_model()
    embeddings, metadata = embed_directory(
        input_dir, processor, model, device, limit=limit
    )
    if not metadata:
        raise FileNotFoundError(
            f"No supported images found in '{input_dir}'. "
            "Expected jpg, jpeg, png, bmp, tif, tiff, or webp files."
        )
    client = get_client()
    try:
        await ensure_collection(client)
        await upsert_embeddings(client, embeddings, metadata)
    finally:
        await client.close()
    print(f"Indexed {len(metadata)} images into Qdrant.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Biomedical image retrieval tools")
    parser.add_argument("command", choices=["index"])
    parser.add_argument("input_dir")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(index_directory(args.input_dir, args.limit))
