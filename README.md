# MedClip-pipeline

Multimodal biomedical image retrieval: text queries → [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) embeddings → nearest neighbors in [Qdrant](https://qdrant.tech/).

Index dermatology images streamed from Hugging Face (no local dataset zips), search them from a FastAPI API or a small browser UI.

## Stack

| Layer | Technology |
| --- | --- |
| Model | BiomedCLIP via `open_clip` (512-d image/text embeddings) |
| Indexing | Hugging Face `datasets` streaming + OpenCV/Pillow preprocess |
| Vector DB | Qdrant (cosine similarity) |
| API | FastAPI + Uvicorn |
| UI | Static web app served by FastAPI |
| Tooling | `uv`, Docker Compose, GitHub Actions |

## Quick start

### 1. Environment

```bash
cp .env.example .env
```

Defaults point Qdrant at `localhost:6333`.

### 2. Start Qdrant

```bash
cd docker
docker compose up -d qdrant
```

### 3. Install & index sample data

```bash
uv sync
uv run python stream_index.py --limit 200
```

This reservoir-samples images from `marmal88/skin_cancer`, caches JPEGs under `data/cache/`, and (re)builds the Qdrant collection.

### 4. Run the API + UI

```bash
uv run uvicorn main:app --reload --port 8000
```

Open **http://127.0.0.1:8000/** — or call the API:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search `
  -ContentType "application/json" `
  -Body '{"query":"melanoma","top_k":5}' | ConvertTo-Json -Depth 6
```

Example queries: `melanoma`, `basal cell carcinoma`, `dermatoscopic image of actinic keratosis`.

## Project layout

```
main.py           FastAPI app (POST /search, UI, /media)
preprocess.py     BiomedCLIP load + embed helpers
load_frames.py    Local frame/tile loader CLI
stream_index.py   HF stream → embed → Qdrant + image cache
database.py       Async Qdrant client helpers
web/              Browser search UI
docker/           Compose (Qdrant + optional API image)
tests/            Lightweight unit tests
```

## Design notes

- Embeddings are L2-normalized; Qdrant uses cosine distance.
- Point IDs are stable UUIDv5 values derived from image filenames so re-indexes upsert cleanly.
- `stream_index.py` recreates the collection by default to avoid stale vectors without cached thumbnails.
- Absolute CLIP scores are often ~0.3–0.5; ranking and returned `class` labels matter more than hitting 0.9.

## Tests & CI

```bash
uv run pytest tests/ -v
```

## License / data

Use of `marmal88/skin_cancer` (HAM10000-derived) is subject to the dataset’s own terms on Hugging Face. This repo does not redistribute the images.
