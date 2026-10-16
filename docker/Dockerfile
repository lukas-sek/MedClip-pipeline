FROM python:3.13-slim

WORKDIR /app

# Install uv for fast, reproducible dependency management.
RUN pip install --no-cache-dir uv

# Copy lockfile first so Docker layer cache is only invalidated on dep changes.
COPY pyproject.toml uv.lock ./

# Install production dependencies only (no dev tools like ruff in the image).
RUN uv sync --frozen --no-dev

# Copy the application package.
COPY MedClip_pipeline/ ./MedClip_pipeline/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "MedClip_pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
