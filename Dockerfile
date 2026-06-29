# Use official Python image
# Use a slim Debian-based Python image to reduce attack surface and known vulnerabilities
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Install OS dependencies
RUN apt-get update \
	&& apt-get install -y --no-install-recommends build-essential poppler-utils curl \
	&& rm -rf /var/lib/apt/lists/*

# Install uv (Python package/dependency manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_LINK_MODE=copy
ENV PYTHONPATH="/app:/app/multi_doc_chat"

# Copy dependency manifests for better layer caching
COPY requirements.txt ./

# Install dependencies into the system interpreter using uv pip
RUN uv pip install --system -r requirements.txt

# Pre-fetch the local embedding + reranker models so the first user request is fast
# (otherwise the first upload would block while ~150MB of onnx models download).
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')" && \
    python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')"

# Copy project files
COPY . .

# Informational; the platform routes to the port the app binds below.
EXPOSE 8080

# Bind to the platform-provided $PORT (Render/Railway/Fly), defaulting to 8080 locally.
# sh -c so ${PORT} expands. Single worker — sessions + FAISS index are per-process.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]

# For local development with autoreload, override the CMD:
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]