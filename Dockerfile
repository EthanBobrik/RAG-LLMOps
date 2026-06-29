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

# Copy project files
COPY . .


# Expose port
EXPOSE 8080

# Run FastAPI with uvicorn (production: no --reload; scale with workers).
# Note: the in-memory SESSIONS dict and on-disk FAISS index are per-process,
# so multiple workers require shared/sticky session storage before scaling up.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

# For local development with autoreload, override the CMD:
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]