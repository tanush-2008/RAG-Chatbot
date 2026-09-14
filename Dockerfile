# Domain-Specific RAG Chatbot - Streamlit app container.
#
# Build:  docker build -t rag-chatbot .
# Run:    docker run -p 8501:8501 --env-file .env -v rag_data:/app/vector_store/saved_index rag-chatbot
#
# Running in a standard Linux container also sidesteps a real issue this
# project hit on a locked-down Windows dev machine: an Application Control
# (WDAC) policy there blocked torch's native DLLs, forcing a lower-accuracy
# offline embedding fallback (see embeddings.py). Torch has no such
# restriction here, so the real all-MiniLM-L6-v2 model loads normally.
#
# Runs as a non-root user (standard container hardening - a compromised
# process can't write outside its own files or touch the host as root).

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code-only rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p vector_store/saved_index feedback && \
    useradd --create-home --shell /bin/bash --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER 1000:1000

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD \
    ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
