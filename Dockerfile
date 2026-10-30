# syntax=docker/dockerfile:1
#
# One container serves the React app and the JSON API from the same origin.
#   docker build -t ai-stock-analyzer .
#   docker run -p 8000:8000 --env-file .env ai-stock-analyzer
#
# If Docker Hub rate-limits anonymous pulls (429), build from a mirror:
#   --build-arg NODE_IMAGE=mirror.gcr.io/library/node:22-slim
#   --build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.11-slim
ARG NODE_IMAGE=node:22-slim
ARG PYTHON_IMAGE=python:3.11-slim

# ---- build the React app ----------------------------------------------------
FROM ${NODE_IMAGE} AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
# Typecheck and build. A failure fails the image, rather than shipping an API
# whose / quietly 404s because there is no frontend to serve.
RUN npm run build

# ---- runtime ----------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd --create-home --uid 10001 app
WORKDIR /app

# Streamlit and Plotly are used only by app.py, the legacy UI, and pull in
# ~250 MB (pyarrow, pydeck, altair) that the API never imports.
COPY requirements.txt ./
RUN grep -vE '^\s*(streamlit|plotly)\b' requirements.txt > /tmp/requirements-api.txt \
 && pip install -r /tmp/requirements-api.txt

COPY --chown=app:app api/ api/
COPY --chown=app:app services/ services/
COPY --chown=app:app agents/ agents/
COPY --chown=app:app models/ models/
COPY --chown=app:app data/ data/
COPY --chown=app:app *.py ./
COPY --chown=app:app --from=web /web/dist web/dist
# ChromaDB persists its vector stores under /app at runtime.
RUN chown app:app /app

USER app
# Bake the embedding model into the image. Otherwise every cold start downloads
# ~80 MB from S3 before the first analysis, and fails outright where the
# filesystem is read-only. Runs as `app` so it lands in the runtime user's home.
RUN python -c "from alpha_vantage_vector_db import shared_embedding_model; shared_embedding_model()(['warm up'])"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '8000'), timeout=4)" || exit 1

# PORT is set by most hosts (Render, Railway, Fly). --proxy-headers lets the app
# see https behind the host's TLS terminator, so the session cookie is marked
# Secure; trusting any forwarder is safe only because the container is reachable
# solely through that proxy.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
