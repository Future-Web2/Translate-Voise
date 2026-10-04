FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/data/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    XDG_CACHE_HOME=/data/models/.cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 botuser \
    && mkdir -p /app /data/models \
    && chown -R botuser:botuser /app /data

WORKDIR /app
COPY requirements.lock pyproject.toml ./
RUN pip install -r requirements.lock
COPY src ./src
RUN pip install --no-deps .

# Railway Volumes can be mounted as root. Runtime creates /data/models before dropping UID.
COPY docker-entrypoint.py ./
ENTRYPOINT ["python", "/app/docker-entrypoint.py"]
