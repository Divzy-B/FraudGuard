FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRAUDGUARD_MODEL_PATH=artifacts/fraudguard.joblib

WORKDIR /app
COPY pyproject.toml README.md ./
COPY fraudguard ./fraudguard
RUN pip install --no-cache-dir . && \
    python -m fraudguard.train --rows 12000 --output "$FRAUDGUARD_MODEL_PATH"

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "fraudguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
