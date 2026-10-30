FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    RECSYS_HOME=/app
WORKDIR /app

# Install the package and the API dependencies (no Streamlit in this image).
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"

# Train at build time: containers start instantly and never download data at runtime.
RUN python -m recsys.train

COPY api ./api
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
# Hosting platforms such as Render set $PORT; default to 8000 locally.
CMD uvicorn main:app --app-dir api --host 0.0.0.0 --port ${PORT:-8000}
