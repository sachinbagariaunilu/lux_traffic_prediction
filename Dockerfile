FROM python:3.9-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/
# Recorded 2025 counts served by /actuals/{poste_id}. ~30 MB, and the image is
# the only place the API looks for them -- data/ is excluded by .dockerignore.
COPY actuals/ ./actuals/

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
