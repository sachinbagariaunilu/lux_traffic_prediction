FROM python:3.9-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/

# The prediction code, vendored from the training repo by
# scripts/deploy_to_backend.sh. REQUIRED, and not only for the import in
# app/main.py: the bundle pickles a forecast.features.Profiles dataclass, so
# joblib.load() fails with ModuleNotFoundError without this -- which reads as a
# corrupt model file rather than a missing package.
COPY forecast/ ./forecast/

COPY models/ ./models/

# The holiday calendar and site metadata, read at CALL TIME rather than baked
# into the .pkl. That is what lets the calendar be extended to 2030 without
# retraining -- and it is why the .pkl alone is not a deployable artifact.
# Without this every date fails the coverage guard.
COPY data/external/ ./data/external/

# Which series each model serves, and which of them have 2025 actuals to be
# scored against. Derived from the bundles, so regenerated on deploy rather than
# written by hand. See COUNTER_MANIFEST.md for the counts and the exclusions.
# One manifest per model. Both are required: app/main.py loads them at import
# and cannot start with either missing.
COPY counter_manifest_2024.json counter_manifest_2024_2025.json ./

# Recorded 2025 counts served by /actuals/{poste_id}. ~30 MB, and the image is
# the only place the API looks for them.
COPY actuals/ ./actuals/

# NOTE on versions. requirements.txt pins numpy 2.0.2 / lightgbm 4.6.0 while the
# model is trained against numpy 2.4.6 / lightgbm 4.7.0. Verified working -- the
# bundle loads and predicts identically across that gap -- but pickle
# compatibility is not guaranteed in general. If a future bundle fails to load
# here, align these pins with the training environment before debugging
# anything else.

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
