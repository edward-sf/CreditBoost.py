# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src ./src
# Base package only — never [train]. scikit-learn stays out of the runtime.
RUN pip install --no-cache-dir .

# The model's bytes live in a GitHub Release, not in git. Fetch them here and
# refuse to continue unless they match the committed lockfile: an image
# containing an unverified, fixture-provenance, or ECOA-violating artifact
# cannot be built at all. urllib is used rather than curl so this stage needs
# no apt-get layer.
COPY models/model.lock.json ./models/model.lock.json
RUN creditboost-artifact fetch  --dir /build/models --lockfile /build/models/model.lock.json \
 && creditboost-artifact verify --dir /build/models --lockfile /build/models/model.lock.json

FROM python:3.12-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    CREDITBOOST_MODEL_DIR=/app/models
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /build/models /app/models
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"
CMD ["uvicorn", "creditboost.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
