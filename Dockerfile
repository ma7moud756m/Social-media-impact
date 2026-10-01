# Builder stage: compile dependencies
FROM python:3.11-slim as builder

WORKDIR /tmp

# Install build dependencies only in builder stage
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
COPY requirements.txt .
RUN pip install --user --no-cache-dir --no-warn-script-location --upgrade pip && \
    pip install --user --no-cache-dir --no-warn-script-location -r requirements.txt

# Runtime stage: minimal production image
FROM python:3.11-slim

# Set environment variables:
# - PYTHONDONTWRITEBYTECODE: prevents python from writing .pyc files
# - PYTHONUNBUFFERED: ensures stdout/stderr are sent straight to terminal
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# Install runtime dependencies only (no build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application packages and model artifacts
COPY api/ /code/api/
COPY dashboard/ /code/dashboard/
COPY models/ /code/models/
COPY app.py /code/app.py

# Expose ports: 8000 (FastAPI API) and 8501 (Streamlit Dashboard)
EXPOSE 8000 8501

# Health check to ensure container readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: Launch FastAPI REST server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
