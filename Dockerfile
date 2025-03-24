# Multi-stage build for the Supply Chain Forecaster
# This Dockerfile supports both API and dashboard services

# Base stage with common dependencies
FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Create a non-root user to run the application
RUN groupadd -r app && useradd -r -g app app

# Install common Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage for either service
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY . .

# Change ownership of the application code
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Default to API service in development
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production builder stage for optimizing dependencies
FROM base as builder

# Install production dependencies only
RUN pip install --no-cache-dir --user wheel setuptools

# Install application as a package
COPY setup.py .
COPY pyproject.toml .
RUN python -m pip install --user --no-cache-dir -e .

# Production stage for API service
FROM python:3.10-slim as api-production

WORKDIR /app

# Copy only necessary dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Set production environment
ENV ENV=production \
    LOG_LEVEL=INFO \
    PYTHONPATH=/app

# Create a non-root user
RUN groupadd -r app && useradd -r -g app app

# Copy only the necessary application code
COPY ./api ./api
COPY ./models ./models
COPY ./config ./config
COPY ./utils ./utils
COPY ./data ./data

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data/models && \
    chown -R app:app /app

# Add health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER app

# Expose the API port
EXPOSE 8000

# Start the API server with gunicorn for production performance
CMD ["gunicorn", "api.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# Production stage for Dashboard service
FROM python:3.10-slim as dashboard-production

WORKDIR /app

# Copy only necessary dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Set production environment
ENV ENV=production \
    LOG_LEVEL=INFO \
    PYTHONPATH=/app

# Install dashboard-specific dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -r app && useradd -r -g app app

# Copy only the necessary application code
COPY ./dashboard ./dashboard
COPY ./config ./config
COPY ./utils ./utils

# Create directories for logs and assets
RUN mkdir -p /app/logs /app/dashboard/assets && \
    chown -R app:app /app

# Add health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

# Switch to non-root user
USER app

# Expose the dashboard port
EXPOSE 8050

# Start the Dashboard server with gunicorn for production performance
CMD ["gunicorn", "dashboard.main:server", "--workers", "2", "--bind", "0.0.0.0:8050"]