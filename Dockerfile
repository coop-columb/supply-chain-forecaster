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

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership of the application code
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Stage for development
FROM base as development
RUN pip install --no-cache-dir -r requirements-dev.txt
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage for production
FROM base as production
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]