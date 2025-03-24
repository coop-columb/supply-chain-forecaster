# Docker Production Configuration

This document describes the production Docker configuration for the Supply Chain Forecaster.

## Overview

The Supply Chain Forecaster uses a multi-stage Docker build to optimize container images for production deployment. The configuration provides:

- Separation of development and production environments
- Optimized image sizes through multi-stage builds
- Service-specific images for API and Dashboard components
- Health monitoring through container health checks
- Resource limiting capabilities
- Secure execution with non-root users
- Production-grade web servers with Gunicorn

## Production Images

Two separate production images are created:

1. **API Service**: Contains the FastAPI application with forecasting and anomaly detection services
2. **Dashboard Service**: Contains the Dash dashboard application for data visualization and interaction

## Building Production Images

Use the provided script to build the production images:

```bash
# Default version is "latest"
./scripts/build_push_images.sh

# Specify a version
./scripts/build_push_images.sh 1.0.0

# Build and push to a registry
./scripts/build_push_images.sh 1.0.0 your-registry.example.com
```

Or build manually with Docker:

```bash
# Build API image
docker build --target api-production -t supply-chain-forecaster-api:latest .

# Build Dashboard image
docker build --target dashboard-production -t supply-chain-forecaster-dashboard:latest .
```

## Local Production Testing

The docker-compose.yml file includes services for local production testing:

```bash
# Copy production environment template
cp .env.prod.template .env.prod

# Edit .env.prod with your configuration
nano .env.prod

# Start production services
docker-compose up -d api-prod dashboard-prod
```

## Production Environment Variables

The following environment variables can be configured for production:

| Variable | Description | Default |
|----------|-------------|---------|
| ENV | Environment name | production |
| LOG_LEVEL | Logging level | INFO |
| API_HOST | API service host | 0.0.0.0 |
| API_PORT | API service port | 8000 |
| API_WORKERS | Number of Gunicorn workers for API | 4 |
| DASHBOARD_HOST | Dashboard service host | 0.0.0.0 |
| DASHBOARD_PORT | Dashboard service port | 8050 |
| DASHBOARD_WORKERS | Number of Gunicorn workers for Dashboard | 2 |
| API_URL | URL for Dashboard to connect to API | http://api-prod:8000 |
| MODEL_STORAGE_PATH | Path to store trained models | /app/data/models |

## Container Health Checks

Both containers include health checks to monitor application status:

- **API Service**: Checks the `/health` endpoint every 30 seconds
- **Dashboard Service**: Checks the root path (`/`) every 30 seconds

## Security Considerations

The production Docker configuration includes several security enhancements:

1. Non-root user execution for both services
2. Minimal base images with only required dependencies
3. Properly set file permissions
4. Removed development tools and packages
5. Mount-only access to persistent volumes

## Resource Limits

The docker-compose.yml file includes resource limits for containers:

```yaml
deploy:
  resources:
    limits:
      cpus: '0.5'
      memory: 512M
    reservations:
      cpus: '0.25'
      memory: 256M
```

These can be adjusted based on your requirements.

## Kubernetes Deployment

For deploying to Kubernetes, refer to the [Kubernetes Deployment Guide](../../k8s/README.md).

## Monitoring and Logging

The production configuration includes a comprehensive monitoring and observability solution:

- **Structured Logging**: JSON formatted logs with distributed request IDs
- **Metrics Collection**: Prometheus metrics exposed via dedicated endpoints
- **Visualization**: Grafana dashboards for API, model, and system performance
- **Health Checks**: Endpoints for Kubernetes readiness and liveness probes
- **Alert Rules**: Prometheus alerting for critical conditions
- **Resource Monitoring**: CPU and memory usage tracking

The monitoring stack (Prometheus and Grafana) is included in the docker-compose.yml file:

```bash
# Start the production environment with monitoring
docker-compose up -d api-prod dashboard-prod prometheus grafana
```

For detailed information about the monitoring implementation, refer to the [Monitoring and Observability Guide](monitoring.md).

## Troubleshooting

If you encounter issues with the production Docker configuration:

1. Check container logs: `docker-compose logs api-prod`
2. Verify health checks: `docker inspect --format "{{.State.Health.Status}}" container_id`
3. Ensure persistent volumes have correct permissions
4. Verify environment variables are properly set