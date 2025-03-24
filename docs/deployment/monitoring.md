# Monitoring and Observability

This document describes the monitoring and observability setup for the Supply Chain Forecaster application.

## Overview

The monitoring infrastructure for Supply Chain Forecaster consists of:

1. **Structured Logging**: JSON-formatted logs with request IDs for distributed tracing
2. **Prometheus Metrics**: Real-time metrics collection for API and model performance
3. **Grafana Dashboards**: Visualization of application metrics and system resources
4. **Health Check Endpoints**: Ready-to-use endpoints for Kubernetes health probes

## Architecture

```
┌───────────────┐     ┌──────────────┐     ┌───────────────┐
│ API Service   │────▶│  Prometheus  │────▶│    Grafana    │
└───────────────┘     └──────────────┘     └───────────────┘
        ▲                     ▲                    ▲
        │                     │                    │
┌───────────────┐             │                    │
│   Dashboard   │─────────────┘                    │
└───────────────┘                                  │
        ▲                                          │
        │                                          │
┌───────────────┐                                  │
│     Logs      │──────────────────────────────────┘
└───────────────┘
```

## Metrics Collection

### API Metrics

The following API metrics are collected:

- **HTTP Request Counts**: Total number of requests by endpoint, method, and status code
- **Request Duration**: Histogram of request durations in seconds
- **Active Requests**: Real-time gauge of currently active requests
- **Error Counts**: Total errors by error type

### Model Metrics

The following model metrics are collected:

- **Prediction Counts**: Total number of model predictions by model name and status
- **Prediction Duration**: Histogram of prediction durations in seconds
- **Model Latency**: Summary of model prediction latency

### System Metrics

The following system metrics are collected:

- **Memory Usage**: Current memory usage in bytes
- **CPU Usage**: Current CPU usage percentage

## Logs

Logs are structured as JSON for better parsing by log management systems:

```json
{
  "timestamp": "2025-03-24T12:34:56.789Z",
  "level": "INFO",
  "message": "Request GET /health completed with status 200 in 1.23ms",
  "name": "api.request",
  "function": "log_request",
  "line": 286,
  "file": "/app/utils/logging.py",
  "process": 1234,
  "thread": 5678,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "GET",
  "url": "/health",
  "status_code": 200,
  "duration_ms": 1.23
}
```

All logs include:
- Timestamp with millisecond precision
- Log level
- Source code location (file, function, line)
- Process and thread IDs
- Request ID for distributed tracing

## Dashboards

### API Performance Dashboard

The API Performance Dashboard shows:
- Request rate by endpoint
- Response time (p95) by endpoint
- Error rate by endpoint
- Number of active requests

### Model Performance Dashboard

The Model Performance Dashboard shows:
- Prediction rate by model
- Prediction latency (p95) by model
- Success vs. error rate by model
- Average model latency

### System Resources Dashboard

The System Resources Dashboard shows:
- CPU usage over time
- Memory usage over time
- Error rate by error type
- Current active requests

## Health Checks

Three types of health check endpoints are available:

1. **Basic Health Check** (`/health`): Quick check to verify the application is running
2. **Readiness Check** (`/health/readiness`): Check if the application is ready to receive traffic
3. **Liveness Check** (`/health/liveness`): Check if the application is alive and functioning

These endpoints are designed to work with Kubernetes health probes.

## Alerting

Alerting rules are defined in Prometheus for critical conditions:

- **API Down**: Alert when the API service is down for more than 30 seconds
- **Dashboard Down**: Alert when the Dashboard service is down for more than 30 seconds
- **High Request Latency**: Alert when average latency exceeds 500ms for more than 1 minute
- **High Error Rate**: Alert when error rate exceeds 1% for more than 2 minutes
- **High Resource Usage**: Alert when CPU or memory usage is consistently high

## Local Setup and Testing

To start the monitoring stack locally:

```bash
# Start the complete stack with monitoring
docker-compose up -d api-prod dashboard-prod prometheus grafana

# Access Grafana dashboard (default credentials: admin/admin)
open http://localhost:3000

# Access Prometheus console
open http://localhost:9090
```

## Monitoring in Kubernetes

In a Kubernetes environment, monitoring is set up as follows:

1. API and Dashboard deployments expose metrics on their metrics endpoints
2. Prometheus scrapes these endpoints based on the configuration
3. Grafana accesses Prometheus as a datasource
4. Dashboards are pre-loaded via the provisioning configuration

## Troubleshooting

If monitoring is not working correctly:

1. Check that the services are running: `docker-compose ps`
2. Verify metrics endpoints are accessible: `curl http://localhost:8000/metrics`
3. Check Prometheus targets in the UI: http://localhost:9090/targets
4. Look for errors in the Prometheus logs: `docker-compose logs prometheus`
5. Ensure Grafana can connect to Prometheus: Check datasource status in Grafana

## Extending Monitoring

To add new metrics:

1. Define new metrics in `utils/monitoring.py`
2. Use the metrics in your code (e.g., with decorators or direct calls)
3. Update Prometheus scrape configuration if necessary
4. Create or update Grafana dashboards to visualize the new metrics