# Supply Chain Forecaster

A production-grade supply chain forecasting system with advanced ML models.

## Features

- Time series forecasting for inventory optimization
- Demand planning and prediction
- Anomaly detection for supply chain disruptions
- Interactive dashboards for visualization
- API for integration with existing systems
- Automated CI/CD pipeline
- Containerized deployment
- Comprehensive monitoring and observability

## Tech Stack

- Python 3.10+
- FastAPI
- Pandas, NumPy, Scikit-learn
- Prophet, XGBoost, LSTM models
- Plotly Dash for visualization
- Docker for containerization
- GitHub Actions for CI/CD
- Kubernetes for orchestration
- Prometheus and Grafana for monitoring

## Installation

Detailed installation instructions are provided in the [Installation Guide](docs/installation.md).

### Quick Start with Docker

For development:

```bash
# Start the development environment
docker-compose up -d api dashboard

# Run tests
docker-compose run tests
```

For production:

```bash
# Create a production environment file
cp .env.prod.template .env.prod
# Edit the .env.prod file with your configuration

# Start the production environment without monitoring
docker-compose up -d api-prod dashboard-prod

# Start the production environment with monitoring stack
docker-compose up -d api-prod dashboard-prod prometheus grafana
```

### Monitoring and Observability

The project includes a comprehensive monitoring and observability stack:

- Structured JSON logging with distributed tracing
- Prometheus metrics for API and model performance
- Grafana dashboards for visualization
- Health check endpoints for Kubernetes
- Performance profiling for application optimization

For more details, see the [Monitoring and Observability Guide](docs/deployment/monitoring.md).

### Performance Profiling and Optimization

The system includes built-in performance profiling and optimization features:

```bash
# Run the API with profiling enabled
./scripts/run_with_profiling.sh

# Run the profiling analysis script
python scripts/profile_application.py

# Analyze profiling results
python scripts/analyze_profiling.py

# Test optimization improvements
python scripts/run_model_optimization_tests.py
```

Performance features include:
- Comprehensive profiling for API endpoints, models, and dashboard components
- Model instance caching to reduce load times
- Prediction result caching to avoid redundant computations
- Optimized LSTM sequence creation algorithms
- Efficient ARIMA parameter selection

Access profiling data directly via the API endpoint:
```
http://localhost:8000/profiling/stats
```

All caching behavior is configurable via environment variables or configuration files.

## Deployment

### Docker Production Deployment

The system includes optimized Docker configurations for production:

1. Multi-stage builds for smaller image sizes
2. Separate production targets for API and Dashboard services
3. Health checks for container monitoring
4. Resource limits for better stability
5. Production-ready Gunicorn servers
6. Secure non-root user execution

```bash
# Build production images
docker build --target api-production -t supply-chain-forecaster-api:latest .
docker build --target dashboard-production -t supply-chain-forecaster-dashboard:latest .
```

### Kubernetes Deployment

For production Kubernetes deployment, see the [Kubernetes Deployment Guide](k8s/README.md).

## Usage

Comprehensive documentation is available to help you get started:

- [Quick Start Guide](docs/usage/quickstart.md) - Get up and running quickly
- [User Documentation](docs/usage/index.md) - Complete user documentation
- [Dashboard Walkthrough](docs/usage/dashboard_walkthrough.md) - Guide to using the dashboard
- [API Examples](docs/usage/api_examples.md) - Examples of using the API programmatically
- [Performance Optimization](docs/performance_optimization.md) - Guide to optimizing system performance
- [Troubleshooting](docs/usage/troubleshooting.md) - Solutions for common issues

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.