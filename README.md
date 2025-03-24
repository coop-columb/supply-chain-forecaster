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

## Tech Stack

- Python 3.10+
- FastAPI
- Pandas, NumPy, Scikit-learn
- Prophet, XGBoost, LSTM models
- Plotly Dash for visualization
- Docker for containerization
- GitHub Actions for CI/CD
- Kubernetes for orchestration

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

# Start the production environment
docker-compose up -d api-prod dashboard-prod
```

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

Usage examples are available in the [Usage Guide](docs/usage.md).

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.