# Supply Chain Forecaster Documentation

This directory contains comprehensive documentation for the supply chain forecaster system.

## Table of Contents

- [Installation Guide](installation/installation.md)
- [Usage Guide](usage/usage.md)
- [API Documentation](api/api.md)
- [Model Documentation](models/models.md)
- [Deployment Documentation](deployment/docker_production.md)
- [CI Troubleshooting](CI_TROUBLESHOOTING.md)

## Overview

The Supply Chain Forecaster is a production-grade system for forecasting supply chain metrics and detecting anomalies. It provides a RESTful API and an interactive dashboard for managing models and generating forecasts.

## Features

- Time series forecasting for inventory optimization
- Demand planning and prediction
- Anomaly detection for supply chain disruptions
- Interactive dashboards for visualization
- API for integration with existing systems
- Automated CI/CD pipeline
- Containerized deployment

## Project Structure
supply-chain-forecaster/
├── api/               # API module
├── config/            # Configuration module
├── dashboard/         # Dashboard module
├── data/              # Data handling module
├── models/            # Models module
├── utils/             # Utilities module
├── docs/              # Documentation
├── tests/             # Tests
├── .github/           # GitHub CI/CD workflows
├── docker-compose.yml # Docker Compose configuration
├── Dockerfile         # Docker configuration
├── requirements.txt   # Python dependencies
└── README.md          # Project overview

## Getting Started

Refer to the [Installation Guide](installation/installation.md) and [Usage Guide](usage/usage.md) for detailed instructions on how to set up and use the system.

## Deployment

For production deployment information, see the [Docker Production Guide](deployment/docker_production.md) and [Kubernetes Deployment Guide](../k8s/README.md).