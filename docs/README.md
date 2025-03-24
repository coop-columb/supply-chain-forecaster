# Supply Chain Forecaster Documentation

This directory contains comprehensive documentation for the supply chain forecaster system.

## Table of Contents

- [User Documentation](usage/index.md)
  - [Quick Start Guide](usage/quickstart.md)
  - [Dashboard Walkthrough](usage/dashboard_walkthrough.md)
  - [Common Workflows](usage/common_workflows.md)
  - [API Examples](usage/api_examples.md)
  - [Troubleshooting Guide](usage/troubleshooting.md)
- [Installation Guide](installation/installation.md)
- [API Documentation](api/api.md)
- [Model Documentation](models/models.md)
- [Deployment Documentation](deployment/docker_production.md)
- [Monitoring Documentation](deployment/monitoring.md)
- [CI Troubleshooting](CI_TROUBLESHOOTING.md)

## Overview

The Supply Chain Forecaster is a production-grade system for forecasting supply chain metrics and detecting anomalies. It provides a RESTful API and an interactive dashboard for managing models and generating forecasts.

## Features

- Time series forecasting for inventory optimization
- Demand planning and prediction
- Anomaly detection for supply chain disruptions
- Interactive dashboards for visualization
- API for integration with existing systems
- Comprehensive monitoring and observability
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
├── monitoring/        # Monitoring configuration
├── docs/              # Documentation
├── tests/             # Tests
├── .github/           # GitHub CI/CD workflows
├── docker-compose.yml # Docker Compose configuration
├── Dockerfile         # Docker configuration
├── requirements.txt   # Python dependencies
└── README.md          # Project overview

## Getting Started

Refer to the [Quick Start Guide](usage/quickstart.md) and [User Documentation](usage/index.md) for detailed instructions on how to use the system.

## Installation

See the [Installation Guide](installation/installation.md) for detailed setup instructions.

## Deployment

For production deployment information, see the [Docker Production Guide](deployment/docker_production.md) and [Kubernetes Deployment Guide](../k8s/README.md).

## Monitoring

For information about monitoring and observability, see the [Monitoring Guide](deployment/monitoring.md).