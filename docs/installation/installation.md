# Installation Guide

This guide provides step-by-step instructions for installing and setting up the Supply Chain Forecaster system.

## Prerequisites

- Python 3.9+ 
- Docker and Docker Compose (optional, for containerized deployment)
- Git
- GitHub CLI (optional, for repository setup)

## Option 1: Local Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/supply-chain-forecaster.git
cd supply-chain-forecaster
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

### 4. Configure the Application

Create a `.env` file in the project root directory with your configuration settings:

```bash
cp .env.example .env
```

Edit the `.env` file to adjust settings as needed.

### 5. Run the Application

Start the API:

```bash
python -m api.main
```

Start the dashboard (in a separate terminal):

```bash
python -m dashboard.main
```

## Option 2: Docker Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/supply-chain-forecaster.git
cd supply-chain-forecaster
```

### 2. Configure the Application

Create a `.env` file in the project root directory with your configuration settings:

```bash
cp .env.example .env
```

Edit the `.env` file to adjust settings as needed.

### 3. Build and Run with Docker Compose

```bash
docker-compose up
```

This will build and start both the API and dashboard containers.

## Option 3: Development Setup

For development purposes, you may want to install additional tools:

### 1. Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

### 2. Set Up Pre-commit Hooks

```bash
pre-commit install
```

### 3. Set Up VS Code

Open the project in VS Code:

```bash
code .
```

Install recommended extensions from `.vscode/extensions.json`.

## Troubleshooting

If you encounter issues during installation, here are some common solutions:

### Package Installation Errors

If you encounter errors with package installation, try updating pip:

```bash
pip install --upgrade pip
```

### Docker Issues

If you encounter Docker issues, make sure Docker daemon is running:

```bash
docker info
```

### Permission Issues

If you encounter permission issues on Linux or macOS:

```bash
chmod +x scripts/*.py
```

### Port Conflicts

If you see "Address already in use" errors, check if the ports are already in use:

```bash
lsof -i :8000  # Check if API port is in use
lsof -i :8050  # Check if dashboard port is in use
```

## Next Steps

After installation, proceed to the [Usage Guide](../usage/usage.md) to learn how to use the system.