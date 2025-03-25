#!/bin/bash
# Test script for generating CD secrets (will be removed after testing)

# Stop on error
set -e

echo "Generating test secrets for GitHub Environments (CD Pipeline)..."

# Create an output directory if it doesn't exist
mkdir -p .github_environment_setup

# Generate Kubernetes configs for staging
echo "Generating Kubernetes config for staging environment..."
kubectl config view --raw > .github_environment_setup/kube_config_staging.yaml
cat .github_environment_setup/kube_config_staging.yaml | base64 > .github_environment_setup/kube_config_staging_base64.txt

# For test purposes, use the same config for production
echo "Generating Kubernetes config for production environment..."
cp .github_environment_setup/kube_config_staging.yaml .github_environment_setup/kube_config_production.yaml
cp .github_environment_setup/kube_config_staging_base64.txt .github_environment_setup/kube_config_production_base64.txt

# Generate API keys
echo "Generating random API keys..."
openssl rand -hex 32 > .github_environment_setup/api_key_staging.txt
openssl rand -hex 32 > .github_environment_setup/api_key_production.txt

echo ""
echo "Secret generation complete!"
echo ""
echo "Secret files are stored in the .github_environment_setup directory:"
echo "- kube_config_staging_base64.txt: Base64-encoded Kubernetes config for staging"
echo "- kube_config_production_base64.txt: Base64-encoded Kubernetes config for production"
echo "- api_key_staging.txt: API key for staging"
echo "- api_key_production.txt: API key for production"