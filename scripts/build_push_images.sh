#!/bin/bash
# Script to build and push Docker images for production deployment

set -e  # Exit on any error

# Set variables from arguments or use defaults
VERSION=${1:-latest}
DOCKER_REGISTRY=${2:-""}  # Default to no registry for local builds

# Add trailing slash to registry if provided
if [ -n "$DOCKER_REGISTRY" ]; then
  DOCKER_REGISTRY="${DOCKER_REGISTRY}/"
fi

# Display information
echo "Building images with version: $VERSION"
if [ -n "$DOCKER_REGISTRY" ]; then
  echo "Will push to registry: $DOCKER_REGISTRY"
else
  echo "No registry specified, will build locally only"
fi

# Build API image
echo "Building API image..."
docker build \
  --target api-production \
  -t "${DOCKER_REGISTRY}supply-chain-forecaster-api:${VERSION}" \
  .

# Build Dashboard image
echo "Building Dashboard image..."
docker build \
  --target dashboard-production \
  -t "${DOCKER_REGISTRY}supply-chain-forecaster-dashboard:${VERSION}" \
  .

# Push images if registry provided
if [ -n "$DOCKER_REGISTRY" ]; then
  echo "Pushing API image to registry..."
  docker push "${DOCKER_REGISTRY}supply-chain-forecaster-api:${VERSION}"
  
  echo "Pushing Dashboard image to registry..."
  docker push "${DOCKER_REGISTRY}supply-chain-forecaster-dashboard:${VERSION}"
fi

echo "Build and push process completed successfully!"