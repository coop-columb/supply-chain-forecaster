#!/bin/bash
# Script to set up a local Kubernetes cluster for development and CD pipeline testing
# This enables the transition from simulation to real Kubernetes deployments
#
# IMPORTANT: This script is part of the Phase 9 Kubernetes implementation plan
# See ROADMAP.md for full details on the planned implementation stages

set -e

echo "=== Supply Chain Forecaster: Local Kubernetes Setup ==="
echo ""
echo "This script will set up a local Kubernetes cluster using k3d"
echo "and configure it for use with the CD pipeline."
echo ""

# Check for k3d
if ! command -v k3d &> /dev/null; then
    echo "k3d not found. Would you like to install it? (y/n)"
    read -r install_k3d
    if [[ "$install_k3d" =~ ^[Yy]$ ]]; then
        echo "Installing k3d..."
        curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
    else
        echo "Please install k3d manually and run this script again."
        exit 1
    fi
fi

# Check for kubectl
if ! command -v kubectl &> /dev/null; then
    echo "kubectl not found. Would you like to install it? (y/n)"
    read -r install_kubectl
    if [[ "$install_kubectl" =~ ^[Yy]$ ]]; then
        echo "Installing kubectl..."
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/$(uname -s | tr '[:upper:]' '[:lower:]')/$(uname -m)/kubectl"
        chmod +x kubectl
        sudo mv kubectl /usr/local/bin/
    else
        echo "Please install kubectl manually and run this script again."
        exit 1
    fi
fi

# Create a k3d cluster
echo "Creating a k3d cluster for supply-chain-forecaster..."
k3d cluster create supply-chain-cluster \
  --port "8081:80@loadbalancer" \
  --port "8443:443@loadbalancer" \
  --agents 1

echo "Cluster created successfully!"

# Create namespaces
echo "Creating namespaces..."
kubectl create namespace staging
kubectl create namespace production

# Create sample secrets
echo "Creating sample API keys as secrets..."
kubectl create secret generic api-key --from-literal=api-key=staging-api-key-$(date +%s) -n staging
kubectl create secret generic api-key --from-literal=api-key=production-api-key-$(date +%s) -n production

# Get kubeconfig for GitHub Actions
echo "Generating kubeconfig files for GitHub Actions..."
mkdir -p .github_environment_setup

# Get the kubeconfig
k3d kubeconfig get supply-chain-cluster > .github_environment_setup/kube_config.yaml

# Base64 encode for GitHub secrets
cat .github_environment_setup/kube_config.yaml | base64 > .github_environment_setup/kube_config_base64.txt

echo "Kubeconfig generated at .github_environment_setup/kube_config_base64.txt"
echo ""
echo "To set up GitHub Environments with this kubeconfig:"
echo "1. Go to your GitHub repository settings"
echo "2. Navigate to Environments"
echo "3. Select or create the 'staging' environment"
echo "4. Add a secret named KUBE_CONFIG_STAGING with the content of .github_environment_setup/kube_config_base64.txt"
echo "5. Repeat for the 'production' environment with KUBE_CONFIG_PRODUCTION"
echo ""
echo "The CD pipeline will now use real Kubernetes deployments instead of simulation mode!"
echo ""
echo "To view your cluster:"
echo "kubectl get pods --all-namespaces"
echo ""
echo "To delete the cluster when you're done:"
echo "k3d cluster delete supply-chain-cluster"