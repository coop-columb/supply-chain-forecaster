#!/bin/bash
# Script to set up a GKE cluster and configure it for the Supply Chain Forecaster CD pipeline
# This enables real Kubernetes deployments with GitHub Actions integration

set -e

# Configuration variables
CLUSTER_NAME="supply-chain-forecaster"
ZONE="us-central1-a"
NODE_COUNT=2
MACHINE_TYPE="e2-standard-2"
PROJECT_ID=$(gcloud config get-value project)

echo "=== Supply Chain Forecaster: Cloud Kubernetes Setup ==="
echo ""
echo "This script will set up a GKE cluster and configure it for use with the CD pipeline."
echo "It will create the following resources in Google Cloud:"
echo "- GKE cluster: ${CLUSTER_NAME}"
echo "- Zone: ${ZONE}"
echo "- Node count: ${NODE_COUNT}"
echo "- Machine type: ${MACHINE_TYPE}"
echo "- Project: ${PROJECT_ID}"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 1
fi

# Check for gcloud
if ! command -v gcloud &> /dev/null; then
    echo "Google Cloud SDK (gcloud) is not installed or not in PATH."
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check for kubectl
if ! command -v kubectl &> /dev/null; then
    echo "kubectl not found. Would you like to install it using gcloud? (y/n)"
    read -r install_kubectl
    if [[ "$install_kubectl" =~ ^[Yy]$ ]]; then
        echo "Installing kubectl..."
        gcloud components install kubectl
    else
        echo "Please install kubectl manually and run this script again."
        exit 1
    fi
fi

# Ensure user is logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "You are not logged in to Google Cloud. Please run 'gcloud auth login' first."
    exit 1
fi

# Check if project is set
if [ -z "$PROJECT_ID" ]; then
    echo "No Google Cloud project is set. Please run 'gcloud config set project YOUR_PROJECT_ID' first."
    exit 1
fi

# Enable required APIs
echo "Enabling required Google Cloud APIs..."
gcloud services enable container.googleapis.com

# Create GKE cluster
echo "Creating GKE cluster..."
gcloud container clusters create ${CLUSTER_NAME} \
  --zone ${ZONE} \
  --num-nodes ${NODE_COUNT} \
  --machine-type ${MACHINE_TYPE} \
  --release-channel regular

echo "Cluster created successfully!"

# Get credentials for the cluster
echo "Getting cluster credentials..."
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${ZONE}

# Create namespaces
echo "Creating namespaces..."
kubectl create namespace staging
kubectl create namespace production

# Create sample secrets
echo "Creating sample API keys as secrets..."
kubectl create secret generic api-key --from-literal=api-key=staging-api-key-$(date +%s) -n staging
kubectl create secret generic api-key --from-literal=api-key=production-api-key-$(date +%s) -n production

# Generate kubeconfig for GitHub Actions
echo "Generating kubeconfig files for GitHub Actions..."
mkdir -p .github_environment_setup

# Get the kubeconfig
kubectl config view --raw > .github_environment_setup/kube_config.yaml

# Base64 encode for GitHub secrets (in a way that's compatible with GitHub Actions)
cat .github_environment_setup/kube_config.yaml | base64 -w 0 > .github_environment_setup/kube_config_base64.txt

echo "Kubeconfig generated at .github_environment_setup/kube_config_base64.txt"
echo ""
echo "To set up GitHub Environments with this kubeconfig:"
echo "1. Go to your GitHub repository settings"
echo "2. Navigate to Environments"
echo "3. Select or create the 'staging' environment"
echo "4. Add a secret named KUBE_CONFIG_STAGING with the content of .github_environment_setup/kube_config_base64.txt"
echo "5. Repeat for the 'production' environment with KUBE_CONFIG_PRODUCTION"
echo ""

# Check if gh CLI is installed
if command -v gh &> /dev/null; then
    echo "GitHub CLI detected. Would you like to set up the GitHub environment secrets now? (y/n)"
    read -r setup_gh_secrets
    if [[ "$setup_gh_secrets" =~ ^[Yy]$ ]]; then
        echo "Setting up GitHub environment secrets..."
        
        # Check GitHub authentication
        if ! gh auth status &> /dev/null; then
            echo "You are not logged in to GitHub. Please run 'gh auth login' first."
            exit 1
        fi
        
        # Set the secrets
        cat .github_environment_setup/kube_config_base64.txt | gh secret set KUBE_CONFIG_STAGING -e staging
        cat .github_environment_setup/kube_config_base64.txt | gh secret set KUBE_CONFIG_PRODUCTION -e production
        
        # Reset simulation mode
        echo "Would you like to reset the simulation mode in the CD pipeline? (y/n)"
        read -r reset_simulation
        if [[ "$reset_simulation" =~ ^[Yy]$ ]]; then
            echo "Resetting simulation mode..."
            ./scripts/ci_cd_utils.sh reset-simulation
        fi
    fi
fi

echo ""
echo "===== Setup Complete ====="
echo ""
echo "Your GKE cluster is now ready to use with the Supply Chain Forecaster CD pipeline."
echo ""
echo "To view your cluster:"
echo "kubectl get pods --all-namespaces"
echo ""
echo "To test the CD pipeline with real Kubernetes deployments:"
echo "gh workflow run 'Continuous Deployment' --ref main"
echo ""
echo "Cost considerations:"
echo "This cluster will incur charges on your Google Cloud account. To delete it when not needed:"
echo "gcloud container clusters delete ${CLUSTER_NAME} --zone ${ZONE}"