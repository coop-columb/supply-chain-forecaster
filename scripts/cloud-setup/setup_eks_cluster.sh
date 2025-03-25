#!/bin/bash
# Script to set up an AWS EKS cluster and configure it for the Supply Chain Forecaster CD pipeline
# This enables real Kubernetes deployments with GitHub Actions integration

set -e

# Configuration variables
CLUSTER_NAME="supply-chain-forecaster"
REGION="us-east-1"
NODE_GROUP_NAME="${CLUSTER_NAME}-nodes"
NODE_TYPE="t3.medium"
NODE_COUNT=2

echo "=== Supply Chain Forecaster: AWS EKS Setup ==="
echo ""
echo "This script will set up an EKS cluster and configure it for use with the CD pipeline."
echo "It will create the following resources in AWS:"
echo "- EKS cluster: ${CLUSTER_NAME}"
echo "- Region: ${REGION}"
echo "- Node group: ${NODE_GROUP_NAME}"
echo "- Instance type: ${NODE_TYPE}"
echo "- Node count: ${NODE_COUNT}"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 1
fi

# Check for AWS CLI
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed or not in PATH."
    echo "Please install it from: https://aws.amazon.com/cli/"
    exit 1
fi

# Check for eksctl
if ! command -v eksctl &> /dev/null; then
    echo "eksctl not found. Would you like to install it? (y/n)"
    read -r install_eksctl
    if [[ "$install_eksctl" =~ ^[Yy]$ ]]; then
        echo "Installing eksctl..."
        # This installation method works for macOS with Homebrew
        if command -v brew &> /dev/null; then
            brew install weaveworks/tap/eksctl
        else
            echo "Please install eksctl manually using the instructions at: https://eksctl.io/installation/"
            exit 1
        fi
    else
        echo "Please install eksctl manually and run this script again."
        exit 1
    fi
fi

# Check for kubectl
if ! command -v kubectl &> /dev/null; then
    echo "kubectl not found. Would you like to install it? (y/n)"
    read -r install_kubectl
    if [[ "$install_kubectl" =~ ^[Yy]$ ]]; then
        echo "Installing kubectl..."
        # This installation method works for macOS with Homebrew
        if command -v brew &> /dev/null; then
            brew install kubectl
        else
            echo "Please install kubectl manually using the instructions at: https://kubernetes.io/docs/tasks/tools/install-kubectl/"
            exit 1
        fi
    else
        echo "Please install kubectl manually and run this script again."
        exit 1
    fi
fi

# Ensure user is logged in to AWS
if ! aws sts get-caller-identity &> /dev/null; then
    echo "You are not authenticated with AWS. Please run 'aws configure' first."
    exit 1
fi

# Create EKS cluster with node group
echo "Creating EKS cluster (this may take 15-20 minutes)..."
eksctl create cluster \
    --name ${CLUSTER_NAME} \
    --region ${REGION} \
    --node-type ${NODE_TYPE} \
    --nodes ${NODE_COUNT} \
    --nodes-min 1 \
    --nodes-max 3 \
    --with-oidc \
    --ssh-access \
    --ssh-public-key ~/.ssh/id_rsa.pub \
    --managed

echo "Cluster created successfully!"

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
aws eks update-kubeconfig --name ${CLUSTER_NAME} --region ${REGION}
kubectl config view --raw > .github_environment_setup/kube_config.yaml

# Base64 encode for GitHub secrets
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
echo "Your EKS cluster is now ready to use with the Supply Chain Forecaster CD pipeline."
echo ""
echo "To view your cluster:"
echo "kubectl get pods --all-namespaces"
echo ""
echo "To test the CD pipeline with real Kubernetes deployments:"
echo "gh workflow run 'Continuous Deployment' --ref main"
echo ""
echo "Cost considerations:"
echo "This cluster will incur charges on your AWS account. To delete it when not needed:"
echo "eksctl delete cluster --name ${CLUSTER_NAME} --region ${REGION}"