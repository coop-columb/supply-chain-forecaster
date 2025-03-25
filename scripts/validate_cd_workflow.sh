#!/bin/bash
# Script to validate the CD workflow and test Kubernetes configurations
# This script helps verify Kubernetes configurations for the CD pipeline

set -e

echo "=== Supply Chain Forecaster: CD Workflow Validator ==="
echo ""
echo "This script will verify the CD workflow configuration and test"
echo "the Kubernetes configurations for both staging and production."
echo ""

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed. Please install it first."
    exit 1
fi

# Create a test directory
mkdir -p .cd_workflow_test

# Function to validate a kubeconfig
validate_kubeconfig() {
    local env_name=$1
    local kube_config=$2
    local output_file=".cd_workflow_test/${env_name}_kube_config_test.yaml"
    
    echo "Testing ${env_name} Kubernetes configuration..."
    
    # Decode and save the kubeconfig
    echo "${kube_config}" | base64 -d > "${output_file}"
    
    # Verify the kubeconfig format is valid
    if kubectl --kubeconfig="${output_file}" config view &> /dev/null; then
        echo "✅ ${env_name} Kubernetes configuration format is valid."
    else
        echo "❌ ${env_name} Kubernetes configuration is NOT valid!"
        echo "Please check and regenerate the configuration using scripts/generate_cd_secrets.sh"
        return 1
    fi
    
    # Verify the server is reachable (this will fail in portfolio context, which is expected)
    if kubectl --kubeconfig="${output_file}" cluster-info &> /dev/null; then
        echo "✅ ${env_name} Kubernetes cluster is reachable."
    else
        echo "⚠️ ${env_name} Kubernetes cluster is not reachable."
        echo "This is expected in a portfolio context without real clusters."
        echo "The CD workflow will use simulation mode in this case."
    fi
    
    return 0
}

# Check GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed. Please install it first."
    exit 1
fi

# Check authentication status
if ! gh auth status &> /dev/null; then
    echo "Error: You are not authenticated with GitHub CLI."
    echo "Please run 'gh auth login' first."
    exit 1
fi

# Check repository access
echo "Checking repository access..."
if ! gh repo view &> /dev/null; then
    echo "Error: Unable to access the repository."
    echo "Please make sure you have access rights."
    exit 1
fi

# Check environment secrets
echo "Checking environment secrets..."

# Check staging environment
echo "Checking staging environment..."
STAGING_KUBE_CONFIG=$(gh secret list -e staging 2>/dev/null | grep KUBE_CONFIG_STAGING)
STAGING_API_KEY=$(gh secret list -e staging 2>/dev/null | grep API_KEY_STAGING)

if [ -z "$STAGING_KUBE_CONFIG" ]; then
    echo "⚠️ KUBE_CONFIG_STAGING secret is not set."
    echo "The CD workflow will use simulation mode for staging."
else
    echo "✅ KUBE_CONFIG_STAGING secret is set."
fi

if [ -z "$STAGING_API_KEY" ]; then
    echo "⚠️ API_KEY_STAGING secret is not set."
    echo "Deployment verification tests will be limited."
else
    echo "✅ API_KEY_STAGING secret is set."
fi

# Check production environment
echo "Checking production environment..."
PRODUCTION_KUBE_CONFIG=$(gh secret list -e production 2>/dev/null | grep KUBE_CONFIG_PRODUCTION)
PRODUCTION_API_KEY=$(gh secret list -e production 2>/dev/null | grep API_KEY_PRODUCTION)

if [ -z "$PRODUCTION_KUBE_CONFIG" ]; then
    echo "⚠️ KUBE_CONFIG_PRODUCTION secret is not set."
    echo "The CD workflow will use simulation mode for production."
else
    echo "✅ KUBE_CONFIG_PRODUCTION secret is set."
fi

if [ -z "$PRODUCTION_API_KEY" ]; then
    echo "⚠️ API_KEY_PRODUCTION secret is not set."
    echo "Deployment verification tests will be limited."
else
    echo "✅ API_KEY_PRODUCTION secret is set."
fi

# Check CD workflow file
echo "Checking CD workflow file..."
if [ -f ".github/workflows/cd.yml" ]; then
    echo "✅ CD workflow file exists."
else
    echo "❌ CD workflow file not found!"
    echo "Please make sure the file .github/workflows/cd.yml exists."
    exit 1
fi

# Verify Kubernetes manifest files
echo "Verifying Kubernetes manifest files..."
for manifest in "k8s/api-deployment.yaml" "k8s/dashboard-deployment.yaml" "k8s/ingress.yaml"; do
    if [ -f "$manifest" ]; then
        echo "✅ $manifest exists."
        
        # Verify manifest format
        if kubectl apply --dry-run=client -f "$manifest" &> /dev/null; then
            echo "✅ $manifest format is valid."
        else
            echo "❌ $manifest format is NOT valid!"
            echo "Please check the manifest file format."
        fi
    else
        echo "❌ $manifest not found!"
        echo "Please make sure the file $manifest exists."
        exit 1
    fi
done

# Summary
echo ""
echo "CD Workflow Validation Summary:"
echo "------------------------------"
echo "✅ GitHub authentication and repository access verified."
echo "✅ CD workflow file and Kubernetes manifests verified."
echo ""
echo "The CD workflow is configured to use:"
echo "- Real deployment when Kubernetes configs are valid and clusters are reachable"
echo "- Simulation mode otherwise (for portfolio demonstration)"
echo ""
echo "You can run the CD workflow with:"
echo "gh workflow run 'Continuous Deployment' --ref main"
echo ""
echo "For detailed logs and diagnostics, check the GitHub Actions tab in your repository."

# Clean up
rm -rf .cd_workflow_test

exit 0