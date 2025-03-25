#!/bin/bash
# Script to generate Kubernetes configuration secrets for the CD pipeline

set -e

echo "=== Supply Chain Forecaster: Generate CD Secrets ==="
echo ""
echo "This script helps you generate the required Kubernetes configuration"
echo "secrets for the Continuous Deployment (CD) pipeline."
echo ""

# Check for kubectl
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed. Please install it first."
    exit 1
fi

# Check for valid kubeconfig
if ! kubectl cluster-info &> /dev/null; then
    echo "Error: Unable to connect to Kubernetes cluster."
    echo "Please make sure your kubeconfig is correctly set up."
    exit 1
fi

# Create output directory
mkdir -p .github_environment_setup

# Generate kubeconfig for GitHub secrets
echo "Generating kubeconfig files for GitHub Actions..."
KUBE_CONFIG_PATH=".github_environment_setup/kube_config.yaml"
kubectl config view --raw > "$KUBE_CONFIG_PATH"

# Base64 encode the kubeconfig
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS requires the -b 0 flag to disable line wrapping
    base64 -b 0 < "$KUBE_CONFIG_PATH" > ".github_environment_setup/kube_config_base64.txt"
else
    # Linux (and others) don't need the flag
    base64 -w 0 < "$KUBE_CONFIG_PATH" > ".github_environment_setup/kube_config_base64.txt"
fi

# Create API keys for staging and production
STAGING_API_KEY="staging-api-key-$(date +%s)"
PRODUCTION_API_KEY="production-api-key-$(date +%s)"

echo "Generated staging API key: $STAGING_API_KEY"
echo "Generated production API key: $PRODUCTION_API_KEY"

# Print instructions
echo ""
echo "===== GitHub Environment Setup Instructions ====="
echo ""
echo "1. Go to your GitHub repository settings"
echo "2. Navigate to Environments"
echo "3. Select or create the 'staging' environment"
echo "4. Add the following secrets:"
echo "   - KUBE_CONFIG_STAGING: Copy the contents of .github_environment_setup/kube_config_base64.txt"
echo "   - API_KEY_STAGING: $STAGING_API_KEY"
echo "5. Select or create the 'production' environment"
echo "6. Add the following secrets:"
echo "   - KUBE_CONFIG_PRODUCTION: Copy the contents of .github_environment_setup/kube_config_base64.txt"
echo "   - API_KEY_PRODUCTION: $PRODUCTION_API_KEY"
echo ""
echo "The kubeconfig has been saved to .github_environment_setup/kube_config.yaml"
echo "The base64-encoded kubeconfig has been saved to .github_environment_setup/kube_config_base64.txt"
echo ""
echo "To verify your setup, run the validation script:"
echo "./scripts/validate_cd_workflow.sh"
echo ""
echo "After setting up the GitHub environment secrets, you can run the CD workflow:"
echo "gh workflow run 'Continuous Deployment' -f environment=staging -f real_deployment=true"
echo ""
echo "IMPORTANT: The .github_environment_setup directory contains sensitive information."
echo "          Do not commit it to your repository. Run 'scripts/cleanup_temp_files.sh'"
echo "          when you're done to remove it securely."

exit 0