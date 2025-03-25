#!/bin/bash
# Script to generate secrets for GitHub Environments CD setup
# This script helps generate all the necessary secrets for real deployment mode

# Stop on error
set -e

echo "=== Supply Chain Forecaster: Kubernetes Configuration Generator ==="
echo ""
echo "This script will generate Kubernetes configurations and API keys required"
echo "for the real deployment mode of the CD pipeline."
echo ""
echo "Note: By default, the CD pipeline uses simulation mode, which doesn't require"
echo "these configurations. You only need to run this script if you want to use"
echo "real deployment mode."
echo ""
echo "This script will generate:"
echo "1. Base64-encoded Kubernetes configs for staging and production"
echo "2. Random API keys for both environments"
echo ""
echo "You'll need to add these secrets to your GitHub repository environments manually."
echo ""

# Confirm the user wants to proceed
read -p "Do you want to continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Operation cancelled."
    exit 0
fi

# Create an output directory if it doesn't exist
mkdir -p .github_environment_setup

# Generate Kubernetes configs for staging
echo ""
echo "Generating Kubernetes config for staging environment..."
echo "Please switch kubectl to your staging context now if needed."
read -p "Press Enter to continue or Ctrl+C to abort..."

kubectl config view --raw > .github_environment_setup/kube_config_staging.yaml
cat .github_environment_setup/kube_config_staging.yaml | base64 > .github_environment_setup/kube_config_staging_base64.txt

# Generate Kubernetes configs for production
echo ""
echo "Generating Kubernetes config for production environment..."
echo "Please switch kubectl to your production context now if needed."
read -p "Press Enter to continue or Ctrl+C to abort..."

kubectl config view --raw > .github_environment_setup/kube_config_production.yaml
cat .github_environment_setup/kube_config_production.yaml | base64 > .github_environment_setup/kube_config_production_base64.txt

# Generate API keys
echo ""
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
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Go to your GitHub repository Settings > Environments"
echo "2. Confirm you have two environments created:"
echo "   - staging"
echo "   - production"
echo ""
echo "3. Add these secrets to each environment:"
echo "   - For staging:"
echo "     - KUBE_CONFIG_STAGING: content of kube_config_staging_base64.txt"
echo "     - API_KEY_STAGING: content of api_key_staging.txt"
echo ""
echo "   - For production:"
echo "     - KUBE_CONFIG_PRODUCTION: content of kube_config_production_base64.txt"
echo "     - API_KEY_PRODUCTION: content of api_key_production.txt"
echo ""
echo "4. To use real deployment mode:"
echo "   - Go to the Actions tab in your repository"
echo "   - Select the 'Continuous Deployment' workflow"
echo "   - Click 'Run workflow'"
echo "   - Select the environment (staging or production)"
echo "   - Check the 'real_deployment' option"
echo "   - Click 'Run workflow'"
echo ""
echo "IMPORTANT: Delete the .github_environment_setup directory after adding the secrets to GitHub"
echo "           as these are sensitive credentials."