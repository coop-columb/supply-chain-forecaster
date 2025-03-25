#!/bin/bash
# Script to generate secrets for GitHub Environments CD setup
# This script helps generate all the necessary secrets for CD pipeline setup

# Stop on error
set -e

echo "Generating secrets for GitHub Environments (CD Pipeline)..."
echo ""
echo "This script will generate:"
echo "1. Base64-encoded Kubernetes configs for staging and production"
echo "2. Random API keys for both environments"
echo ""
echo "You'll need to add these secrets to your GitHub repository environments manually."
echo ""

# Create an output directory if it doesn't exist
mkdir -p .github_environment_setup

# Generate Kubernetes configs for staging
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
echo "2. Create two environments:"
echo "   - staging"
echo "   - production"
echo ""
echo "3. Configure protection rules for each environment:"
echo "   - For staging: No required reviewers needed"
echo "   - For production: Add required reviewers and optionally set a wait timer"
echo ""
echo "4. Add secrets to each environment:"
echo "   - For staging:"
echo "     - KUBE_CONFIG_STAGING: content of kube_config_staging_base64.txt"
echo "     - API_KEY_STAGING: content of api_key_staging.txt"
echo ""
echo "   - For production:"
echo "     - KUBE_CONFIG_PRODUCTION: content of kube_config_production_base64.txt"
echo "     - API_KEY_PRODUCTION: content of api_key_production.txt"
echo ""
echo "5. (Optional) Add repository-level secrets if using external registry:"
echo "   - REGISTRY_USERNAME: Your container registry username"
echo "   - REGISTRY_PASSWORD: Your container registry password"
echo ""
echo "After setting up the GitHub Environments and secrets, you can test the CD pipeline by:"
echo "- Going to the Actions tab in your repository"
echo "- Selecting the 'Continuous Deployment' workflow"
echo "- Clicking 'Run workflow'"
echo "- Selecting 'staging' as the environment"
echo "- Clicking 'Run workflow'"
echo ""
echo "Don't forget to ensure your Kubernetes clusters are properly configured with:"
echo "- Namespace for the application"
echo "- RBAC permissions for deployments"
echo "- Network policies if required"
echo ""
echo "IMPORTANT: Delete the .github_environment_setup directory after adding the secrets to GitHub"
echo "           as these are sensitive credentials."