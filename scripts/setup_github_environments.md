# Setting Up GitHub Environments for CD Pipeline

This guide explains how to set up the GitHub environments required for the Continuous Deployment (CD) pipeline.

## Implementation Status

**Note:** The GitHub Environments have been successfully configured. The CD pipeline is fully operational with dual-mode capability:

- **Simulation Mode**: Default for CI/CD testing without requiring Kubernetes clusters
- **Real Deployment Mode**: Optional for actual Kubernetes deployments when needed

## Required Environments

The CD pipeline uses two environments:

1. **staging** - For deployment to the staging environment
2. **production** - For deployment to the production environment

## Step 1: Create Environments in GitHub

1. Navigate to your GitHub repository
2. Click on "Settings" tab
3. In the left sidebar, click on "Environments"
4. Click on "New environment" button

### Staging Environment

1. Name: `staging`
2. Configure environment protection rules:
   - No required reviewers needed for staging
   - No wait timer needed

3. For simulation mode, no secrets are required
4. For real deployment mode, add the following environment secrets:
   - `KUBE_CONFIG_STAGING`: The base64-encoded Kubernetes config for your staging cluster
   - `API_KEY_STAGING`: API key for authentication in the staging environment

### Production Environment

1. Name: `production`
2. Configure environment protection rules:
   - Enable "Required reviewers" and add at least one reviewer
   - Optional: Set a wait timer (e.g., 10 minutes)

3. For simulation mode, no secrets are required
4. For real deployment mode, add the following environment secrets:
   - `KUBE_CONFIG_PRODUCTION`: The base64-encoded Kubernetes config for your production cluster
   - `API_KEY_PRODUCTION`: API key for authentication in the production environment

## Step 2: Configure Repository Secrets

Some secrets are needed at the repository level if using external registries:

1. Navigate to "Settings" > "Secrets and variables" > "Actions"
2. Click on "New repository secret"
3. Add the following secrets if needed:
   - `REGISTRY_USERNAME`: Username for container registry (if using external registry)
   - `REGISTRY_PASSWORD`: Password for container registry (if using external registry)

## Step 3: Configure Branch Protection Rules

1. Navigate to "Settings" > "Branches"
2. Click on "Add rule"
3. Configure protection for the `main` branch:
   - Require pull request reviews before merging
   - Require status checks to pass before merging
   - Select the CI workflow as a required status check

## Using the CD Pipeline

### Simulation Mode (Default)

By default, the CD pipeline runs in simulation mode, which doesn't require any Kubernetes clusters or configuration:

1. Automatic trigger:
   - Push to the `main` branch - This triggers simulation for staging environment
   - Push to the `main` branch with `[deploy-prod]` in the commit message - This triggers simulation for both staging and production

2. Manual trigger:
   - Go to the Actions tab in your repository
   - Select the "Continuous Deployment" workflow
   - Click "Run workflow"
   - Select the environment (staging or production)
   - Ensure "real_deployment" is unchecked
   - Click "Run workflow"

### Real Deployment Mode (Optional)

To use real deployment mode, which performs actual Kubernetes deployments:

1. First, ensure you have configured the necessary secrets:
   - For staging: `KUBE_CONFIG_STAGING` and `API_KEY_STAGING`
   - For production: `KUBE_CONFIG_PRODUCTION` and `API_KEY_PRODUCTION`

2. Run the workflow manually:
   - Go to the Actions tab in your repository
   - Select the "Continuous Deployment" workflow
   - Click "Run workflow"
   - Select the environment (staging or production)
   - Check the "real_deployment" option
   - Click "Run workflow"

## Generating Kubernetes Configs for Real Deployment

If you're setting up real deployment mode, you can use the provided script to generate the necessary Kubernetes configurations:

```bash
# Make the script executable
chmod +x scripts/generate_cd_secrets.sh

# Run the script
./scripts/generate_cd_secrets.sh
```

The script will guide you through the process of generating the required configs and secrets for real Kubernetes deployments.