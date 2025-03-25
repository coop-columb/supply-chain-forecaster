# Setting Up GitHub Environments for CD Pipeline

This guide explains how to set up the GitHub environments required for the Continuous Deployment (CD) pipeline.

## Required Environments

The CD pipeline uses two environments:

1. **staging** - For deploying to the staging environment
2. **production** - For deploying to the production environment

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

3. Add the following environment secrets:
   - `KUBE_CONFIG_STAGING`: The base64-encoded Kubernetes config for your staging cluster
   - `API_KEY_STAGING`: API key for authentication in the staging environment

### Production Environment

1. Name: `production`
2. Configure environment protection rules:
   - Enable "Required reviewers" and add at least one reviewer
   - Optional: Set a wait timer (e.g., 10 minutes)

3. Add the following environment secrets:
   - `KUBE_CONFIG_PRODUCTION`: The base64-encoded Kubernetes config for your production cluster
   - `API_KEY_PRODUCTION`: API key for authentication in the production environment

## Step 2: Configure Repository Secrets

Some secrets are needed at the repository level:

1. Navigate to "Settings" > "Secrets and variables" > "Actions"
2. Click on "New repository secret"
3. Add the following secrets:
   - `REGISTRY_USERNAME`: Username for container registry (if using external registry)
   - `REGISTRY_PASSWORD`: Password for container registry (if using external registry)

## Step 3: Configure Branch Protection Rules

1. Navigate to "Settings" > "Branches"
2. Click on "Add rule"
3. Configure protection for the `main` branch:
   - Require pull request reviews before merging
   - Require status checks to pass before merging
   - Select the CI workflow as a required status check

## Testing the CD Pipeline

### Manual Trigger

1. Navigate to the "Actions" tab in your repository
2. Select the "Continuous Deployment" workflow
3. Click on "Run workflow"
4. Select the environment (staging or production)
5. Click "Run workflow"

### Automatic Triggers

- Pushing to the `main` branch will automatically deploy to staging
- To deploy to production automatically, include `[deploy-prod]` in your commit message when pushing to main