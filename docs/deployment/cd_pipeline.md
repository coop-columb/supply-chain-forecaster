# Continuous Deployment Pipeline

This document describes the Continuous Deployment (CD) pipeline for the Supply Chain Forecaster project.

## Overview

The CD pipeline automates the process of building, testing, and deploying the application to staging and production environments. It uses GitHub Actions to orchestrate the pipeline and supports both simulation and real Kubernetes deployments.

## Implementation Status

**Note:** The CD pipeline is now fully implemented:

- ✅ The workflow configuration is complete (GitHub Action workflow file)
- ✅ The build and push Docker images stage is functional
- ✅ GitHub Environments have been configured for staging and production
- ✅ The workflow supports both simulation and real deployment modes
- ✅ The workflow has been successfully tested and verified
- ✅ Default mode is simulation for safe CI/CD without requiring Kubernetes clusters
- ✅ Optional real deployment mode for actual production deployments when needed

See [Setting Up GitHub Environments](../../scripts/setup_github_environments.md) for details on the environment configuration.

## Pipeline Stages

The CD pipeline consists of the following stages:

1. **Build and Push Docker Images**
   - Builds the API and Dashboard Docker images
   - Pushes the images to GitHub Container Registry (GHCR)
   - Tags images with the commit SHA and "latest"

2. **Deploy to Staging**
   - Automatically runs for the staging environment
   - Updates Kubernetes manifests with new image tags
   - Deploys or simulates deployment based on selected mode
   - Runs or simulates verification tests
   - Creates a deployment record

3. **Deploy to Production**
   - Requires successful staging deployment
   - Manually triggered or via special commit message
   - Updates Kubernetes manifests with new image tags
   - Deploys or simulates deployment based on selected mode
   - Runs or simulates verification tests
   - Creates a deployment record

## Deployment Modes

The pipeline supports two deployment modes:

1. **Simulation Mode (Default):**
   - Default for all automated workflows
   - Simulates Kubernetes operations with echo statements
   - Does not require actual Kubernetes clusters or configuration
   - Safe for CI/CD testing and demonstration
   - Used when no mode is explicitly specified

2. **Real Deployment Mode:**
   - Available through manual workflow triggers
   - Performs actual Kubernetes deployments to real clusters
   - Requires properly configured Kubernetes secrets
   - Must be explicitly enabled with the `real_deployment` input
   - Only available in manual workflow runs

## Triggering the Pipeline

The CD pipeline can be triggered in the following ways:

1. **Automatic Staging Deployment (Simulation Mode):**
   - Push to the `main` branch
   - This automatically runs simulation for staging environment

2. **Manual Deployment (Simulation or Real Mode):**
   - Go to the Actions tab in GitHub
   - Select the "Continuous Deployment" workflow
   - Click "Run workflow"
   - Choose the environment (staging or production)
   - Optionally check "real_deployment" for actual Kubernetes deployment
   - Optionally specify a version tag (defaults to commit SHA)

3. **Automatic Production Deployment (Simulation Mode):**
   - Push to the `main` branch with `[deploy-prod]` in the commit message
   - This runs simulation for staging first, then production if successful

## Environments

The CD pipeline uses the following environments:

1. **Staging:**
   - Used for testing and verification before production
   - Automatically deployed on push to main
   - No approval required
   - Supports both simulation and real deployment modes

2. **Production:**
   - The live environment
   - Requires approval or special commit message
   - Requires successful deployment to staging
   - Supports both simulation and real deployment modes

## Kubernetes Configuration

For real deployment mode, the CD pipeline requires the following secrets:

- `KUBE_CONFIG_STAGING`: Base64-encoded Kubernetes config for staging
- `KUBE_CONFIG_PRODUCTION`: Base64-encoded Kubernetes config for production
- `API_KEY_STAGING`: API key for staging
- `API_KEY_PRODUCTION`: API key for production

The CD pipeline uses the following Kubernetes manifests:

- `k8s/api-deployment.yaml` - API deployment configuration
- `k8s/dashboard-deployment.yaml` - Dashboard deployment configuration
- `k8s/ingress.yaml` - Ingress configuration for routing

## Setting Up Real Deployments

To use the real deployment mode:

1. Configure Kubernetes clusters for staging and production
2. Generate Kubernetes configuration files using `scripts/generate_cd_secrets.sh`
3. Add the necessary secrets to GitHub Environments
4. Run the workflow manually with the "real_deployment" option checked

## Troubleshooting

If the CD pipeline fails, check the following:

1. **Build Failures:**
   - Check the Docker build logs
   - Verify that the Dockerfile is correct
   - Ensure the necessary dependencies are available

2. **Deployment Failures:**
   - Check the logs for errors in the deployment steps
   - If using real mode, verify that the Kubernetes manifests are correct
   - If using real mode, ensure the Kubernetes cluster is accessible
   - If using real mode, check that the Kubernetes configuration is valid

3. **Mode Selection Issues:**
   - Verify that the workflow is using the expected deployment mode
   - Check if real_deployment is properly set for manual workflow runs
   - Review the logs to confirm which mode is being used