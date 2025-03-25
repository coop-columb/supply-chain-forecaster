# Continuous Deployment Pipeline

This document describes the Continuous Deployment (CD) pipeline for the Supply Chain Forecaster project.

## Overview

The CD pipeline automates the process of building, testing, and deploying the application to staging and production environments. It uses GitHub Actions to orchestrate the pipeline and Kubernetes for deployments.

## Implementation Status

**Note:** The CD pipeline is now fully implemented:

- ✅ The workflow configuration is complete (GitHub Action workflow file)
- ✅ The build and push Docker images stage is functional
- ✅ Automation script for generating required secrets is available
- ✅ GitHub Environments and secrets have been configured
- ✅ The workflow has been refined to directly use Kubernetes commands
- ✅ The workflow has been successfully tested and verified

See [Setting Up GitHub Environments](../../scripts/setup_github_environments.md) for details on the environment configuration. You can use the provided automation script (`scripts/generate_cd_secrets.sh`) to generate all required secrets.

## Pipeline Stages

The CD pipeline consists of the following stages:

1. **Build and Push Docker Images**
   - Builds the API and Dashboard Docker images
   - Pushes the images to GitHub Container Registry (GHCR)
   - Tags images with the commit SHA and "latest"

2. **Deploy to Staging**
   - Automatically deploys to the staging environment
   - Updates Kubernetes manifests with new image tags
   - Applies the Kubernetes manifests
   - Runs deployment verification tests
   - Creates a deployment record

3. **Deploy to Production**
   - Requires successful deployment to staging
   - Manually triggered or via special commit message
   - Updates Kubernetes manifests with new image tags
   - Applies the Kubernetes manifests
   - Runs deployment verification tests
   - Creates a deployment record

## Triggering the Pipeline

The CD pipeline can be triggered in the following ways:

1. **Automatic Staging Deployment:**
   - Push to the `main` branch
   - This automatically deploys to the staging environment

2. **Manual Deployment:**
   - Go to the Actions tab in GitHub
   - Select the "Continuous Deployment" workflow
   - Click "Run workflow"
   - Choose the environment (staging or production)
   - Optionally specify a version tag (defaults to commit SHA)

3. **Automatic Production Deployment:**
   - Push to the `main` branch with `[deploy-prod]` in the commit message
   - This deploys to staging first, then to production if successful

## Deployment Verification

After each deployment, the pipeline runs automated tests to verify that the application is functioning correctly. These tests include:

- API health check
- Dashboard accessibility check
- Basic functionality tests
- End-to-end workflow test (skipped in production)

## Environments

The CD pipeline uses the following environments:

1. **Staging:**
   - Used for testing and verification before production
   - Automatically deployed on push to main
   - No approval required

2. **Production:**
   - The live environment
   - Requires approval or special commit message
   - Requires successful deployment to staging

## Kubernetes Configuration

The CD pipeline requires the following Kubernetes manifests:

- `k8s/api-deployment.yaml` - API deployment configuration
- `k8s/dashboard-deployment.yaml` - Dashboard deployment configuration
- `k8s/ingress.yaml` - Ingress configuration for routing

## Secrets and Environment Variables

The CD pipeline uses the following secrets:

- `KUBE_CONFIG_STAGING` - Kubernetes config for staging
- `KUBE_CONFIG_PRODUCTION` - Kubernetes config for production
- `API_KEY_STAGING` - API key for staging
- `API_KEY_PRODUCTION` - API key for production

## Troubleshooting

If the CD pipeline fails, check the following:

1. **Build Failures:**
   - Check the Docker build logs
   - Verify that the Dockerfile is correct
   - Ensure the necessary dependencies are available

2. **Deployment Failures:**
   - Check the Kubernetes logs
   - Verify that the Kubernetes manifests are correct
   - Ensure the Kubernetes cluster is accessible

3. **Verification Test Failures:**
   - Check the test logs
   - Verify that the API and Dashboard are running
   - Check the application logs for errors

## Adding New Environments

To add a new environment to the CD pipeline:

1. Create a new environment in GitHub
2. Add the necessary secrets
3. Update the CD workflow to include the new environment
4. Create Kubernetes manifests for the new environment