# GitHub Environments Setup Checklist

This checklist provides step-by-step instructions for setting up GitHub Environments for the CD pipeline.

## Prerequisites ✓
- [ ] Access to the GitHub repository settings
- [ ] Generated secrets using the `generate_cd_secrets.sh` script or the test script
- [ ] Verified CD workflow configuration

## Step 1: Set Up Staging Environment
- [ ] Go to your GitHub repository
- [ ] Click on "Settings" tab
- [ ] In the left sidebar, click on "Environments"
- [ ] Click on "New environment" button
- [ ] Enter name: `staging`
- [ ] Configure protection rules:
  - [ ] No required reviewers
  - [ ] No wait timer
- [ ] Add environment secrets:
  - [ ] Name: `KUBE_CONFIG_STAGING`
    - [ ] Value: Copy content from `.github_environment_setup/kube_config_staging_base64.txt`
  - [ ] Name: `API_KEY_STAGING`
    - [ ] Value: Copy content from `.github_environment_setup/api_key_staging.txt`
- [ ] Click "Save protection rules"

## Step 2: Set Up Production Environment
- [ ] Go to "Environments" in Settings
- [ ] Click on "New environment" button
- [ ] Enter name: `production`
- [ ] Configure protection rules:
  - [ ] Enable "Required reviewers"
  - [ ] Add at least one reviewer (your GitHub username)
  - [ ] Optional: Set a wait timer (e.g., 10 minutes)
- [ ] Add environment secrets:
  - [ ] Name: `KUBE_CONFIG_PRODUCTION`
    - [ ] Value: Copy content from `.github_environment_setup/kube_config_production_base64.txt`
  - [ ] Name: `API_KEY_PRODUCTION`
    - [ ] Value: Copy content from `.github_environment_setup/api_key_production.txt`
- [ ] Click "Save protection rules"

## Step 3: Add Repository Secrets (if using external registry)
- [ ] Go to "Settings" > "Secrets and variables" > "Actions"
- [ ] Click on "New repository secret"
- [ ] Add registry secrets if needed:
  - [ ] Name: `REGISTRY_USERNAME`
    - [ ] Value: Your container registry username
  - [ ] Name: `REGISTRY_PASSWORD`
    - [ ] Value: Your container registry password

## Step 4: Test the CD Pipeline
- [ ] Go to the "Actions" tab in your repository
- [ ] Select the "Continuous Deployment" workflow
- [ ] Click on "Run workflow"
- [ ] Select "staging" as the environment
- [ ] Click "Run workflow"
- [ ] Monitor the workflow for successful completion
- [ ] Verify the application is deployed and functioning correctly

## Step 5: Update Documentation
- [ ] Update `ROADMAP.md` to mark CD implementation as complete
- [ ] Clean up any test files used during setup

## Step 6: Test Production Deployment (Optional)
- [ ] Go to the "Actions" tab in your repository
- [ ] Select the "Continuous Deployment" workflow
- [ ] Click on "Run workflow"
- [ ] Select "production" as the environment
- [ ] Click "Run workflow"
- [ ] Approve the deployment when prompted
- [ ] Monitor the workflow for successful completion
- [ ] Verify the application is deployed and functioning correctly in production

## Security Follow-up
- [ ] Delete the `.github_environment_setup` directory after adding the secrets to GitHub
- [ ] Consider rotating API keys periodically
- [ ] Review Kubernetes RBAC permissions to ensure they follow the principle of least privilege