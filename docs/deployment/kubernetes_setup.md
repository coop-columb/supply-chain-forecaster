# Kubernetes Cluster Setup for CD Pipeline

This guide explains how to set up a Kubernetes cluster for the Continuous Deployment (CD) pipeline of the Supply Chain Forecaster project.

> **IMPORTANT: Implementing Real Kubernetes Deployments**
> 
> While the CD pipeline currently supports a simulation mode for development and testing,
> implementing real Kubernetes deployments is a high-priority task in the project roadmap.
> This document outlines the steps to set up the necessary infrastructure to transition
> from simulation to real deployments.

## Overview

The CD pipeline is designed to support both simulation mode (for testing without real infrastructure) and real mode (deploying to actual Kubernetes clusters). To enable real deployments, you need to:

1. Set up a Kubernetes cluster
2. Configure the GitHub repository with the appropriate secrets
3. Run the CD pipeline in real deployment mode

## Option 1: Local Development Cluster

For local development or demonstration purposes, you can use a lightweight local Kubernetes cluster. We provide a script that automates the setup of a k3d-based Kubernetes cluster.

### Prerequisites

- Docker (installed and running)
- kubectl (optional, will be installed by the script if missing)
- k3d (optional, will be installed by the script if missing)

### Steps

1. Run the setup script:

```bash
./scripts/setup_local_k8s.sh
```

2. The script will:
   - Install k3d and kubectl if needed
   - Create a local Kubernetes cluster
   - Configure it for use with the CD pipeline
   - Generate the required kubeconfig files
   - Guide you on setting up GitHub environment secrets

3. Follow the instructions from the script to configure GitHub repository environments.

## Option 2: Cloud Provider Kubernetes Service

For a more production-like environment, you can use a managed Kubernetes service from a cloud provider.

### Option 2A: Google Kubernetes Engine (GKE)

1. Create a GKE cluster:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Navigate to Kubernetes Engine
   - Click "Create Cluster"
   - Configure your cluster (a small cluster with 1-3 nodes is sufficient for testing)
   - Create the cluster

2. Get the kubeconfig:
   ```bash
   gcloud container clusters get-credentials CLUSTER_NAME --zone ZONE --project PROJECT_ID
   ```

3. Base64 encode the kubeconfig:
   ```bash
   kubectl config view --raw | base64 > gke_kube_config_base64.txt
   ```

4. Configure GitHub environments with this kubeconfig.

### Option 2B: Amazon EKS

1. Create an EKS cluster:
   - Go to the [AWS Management Console](https://aws.amazon.com/console/)
   - Navigate to EKS
   - Click "Create cluster"
   - Configure your cluster
   - Create the cluster

2. Get the kubeconfig:
   ```bash
   aws eks update-kubeconfig --name CLUSTER_NAME --region REGION
   ```

3. Base64 encode the kubeconfig:
   ```bash
   kubectl config view --raw | base64 > eks_kube_config_base64.txt
   ```

4. Configure GitHub environments with this kubeconfig.

### Option 2C: Azure Kubernetes Service (AKS)

1. Create an AKS cluster:
   - Go to the [Azure Portal](https://portal.azure.com/)
   - Navigate to Kubernetes Services
   - Click "Create Kubernetes cluster"
   - Configure your cluster
   - Create the cluster

2. Get the kubeconfig:
   ```bash
   az aks get-credentials --resource-group RESOURCE_GROUP --name CLUSTER_NAME
   ```

3. Base64 encode the kubeconfig:
   ```bash
   kubectl config view --raw | base64 > aks_kube_config_base64.txt
   ```

4. Configure GitHub environments with this kubeconfig.

## Setting up GitHub Environment Secrets

Once you have a Kubernetes cluster and kubeconfig, you need to configure GitHub environments:

1. Go to your GitHub repository
2. Navigate to Settings > Environments
3. Create or select the `staging` environment
4. Add a secret named `KUBE_CONFIG_STAGING` with the base64-encoded kubeconfig
5. Add a secret named `API_KEY_STAGING` with a suitable API key
6. Repeat for the `production` environment (with `KUBE_CONFIG_PRODUCTION` and `API_KEY_PRODUCTION`)

## Verify Setup

To verify that your setup is working correctly:

1. Run the CD pipeline manually:
   ```bash
   gh workflow run "Continuous Deployment" --ref main
   ```

2. Check the logs to ensure it's using real deployments instead of simulation mode.

3. Verify the deployment in your Kubernetes cluster:
   ```bash
   kubectl get deployments -n staging
   kubectl get services -n staging
   ```

## Troubleshooting

If you encounter issues with the Kubernetes setup or CD pipeline:

1. Run the validation script to check your configuration:
   ```bash
   ./scripts/validate_cd_workflow.sh
   ```

2. Check the workflow logs in GitHub Actions for detailed error messages.

3. Verify that the kubeconfig is correctly formatted and base64-encoded:
   ```bash
   # Decode the base64 secret to verify it's valid YAML
   echo "YOUR_BASE64_STRING" | base64 -d
   ```

4. Test connectivity to the Kubernetes cluster:
   ```bash
   # Test using the decoded kubeconfig
   kubectl --kubeconfig=temp_kubeconfig.yaml cluster-info
   ```

## Cleanup

To clean up a local k3d cluster when you're done:

```bash
k3d cluster delete supply-chain-cluster
```

For cloud-based clusters, use the respective cloud provider's console or CLI to delete the cluster when no longer needed.