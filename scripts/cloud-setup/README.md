# Cloud Kubernetes Setup Scripts

This directory contains scripts to set up cloud-based Kubernetes clusters for the Supply Chain Forecaster project. These scripts enable real Kubernetes deployments with the CD pipeline, allowing GitHub Actions to deploy directly to cloud infrastructure.

## Available Scripts

### Google Kubernetes Engine (GKE)

```bash
./scripts/cloud-setup/setup_gke_cluster.sh
```

This script sets up a Google Kubernetes Engine cluster with all necessary configurations. It requires the Google Cloud SDK (`gcloud`) to be installed and authenticated.

### Azure Kubernetes Service (AKS)

```bash
./scripts/cloud-setup/setup_aks_cluster.sh
```

This script sets up an Azure Kubernetes Service cluster with all necessary configurations. It requires the Azure CLI (`az`) to be installed and authenticated.

### Amazon EKS

```bash
./scripts/cloud-setup/setup_eks_cluster.sh
```

This script sets up an Amazon EKS cluster with all necessary configurations. It requires the AWS CLI (`aws`) and `eksctl` to be installed and authenticated.

## Workflow

All scripts follow a similar workflow:

1. Create a cloud-based Kubernetes cluster
2. Configure the cluster with appropriate namespaces and secrets
3. Generate a kubeconfig file compatible with GitHub Actions
4. Provide instructions for setting up GitHub environment secrets
5. Optionally configure the secrets directly if GitHub CLI is available

## After Setup

After running one of these scripts:

1. The kubeconfig will be saved to `.github_environment_setup/kube_config_base64.txt`
2. This config can be used to set up the `KUBE_CONFIG_STAGING` and `KUBE_CONFIG_PRODUCTION` secrets in GitHub
3. Reset the simulation mode using `./scripts/ci_cd_utils.sh reset-simulation`
4. Trigger the CD workflow using `gh workflow run 'Continuous Deployment' --ref main`

## Cost Considerations

The clusters created by these scripts will incur charges on your cloud provider account. Each script includes instructions for deleting the resources when they are no longer needed.