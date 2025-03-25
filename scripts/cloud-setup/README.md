# Cloud Kubernetes Setup Scripts

This directory contains scripts to set up Kubernetes clusters on major cloud providers for use with the Supply Chain Forecaster CD pipeline.

## Available Scripts

- `setup_gke_cluster.sh` - Sets up a Google Kubernetes Engine (GKE) cluster
- `setup_aks_cluster.sh` - Sets up an Azure Kubernetes Service (AKS) cluster
- `setup_eks_cluster.sh` - Sets up an Amazon Elastic Kubernetes Service (EKS) cluster

## Usage

Each script is designed to be self-contained and interactive. They will:

1. Check for required CLI tools and credentials
2. Create a Kubernetes cluster with appropriate settings
3. Configure namespaces and other required resources
4. Generate kubeconfig files for GitHub environment secrets
5. Provide instructions for setting up GitHub secrets

### Google Kubernetes Engine (GKE)

```bash
# Make the script executable
chmod +x scripts/cloud-setup/setup_gke_cluster.sh

# Run the script
./scripts/cloud-setup/setup_gke_cluster.sh
```

Requirements:
- Google Cloud SDK (`gcloud`)
- `kubectl`
- Active Google Cloud account with appropriate permissions

### Azure Kubernetes Service (AKS)

```bash
# Make the script executable
chmod +x scripts/cloud-setup/setup_aks_cluster.sh

# Run the script
./scripts/cloud-setup/setup_aks_cluster.sh
```

Requirements:
- Azure CLI (`az`)
- `kubectl`
- Active Azure account with appropriate permissions

### Amazon Elastic Kubernetes Service (EKS)

```bash
# Make the script executable
chmod +x scripts/cloud-setup/setup_eks_cluster.sh

# Run the script
./scripts/cloud-setup/setup_eks_cluster.sh
```

Requirements:
- AWS CLI (`aws`)
- `eksctl`
- `kubectl`
- Active AWS account with appropriate permissions

## Setting Up GitHub Environment Secrets

After creating a cloud-based Kubernetes cluster, follow these steps to configure GitHub environment secrets:

1. Go to your GitHub repository settings
2. Navigate to "Environments"
3. Select or create the "staging" environment
4. Add the following secrets:
   - `KUBE_CONFIG_STAGING`: Copy the contents of the generated `kube_config_base64.txt` file
   - `API_KEY_STAGING`: Create an API key for the staging environment
5. Repeat for the "production" environment with different values

## Cleanup

To clean up resources when no longer needed, each script includes cleanup instructions or can be run with the `--cleanup` flag:

```bash
# Example for GKE
./scripts/cloud-setup/setup_gke_cluster.sh --cleanup
```

## Troubleshooting

If you encounter issues with the scripts or Kubernetes configuration:

1. Check the script output for error messages
2. Ensure you have the correct CLI tools installed and updated
3. Verify your cloud provider account has the necessary permissions
4. Run the validation script to check your configuration:
   ```bash
   ./scripts/validate_cd_workflow.sh
   ```
5. See the main documentation for more troubleshooting tips

## Security Considerations

- The generated Kubernetes configurations contain sensitive information, handle them securely
- The scripts create minimal IAM roles and permissions for secure operation
- Configure network policies and firewall rules according to your security requirements
- Consider enabling additional security features like pod security policies

## Next Steps

After setting up your cloud-based Kubernetes cluster and configuring GitHub environment secrets:

1. Reset the simulation mode:
   ```bash
   ./scripts/ci_cd_utils.sh reset-simulation
   ```
2. Run the CD workflow with real deployment mode:
   ```bash
   gh workflow run "Continuous Deployment" -f environment=staging -f real_deployment=true
   ```
3. Verify the deployment in your cloud-based cluster

For more details, see the [Kubernetes Deployment Guide](../../docs/deployment/kubernetes_setup.md) and [CD Pipeline Documentation](../../docs/deployment/cd_pipeline.md).