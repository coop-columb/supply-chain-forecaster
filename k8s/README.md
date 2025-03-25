# Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the Supply Chain Forecaster to a Kubernetes cluster.

## Deployment Methods

There are two ways to deploy the application to Kubernetes:

1. **Automated Deployment (Recommended)**: Using the CI/CD pipeline
2. **Manual Deployment**: Using kubectl directly

## Automated Deployment with CI/CD

The project includes a comprehensive CI/CD pipeline for automated deployments. The pipeline will:

1. Build and push Docker images to GitHub Container Registry (ghcr.io)
2. Deploy to the staging environment automatically on pushes to main
3. Deploy to production after approval or with a specially tagged commit

### Prerequisites for CI/CD

- GitHub Repository with GitHub Actions enabled
- GitHub Environments set up (staging and production)
- Kubernetes clusters for staging and production
- Kubernetes configuration files encoded as secrets

For detailed setup instructions, see [Setting Up GitHub Environments](../scripts/setup_github_environments.md).

## Manual Deployment Steps

### 1. Build and Push Docker Images

```bash
# Build and push using the script
./scripts/build_push_images.sh 1.0.0 ghcr.io/your-username

# Or manually:
docker build -t ghcr.io/your-username/supply-chain-forecaster-api:1.0.0 \
  --target api-production .

docker build -t ghcr.io/your-username/supply-chain-forecaster-dashboard:1.0.0 \
  --target dashboard-production .

docker push ghcr.io/your-username/supply-chain-forecaster-api:1.0.0
docker push ghcr.io/your-username/supply-chain-forecaster-dashboard:1.0.0
```

### 2. Apply Kubernetes Manifests

```bash
# Update image references in manifests
sed -i "s|image: ghcr.io/coop-columb/supply-chain-forecaster-api:latest|image: ghcr.io/your-username/supply-chain-forecaster-api:1.0.0|g" k8s/api-deployment.yaml
sed -i "s|image: ghcr.io/coop-columb/supply-chain-forecaster-dashboard:latest|image: ghcr.io/your-username/supply-chain-forecaster-dashboard:1.0.0|g" k8s/dashboard-deployment.yaml

# Apply manifests
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/dashboard-deployment.yaml
kubectl apply -f k8s/ingress.yaml
```

### 3. Verify Deployment

```bash
# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services
kubectl get ingress
```

## Configuration

The deployment can be customized by modifying the following files:

- `api-deployment.yaml`: API service deployment and service configuration
- `dashboard-deployment.yaml`: Dashboard deployment and service configuration
- `ingress.yaml`: Ingress rules for routing external traffic

## Environment Variables

- `ENV`: Set to "production" for production deployments
- `API_URL`: The URL of the API service (for dashboard configuration)
- `LOG_LEVEL`: The logging level (INFO, DEBUG, etc.)

## Scaling

To scale the deployments:

```bash
kubectl scale deployment supply-chain-api --replicas=5
kubectl scale deployment supply-chain-dashboard --replicas=3
```

## Monitoring and Observability

The deployments include health check endpoints configured as readiness and liveness probes for Kubernetes monitoring.

A complete monitoring solution has been implemented:

- **Prometheus Metrics**: The API and Dashboard expose metrics endpoints on `/metrics`
- **Grafana Dashboards**: Pre-configured dashboards for visualizing performance metrics
- **Structured Logging**: JSON-formatted logs with distributed tracing via request IDs
- **Alert Rules**: Configured in Prometheus for critical conditions

### Deploying Monitoring Stack

To deploy the monitoring components to Kubernetes:

```bash
# Create Prometheus and Grafana deployments
kubectl apply -f k8s/prometheus-deployment.yaml
kubectl apply -f k8s/grafana-deployment.yaml

# Create configuration ConfigMaps
kubectl create configmap prometheus-config --from-file=monitoring/prometheus/
kubectl create configmap grafana-dashboards --from-file=monitoring/grafana/provisioning/dashboards/
kubectl create configmap grafana-datasources --from-file=monitoring/grafana/provisioning/datasources/
```

For detailed information about the monitoring implementation, refer to the [Monitoring and Observability Guide](../docs/deployment/monitoring.md).

## Troubleshooting

If you encounter issues with the deployment:

1. Check pod logs: `kubectl logs <pod-name>`
2. Check pod events: `kubectl describe pod <pod-name>`
3. Verify service connectivity: `kubectl exec -it <pod-name> -- curl http://supply-chain-api/health`