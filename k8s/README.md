# Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the Supply Chain Forecaster to a Kubernetes cluster.

## Prerequisites

- A Kubernetes cluster
- kubectl configured to communicate with your cluster
- A container registry where you can push Docker images

## Deployment Steps

### 1. Build and Push Docker Images

```bash
# Set version and registry
export VERSION=1.0.0
export DOCKER_REGISTRY=your-registry.example.com

# Build API image
docker build -t ${DOCKER_REGISTRY}/supply-chain-forecaster-api:${VERSION} \
  --target production \
  --build-arg SERVICE=api .

# Build Dashboard image
docker build -t ${DOCKER_REGISTRY}/supply-chain-forecaster-dashboard:${VERSION} \
  --target production \
  --build-arg SERVICE=dashboard .

# Push images to registry
docker push ${DOCKER_REGISTRY}/supply-chain-forecaster-api:${VERSION}
docker push ${DOCKER_REGISTRY}/supply-chain-forecaster-dashboard:${VERSION}
```

### 2. Apply Kubernetes Manifests

```bash
# Replace placeholder variables in manifests
envsubst < k8s/api-deployment.yaml | kubectl apply -f -
envsubst < k8s/dashboard-deployment.yaml | kubectl apply -f -
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