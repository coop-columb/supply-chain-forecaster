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

## Monitoring

The deployments include readiness and liveness probes for basic health monitoring.

For advanced monitoring, consider implementing:

- Prometheus for metrics collection
- Grafana for visualization
- ELK stack for log aggregation

## Troubleshooting

If you encounter issues with the deployment:

1. Check pod logs: `kubectl logs <pod-name>`
2. Check pod events: `kubectl describe pod <pod-name>`
3. Verify service connectivity: `kubectl exec -it <pod-name> -- curl http://supply-chain-api/health`