# Investigating Local Kubernetes Integration with GitHub Actions

This document outlines our ongoing investigation into connecting local Kubernetes clusters with GitHub Actions for development and testing purposes.

## Background

Our CD pipeline currently supports:
1. **Simulation mode** - for development and testing without real infrastructure
2. **Real deployment mode** - for deploying to cloud-based Kubernetes clusters (GKE, AKS, EKS)

However, we're exploring options to enable GitHub Actions to interact with locally hosted Kubernetes clusters, which would provide more flexibility for development and testing.

## Current Challenges

The primary challenge is that GitHub Actions runners operate in a remote environment, which cannot directly access services running on your local machine, such as a local Kubernetes cluster created with tools like k3d, minikube, or kind.

Our previous attempts included:
- Providing the kubeconfig from a local cluster to GitHub Actions via environment secrets
- Adjusting the kubeconfig format and encoding
- Various other configuration approaches

However, these approaches did not succeed because:
1. The GitHub Actions runner environment is isolated from your local network
2. Local Kubernetes clusters typically bind to localhost or private IPs
3. Authentication credentials in the kubeconfig may not be transferable

## Potential Solutions to Investigate

### 1. GitHub Actions Self-Hosted Runners

Self-hosted runners allow you to run GitHub Actions workflows on your own infrastructure, which could potentially access local Kubernetes clusters.

**Implementation Approach:**
- Set up a self-hosted runner on the same machine as your local Kubernetes cluster
- Configure the runner to have access to the local kubeconfig
- Use this runner for Kubernetes-related workflows

**Considerations:**
- Security implications of exposing self-hosted runners
- Resource requirements for running workflows locally
- Managing the lifecycle of self-hosted runners

### 2. Secure Tunneling Solutions

Create a secure tunnel that exposes your local Kubernetes API server to the internet temporarily.

**Implementation Approach:**
- Use tools like ngrok, inlets, or Cloudflare Tunnel to create a secure tunnel
- Expose the Kubernetes API server via the tunnel
- Update the kubeconfig to use the public endpoint

**Considerations:**
- Security implications of exposing the Kubernetes API
- Potential latency and bandwidth limitations
- Managing tunnel lifecycle and authentication

### 3. API Proxy or Gateway

Create a proxy server that sits between GitHub Actions and your local Kubernetes cluster.

**Implementation Approach:**
- Deploy a proxy server in a cloud environment
- Configure the proxy to forward requests to your local Kubernetes cluster
- Use a secure connection between the proxy and your local environment

**Considerations:**
- Complexity of setup and maintenance
- Security and authentication between proxy and local cluster
- Performance impact of proxying all requests

### 4. Kubernetes Federation

Use Kubernetes federation to create a multi-cluster setup with both cloud and local clusters.

**Implementation Approach:**
- Set up a federation control plane in a cloud environment
- Join both cloud and local clusters to the federation
- Configure GitHub Actions to interact with the federation control plane

**Considerations:**
- Significant complexity in setup and management
- Potential limitations in federation capabilities
- Resource overhead of running federation components

## Next Steps

Our immediate action items for investigating these approaches:

1. **Self-Hosted Runners Evaluation** (Highest Priority)
   - Set up a test environment with a self-hosted runner
   - Create test workflows that interact with a local Kubernetes cluster
   - Document findings and implications

2. **Secure Tunneling Evaluation**
   - Test ngrok and Cloudflare Tunnel for exposing Kubernetes API
   - Assess security implications and develop mitigation strategies
   - Document setup process and limitations

3. **Research Other Projects' Approaches**
   - Identify open-source projects with similar requirements
   - Study their solutions for local development integration with CI/CD
   - Adapt relevant approaches to our needs

## References

- [GitHub Actions Self-Hosted Runners Documentation](https://docs.github.com/en/actions/hosting-your-own-runners/about-self-hosted-runners)
- [ngrok Documentation](https://ngrok.com/docs)
- [Cloudflare Tunnel Documentation](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps)
- [Kubernetes Federation Documentation](https://kubernetes.io/docs/concepts/architecture/control-plane-node-communication/)

## Conclusion

While cloud-based Kubernetes clusters remain the recommended approach for production deployments, we are actively investigating these alternative approaches for local development and testing integration with GitHub Actions. Our findings will be documented here as we explore each option.

Last updated: March 25, 2025