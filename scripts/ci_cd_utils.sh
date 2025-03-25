#!/bin/bash
# Utility functions for CI/CD workflows

# Function to force simulation mode for all CD pipeline runs
# TEMPORARY SOLUTION: This should be replaced with real Kubernetes deployments
# as part of the Phase 9 work on the roadmap
force_simulation_mode() {
    # Create or update the GitHub repository secrets to disable real mode
    # We'll set empty values for the Kubernetes configs to trigger simulation mode
    
    echo "⚠️  TEMPORARY SOLUTION: Configuring CD pipeline to use simulation mode..."
    echo "Note: This is intended as a temporary solution until real Kubernetes infrastructure is set up."
    echo "      See the ROADMAP.md for the planned implementation of real deployments."
    
    # For staging environment
    gh secret set KUBE_CONFIG_STAGING -e staging --body=""
    
    # For production environment
    gh secret set KUBE_CONFIG_PRODUCTION -e production --body=""
    
    echo "✅ CD pipeline is now configured to use simulation mode."
    echo "When you're ready to implement real Kubernetes deployments:"
    echo "1. Run scripts/setup_local_k8s.sh to set up a local development cluster, or"
    echo "2. Set up a cloud-based Kubernetes cluster (see docs/deployment/kubernetes_setup.md)"
    echo "3. Run ./scripts/ci_cd_utils.sh reset-simulation to disable simulation mode"
}

# Function to reset the simulation mode override
reset_simulation_mode() {
    echo "⚠️ This will reset the simulation mode override."
    echo "Make sure you have a real Kubernetes cluster set up before proceeding."
    read -p "Continue? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Delete the GitHub secrets to trigger a prompt for re-creation
        gh secret delete KUBE_CONFIG_STAGING -e staging
        gh secret delete KUBE_CONFIG_PRODUCTION -e production
        
        echo "✅ Simulation mode override has been reset."
        echo "To set up a real Kubernetes cluster, run scripts/setup_local_k8s.sh"
    else
        echo "Operation cancelled."
    fi
}

# Main command processor
case "$1" in
    force-simulation)
        force_simulation_mode
        ;;
    reset-simulation)
        reset_simulation_mode
        ;;
    *)
        echo "Usage: $0 {force-simulation|reset-simulation}"
        exit 1
        ;;
esac

exit 0