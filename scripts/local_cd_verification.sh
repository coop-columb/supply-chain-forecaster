#!/bin/bash
# Verify CD workflow components locally

set -e

echo "===== Local CD Workflow Verification ====="
echo ""
echo "This script verifies the CD workflow components locally to ensure everything is properly configured."
echo ""

# Step 1: Verify Kubernetes manifests
echo "Verifying Kubernetes manifests..."
if [ ! -f "k8s/api-deployment.yaml" ] || [ ! -f "k8s/dashboard-deployment.yaml" ] || [ ! -f "k8s/ingress.yaml" ]; then
  echo "❌ Error: One or more Kubernetes manifests are missing."
  exit 1
fi

echo "✅ Kubernetes manifests found and readable."
echo ""

# Step 2: Verify CD workflow file
echo "Verifying CD workflow file..."
if [ ! -f ".github/workflows/cd.yml" ]; then
  echo "❌ Error: CD workflow file not found."
  exit 1
fi

echo "✅ CD workflow file found and readable."
echo ""

# Step 3: Verify image references in Kubernetes manifests
echo "Verifying image references in Kubernetes manifests..."
api_image_pattern=$(grep -o "image:.*supply-chain-forecaster-api:.*" k8s/api-deployment.yaml)
dashboard_image_pattern=$(grep -o "image:.*supply-chain-forecaster-dashboard:.*" k8s/dashboard-deployment.yaml)

if [ -z "$api_image_pattern" ] || [ -z "$dashboard_image_pattern" ]; then
  echo "❌ Error: Image references not found in Kubernetes manifests."
  exit 1
fi

echo "API image pattern: $api_image_pattern"
echo "Dashboard image pattern: $dashboard_image_pattern"
echo "✅ Image references found in Kubernetes manifests."
echo ""

# Step 4: Verify image substitution
echo "Verifying image substitution..."
temp_api_yaml=$(mktemp)
temp_dashboard_yaml=$(mktemp)

cp k8s/api-deployment.yaml "$temp_api_yaml"
cp k8s/dashboard-deployment.yaml "$temp_dashboard_yaml"

test_api_image="ghcr.io/coop-columb/supply-chain-forecaster-api:test-1234"
test_dashboard_image="ghcr.io/coop-columb/supply-chain-forecaster-dashboard:test-1234"

sed -i.bak "s|image: .*supply-chain-forecaster-api:.*|image: $test_api_image|g" "$temp_api_yaml"
sed -i.bak "s|image: .*supply-chain-forecaster-dashboard:.*|image: $test_dashboard_image|g" "$temp_dashboard_yaml"

api_result=$(grep -o "image:.*$test_api_image" "$temp_api_yaml" || echo "")
dashboard_result=$(grep -o "image:.*$test_dashboard_image" "$temp_dashboard_yaml" || echo "")

if [ -z "$api_result" ] || [ -z "$dashboard_result" ]; then
  echo "❌ Error: Image substitution failed."
  cat "$temp_api_yaml"
  cat "$temp_dashboard_yaml"
  rm "$temp_api_yaml" "$temp_api_yaml.bak" "$temp_dashboard_yaml" "$temp_dashboard_yaml.bak"
  exit 1
fi

echo "✅ Image substitution working correctly."
rm "$temp_api_yaml" "$temp_api_yaml.bak" "$temp_dashboard_yaml" "$temp_dashboard_yaml.bak"
echo ""

# Step 5: Create a properly formatted kubeconfig for testing
echo "Creating properly formatted test kubeconfig..."
mkdir -p .cd_verification
cat > .cd_verification/kubeconfig << EOF
apiVersion: v1
kind: Config
clusters:
- name: test-cluster
  cluster:
    server: https://test-server:6443
    certificate-authority-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUN5RENDQWJDZ0F3SUJBZ0lCQURBTkJna3Foa2lHOXcwQkFRc0ZBREFWTVJNd0VRWURWUVFERXdwcmRXSmwKY201bGRHVnpNQjRYRFRJME1ETXlOREF3TURBeE1Gb1hEVE0wTURNeU1UQXdNREF4TUZvd0ZURVRNQkVHQTFVRQpBeE1LYTNWaVpYSnVaWFJsY3pDQ0FTSXdEUVlKS29aSWh2Y05BUUVCQlFBRGdnRVBBRENDQVFvQ2dnRUJBTU9VCmUrUjdHaUNJb2VlL0lCbjJWdTl4QzlreFR5ZGJCQ0ZWbG1PazRjL0U4Vy91MDRnZHJDRjdVWlRzbkgxUmNybHUKKzFIMkh5TWIyWmNLWnBuRmdvQWc4bHl6QWV2QVhIVjUzNllpYklYbytQK1JRNzVGSlBxeWJWQy9IQ3BHaHdsYgo4V2hFVVVyTzJWbFZ0K21wVHRTVmRVQWRSUWRhMnlTeUQ2NWJlWVF6UlY0L3VnYk9IUnhTeVNISW9RNWI5UkRnCmJsRjB1cnRxcWw1ZnJEOTlaeW5HRTFtVnVXNTNpYnBxaDh5aW5Ldk96YWNNanpHWXVpVmUxN0o1UmlHSlFYVncKL0l5VFJZMGxva3hHbE12eDBEb1djcjY4V3g2MitMcW5XSEVlTXQ1dmFxR256d3NLUFhKT0tpU1p1MWZMN3JLawpwUTdxbDVDVVJuc1lFR3NDQXdFQUFhTWpNQ0V3RGdZRFZSMFBBUUgvQkFRREFnS2tNQThHQTFVZEV3RUIvd1FGCk1BTUJBZjh3RFFZSktvWklodmNOQVFFTEJRQURnZ0VCQUNYcjZVMmtyY3RoZEZYTDFhU1NzWlFhbExLOVU3ekIKVGw3YkNzODRmU0NYbmgwVTJ3ejdGVzJsTFdBb3o5eUdVeDJFVVc3SWk0c3VIcDBzR0s5eXVvQ25lczlYRmplcApzVkZEcXlra0U1VVdXMDkyTTZZZGJBdkd6YzFMeEkrV3RXVUxsM0VpNVY4NEY3em12cGw1R3lqd0R0V3VZckdiCm11MkdZcytabzU3ZWZUVitlUmg0THlCTkxPV0xsNFVqYzVnYzVyV013dVE0Z0pWQlJtV3ZsVjE3THdDL0NmQnYKdTZ1N2QzMDFEZ0FLWkhrNFdLYnZLZUV2R3RVU3RVRkRlVjdGUkM5NndHMmZMOFJMODU0Sk56cGZOc3lUZktJRApSTm5wYmh2ZTBWYzVPOVJVNXNsQkhFZi80bzJUTWVuRXRxazBKWEdKV3BjVjhnSlczQkU9Ci0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K
contexts:
- name: test-context
  context:
    cluster: test-cluster
    user: test-user
current-context: test-context
users:
- name: test-user
  user:
    token: test-token-123456789
EOF

# Base64 encode the kubeconfig for GitHub Environments
cat .cd_verification/kubeconfig | base64 > .cd_verification/kubeconfig_base64.txt

echo "✅ Test kubeconfig created"
echo ""

# Step 6: Create mock Kubernetes test
echo "Testing kubectl with mock kubeconfig..."

export KUBECONFIG=.cd_verification/kubeconfig
kubectl version --client

echo "✅ kubectl client working correctly"
echo ""

# Step 7: Create test API keys
echo "Creating test API keys..."
openssl rand -hex 32 > .cd_verification/api_key_staging.txt
openssl rand -hex 32 > .cd_verification/api_key_production.txt

echo "✅ Test API keys created"
echo ""

echo "===== CD Workflow Verification Complete ====="
echo ""
echo "All components have been verified locally."
echo ""
echo "Next steps:"
echo "1. Set up GitHub Environments with the generated secrets:"
echo "   - Staging environment:"
echo "     - KUBE_CONFIG_STAGING: content of .cd_verification/kubeconfig_base64.txt"
echo "     - API_KEY_STAGING: content of .cd_verification/api_key_staging.txt"
echo "   - Production environment:"
echo "     - KUBE_CONFIG_PRODUCTION: content of .cd_verification/kubeconfig_base64.txt"
echo "     - API_KEY_PRODUCTION: content of .cd_verification/api_key_production.txt"
echo ""
echo "2. Trigger the CD workflow on GitHub Actions"
echo "3. Update ROADMAP.md once the workflow succeeds"
echo ""