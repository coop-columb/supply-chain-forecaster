#!/bin/bash
# Script to clean up CD test files after setup is complete

# Stop on error
set -e

echo "Cleaning up CD test files..."

# Remove test files
rm -f scripts/generate_cd_secrets_test.sh
rm -f scripts/test_cd_workflow.sh
rm -f scripts/roadmap_update_template.md
rm -f scripts/github_environments_checklist.md
rm -f scripts/cleanup_cd_test_files.sh

# Remove the .github_environment_setup directory if it exists
if [ -d ".github_environment_setup" ]; then
  echo "Removing .github_environment_setup directory..."
  rm -rf .github_environment_setup
fi

echo "Cleanup complete. You can add and commit the changes if needed."
echo "git add scripts/"
echo "git commit -m \"chore: remove CD test files\""
echo "git push"