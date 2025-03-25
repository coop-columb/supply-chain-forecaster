#!/bin/bash
# Script to clean up temporary files created during development and testing

set -e

echo "=== Supply Chain Forecaster: Temporary Files Cleanup ==="
echo ""

# List of directories and files to clean up
TEMP_DIRS=(
  ".github_environment_setup"
  ".cd_workflow_test"
)

echo "The following temporary directories will be removed:"
for dir in "${TEMP_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    echo "- $dir"
  else
    echo "- $dir (not found)"
  fi
done

echo ""
echo "This script requires sudo privileges to remove some files with restricted permissions."
echo "Please run with sudo to ensure all files can be removed properly."
echo ""

read -p "Continue with cleanup? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Cleanup cancelled."
  exit 0
fi

# Remove temp directories
for dir in "${TEMP_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    echo "Removing $dir..."
    rm -rf "$dir" || { echo "Failed to remove $dir, try running with sudo"; }
  fi
done

echo ""
echo "Cleanup completed!"
echo ""
echo "Note: If any errors were reported, please run this script with sudo:"
echo "sudo ./scripts/cleanup_temp_files.sh"

exit 0