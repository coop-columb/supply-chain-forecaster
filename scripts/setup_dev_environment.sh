#!/bin/bash
# Script to set up the development environment for Supply Chain Forecaster

# Stop on error
set -e

echo "Setting up development environment for Supply Chain Forecaster..."

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Run pre-commit hooks on all files to ensure everything is clean
echo "Running pre-commit hooks on all files..."
pre-commit run --all-files || {
    echo "Some pre-commit hooks failed. You may need to fix these issues before committing."
    echo "You can run 'pre-commit run --all-files' again after making changes."
}

echo "Development environment setup complete!"
echo ""
echo "To activate this environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests, use:"
echo "  pytest"
echo ""
echo "To check code formatting, use:"
echo "  black --check ."
echo "  isort --check --profile black ."
echo ""
echo "Happy coding!"