#!/bin/bash
# Run the application with profiling enabled

# Set environment variables for profiling
export ENABLE_PROFILING=true
export PROFILING_SAMPLE_RATE=1.0  # Profile all requests during the test

# Determine the parent directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Starting API with profiling enabled..."
echo "API will be available at http://localhost:8000"
echo "Press Ctrl+C to stop the server"

# Start the API server
cd "$PROJECT_ROOT" && python -m api.main

# Note: This script only runs the API.
# To collect profiling data for both API and dashboard, you would need to run both processes.

# After running this script, you can access profiling data at:
# http://localhost:8000/profiling/stats

# To run the profile_application.py script after collecting some data:
# python scripts/profile_application.py