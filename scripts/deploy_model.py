#!/usr/bin/env python
"""
Model deployment script for CI/CD pipeline.

This script deploys models that have passed evaluation to the production environment.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deploy models to production.")
    parser.add_argument(
        "--evaluation-file",
        type=str,
        default="models/evaluation_results.json",
        help="File containing evaluation results",
    )
    parser.add_argument(
        "--production-dir",
        type=str,
        default="models/production",
        help="Directory for production models",
    )
    parser.add_argument(
        "--deployment-log",
        type=str,
        default="models/deployment_history.json",
        help="Log file for deployment history",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force deployment even if models failed evaluation",
    )
    return parser.parse_args()


def load_evaluation_results(evaluation_file):
    """Load model evaluation results from file."""
    if not os.path.exists(evaluation_file):
        logger.error(f"Evaluation file {evaluation_file} not found.")
        return None

    try:
        with open(evaluation_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing evaluation file: {e}")
        return None


def update_deployment_log(deployment_log, deployed_models):
    """Update the deployment history log file."""
    # Create log entry
    deployment_entry = {
        "timestamp": datetime.now().isoformat(),
        "models": deployed_models,
    }

    # Load existing log if it exists
    if os.path.exists(deployment_log):
        try:
            with open(deployment_log, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = {"deployments": []}
    else:
        history = {"deployments": []}

    # Add new entry
    history["deployments"].append(deployment_entry)

    # Write updated log
    os.makedirs(os.path.dirname(deployment_log), exist_ok=True)
    with open(deployment_log, "w") as f:
        json.dump(history, f, indent=2)


def deploy_models(models_to_deploy, production_dir):
    """Deploy models to the production directory."""
    deployed_models = []

    # Create production directory if it doesn't exist
    os.makedirs(production_dir, exist_ok=True)

    # Create subdirectories for each model type
    os.makedirs(os.path.join(production_dir, "forecasting"), exist_ok=True)
    os.makedirs(os.path.join(production_dir, "anomaly"), exist_ok=True)

    # Deploy each model
    for model_path in models_to_deploy:
        try:
            # Determine model type from path
            if "forecasting" in model_path:
                model_type = "forecasting"
            elif "anomaly" in model_path:
                model_type = "anomaly"
            else:
                logger.warning(f"Unknown model type for {model_path}, skipping")
                continue

            # Get model filename
            model_filename = os.path.basename(model_path)

            # Copy model file to production directory
            dest_path = os.path.join(production_dir, model_type, model_filename)
            shutil.copy2(model_path, dest_path)

            # Also copy metadata if it exists
            metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
            if os.path.exists(metadata_path):
                metadata_dest = os.path.join(
                    production_dir,
                    model_type,
                    f"{os.path.splitext(model_filename)[0]}_metadata.json",
                )
                shutil.copy2(metadata_path, metadata_dest)

            logger.info(f"Deployed model {model_path} to {dest_path}")
            deployed_models.append(
                {
                    "source": model_path,
                    "destination": dest_path,
                    "model_type": model_type,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            logger.error(f"Error deploying model {model_path}: {e}")

    return deployed_models


def create_deployment_manifest(production_dir, deployed_models):
    """Create a manifest file listing all deployed models."""
    manifest = {"deployed_at": datetime.now().isoformat(), "models": {}}

    # Group models by type
    for model in deployed_models:
        model_type = model["model_type"]
        if model_type not in manifest["models"]:
            manifest["models"][model_type] = []

        manifest["models"][model_type].append(
            {
                "file": os.path.basename(model["destination"]),
                "deployed_at": model["timestamp"],
            }
        )

    # Write manifest file
    manifest_path = os.path.join(production_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Created deployment manifest at {manifest_path}")


def main():
    """Main function to deploy models."""
    args = parse_arguments()

    # Load evaluation results
    results = load_evaluation_results(args.evaluation_file)

    if not results:
        logger.error("No evaluation results found. Aborting deployment.")
        return 1

    # Get list of models to deploy
    if args.force:
        logger.warning("Forcing deployment of all evaluated models")
        # Deploy all evaluated models if forced
        models_to_deploy = [
            m["model_path"] for m in results.get("evaluated_models", [])
        ]
    else:
        # Only deploy models that passed evaluation
        models_to_deploy = results.get("passed_models", [])

    if not models_to_deploy:
        logger.warning("No models to deploy. Deployment skipped.")
        return 0

    logger.info(f"Deploying {len(models_to_deploy)} models to production")

    # Deploy models
    deployed_models = deploy_models(models_to_deploy, args.production_dir)

    if not deployed_models:
        logger.error("No models were deployed. Deployment failed.")
        return 1

    # Create deployment manifest
    create_deployment_manifest(args.production_dir, deployed_models)

    # Update deployment log
    update_deployment_log(args.deployment_log, deployed_models)

    logger.info(f"Successfully deployed {len(deployed_models)} models to production")
    return 0


if __name__ == "__main__":
    sys.exit(main())
