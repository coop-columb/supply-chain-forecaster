"""Model service for the supply chain forecaster API."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from models.base import ModelBase, ModelRegistry
from utils import ModelError, NotFoundError, get_logger, safe_execute

logger = get_logger(__name__)


class ModelService:
    """Service for managing models in the API."""

    def __init__(
        self,
        model_registry: ModelRegistry = None,
        training_dir: Optional[Union[str, Path]] = None,
        deployment_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the model service.
        
        Args:
            model_registry: Model registry instance.
            training_dir: Directory containing training models.
            deployment_dir: Directory containing deployed models.
        """
        from config import config
        
        self.model_registry = model_registry or ModelRegistry()
        self.training_dir = Path(training_dir) if training_dir else config.TRAINING_MODELS_DIR
        self.deployment_dir = Path(deployment_dir) if deployment_dir else config.DEPLOYMENT_MODELS_DIR
        
        logger.info(
            f"Initialized model service with training directory '{self.training_dir}' "
            f"and deployment directory '{self.deployment_dir}'"
        )

    def get_available_models(self) -> List[str]:
        """
        Get list of available model types.
        
        Returns:
            List of model type names.
        """
        return self.model_registry.list_models()

    def create_model(self, model_type: str, **params) -> ModelBase:
        """
        Create a new model instance.
        
        Args:
            model_type: Type of model to create.
            **params: Model parameters.
        
        Returns:
            Initialized model instance.
        """
        try:
            return self.model_registry.create_model(model_type, **params)
        except NotFoundError:
            raise ModelError(f"Unknown model type: {model_type}", model_name=model_type)
        except Exception as e:
            raise ModelError(
                f"Error creating model: {str(e)}", model_name=model_type
            )

    def get_trained_models(self) -> Dict[str, Dict]:
        """
        Get list of trained models.
        
        Returns:
            Dictionary of model information.
        """
        return self.model_registry.find_models(self.training_dir)

    def get_deployed_models(self) -> Dict[str, Dict]:
        """
        Get list of deployed models.
        
        Returns:
            Dictionary of model information.
        """
        return self.model_registry.find_models(self.deployment_dir)

    def load_model(
        self, model_name: str, model_type: str, from_deployment: bool = True
    ) -> ModelBase:
        """
        Load a model by name.
        
        Args:
            model_name: Name of the model to load.
            model_type: Type of the model.
            from_deployment: Whether to load from deployment or training directory.
        
        Returns:
            Loaded model instance.
        """
        logger.info(f"Loading model '{model_name}' of type '{model_type}'")
        
        # Determine which directory to look in
        model_dir = self.deployment_dir if from_deployment else self.training_dir
        
        # Try to find the model file
        model_files = list(model_dir.glob(f"{model_name}*.joblib"))
        
        if not model_files:
            raise ModelError(
                f"Model '{model_name}' not found in {model_dir}", model_name=model_name
            )
        
        # Use the most recent model file if multiple exist
        model_file = sorted(model_files)[-1]
        
        logger.info(f"Found model file: {model_file}")
        
        # Get the model class
        try:
            model_class = self.model_registry.get_model(model_type)
        except NotFoundError:
            raise ModelError(f"Unknown model type: {model_type}", model_name=model_name)
        
        # Load the model
        try:
            model = model_class.load(model_file)
            logger.info(f"Successfully loaded model '{model_name}'")
            return model
        except Exception as e:
            raise ModelError(
                f"Error loading model: {str(e)}", model_name=model_name
            )

    def save_model(
        self, model: ModelBase, model_name: Optional[str] = None
    ) -> Dict:
        """
        Save a model to the training directory.
        
        Args:
            model: Model to save.
            model_name: Optional name to use for the model.
        
        Returns:
            Model metadata.
        """
        if model_name:
            model.name = model_name
        
        logger.info(f"Saving model '{model.name}'")
        
        # Create directory if it doesn't exist
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate model path
        model_path = self.training_dir / f"{model.name}.joblib"
        
        # Save the model
        try:
            saved_path = model.save(model_path)
            logger.info(f"Successfully saved model to {saved_path}")
            return model.metadata
        except Exception as e:
            raise ModelError(
                f"Error saving model: {str(e)}", model_name=model.name
            )

    def deploy_model(self, model_name: str) -> Dict:
        """
        Deploy a model from training to deployment directory.
        
        Args:
            model_name: Name of the model to deploy.
        
        Returns:
            Deployed model metadata.
        """
        logger.info(f"Deploying model '{model_name}'")
        
        # Find the model file
        model_files = list(self.training_dir.glob(f"{model_name}*.joblib"))
        
        if not model_files:
            raise ModelError(
                f"Model '{model_name}' not found in {self.training_dir}", model_name=model_name
            )
        
        # Use the most recent model file if multiple exist
        model_file = sorted(model_files)[-1]
        
        # Deploy the model
        try:
            deployed_path = self.model_registry.deploy_model(model_file, self.deployment_dir)
            logger.info(f"Successfully deployed model to {deployed_path}")
            
            # Load metadata to return
            metadata_path = deployed_path.with_suffix(".json")
            if metadata_path.exists():
                import json
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                return metadata
            else:
                return {"name": model_name, "path": str(deployed_path)}
        except Exception as e:
            raise ModelError(
                f"Error deploying model: {str(e)}", model_name=model_name
            )

    def delete_model(self, model_name: str, from_deployment: bool = False) -> bool:
        """
        Delete a model.
        
        Args:
            model_name: Name of the model to delete.
            from_deployment: Whether to delete from deployment or training directory.
        
        Returns:
            True if model was deleted.
        """
        logger.info(f"Deleting model '{model_name}'")
        
        # Determine which directory to look in
        model_dir = self.deployment_dir if from_deployment else self.training_dir
        
        # Find all files related to the model
        model_files = list(model_dir.glob(f"{model_name}*"))
        
        if not model_files:
            raise ModelError(
                f"Model '{model_name}' not found in {model_dir}", model_name=model_name
            )
        
        # Delete the files
        try:
            for file in model_files:
                file.unlink()
                logger.info(f"Deleted {file}")
            
            return True
        except Exception as e:
            raise ModelError(
                f"Error deleting model: {str(e)}", model_name=model_name
            )
